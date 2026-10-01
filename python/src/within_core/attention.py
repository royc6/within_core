"""Gemma 4 self-attention with host-held KV.

Pipeline position: one :class:`Attention` per decoder block (:mod:`within_core.chunks`).
The graph never owns or updates a cache. It reads the host's cache as a read-only input,
appends this step's fresh K/V inside the graph, attends, and returns the fresh K/V so that
the host can write them into its cache after the call. Layers past the sharing boundary
compute no K/V; they attend over their donor layer's cache plus the donor's fresh K/V.

Math reference: Transformers 5.12.1 ``Gemma4TextAttention`` (Apache-2.0).
"""

import torch
from torch import Tensor, nn

from within_core.layers import RMSNorm, projection


class Attention(nn.Module):
    """Grouped-query attention in channels-first layout.

    Queries are ``[B, D, heads, S]`` and caches are ``[B, kv_heads, D, K]``. Attention is
    computed per KV head with all of its query heads at once: two matrix products against
    the channels-first cache, never a per-query-head copy of it, which keeps the
    full-attention layers cheap at long contexts.
    """

    def __init__(self, reference: nn.Module) -> None:
        """Initialize from a Transformers ``Gemma4TextAttention``.

        Args:
            reference: The source attention module.

        Raises:
            ValueError: If the layer ties its global K and V projections (not used by E4B).
        """
        super().__init__()
        if reference.use_alternative_attention:
            raise ValueError("Tied global K/V projections are not supported")
        self.head_dim = reference.head_dim
        self.heads = reference.config.num_attention_heads
        self.kv_heads = reference.config.num_key_value_heads
        self.shared = reference.is_kv_shared_layer
        self.q = projection(reference.q_proj)
        self.o = projection(reference.o_proj)
        self.q_norm = RMSNorm.from_module(reference.q_norm)
        if not self.shared:
            self.k = projection(reference.k_proj)
            self.v = projection(reference.v_proj)
            self.k_norm = RMSNorm.from_module(reference.k_norm)
            eps = reference.config.rms_norm_eps
            self.v_norm = RMSNorm.unit(self.head_dim, eps, reference.v_proj.weight)

    def split_heads(self, x: Tensor, heads: int) -> Tensor:
        """Reshape ``[B, heads * D, 1, S]`` to ``[B, D, heads, S]``.

        Channels hold the head dimension so that the per-head norms reduce over channels.
        """
        return x.reshape(x.shape[0], heads, self.head_dim, x.shape[-1]).transpose(1, 2)

    def rotary(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply rotary position embedding along the head dimension.

        The host supplies full-width cosine and sine vectors for the layer's RoPE variant
        (default RoPE for sliding layers, proportional partial RoPE for global layers),
        including the unrotated axes, so one formula serves both.
        """
        half = self.head_dim // 2
        rotated = torch.cat((-x[:, half:], x[:, :half]), dim=1)
        return x * cos + rotated * sin

    def forward(
        self,
        hidden: Tensor,
        cos: Tensor,
        sin: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Attend from ``hidden`` over the cache and this step's keys.

        Args:
            hidden: Normalized input, ``[B, H, 1, S]``.
            cos: Rotary cosines, ``[B, D, 1, S]``.
            sin: Rotary sines, ``[B, D, 1, S]``.
            key_cache: Read-only keys, ``[B, kv_heads, D, K]``. For a shared layer, the
                donor's cache already followed by the donor's fresh keys.
            value_cache: Read-only values, same layout as ``key_cache``.
            mask: Additive mask ``[B, 1, S, K + S]`` (``[B, 1, S, K]`` for a shared layer).

        Returns:
            The attention output ``[B, H, 1, S]`` and the fresh keys and values
            ``[B, kv_heads, D, S]``, or ``None`` for a shared layer.
        """
        query = self.split_heads(self.q(hidden), self.heads)
        query = self.rotary(self.q_norm(query), cos, sin)
        if self.shared:
            context = self.attend(query, key_cache, value_cache, mask)
            return self.o(context), None, None
        fresh_key = self.k_norm(self.split_heads(self.k(hidden), self.kv_heads))
        fresh_key = self.rotary(fresh_key, cos, sin)
        fresh_value = self.v_norm(self.split_heads(self.v(hidden), self.kv_heads))
        fresh_key = fresh_key.transpose(1, 2)
        fresh_value = fresh_value.transpose(1, 2)
        key = torch.cat((key_cache, fresh_key), dim=-1)
        value = torch.cat((value_cache, fresh_value), dim=-1)
        context = self.attend(query, key, value, mask)
        return self.o(context), fresh_key, fresh_value

    def attend(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor) -> Tensor:
        """Return the attention context ``[B, heads * D, 1, S]``.

        Gemma 4 uses a softmax scale of 1.0; its query and key norms take the place of the
        usual ``1 / sqrt(D)``.
        """
        groups = self.heads // self.kv_heads
        mask = mask.permute(0, 3, 1, 2)
        batch, dimension, _, length = query.shape
        outputs = []
        for kv_head in range(self.kv_heads):
            heads = query[:, :, kv_head * groups : (kv_head + 1) * groups]
            q = heads.reshape(batch, dimension, groups * length)
            k, v = key[:, kv_head], value[:, kv_head]
            scores = torch.matmul(k.transpose(1, 2), q).reshape(batch, -1, groups, length) + mask
            probabilities = torch.softmax(scores, dim=1)
            context = torch.matmul(v, probabilities.reshape(batch, -1, groups * length))
            context = context.reshape(batch, dimension, groups, length).permute(0, 2, 1, 3)
            outputs.append(context.reshape(batch, groups * dimension, 1, length))
        return torch.cat(outputs, dim=1)
