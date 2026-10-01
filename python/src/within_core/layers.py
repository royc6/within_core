"""Channels-first building blocks of the Gemma 4 decoder for the Neural Engine.

Pipeline position: :mod:`within_core.chunks` assembles decoder blocks and chunks from
these modules, and :mod:`within_core.export` quantizes and authors them. Activations use
the layout ``[batch, channels, 1, sequence]``, every projection is a 1x1 convolution, and
every reduction is written so that FP16 arithmetic on the Neural Engine stays in range.

Math reference: Transformers 5.12.1 ``Gemma4TextModel`` and ``Gemma4TextDecoderLayer``
(Apache-2.0).
"""

import torch
from coreai_models.primitives.ios.gelu import GELUReauthored
from torch import Tensor, nn


def pointwise(weight: Tensor, bias: Tensor | None = None) -> nn.Conv2d:
    """Return a 1x1 convolution that computes the linear map ``weight``.

    Args:
        weight: Linear weight of shape ``[out_features, in_features]``.
        bias: Optional bias of shape ``[out_features]``.

    Returns:
        A convolution with the weight's dtype and device, holding copies of the tensors.
    """
    conv = nn.Conv2d(weight.shape[1], weight.shape[0], 1, bias=bias is not None, device="meta")
    state = {"weight": weight.detach()[:, :, None, None].clone()}
    if bias is not None:
        state["bias"] = bias.detach().clone()
    conv.load_state_dict(state, assign=True)
    return conv


def projection(reference: nn.Linear) -> nn.Conv2d:
    """Return a 1x1 convolution equivalent to a Transformers linear layer."""
    return pointwise(reference.weight, reference.bias)


class RMSNorm(nn.Module):
    """RMS normalization over channels, written as a layer norm for FP16 hardware.

    ``[x, -x]`` has zero mean, so a layer norm over it equals an RMS norm over ``x``. The
    layer norm keeps its dedicated, accumulation-safe implementation through export (the
    same identity as CoreML-LLM's ``ANERMSNorm``, MIT). Scaling the mirrored input by a
    power of two scales epsilon by its square, which turns Gemma's 1e-6 into a normal FP16
    value without changing the equation.
    """

    def __init__(self, weight: Tensor, eps: float) -> None:
        """Initialize from a learned scale.

        Args:
            weight: Learned scale of shape ``[channels]``.
            eps: Epsilon of the RMS equation.
        """
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone().reshape(1, -1, 1, 1))
        self.eps = eps
        # The graph divides the normalization scale by this input scale. Trained Gemma 4
        # activations need no pre-scaling, so it is 1.0; the division stays in the graph
        # because it is part of the published program.
        self.input_scale = 1.0
        channels = self.weight.shape[1]
        self.register_buffer("zero_bias", self.weight.new_zeros(2 * channels), persistent=False)
        self.register_buffer("normalization_scale", self.weight.new_tensor(8.0), persistent=False)

    @classmethod
    def from_module(cls, reference: nn.Module) -> "RMSNorm":
        """Return the norm equivalent to a Transformers ``Gemma4RMSNorm``."""
        return cls(reference.weight, reference.eps)

    @classmethod
    def unit(cls, channels: int, eps: float, like: Tensor) -> "RMSNorm":
        """Return a norm without a learned scale (Gemma's value normalization).

        Args:
            channels: Number of normalized channels.
            eps: Epsilon of the RMS equation.
            like: Tensor whose dtype and device the unit scale takes.

        Returns:
            A norm whose scale is all ones.
        """
        return cls(like.new_ones(channels), eps)

    def forward(self, x: Tensor) -> Tensor:
        """Normalize ``x`` of shape ``[batch, channels, rows, sequence]`` over channels."""
        channels_last = x.permute(0, 2, 3, 1)
        mirrored = torch.cat((channels_last, -channels_last), dim=-1) * (
            self.normalization_scale / self.input_scale
        )
        # Fusing Gemma's learned scale into the layer norm rounds the normalized values to
        # FP16 once instead of twice.
        affine = self.weight.flatten().repeat(2)
        normalized = nn.functional.layer_norm(
            mirrored, (affine.numel(),), weight=affine, bias=self.zero_bias, eps=self.eps * 64.0
        )
        return normalized[..., : self.weight.shape[1]].permute(0, 3, 1, 2)


class PerLayerProjection(nn.Module):
    """Per-layer embedding (PLE) inputs for a range of decoder layers.

    Combines the context signal (a projection of the token embedding) with the token
    identity signal (the host's lookup in ``perLayerEmbeddings.f16``). The output has shape
    ``[batch, ple_dim, layers, sequence]``; layer ``i`` of the range reads row ``i``.
    """

    def __init__(
        self,
        projection_weight: Tensor,
        norm_weight: Tensor,
        *,
        eps: float,
        layers: int,
        projection_scale: float,
        input_scale: float,
    ) -> None:
        """Initialize from checkpoint tensors.

        Args:
            projection_weight: Rows of ``per_layer_model_projection`` for these layers,
                shape ``[layers * ple_dim, hidden]``.
            norm_weight: ``per_layer_projection_norm`` scale, shape ``[ple_dim]``.
            eps: Epsilon of the norm.
            layers: Number of decoder layers in the range.
            projection_scale: Scale of the projected embedding (``hidden ** -0.5``).
            input_scale: Scale of the combined signal (``2 ** -0.5``).
        """
        super().__init__()
        self.layers = layers
        self.ple_dim = norm_weight.numel()
        self.projection = pointwise(projection_weight)
        self.norm = RMSNorm(norm_weight, eps)
        self.projection_scale = projection_scale
        self.input_scale = input_scale

    def forward(self, embedded: Tensor, token_identity: Tensor) -> Tensor:
        """Return PLE inputs from ``embedded`` ``[B, H, 1, S]`` and identity ``[B, P, L, S]``."""
        projected = self.projection(embedded) * self.projection_scale
        batch, _, _, length = projected.shape
        projected = projected.reshape(batch, self.layers, self.ple_dim, length).transpose(1, 2)
        return (self.norm(projected) + token_identity) * self.input_scale


class FeedForwardAndPLE(nn.Module):
    """The post-attention residual path of a dense Gemma 4 layer.

    Gated MLP, then the per-layer embedding gate and projection, then the layer scalar.
    GELU uses Apple's sigmoid form of the tanh approximation, which avoids ``tanh`` on the
    Neural Engine.
    """

    def __init__(self, reference: nn.Module) -> None:
        """Initialize from a Transformers ``Gemma4TextDecoderLayer``.

        Args:
            reference: The source layer.

        Raises:
            ValueError: If the layer is a mixture-of-experts layer or has no PLE input.
        """
        super().__init__()
        if reference.enable_moe_block or not reference.hidden_size_per_layer_input:
            raise ValueError("Only dense layers with per-layer embeddings are supported")
        self.pre_norm = RMSNorm.from_module(reference.pre_feedforward_layernorm)
        self.post_norm = RMSNorm.from_module(reference.post_feedforward_layernorm)
        self.gate = projection(reference.mlp.gate_proj)
        self.up = projection(reference.mlp.up_proj)
        self.down = projection(reference.mlp.down_proj)
        self.ple_gate = projection(reference.per_layer_input_gate)
        self.ple_projection = projection(reference.per_layer_projection)
        self.ple_norm = RMSNorm.from_module(reference.post_per_layer_input_norm)
        self.act = GELUReauthored()
        # Like RMSNorm.input_scale: 1.0 for trained weights, kept as an explicit multiply
        # because it is part of the published program.
        self.register_buffer(
            "projection_scale", reference.layer_scalar.new_tensor(1.0), persistent=False
        )
        self.register_buffer("layer_scalar", reference.layer_scalar.detach().clone())

    def forward(self, hidden: Tensor, per_layer_input: Tensor) -> Tensor:
        """Apply the residual path to ``hidden`` ``[B, H, 1, S]``."""
        normalized = self.pre_norm(hidden)
        gated = self.act(self.gate(normalized)) * self.projection_scale
        hidden = hidden + self.post_norm(self.down(gated * self.up(normalized)))
        ple_gated = self.act(self.ple_gate(hidden)) * self.projection_scale
        ple = self.ple_projection(ple_gated * per_layer_input)
        return (hidden + self.ple_norm(ple)) * self.layer_scalar
