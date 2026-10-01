# Running Gemma 4 E4B on the Apple Neural Engine: partitioning, placement, precision and verification

Technical report (draft). This is the engine of the iPhone app within; the app itself is not part of this
repository. Every number below comes from a summary file in [`results/`](results/); Appendix A lists what is
needed to reproduce them.

## Abstract

We describe and evaluate a way to run Gemma 4 E4B (4.5 billion effective parameters, 8 billion including
embedding tables) on a phone system-on-chip so that the decoder's arithmetic runs on the Neural Engine (ANE),
inside the 6 GiB a single app process may use, with a measured and bounded loss of quality. The 42 decoder
layers are exported as seven Core AI assets of six layers each and an eighth asset for the vocabulary head,
which returns its 262,144 logits as 16 slices. The app holds the KV caches: every layer receives its cache as a
read-only input and returns only the keys and values of the current step, because in this graph the ANE
compiler rejects KV kept as model state whenever compressed weights are present. MLP weights are 6-bit
palettized, attention weights INT8, the head INT4; the 6.98 GB of embedding tables stay on the CPU as
memory-mapped FP16 lookups.

The evidence has four layers. Placement: in one trace of three interleaved arms over all eight assets, 96 of
96 Neural Engine calls contain a Neural Engine prediction and none overlaps GPU work of the process, while the
GPU control shows the opposite pattern and two idle recordings contain no predictions. Numerics: all 71 outputs
are within PSNR 35 dB and cosine 0.999 of an FP32 recomputation of the same quantized graph (worst 45.6 dB and
0.99906). Quality: on a frozen set of 200 prompts the negative log-likelihood rises by 0.0213 nats/token over
the FP32 original (limit 0.05). Performance on an iPhone 18 Pro with iOS 27.2: 16.4 tokens/s decode with a
16,384-token KV capacity, 0.62–0.64 s to the first token of a 32-token prompt, a peak footprint of 5.73 GB
during an 11-minute run on battery, and about 9 tokens/s once the device warms and the engine paces itself.
We also report the approaches that failed and what each failure rules out.

## Glossary

| Term | Meaning here |
| --- | --- |
| chunk | One compiled Core AI asset: a contiguous range of decoder layers, or the vocabulary head. Called once per step. |
| step | One call of the decoder for a fixed number of new tokens. Functions are named by width: `step_1` (one token; returns the hidden state or logits, and the fresh K/V) and `step_N` for a wider static width. The shipped package has `step_1` and `prefill_16`. |
| prefill | Feeding the prompt. All prompt tokens but the last pass through the KV-owning chunks only, 16 per call; the last token runs `step_1` through all chunks and yields the first logits. |
| decode | Generating one token per step with `step_1` through all eight chunks. |
| host-held KV | KV caches kept in app memory. Each layer takes its cache as a read-only input and returns the fresh K/V of this step; the host writes them into the cache after the step. |
| owner, donor, consumer | Layers 0–23 own KV. Layers 24–41 own none and reuse the KV of a donor: layer 22 (sliding attention) or layer 23 (full attention). |
| capacity | The number of cache slots of a full-attention layer, fixed when the package is exported (16,384 here). |
| placement | The compute unit that actually executes a call, as observed in a trace, as opposed to the unit requested when loading. |
| arm | One way of loading the same asset in the placement test: Neural Engine preference (candidate), GPU preference or CPU preference (controls). |
| window | The span of one call in a trace, from the start of its inference interval to the end of its readback interval. The unit the placement test counts. |
| lut6 | Weight palettization: each weight is a 6-bit index into a 64-entry table; one table per 16 output channels, with a per-channel scale. |
| PLE | Per-layer embeddings: 256 values per layer per token, looked up from a 262,144 × 10,752 table and merged inside each layer. |
| thermal state | iOS `ProcessInfo.thermalState`: nominal, fair, serious, critical. |

## Results at a glance

| Question | Result | Conditions | File |
| --- | --- | --- | --- |
| Does the decoder run on the ANE? | All 8 chunks: 96/96 candidate windows contain a Neural Engine prediction, 0/96 overlap GPU work of the process; GPU control 0/96 and 96/96; CPU control 0/96 and 0/96; idle recordings 0 predictions in 41.9 s each | iPhone 18 Pro, iOS 27.2, Release build, `step_1` of every chunk | `placement.json` |
| Is each call's arithmetic right? | 71/71 outputs pass; worst PSNR 45.6 dB, worst cosine 0.99906; head keeps the top-1 token, KL 8.5 × 10⁻⁵ nats | against an FP32 recomputation of the same quantized graph on the same input bytes | `numerics.json` |
| Is the whole chain right? | Largest per-item NLL difference 0.0047 nats/token over 28 items (tolerance 0.005) | phone engine against an independent full-sequence recomputation | `numerics.json` |
| How much quality is lost? | +0.0213 nats/token (limit 0.05); worst subset 0.029, worst item 0.102; greedy agreement 95.1% | 200 frozen prompts, 25,040 reference tokens | `quality.json` |
| How fast? | 16.4 tokens/s decode; first token 0.62–0.64 s | 32-token prompt, 5 × 512 tokens, nominal thermal state at start | `performance.json` |
| How much memory? | Peak 5.73 GB, at least 0.71 GB always available, of a 6.44 GB limit | 11-minute, 20-turn conversation on battery | `sustained.json`, `memory.json` |
| Does it hold up thermally? | Fair after 62 s, never serious; paced to about 9 tokens/s | same run | `sustained.json` |

## 1 Problem and constraints

### 1.1 Question

How should a decoder-only language model of the E4B class be partitioned, placed, quantized and verified on a
heterogeneous SoC (CPU, GPU and Neural Engine sharing one memory), with about 6 GB available to the process,
so that the main computation demonstrably runs on the Neural Engine and the quality loss stays within a limit
fixed in advance?

The answer is given as a general recipe (Section 2) with Gemma 4 E4B on an iPhone 18 Pro as its first
instance. Section 2.9 lists what carries over to another model and what has to be redone.

### 1.2 The model

| Property | Gemma 4 E4B ([model card][gemma-card]) and the exported configuration |
| --- | --- |
| Parameters | 4.5 B effective, 8 B including embeddings |
| Decoder layers | 42: five sliding-window layers then one full-attention layer, repeated seven times |
| Widths | hidden 2,560; MLP 10,240; 8 query heads and 2 KV heads; head dimension 256 (sliding) and 512 (full) |
| Sliding window | 512 tokens |
| KV sharing | layers 24–41 compute no K/V; they read the cache of layer 22 (sliding) or 23 (full) |
| Per-layer embeddings | a 262,144 × 10,752 table (42 layers × 256) |
| Vocabulary | 262,144 tokens; the head is tied to the token embedding; final logit soft-capping at 30 |
| Context | 128K tokens supported by the model; 16,384 in the exported package |

### 1.3 Platform constraints

| Constraint | What we measured | Consequence for the design |
| --- | --- | --- |
| One memory, one limit | The process may use 6,442,467,328 bytes (about 6 GiB: footprint plus available memory at the start of the sustained run) with the increased-memory-limit entitlement. | Weights, KV, the touched pages of the lookup tables and the app share this budget. |
| The ANE's copy of the weights is charged to the process | About 1.3 × the chunk file bytes on iPhone 18 Pro / iOS 27.2 (a lut6 package with capacity 4,096: 3.57 GB of chunks → 4.69 GB in the neural footprint); about 1.06 × on iPhone 17 Pro / iOS 27.0. | Compiled weights must stay near 3.5 GB; an all-INT8 package (4.65 GB) does not fit (Section 4.2). |
| Compile or fall back | When the ANE compiler rejects a graph, Core AI compiles it for the GPU and the load still succeeds. | Placement must be measured on the device, for each compiled plan, not inferred from the requested unit. |
| State and compressed weights | KV as Core AI state fails ANE compilation in this layer graph whenever INT8 or palettized weights are present. | KV is held by the host (Section 2.4). |
| Static shapes | KV capacity and input widths are fixed at export; one function exists per width. | One package per capacity; prefill width 16. |
| Size of one compiled graph | A 12-layer chunk failed to specialize; a 42-layer graph exceeded the process limit while its functions were prepared; a 32,768-slot full-attention cache failed ANE conversion while 32,000 compiled. | Chunks of six layers; capacity ≤ 32,000. |
| ANE capacity of one process | On iPhone 17 Pro / iOS 27.0, INT8 chunks loaded onto the ANE in one process up to about 3.4 GB; the next chunk, whichever it was, fell back to the GPU. | Compress the weights rather than add chunks (Section 2.5). |
| No process attribution for ANE activity | The trace table of Neural Engine intervals has no process column (GPU intervals do). | Placement is shown by exclusion and coincidence, with control arms and idle recordings (Section 2.8). |
| Heat | Continuous decode reaches the fair state within about a minute; once serious, the device stays there for minutes even at reduced speed. | The engine paces decode from the fair state (Section 3.7). |

Sources: `memory.json`, `negative.json`, `package.json`, `sustained.json`.

### 1.4 Acceptance criteria

Each layer of evidence has an instrument, a pass rule and a denominator, fixed before the measurement reported
in Section 3.

| Layer | Pass rule |
| --- | --- |
| Placement | Every call of every chunk has a complete, ordered set of trace intervals; all chunks are covered; 0/N candidate windows overlap GPU work of the process and N/N GPU-control windows do; N/N candidate windows contain a Neural Engine prediction and the GPU control has 0/N (or a label-permutation test gives p < 0.001); each idle recording's prediction rate is below 1% of the rate inside candidate windows; the per-call numerics rule below holds. N ≥ 90 per arm. |
| Numerics, per call | Every output: PSNR ≥ 35 dB and cosine ≥ 0.999 against an FP32 recomputation of the same quantized graph on the same input bytes; head: same top-1 token and KL ≤ 10⁻³ nats over the full vocabulary. |
| Numerics, whole chain | Per item, mean NLL on the phone within 0.005 nats/token of an independent recomputation. |
| Quality | Mean NLL increase over the FP32 original ≤ 0.05 nats/token overall, ≤ 0.15 for every subset, ≤ 0.40 for every item. |
| Performance | Median decode ≥ 10 tokens/s and first token ≤ 5 s after warm-up. |
| Memory | Steady-state peak ≤ 6.0 GB; available memory > 200 MB at every 1 Hz sample; no termination by the system. |
| Thermal | Never critical; serious for at most 120 s continuously. |

Two thresholds changed during development, and the change is part of the result. The per-call numerics rule
replaced an element-wise rule (|Δ| ≤ 0.003 + 0.003·|reference| and PSNR > 70 dB) that FP16 execution of this
graph cannot meet on any device, the Mac included (Section 4.8). The memory rule started as 55% of the process
limit and became a 6.0 GB steady-state peak when no format that met the quality limit fitted under 55%
(Section 2.5); transients during loading and the first prefill are excluded from the 6.0 GB rule but not from
the 200 MB and no-termination rules.

## 2 Method

The recipe, in the order it is applied:

1. Establish the platform's constraints with small probes on the target device (Section 2.1).
2. Partition the decoder at boundaries that respect the attention pattern, the KV-sharing structure and the
   compiler's limits; separate the vocabulary head (Section 2.2).
3. Place matrix arithmetic on the ANE and lookups, state bookkeeping and sampling on the CPU (Section 2.3).
4. Hold KV on the host behind an explicit chunk contract (Section 2.4).
5. Choose per-block precision by memory first and quality second, judged on a frozen set (Section 2.5).
6. Measure the data flow and memory of the result (Sections 2.6, 2.7) and verify it in four layers (Section 2.8).

### 2.1 Hardware and constraints

![One decode step: what runs where](docs/figures/architecture.svg)

The SoC has three compute units on one unified memory: CPU cores, a GPU and the Neural Engine. Core AI compiles
each asset for a preferred unit on the device, caches the compiled plan, and falls back to the GPU when the ANE
compiler rejects a graph. The constraints in Section 1.3 were established with single-layer and single-chunk
probes built from the real weights (layer 0, or layers 18–23) before any full export: one variable changed per
probe, and the outcome read from the compiler log, the call time and the footprint. Two of them shaped
everything else: the ANE copy of the weights is charged to the process at about 1.3 × their file size, and KV
cannot be model state when the weights are compressed.

### 2.2 Partitioning

![Partition of the 42 decoder layers](docs/figures/partition.svg)

The 42 layers are cut into seven chunks of six; the vocabulary head is an eighth chunk with 16 outputs of
16,384 rows each. The chunk contract (`chunk-contract.json` in the package, validated by `Gemma4ChunkContract`)
names every input and output of every function.

| Chunk | Layers | KV | Functions | `main.mlirb` |
| --- | --- | --- | --- | --- |
| 0–3 | 0–5, 6–11, 12–17, 18–23 | owns its six layers' KV | `step_1`, `prefill_16` | 459.5–459.6 MB |
| 4–6 | 24–29, 30–35, 36–41 | reads donors 22 and 23 | `step_1` | 440.9 MB |
| 7 | vocabulary head | — | `step_1` (16 logit slices) | 336.1 MB |

Why six layers:

- **Upper bound.** Larger graphs failed: a 12-layer chunk failed to specialize (cause not established), a
  single 42-layer FP8 graph exceeded the process limit during function preparation, and a staged FP16 graph
  failed ANE compilation. Six-layer INT8 chunks of at most 0.58 GB each compiled for the ANE. A per-graph size
  limit between six and twelve layers was not located; the process's ANE capacity (Section 1.3) is a separate
  limit on the sum of all chunks.
- **Lower bound.** Each call has a fixed cost. Single-call probes of INT8 host-KV layers took 2.71, 3.85 and
  5.23 ms for one, two and three layers (iPhone 17 Pro): about 1.26 ms per layer plus 1.4 ms per call. Smaller
  chunks multiply the fixed cost; with six layers per chunk a token costs eight calls.
- **Structure.** Six is one period of the attention pattern, so every layer chunk has the same shape (five
  sliding layers, then one full-attention layer), and the KV-sharing boundary at layer 24 falls between chunks:
  chunks 0–3 only own KV and chunks 4–6 only consume it. Owner chunks carry a prefill function; consumer chunks
  need none, because layers 24–41 produce no KV and so only the last prompt token has to pass through them.

Why a separate head split into 16 slices: as its own chunk the head can take its own format (INT4 while the
layers are not), and as a single 262,144-wide output it took 27 ms per call against 14.4 ms as 16 independent
slices (INT8 head, iPhone 17 Pro); the host concatenates the slices. In the first INT8 package the head shared a
chunk with layers 36–41 (1.23 GB), which ran on the GPU at 54 ms per call; later runs traced that fallback to the
process's ANE capacity rather than to the chunk's size (Section 4.2). In the shipped format (INT4) the head's
median Neural Engine busy time is 3.8 ms per call.

What was not done: a sweep over chunk sizes in the shipped format. Six layers is a size that works and is
bounded on both sides by measurements, not a measured optimum.

### 2.3 Placement

| Work | Unit | Reason |
| --- | --- | --- |
| All 42 decoder layers: projections, attention, norms, MLP, PLE merge | ANE (chunks 0–6) | Almost all of the arithmetic; the placement under test. |
| Vocabulary head, 2,560 → 262,144 | ANE (chunk 7) | The largest single matrix product per token. |
| Token and per-layer embedding lookup | CPU | One row per token from 6.98 GB of tables: no arithmetic. As graph constants the tables would count toward the ANE copy charged to the process; memory-mapped, only the touched rows are read. |
| RoPE tables and attention masks | CPU | Small and position-dependent; computing them on the host keeps every graph static. |
| KV write-back | CPU | Follows from host-held KV (Section 2.4). |
| Logit concatenation and sampling | CPU | 262,144 values per token; temperature, top-K and top-P. |
| Tokenizer and chat template | CPU | Text processing. |
| — | GPU | No work in the shipped path; used only as a control arm. The ANE package on Core AI's GPU backend was not usable (Section 4.7). |

Over the 11-minute sustained run the process used 159.1 s of CPU time in 660 s of wall time, about a quarter of
one core, including the app's interface.

### 2.4 State: host-held KV

The contract for an owner layer *n*: inputs `key_n` and `value_n` of shape [1, 2, *D*, *C*], read only; outputs
`fresh_key_n` and `fresh_value_n` of shape [1, 2, *D*, *L*] for the *L* tokens of the call; after the call the
host writes the fresh values into slots (*p* + *i*) mod *C*. Attention inside the graph runs over the *C* cached
slots followed by the *L* fresh ones, with masks from the host.

| Layer kind | Layers with KV | Slots *C* | Host bytes (FP16) |
| --- | --- | --- | --- |
| Sliding attention (*D* = 256) | 20 | 512, used as a ring | 20,971,520 |
| Full attention (*D* = 512) | 4 | 16,384 | 268,435,456 |
| Total |  |  | 289,406,976 (0.29 GB) |

Layers 24–41 own no KV. Chunks 4–6 receive the caches of donor layers 22 and 23 plus this step's fresh K/V of
those two layers from chunk 3, because the host writes the caches back only after the whole step.

Why not model state: probes on two devices and two OS releases (Section 4.1) show that in this layer graph KV
kept as Core AI state compiles for the ANE only without compressed weights, while the same layers with
host-held KV compile and run. The finding is scoped to this graph: Apple's iOS recipe combines palettized
weights with state for other models.

What host-held KV costs and gives:

- Every call binds the full caches of its layers as inputs: 0.49 GB of cache tensors per decode step at
  capacity 16,384 (derived from the contract's shapes). Allocating the caches from the function's own input
  descriptor gave the same layout and footprint, so no zero-copy path was found in the public API. Decode time
  therefore depends on the compiled capacity, not on the position: the same engine decodes at 16.4 tokens/s
  with capacity 16,384 and 21.6 tokens/s with 4,096.
- Capacity is fixed per package; packages that carry several capacities are slower at every capacity
  (Section 4.5).
- The host owns the state. A conversation's caches can be written to disk and read back: after a reload,
  restoring a 13.7K-token conversation gave the first token in 1.27 s, against 54 s to prefill it again.
  A new prompt that extends exactly the tokens in the cache is fed incrementally.
- Ring buffers keep the 20 sliding layers at 512 slots: 0.29 GB of KV in total, against 0.94 GB if those
  layers held all 16,384 slots (derived).

The ring and the prefill schedule are checked against a recomputation without a ring buffer (Section 3.3).

### 2.5 Precision

| Tensors | Format | Package bytes | Reason |
| --- | --- | --- | --- |
| MLP gate, up, down: 42 × 3 × 2,560 × 10,240 = 3.30 B weights | lut6: 6-bit k-means palette, one 64-entry table per 16 output channels, per-channel scale | about 2.48 GB of the 3.50 GB of chunks (derived) | The largest block. The 4-bit formats that the ANE compiles failed the quality limit; INT8 does not fit. |
| Attention q, k, v, o; per-layer-embedding projections | INT8 per output channel, symmetric, weight-only | the rest of the layer chunks | Near-lossless; compressing attention further cost more quality than it saved memory. |
| Vocabulary head (tied to the embedding) | INT4 per output row, 16 slices | 336.1 MB | Half of the INT8 head (about 672 MB); its share of the 4-bit-MLP loss below was about 0.013. |
| Token and per-layer embedding tables | FP16, outside the graphs | 6.98 GB | Lookups only (Section 2.3). |
| Activations, KV, norms | FP16 | — | The ANE's native type. |

The format was chosen by memory first, then quality, on subsets of the frozen quality set (Section 3.4), with
the exporter's own rounding recomputed on the Mac:

| Configuration | Items | NLL increase (nats/token) | Outcome |
| --- | --- | --- | --- |
| FP16 graph, no weight rounding | 20 | 0.00002 | The graph is lossless; the loss comes from rounding. |
| All projections INT8 | 20 | 0.002 | Passes quality; does not fit in memory. |
| MLP INT4 per channel, head INT4 | 20 | 0.286 | Fails. |
| MLP INT4 in blocks of 32 input channels | — | — | Not converted for the ANE (a 6-layer probe fell back to the GPU at 76–78 ms). |
| MLP lut4, head INT4 | 18 | 0.0997 | Fails. |
| MLP lut4 on down only / on gate and up only | 18 | 0.045 / 0.073 | The losses add up; no single projection carries them. |
| MLP lut4 with input-moment-weighted k-means | 18 | 0.103 | No gain. |
| MLP and attention lut4 | 18 | 0.153 | Saves about 275 MB for +0.053. |
| **MLP lut6, head INT4 (shipped)** | 18 | 0.023 | Passes; the full 200-item run on the phone gives 0.0213. |

Replacing one layer's gate and up projections at a time cost at most +0.004 per layer, spread over layers
11–41, so no small set of layers could be given more bits to recover the 4-bit loss. One exporter error found on
the way is worth recording for anyone repeating this: the per-layer-embedding projections live in the same
module as the MLP, and quantizing them to INT4 along with it added +0.47 on its own; keeping them at INT8 costs
about 27 MB.

Sources: `quality.json` (`formatAlternatives`), `negative.json`.

### 2.6 Data flow of one token

![Time of one decode step](docs/figures/decode-step.svg)

Decode, one token at position *p*:

1. CPU: look up the embedding row (scaled by √2,560) and the 42 × 256 per-layer values (scaled by √256);
   compute RoPE cos/sin (θ = 10,000 for sliding layers; θ = 1,000,000 with a quarter of the dimensions rotated
   for full layers) and the masks.
2. ANE, chunks 0–3 (`step_1`): hidden state, embedding row, the chunk's six layers' PLE rows, RoPE, masks and
   caches in; hidden state and twelve fresh K/V tensors out.
3. ANE, chunks 4–6 (`step_1`): the same inputs, with the donor caches and donor fresh K/V instead of own caches;
   hidden state out.
4. ANE, chunk 7: hidden state in, 16 logit slices out. CPU: concatenate, sample.
5. CPU: write the 24 owner layers' fresh K/V into slot *p* mod 512 (sliding) or *p* (full).

Time per step. In the placement recording the median Neural Engine busy time per call is 6.65–7.17 ms for the
layer chunks and 3.80 ms for the head, 52.8 ms in total. The product's median token interval in the performance
procedure is 60.7–61.1 ms per unpaced run (16.4 tokens/s). The difference, about 8 ms, is host work and call
overhead; it is an estimate from two different runs, not a measurement.

Prefill. A prompt of *n* tokens is fed as ⌊(*n* − 1)/16⌋ calls of `prefill_16` and (*n* − 1) mod 16
single-token calls, each through chunks 0–3 only, then one `step_1` through all eight chunks for the last token.
A 32-token prompt is therefore 1 + 15 owner-chunk calls and one full step; its first token arrives after a
median of 0.62–0.64 s. Long prompts: in the sustained run, two turns whose cache could not be reused
re-prefilled 1,419 and 2,544 tokens and produced their first token after 5.24 s and 9.32 s (about 270 tokens/s,
derived).

Memory per step. Without preallocated outputs, Core AI allocated new output arrays on every call, about
23.2 MiB per token (mostly the head's 16 logit slices, whose output buffers are padded), and released them only every few tokens. Binding the outputs
of `step_1` to arrays allocated once at load (`Gemma4StateStorage`) reduced the growth to about 2.9 MiB per
token and the head's call time from 7.2 to 3.6 ms.

### 2.7 Memory map

![Peak memory footprint against the per-process limit](docs/figures/memory.svg)

| Component | Size | Where | Source |
| --- | --- | --- | --- |
| Process limit | 6,442,467,328 B (about 6 GiB) | — | `memory.json` |
| Chunk files | 3,496,985,782 B | flash; compiled plans cached by the system | `package.json` |
| ANE working copy of the weights | about 1.3 × chunk bytes | charged to the process | `memory.json` |
| Host KV | 289,406,976 B | app heap | `package.json` (derived) |
| Embedding tables | 6,979,321,856 B | memory-mapped read-only; only touched rows are read | `package.json` |
| Steady-state peak, shipped engine | 5.65–5.73 GB | — | `memory.json` |
| Minimum available, 11-minute run | 0.71 GB | — | `sustained.json` |

The first decode after a load used to be the peak. Without preallocated outputs, a cold start with a short
prompt peaked at 5.985 GB, the first decode after a 13.6K-token prompt at 6.195 GB (0.248 GB available) and the
first decode after a reload and KV restore at 6.303 GB (0.139 GB available), while the steady state
mid-conversation was 5.82–5.95 GB. The engine's causes were the per-call output allocation above and file-read buffers
kept alive after a KV restore. With preallocated outputs and reads straight into the caches, the same three
scenarios peaked at 5.692, 5.720 and 5.669 GB (Debug builds). In the 11-minute Release run the peak, 5.73 GB,
came at 283 s, mid-conversation.

There is no exact breakdown of the shipped peak by component: the neural footprint and the app's own regions
were measured on a sister package (capacity 4,096), and the two accounts overlap.

### 2.8 Verification system

| Layer | Instrument | Rules out | Does not rule out |
| --- | --- | --- | --- |
| Placement | One process, three arms (Neural Engine, GPU, CPU preference) of the same assets, interleaved, recorded with all-process tracing; two idle recordings (Section 3.2) | GPU execution of chunk calls by this process; a silent GPU fallback after a compile failure; a chunk missing from the ANE plan; ANE activity of an idle system above 1% of the in-window rate | Which operation ran on the ANE; ANE activity of other processes inside a window (bounded, not excluded); CPU execution (CPU time was not sampled); placement of `prefill_16` (not traced) |
| Numerics, per call | FP32 recomputation of each chunk's quantized graph from the call's own input bytes, with Mac FP16 as a fourth arm (Section 3.3) | Wrong arithmetic in a compiled plan (the CPU arm is caught); errors that appear on one unit only | Quantization loss (same representation by construction); behavior on real conversation caches (inputs are seeded random caches) |
| Numerics, whole chain | Teacher-forced NLL on the phone against an independent full-sequence recomputation with the same rounding (Section 3.3) | Host-side errors: lookups, masks, ring buffer, prefill schedule, KV write-back | Effects specific to the shipped device (measured on an iPhone 17 Pro at capacity 4,096) |
| Quality | NLL increase over the FP32 original on a frozen 200-prompt set (Section 3.4) | Quantization damage beyond the budget, per subset and per item | Task-level accuracy; long generations; thinking mode |
| Performance, memory, thermal | Frozen procedures with fixed prompts, counts and thresholds (Sections 3.5–3.7) | Regressions in speed, memory and heat under the defined workloads | Other devices and ambient conditions; energy |

The layers complement each other: placement alone could pass for a wrong graph and numerics alone could pass for
a GPU plan, which is why the per-call numerics rule is part of the placement verdict.

### 2.9 Transferring the recipe

| Step | Carries over | Redo for a new model or device |
| --- | --- | --- |
| Constraint probes | The probe design: one layer of real weights, one variable at a time, compile log plus call time plus footprint | All results. The residency ratio changed from about 1.06 to about 1.3 between two devices and OS releases. |
| Partition | Cut at multiples of the attention period and at the KV-sharing boundary; split the head into row slices; owner chunks get a prefill function | Chunk size, number of head slices, size limits |
| Placement | Lookups, RoPE, masks and sampling on the CPU; matrix work on the ANE | Check that nothing else is large enough to need the ANE |
| State | The host-held KV contract, rings for sliding layers, donor passing for shared KV | Whether model state compiles with the chosen weight format; capacity |
| Precision | INT8 baseline, memory check, compress the largest block, judge on a frozen set against the original model | Every number; the set's baseline |
| Verification | Arms, idle controls, window counting, FP32 same-representation reference, NLL fidelity, quality limits, procedures | The reference implementation of the new model; the quality set's baseline continuations |

## 3 Evaluation

### 3.1 Setup

| Item | Value |
| --- | --- |
| Device | iPhone 18 Pro (iPhone19,2), iOS 27.2 (24B5089g) |
| App | Release build; its engine is the code in `Sources/` as built into the app when measured (Appendix A.1) |
| Package | The shipped package (Appendix A.2). Part of the evidence was recorded on an earlier export of the same graphs and weights, equivalent by the definition in Appendix A.3. |
| Exceptions | Whole-chain numerics: iPhone 17 Pro, iOS 27.0, Debug build, capacity 4,096 (Section 3.3). Probes in Sections 2 and 4 name their device. |

### 3.2 Placement

![Placement evidence](docs/figures/placement-evidence.svg)

Design. In one process, each chunk's asset is loaded three times, with Neural Engine, GPU and CPU preference,
and its `step_1` function is called 12 times per arm with identical inputs, the arm order rotating every round;
chunks run one at a time, because three copies of the model do not fit. The inputs are the product's decode
inputs for one token at position 1,000 with seeded random caches; each chunk reads the candidate arm's outputs
of the previous chunk. Every call is bracketed by three signposted intervals (input materialization, inference,
readback). The trace covers all processes and starts before the app. After the workload, two idle recordings of
the same length are made with the app terminated.

| Criterion | Neural Engine (candidate) | GPU (control) | CPU (control) |
| --- | --- | --- | --- |
| Calls with complete, ordered intervals | 288 of 288 across the three arms; 0 unfinished, 0 unexpected; trace files unchanged since recording | | |
| Chunks covered | 8 of 8 | 8 of 8 | 8 of 8 |
| Windows overlapping GPU work of this process | 0 / 96 | 96 / 96 | 0 / 96 |
| Windows containing a Neural Engine prediction | 96 / 96 | 0 / 96 | 0 / 96 |
| Share of window time with GPU work of this process | 0.0 | 0.503 | 0.0 |
| Median window | 9.04 ms | 21.57 ms | 23.27 ms |

Every candidate window contains exactly one Neural Engine prediction interval; inside candidate windows they
occur at 100.4 per second. The two idle recordings (41.9 s each) contain none, below the 1% bound. Because the
GPU control had no window with a prediction, the rule's stricter branch decides; the label-permutation test is a
fallback for a control with some predictions and was not exercised (the review stores p = 0.0 for this branch,
which is not a computed p-value). The numerics criterion is reported in Section 3.3. Verdict: pass.

| Chunk | Median candidate window | Median ANE busy | ANE share of window | GPU-control GPU share |
| --- | --- | --- | --- | --- |
| 0 | 9.33 ms | 7.15 ms | 0.779 | 0.555 |
| 1 | 9.23 ms | 7.17 ms | 0.780 | 0.564 |
| 2 | 9.13 ms | 7.11 ms | 0.777 | 0.565 |
| 3 | 9.16 ms | 6.99 ms | 0.782 | 0.568 |
| 4 | 8.25 ms | 6.65 ms | 0.814 | 0.344 |
| 5 | 8.62 ms | 7.03 ms | 0.814 | 0.338 |
| 6 | 8.62 ms | 6.86 ms | 0.803 | 0.337 |
| head | 8.69 ms | 3.80 ms | 0.442 | 0.746 |

The GPU and CPU arms run the package compiled for the ANE; their window times are those of control arms, not a
benchmark of either unit.

What this can and cannot prove:

- The proof is by exclusion plus coincidence, not attribution: it cannot show which operation ran on the ANE.
- ANE hardware rows carry no process: the idle control bounds, and does not exclude, rows from other processes
  inside a window.
- Only the instrumented stages (input materialization, inference, readback) are covered.
- Compute preference, numerical acceptance, capture coverage and hardware placement are separate conclusions.

Two further gaps: the process's CPU time was not sampled (the recording template had no time-profile table), so
the CPU control is identified by the absence of both signals; and the Core AI execution table returned no rows
in this recording, for a reason not yet diagnosed, so per-call Core AI attribution is not part of the verdict.

Source: `placement.json`.

### 3.3 Numerics

Per call. The round-0 inputs and outputs of the placement recording are scored against an FP32 recomputation of
each chunk on the Mac: the chunk is rebuilt from the checkpoint with the exporter's own quantization and
evaluated in FP32 on the phone's input bytes. The same graph evaluated in FP16 on the Mac is scored as a fourth
arm. Each chunk is scored on its own inputs, so errors do not accumulate across chunks here.

| Chunk | Outputs | Neural Engine: min PSNR / min cosine | GPU | CPU | Mac FP16 |
| --- | --- | --- | --- | --- | --- |
| 0 | 13 | 48.3 dB / 0.999892 | 46.9 / 0.999735 | 25.2 / 0.941976 | 54.2 / 0.999952 |
| 1 | 13 | 55.9 / 0.999959 | 54.9 / 0.999969 | 37.4 / 0.998965 | 54.4 / 0.999968 |
| 2 | 13 | 45.6 / 0.999063 | 52.4 / 0.999926 | 37.3 / 0.994197 | 50.4 / 0.999790 |
| 3 | 13 | 55.1 / 0.999842 | 55.2 / 0.999958 | 36.7 / 0.992097 | 52.9 / 0.999901 |
| 4 | 1 | 75.1 / 0.999982 | 80.7 / 0.999995 | 59.4 / 0.999346 | 79.9 / 0.999994 |
| 5 | 1 | 76.6 / 0.999992 | 73.9 / 0.999984 | 60.6 / 0.999649 | 74.6 / 0.999986 |
| 6 | 1 | 83.4 / 0.999997 | 87.1 / 0.999999 | 69.5 / 0.999931 | 86.2 / 0.999999 |
| head | 16 slices | 60.9 / 0.999999 | 79.3 / 1.000000 | 68.2 / 1.000000 | 74.8 / 1.000000 |

Owner chunks return the hidden state and twelve fresh K/V tensors; consumer chunks return only the hidden state.

| | Neural Engine | GPU | CPU | Mac FP16 |
| --- | --- | --- | --- | --- |
| Outputs failing PSNR ≥ 35 dB and cosine ≥ 0.999 | 0 / 71 | 0 / 71 | 16 / 71 | 0 / 71 |
| Outputs passing the old element-wise rule (\|Δ\| ≤ 0.003 + 0.003·\|ref\|, PSNR > 70 dB) | 8 / 71 | 9 / 71 | 0 / 71 | 9 / 71 |
| Head: top-1 equal / KL (nats) / top-10 overlap | yes / 8.5 × 10⁻⁵ / 10 | yes / 3.9 × 10⁻⁶ / 10 | yes / 3.9 × 10⁻⁵ / 10 | yes / 3.3 × 10⁻⁵ / 10 |

The thinnest margin is chunk 2's hidden state: cosine 0.99906 on the ANE (relative L2 error 4.3%), against 2.1%
for the Mac FP16 run and 1.2% for the GPU. The ANE, the GPU and Mac FP16 are at the same level of error; the Core
AI CPU arm is not (worst output 25.2 dB; chunk 0's hidden state at cosine 0.94; failing outputs in chunks 0–3),
so the rule separates a correct plan from an incorrect one. The CPU backend is not on the shipped path and was
not investigated further.

Whole chain. Teacher-forced NLL from the product engine on the phone (lookups, all chunks, host KV with 512-slot
rings, the prefill schedule) was compared item by item with an independent recomputation that builds the same
rounded graph in PyTorch FP16 and runs each sequence in one pass without a ring buffer.

| Items | Coverage | Largest difference | Tolerance |
| --- | --- | --- | --- |
| 22 from the quality set | all five subsets | 0.0022 nats/token | 0.005 |
| 6 window-boundary items | prefill ending at 511, 512, 513; decode across 511–513; decode across the ring's second wrap (1,023–1,025); a control at 300 | 0.0047 nats/token (prefill ending at 512) | 0.005 |

Conditions: iPhone 17 Pro, iOS 27.0, Debug build, a package with the same chunking and formats at capacity
4,096. The three items that cross the window boundary (prefill ending at 511 and at 512, decode across 511–513) use
77–94% of the tolerance, against at most 0.0022 for the
others; the masks were checked to be equivalent and the difference is numerical (ring buffer against
whole-sequence masking), measured on one text at three positions. The result was recomputed from the raw
receipts by three independent scripts with identical results.

Source: `numerics.json`.

### 3.4 Quality

The quality test set is frozen and identified by its SHA-256; its contents are not published, because the long
subset contains unpublished text.

| Subset | Items | Content | Prompt tokens (min / median / max) | Reference tokens |
| --- | --- | --- | --- | --- |
| en | 60 | English, single turn | 14 / 21 / 39 | 7,429 |
| zh | 60 | Simplified Chinese, single turn | 15 / 20 / 39 | 7,553 |
| multi | 40 | four user turns with three fixed assistant turns, 20 English and 20 Chinese | 100 / 121 / 156 | 4,938 |
| code | 20 | write, explain or fix code in nine languages | 24 / 34 / 74 | 2,560 |
| long | 20 | at least 1,500 tokens of first-party documents or source plus a task, English and Chinese | 1,556 / 2,484 / 3,821 | 2,560 |
| total | 200 | | | 25,040 |

Set SHA-256 `a2214c774c535a6f0e5f6478368adab68b45074013be80ca527b5de0d7f3d316`; reference continuations SHA-256
`9880394ec5f99e1c54aa2f225b70b4cd9e671a3eb5f5e9e38609026844512d13`.

Definition. Prompts are rendered with the checkpoint's own chat template (thinking disabled, no system prompt).
The original model generated each reference continuation greedily (at most 128 tokens), and both were frozen
before any candidate was scored. The baseline is the original checkpoint in FP32 on a Mac CPU. The metric is the
mean negative log-likelihood of the reference tokens under teacher forcing, prompt tokens excluded; the reported
delta is candidate minus baseline, weighted by tokens. On the phone the product engine computes it from its FP16
logits, accumulating in double precision.

| Subset | Baseline NLL | Phone NLL | Increase | Limit | Greedy agreement |
| --- | --- | --- | --- | --- | --- |
| en | 0.1806 | 0.1983 | 0.0177 | 0.15 | 95.9% |
| zh | 0.2592 | 0.2882 | 0.0290 | 0.15 | 93.8% |
| multi | 0.2866 | 0.3140 | 0.0275 | 0.15 | 93.5% |
| code | 0.0816 | 0.0880 | 0.0063 | 0.15 | 98.0% |
| long | 0.2428 | 0.2546 | 0.0118 | 0.15 | 97.0% |
| **all** | 0.2215 | 0.2427 | **0.0213** | 0.05 | 95.1% |

No item exceeds 0.40; the largest item increase is 0.102. The 200 per-item NLL sums were identical, bit for bit,
to those of an earlier Debug run of the 4,096-capacity package on an iPhone 17 Pro with iOS 27.0: capacity,
device, OS and build configuration did not change any value.

What the number means: under teacher forcing on the original model's own continuations, the quantized model
assigns them on average e^0.0213 ≈ 1.02 times lower likelihood per token. It is not a task accuracy.

Source: `quality.json`.

### 3.5 Performance

Procedure: a fixed 32-token prompt, no system prompt, thinking off, temperature 0.7, top-K 64, top-P 0.95; three
warm-up generations of 64 tokens; five runs of exactly 512 tokens (stop tokens suppressed), each starting in the
nominal thermal state; speed of a run = 1000 / median interval over tokens 33–512; reported speed = median of the
five runs; first token = request to first sampled token, after warm-up.

| Package | Median speed | Runs (tokens/s) | First token, median / max |
| --- | --- | --- | --- |
| evidence export | 16.43 tokens/s | 16.44, 16.43, 16.38, 16.46, 9.04 | 0.64 s / 0.89 s |
| shipped export | 16.41 tokens/s | 9.06, 16.43, 16.37, 16.47, 16.41 | 0.62 s / 1.55 s |

In each set one run reached the fair state early and was paced for most of its tokens (median interval about
110 ms); the median of five is not affected. Some runs waited up to 235 s for the nominal state before starting.
The first generation after loading took 2.30–2.99 s to its first token. Loading the package for the first time
compiles all eight chunks for the ANE (138 s, Debug build); later launches load from the system's compile cache
(6 s).

Capacity. The same engine with a 4,096-slot package decodes at 21.6 tokens/s against 16.4 with 16,384 (cold start,
512 tokens, Debug build, plugged in), because every step binds the full caches.

Published figures for comparison. Google publishes LiteRT-LM figures for Gemma 4 E4B on an iPhone 17 Pro
([model card][litert-card]; also the [developer page][litert-dev]):

| | This work | LiteRT-LM GPU | LiteRT-LM CPU |
| --- | --- | --- | --- |
| Device | iPhone 18 Pro | iPhone 17 Pro | iPhone 17 Pro |
| Compute unit for the decoder | ANE | GPU | CPU |
| Context capacity | 16,384 | 2,048 | 2,048 |
| Decode | 16.4 tokens/s | 25.1 tokens/s | 9.7 tokens/s |
| Prefill | about 270 tokens/s (derived, Section 2.6) | 1,189 tokens/s | 159 tokens/s |
| First token | 0.62–0.64 s for 32 tokens | 0.9 s for 1,024 tokens | 6.5 s for 1,024 tokens |
| Peak memory (phys_footprint) | 5.73 GB | 3,380 MB | 961 MB |
| Model download | 10.5 GB | 3.66 GB | 3.66 GB |

These are not like-for-like: different device generations, runtimes, compute units, weight formats and context
capacities, and we did not run Google's conditions (1,024 prompt tokens, 256 generated, context 2,048). On the
published numbers, the GPU path on the previous phone generation decodes about 1.5 times faster, prefills about
four times faster and uses less memory than this ANE path; this work keeps the GPU idle and holds a 16,384-token
context. Energy is not compared: Google does not publish it, and we could not measure it (Section 3.7). A
third-party project reports 15.7 tokens/s for a Gemma 4 E4B build on the iPhone 17 Pro's Neural Engine with INT8
chunks and a 2,048-token context ([CoreML-LLM][coreml-llm]).

Source: `performance.json`.

### 3.6 Memory

| Scenario | Peak footprint | Minimum available | Build |
| --- | --- | --- | --- |
| 11-minute sustained run | 5.73 GB | 0.71 GB | Release |
| Cold start, 512 tokens | 5.65 GB | 0.79 GB | Debug, optimized |
| Cold start, 13.7K-token prompt | 5.720 GB | 0.723 GB | Debug |
| Reload and KV restore | 5.669 GB | 0.774 GB | Debug |
| Capacity 4,096, cold start, 512 tokens | 5.027 GB | 1.415 GB | Debug |

All are within the 6.0 GB steady-state rule with more than 0.7 GB available. Earlier variants and rejected
packages are in the figure of Section 2.7 and in Section 4.

Source: `memory.json`.

### 3.7 Sustained use and thermal behaviour

Workload: one conversation of 20 turns of exactly 256 tokens, 5 s between turns, at least 10 minutes, twelve fixed
English and Chinese prompts, temperature 0.7, top-K 64, top-P 0.95, thinking off. Environment: airplane mode,
screen on at 50% brightness, unplugged, battery at 85%, nominal thermal state at the start, Release build. Ambient
temperature was not recorded.

| Measure | Result | Rule |
| --- | --- | --- |
| Duration, tokens | 660 s, 5,120 output tokens | ≥ 600 s |
| Crashes, terminations | none; no unclean previous launch | none |
| Peak footprint | 5.73 GB (at 283 s) | ≤ 6.0 GB |
| Minimum available | 0.71 GB | > 200 MB |
| Thermal samples (every 5 s) | 12 nominal, 117 fair, 0 serious, 0 critical; fair from 62 s | never critical, serious ≤ 120 s |
| Decode interval | 61–63 ms for turns 1–3, about 110 ms from turn 4 | — |
| First token | 2.73 s for turn 1 (after load); 0.38–0.74 s when the cache was reused (17 of 19 later turns) | — |
| Energy | not measurable: the battery level stayed at 85% | recorded only |

The engine paces decode once the device leaves the nominal state: a token takes at least 110 ms in the fair state
and 200 ms in serious or critical. The reason is that iOS leaves the serious state slowly: with 92 ms pacing at
fair, a plugged-in run of the same format reached serious at 484 s and stayed there for 204 s, longer than the
120 s rule. The cost is visible above: after about a minute of continuous generation the speed is about
9 tokens/s, below the 10 tokens/s floor of the performance rule (Section 1.4), which is measured from a nominal start.

Two later turns did not reuse the cache because the conversation's text re-tokenized differently from the
generated tokens; they re-prefilled 1,419 and 2,544 tokens.

Source: `sustained.json`.

## 4 Negative results

| | Approach | Result | Consequence |
| --- | --- | --- | --- |
| 4.1 | KV as Core AI state with compressed weights | ANE compilation fails in every tested combination | Host-held KV |
| 4.2 | All-INT8 weights | Does not fit: terminated during prefill | Compress the MLP |
| 4.3 | 4-bit MLP formats | Fail the quality limit or the ANE conversion | 6-bit palettes |
| 4.4 | Larger graphs: 12-layer chunks, whole-model graphs | Fail to compile or exceed the process limit | Six-layer chunks |
| 4.5 | Packages with several KV capacities | Slower at every capacity | One capacity per package |
| 4.6 | 32K context | Terminated; 32,768 slots do not convert | Capacity 16,384 |
| 4.7 | This package on Core AI's GPU backend | Terminated, or 13–16 s per token | GPU only as a control |
| 4.8 | Element-wise numerics rule at 70 dB | Not attainable by FP16 execution | PSNR, cosine and KL rule |
| 4.9 | Trusting a cached compiled plan | A recompile can land off the ANE | Placement re-checked per plan |
| 4.10 | Attention per query head | Slower than per KV head | Grouped attention |

**4.1 State with compressed weights.** Single-layer probes (real layer-0 weights, capacity 512):

| Device, OS | Fails ANE compilation | Compiles |
| --- | --- | --- |
| iPhone 17 Pro, iOS 27.0 | INT8 whole layer, INT8 attention only, INT8 MLP only, each with KV as state | FP16 with state; INT8 per-layer-embedding projections only, with state; state update only; INT8 layer with host-held KV |
| iPhone 18 Pro, iOS 27.2 | INT8 with state | FP16 with state; INT8 with host-held KV (1.5 ms per call) |
| iPhone 18 Pro, iOS 27.2 | lut6 MLP; lut6 attention and MLP; the shipped mix; lut6 without per-channel scale; lut4 in groups of 32 without scale (the default iOS palettization preset) — each with state | — |

Every failure reports the same compiler error ("MLIR MPS to ANEC conversion failed"). Removing the per-channel
scale did not help, so the failure is not the palette's table shape. Splitting compressed projections and stateful
attention into separate graphs would cost about three calls per layer (around 126 per token) and was not pursued.
A seven-chunk stateful INT8 export compiled for the GPU instead and exhausted the phone's storage during the
fallback compile.

**4.2 INT8 does not fit.** On the iPhone 18 Pro an all-INT8 package (4.65 GB of chunks, capacity 4,096) reached a
6.18 GB footprint at its first inference and was terminated during prefill; the same measurement on the lut6
package showed the ANE copy at about 1.3 × the chunk bytes for both formats, so the palettes are not being
expanded. On the iPhone 17 Pro with iOS 27.0, chunks loaded onto the ANE in one process up to about 3.4 GB, after
which the next chunk, whichever it was, failed to map and fell back to the GPU (8.38 tokens/s with six chunks on
the ANE). The seventh chunk of that package also held the head (1.23 GB, the only chunk above 1 GB) and ran on
the GPU at 54 ms per call. After the head moved to a chunk of its own, the seventh chunk (layers 36–41 alone)
failed in the same way, also when the first six were loaded from the compile cache without compiling, which
points to the process's capacity rather than to the size of the combined chunk.

**4.3 4-bit MLP formats.** See the table in Section 2.5: INT4 per channel +0.286 and lut4 +0.0997 nats/token on
18–20-item subsets, against a limit of 0.05; INT4 in blocks of 32 input channels did not convert for the ANE.

**4.4 Larger graphs.** A 12-layer chunk failed to specialize (cause not established). A single 42-layer FP8 graph
specialized, but preparing its functions pushed the footprint to 5.5–6.5 GB and the process was killed at the
per-process limit; a staged FP16 graph failed ANE compilation and fell back to a full GPU compile.

**4.5 Several capacities in one package.** Functions for several capacities can share one set of weights in one
asset. On the iPhone 18 Pro (Debug, plugged in, 512 tokens, without preallocated outputs):

| Package, at capacity | Tokens/s | Peak |
| --- | --- | --- |
| single 4,096, at 4,096 | 19.5 | 5.44 GB |
| two capacities, at 4,096 | 12.3 | 5.23 GB |
| four capacities, at 4,096 | 16.8 | 6.27 GB |
| single 16,384, at 16,384 | 14.4 | 5.995 GB |
| two capacities, at 16,384 | 11.5 | 5.45 GB |
| four capacities, at 16,384 | 11.3 | 6.30 GB |

The slowdown is in the ANE calls, not on the host. The four-capacity package also peaked about 0.8 GB higher at
4,096 (6.27 against 5.44 GB) and left 0.12–0.17 GB available.

**4.6 32K context.** The four-capacity package at 32,000 reached 6.44 GB and was terminated after 8 tokens. A
32,768-slot cache failed ANE conversion even alone (GPU at about 90 ms per call for six layers); 32,000 compiled.
Allocating the host caches from the function's input descriptor gave the same contiguous layout and footprint
(6.32 against 6.31 GB at 32,000): no zero-copy binding was found that would remove the per-call cache inputs.

**4.7 The GPU backend for this package.** Loaded with GPU preference, the shipped package ran out of memory: the
first run was terminated after 84 s with the reason `vm-pageout-starvation` (the weight files were paged through
on every token), and a second run with a warm cache took 13–16 s per token. This is a statement about this ANE
package on this backend; a model packaged for the GPU runs there at the speeds in Section 3.5. As a consequence,
energy could not be compared against a GPU arm of the same package.

**4.8 The 70 dB element-wise rule.** The original per-call rule (|Δ| ≤ 0.003 + 0.003·|reference| and PSNR > 70 dB)
is met by 9 of 71 outputs of the Mac's own FP16 run of the same graph, 8 for the ANE and 9 for the GPU
(Section 3.3): FP16 rounding over six layers alone exceeds it. The replacement rule keeps the intent (catch wrong
arithmetic) and is close to public practice for checking converted models; Apple's Stable Diffusion conversion,
for example, requires PSNR ≥ 35 dB ([`ABSOLUTE_MIN_PSNR`][sd-psnr]). It separates the arms: the ANE, the GPU and
Mac FP16 pass every output, the CPU arm fails 16. The element-wise rule remains in use for single-operator checks.

**4.9 A recompiled plan can leave the ANE.** Core AI caches compiled plans on the device. When an app update forced
a recompilation, chunk 0 ran at 24 ms instead of 8.9 ms and decode fell from 16.4 to 13.1 tokens/s; recompiling
that chunk alone restored 8.9 ms. Earlier, the compiler service disconnecting in the middle of a compile left a
cached plan that ran chunk 0 at 97 ms. Placement is a property of the compiled plan on a device, not of the package.

**4.10 Attention per query head.** Replicating the cache per query head made a six-layer step 8.65 ms at 4,096
and 15.09 ms at 16,384; computing attention per KV head (grouped) made it 7.91 and 10.24 ms.

Sources: `negative.json`, `quality.json`.

## 5 Limitations

- **One device for the main evidence.** Placement, quality, performance and the sustained run come from one
  iPhone 18 Pro with iOS 27.2. The whole-chain numerics and several probes come from an iPhone 17 Pro with
  iOS 27.0, where the ANE's copy of the weights was about 1.06 × rather than 1.3 × the chunk bytes.
- **What the placement test sees.** Exclusion and coincidence, not attribution; other processes' ANE activity is
  bounded by idle recordings only; CPU time was not sampled; Core AI's execution rows were empty; only `step_1`
  was traced, not `prefill_16`; the test uses a harness that loads one chunk at a time with seeded random
  caches, not the product's decode loop.
- **What the numerics test sees.** One token at one position with random caches per chunk; the whole-chain check
  covers 28 items, and its boundary items use up to 94% of the tolerance.
- **What the quality number means.** Likelihood of the original model's own continuations, not task accuracy;
  continuations of at most 128 tokens; thinking mode and long generations not evaluated; format alternatives
  judged on 18–20 items.
- **Performance.** One prompt length for first-token latency; power state not recorded for the performance runs;
  the sampler is not seeded (the procedure forces exact token counts, so speed does not depend on the text);
  energy not measured.
- **Sustained speed.** About 9 tokens/s once the device is in the fair state, which it reached after about a
  minute; one 11-minute run, ambient temperature not recorded.
- **Memory.** The ANE residency ratio was measured on a sister package; there is no component breakdown of the
  shipped peak.
- **Partition.** Six-layer chunks are bounded by measurements, not swept.
- **Context.** 16,384 of the model's 128K; each package has one capacity.
- **Cache reuse.** A turn reuses the cache only when its prompt extends exactly the cached tokens. About one
  turn in eight in development use fails this, because the chat template trims the assistant's text and the
  generated tokens re-tokenize differently; 2 of 19 turns did in the sustained run; with thinking enabled every turn re-prefills the
  conversation, because the previous turn's thinking is not kept in the history.
- **Size.** The download is 10.5 GB, two thirds of it FP16 lookup tables.
- **Reproducibility.** Package equivalence has been shown between two exports on the same machine; an independent
  rebuild from the public checkpoint on another machine has not yet been done, and the k-means palettization was
  shown to be deterministic on one machine only.

## 6 Future work

- Quantized lookup tables (INT8 or INT4 rows with per-row scales) to cut the download, judged on the same
  quality set.
- Wider prefill functions to raise prefill throughput.
- More context per byte: INT8 KV inputs, zero-copy input binding, or model state if a future compiler accepts it
  together with compressed weights.
- A runtime placement check: compare each chunk's call time and post-load footprint with the package's baseline
  and recompile a chunk that left the ANE.
- Close the instrument gaps: sample CPU time in the placement recording, explain the empty Core AI execution
  table, trace `prefill_16` and the product loop.
- Measure energy with a power profiler.
- Repeat the evaluation on other devices, and with a smaller-capacity package for devices with less memory.
- Reuse caches across turns by longest common prefix, which needs care with the sliding-window rings.
- Sweep chunk sizes; try calibrated rounding (GPTQ-style) for 4-bit MLP weights.
- Apply the recipe to a second model.

## Appendix A Reproducibility

### A.1 Hardware and software

| Item | Version |
| --- | --- |
| Phone | iPhone 18 Pro (iPhone19,2), iOS 27.2 (24B5089g) |
| Whole-chain numerics and earlier probes | iPhone 17 Pro, iOS 27.0 |
| App build for the placement, quality and first performance runs | Xcode 27.2 beta 2 (27B5028f), Release configuration |
| Trace recording | `xctrace` from the same Xcode; template "Core AI" with the "Points of Interest" and "Activity Monitor" instruments; all processes |
| Swift packages (current pins) | `apple/coreai-models` at `df8119879f125ad1e2e4c6249c2cddded75c190a`; `huggingface/swift-transformers` 1.3.4 |
| Converter (Python), as pinned by the exporter's requirements | torch 2.9.0, transformers 5.12.1, coreai-core 1.0.0b2, coreai-torch 0.4.2, coreai-models at the same commit, huggingface-hub 1.30.0 |
| Checkpoint | `google/gemma-4-E4B-it` at revision `ee0ef6023621cff504d758262d4e04895a5af4a2`, read from a local copy |
| Dates | Evidence recorded in September 2026; the shipped package exported in October 2026; external figures retrieved in October 2026 |

### A.2 Model package

Revision `lut6-c16384-20261001-01` at https://huggingface.co/royc6/within-gemma-4-e4b (tag of the same name):
35 files, 10,508,518,095 bytes, contract `gemma4-chunked-hostkv-v1`, capacity 16,384, sliding cache 512,
prefill width 16. The SHA-256 of every file is in the package's manifest; those of the eight `main.mlirb` files
and the two tables are also in `results/package.json`.

### A.3 Package equivalence

The Core AI converter gives composite subgraphs and resource handles random names (generated with
`secrets.choice`, which cannot be seeded), so two exports of the same graph are never byte-identical. Two packages
are called equivalent when, for every chunk, the program printed without source locations is identical after the
random names are renumbered in order of first appearance, and the bytecode resource section (all weights) is
byte-identical; every other file must be byte-identical except `main.hash`, `metadata.json` and the installed
manifest. The evidence in Section 3 was recorded on an export made in the converter's debug mode
(`lut6-c16384-20260930-01`), which records source paths in each asset; the shipped export was made in release
mode. All 8 chunks have the same program and byte-identical weights, the release export contains no local paths,
and the 11 other files are byte-identical. On the phone the shipped export decoded at 16.16 tokens/s with a
5.65 GB peak (Debug, optimized, 512 tokens), and the performance procedure passed again on it (Section 3.5).

### A.4 Procedures and seeds

| Procedure | Fixed parameters |
| --- | --- |
| Placement | 12 rounds per chunk and arm, arm order rotated each round; token id 1,000 at position 1,000; caches filled with uniform values in [−1, 1) from a 64-bit linear congruential generator (multiplier 6364136223846793005, increment 1442695040888963407, seed `0x5eed0930`); two idle recordings of the workload's length; permutation test (fallback only) with 20,000 rounds and seed 930 |
| Numerics | Round-0 inputs and outputs written by the app outside the measured windows; reference built with the exporter's quantization; PyTorch on the CPU, FP32 and FP16 |
| Whole chain | Frozen token ids; reference with the same rounding, one pass per sequence; tolerance 0.005 per item |
| Quality | Frozen set and references (hashes in Section 3.4); greedy references of at most 128 tokens; teacher forcing, no sampling |
| Performance | 32-token prompt; temperature 0.7, top-K 64, top-P 0.95; 3 × 64 warm-up, 5 × 512 measured; nominal start confirmed within 10 s; sampler not seeded |
| Sustained | 20 turns × 256 tokens, 5 s pause, ≥ 600 s, 12 fixed prompts, same sampling; memory at 1 Hz, thermal state every 5 s |

### A.5 Commands

Package export: one process per chunk, then a finalize step that writes `chunk-contract.json` and
`gemma-runtime.json`; the FP16 tables are exported separately; the converter runs in release mode. The defaults of
`within_core.convert` are the published configuration, so the commands in [`python/`](python/README.md) rebuild the
shipped package from a local checkpoint (`CKPT`) in a work folder (`WORK`):

```sh
for i in 0 1 2 3 4 5 6 7; do
  python -m within_core.convert chunk "$i" --checkpoint "$CKPT" --output "$WORK/export"
done
python -m within_core.convert finalize --output "$WORK/export"
python -m within_core.convert tables --checkpoint "$CKPT" --output "$WORK/tables"
python -m within_core.convert package --export "$WORK/export" --tables "$WORK/tables" \
  --checkpoint "$CKPT" --output "$WORK/release" --revision lut6-c16384-20261001-01 \
  --base-url https://huggingface.co/royc6/within-gemma-4-e4b/resolve/lut6-c16384-20261001-01/
```

Engine: `swift build` and `swift test` in this repository build the runtime and run the contract, prefill-plan,
mask, lookup and quality-score tests; the fixtures under `Tests/` are the shipped package's `gemma-runtime.json`
and `chunk-contract.json`.

### A.6 Result files

| File | Content |
| --- | --- |
| `results/package.json` | Package composition, chunk table, derived sizes, equivalence of the two exports |
| `results/placement.json` | Placement test design, verdict, per-arm and per-chunk summaries, per-window table |
| `results/numerics.json` | Per-call scores by chunk and arm, head distribution, whole-chain NLL fidelity |
| `results/quality.json` | Test set description and hashes, metric, results by subset, format alternatives |
| `results/performance.json` | Procedure, runs, capacity comparison, prefill and load times, published figures |
| `results/memory.json` | Budget, components, peaks by scenario |
| `results/sustained.json` | Workload, per-turn timings, thermal and memory results, pacing |
| `results/negative.json` | Rejected approaches with their conditions |

## Appendix B References

1. Google. Gemma 4 model card. https://ai.google.dev/gemma/docs/core/model_card_4
2. Google. LiteRT-LM Gemma 4 E4B model card (benchmarks, memory method, model size). https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm
3. Google AI Edge. LiteRT-LM: Gemma 4. https://developers.google.com/edge/litert-lm/models/gemma-4
4. CoreML-LLM project README. https://github.com/john-rocky/CoreML-LLM
5. Apple. ml-stable-diffusion, `torch2coreml.py` (`ABSOLUTE_MIN_PSNR`). https://github.com/apple/ml-stable-diffusion/blob/main/python_coreml_stable_diffusion/torch2coreml.py
6. Apple. Core AI models and Swift runtime. https://github.com/apple/coreai-models

[gemma-card]: https://ai.google.dev/gemma/docs/core/model_card_4
[litert-card]: https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm
[litert-dev]: https://developers.google.com/edge/litert-lm/models/gemma-4
[coreml-llm]: https://github.com/john-rocky/CoreML-LLM
[sd-psnr]: https://github.com/apple/ml-stable-diffusion/blob/main/python_coreml_stable_diffusion/torch2coreml.py
