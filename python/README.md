# within_core (Python): Gemma 4 E4B for the Apple Neural Engine

`within_core` converts a local checkpoint of
[google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) into the model package
that the within app downloads: eight uncompiled Core AI assets that the device compiles
for the Neural Engine, plus the tables, contract and manifests the app's runtime needs. The
published package is
[royc6/within-gemma-4-e4b](https://huggingface.co/royc6/within-gemma-4-e4b), revision
`lut6-c16384-20261001-01`; these commands reproduce it.

## What the package contains

| Part | Form |
|---|---|
| Decoder | 7 chunks of 6 layers (`chunk-00` to `chunk-06.aimodel`), each with a `step_1` (decode) entrypoint and, for the four chunks that own KV, a `prefill_16` entrypoint |
| Vocabulary head | `chunk-07.aimodel`: final norm, tied head in 16 row slices of 16384, logit softcap |
| KV | host-held: each chunk reads its layers' caches as inputs and returns this step's fresh K/V; full-attention caches hold 16384 positions, sliding-attention caches are 512-slot rings; layers 24 to 41 reuse the KV of layers 22 and 23 |
| Weights | MLP 6-bit palettes (`lut6`, groups of 16 output channels), attention and per-layer-embedding projections INT8 per channel, head INT4 per channel; activations, norms and KV FP16 |
| Embeddings | `embeddings.f16` and `perLayerEmbeddings.f16`: raw FP16 tables the host reads row by row, outside the graph |
| Metadata | `chunk-contract.json` (every chunk's inputs, outputs and shapes), `gemma-runtime.json` (geometry, RoPE, cache sizes, table formats), the unchanged tokenizer and generation configuration, `LICENSE` and `NOTICE`, and `installed-manifest.json` (equal to the app's bundled `ModelCatalog.json`) |

Terms: a *chunk* is one Core AI asset; *prefill* processes prompt tokens in blocks of 16
and returns only K/V; *decode* (`step_1`) produces logits for one token; *host-held KV*
means the caches live in the app's memory, not in the graph.

## Modules

| Module | Step |
|---|---|
| `checkpoint.py` | verified, bounded, offline reads from the local checkpoint (`reference-config.json` pins the revision and every file's SHA-256) |
| `layers.py` | channels-first RMSNorm (as a layer norm), per-layer embedding projection, feedforward and PLE path |
| `attention.py` | grouped-query attention over host-held KV |
| `chunks.py` | decoder blocks, layer and head chunks, and their interface (`ChunkLayout`, `HeadLayout`) |
| `quantize.py` | mixed-precision weight compression, including the head one row block at a time |
| `export.py` | the plan, building and authoring one chunk, `chunk-contract.json` and `gemma-runtime.json` |
| `tables.py` | the two FP16 embedding tables |
| `package.py` | package assembly and the manifest rules the app enforces |
| `convert.py` | the command line |
| `verify/trace.py` | reading `xctrace export` tables, checked against the recording's table of contents |
| `verify/review.py` | the placement and numerics verdict of a recorded run (see "Verifying placement") |
| `verify/reference.py` | FP32 recomputation of each chunk's outputs from the recorded inputs |
| `verify/__main__.py` | the verification command line |

## Prerequisites

- macOS 26 or later on Apple silicon, and Python 3.12.
- The pinned Python stack: `pip install -e ".[dev]"` in this folder. `torch`,
  `transformers` and Apple's Core AI authoring packages come from PyPI (`coreai-core`, which
  provides the `coreai` import package, `coreai-torch` and `coreai-opt`); `coreai-models`
  comes from [apple/coreai-models](https://github.com/apple/coreai-models) at commit
  `df8119879f125ad1e2e4c6249c2cddded75c190a`.
- A local copy of `google/gemma-4-E4B-it` at revision
  `ee0ef6023621cff504d758262d4e04895a5af4a2`. The conversion reads `model.safetensors`,
  `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json` and
  `chat_template.jinja`, and refuses any file whose size or SHA-256 differs from
  `reference-config.json`. It never downloads anything.
- About 11 GB of free disk (3.3 GB of chunks, 7 GB of tables; the package hard-links them).
  Each command refuses to start unless `--min-free-gb` (default 10) remains free afterwards.
- Time: one layer chunk takes 8 to 9 minutes on an Apple-silicon Mac with 16 GB of memory,
  mostly 6-bit palettization; the whole conversion about an hour.

## Reproducing the published package

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CKPT=/path/to/gemma-4-E4B-it
WORK=/path/to/work

# One process per chunk keeps a single chunk's weights in memory.
for i in 0 1 2 3 4 5 6 7; do
  python -m within_core.convert chunk "$i" --checkpoint "$CKPT" --output "$WORK/export"
done
python -m within_core.convert finalize --output "$WORK/export"
python -m within_core.convert tables --checkpoint "$CKPT" --output "$WORK/tables"
python -m within_core.convert package --export "$WORK/export" --tables "$WORK/tables" \
  --checkpoint "$CKPT" --output "$WORK/release" --revision lut6-c16384-20261001-01 \
  --base-url https://huggingface.co/royc6/within-gemma-4-e4b/resolve/lut6-c16384-20261001-01/
```

The defaults are the published configuration (context 16384, 4 CPU threads). The package is
in `$WORK/release/package`, and `$WORK/release/ModelCatalog.json` is the catalog the app
bundles.

## What "the same package" means

The Core AI converter names composite subgraphs and weight resources with random suffixes
(drawn with `secrets.choice`), so a `main.mlirb` is not byte-reproducible. A reproduction
is equivalent to the published package when:

- for every `chunk-*.aimodel/main.mlirb`, the program text printed without debug
  information is identical after the converter's names (`@<name>_<8 lowercase letters>` and
  `resource_<n>`) are renumbered in order of first appearance, and the bytecode's resource
  section (all weights, with its encoding and padding) is byte-identical;
- `main.mlirb` holds no local file path (the converter runs in release mode);
- every other file is byte-identical, except `main.hash` (the SHA-256 of `main.mlirb`),
  `metadata.json` (the creation time) and, through those digests,
  `installed-manifest.json` and `ModelCatalog.json`.

## Verifying placement

`within_core.verify` decides from a device recording whether the chunks ran on the Neural
Engine (the ANE-EXEC-v1 verdict), and whether their outputs there are numerically right.

What this can and cannot prove: the proof is by exclusion plus coincidence, not
attribution. It cannot show which operation ran on the ANE; the idle control bounds, and
does not exclude, ANE rows from other processes inside a window; it covers only the
instrumented stages.

### The workload a host app runs

The recording needs a host app on the device that runs every chunk of the installed
package and marks each call with signposts. The within app does it when launched with a
developer argument; any app that follows this description works:

- Each chunk's `step_1` runs on three arms of the same asset, loaded with Core AI's
  preferred compute unit kind `.neuralEngine` (the candidate), `.gpu` and `.cpu` (the
  controls). One chunk at a time, all three arms loaded together; the previous chunk's
  arms are released first, since three copies of the whole model do not fit in memory.
- The inputs are the runtime's own decode inputs for one token (token 1000 at position
  1000): the embedding row times the square root of the hidden size, the per-layer
  embedding rows times the square root of their width (sliced to the chunk's layers), the
  RoPE cosines and sines of both attention kinds, the full-attention mask over
  context + 1 slots and the sliding ring mask over window + 1 slots. The KV caches are
  filled with seeded uniform values in [-1, 1) (a 64-bit linear congruential generator,
  rounded to FP16), the same bytes for every arm. Each chunk takes `hidden` and its donors'
  fresh K/V from the candidate arm's outputs of the chunk before it.
- 12 rounds per chunk; each round runs the three arms with identical inputs, rotating the
  order every round so that no arm always runs first: 96 calls per arm for the 8 chunks
  (the review needs at least 90).
- Every call is bracketed by three Points of Interest intervals of one `OSSignposter`
  (the app's subsystem, category `pointsOfInterest`): `PartInputMaterialize` (taking the
  inputs), `PartInference` (the `run` call) and `PartReadback` (copying every output out as
  FP16). Each interval's message is the call's label, `<arm> c<chunk> r<round>`, e.g.
  `neuralEngine c3 r7`.
- The app writes `report.json`: `state` (`complete`), `pid`, `revision`, `configuration`,
  `samples` (one per call: `arm`, `chunk`, `round`, `label`), `chunks` (one per chunk:
  `chunk`, `layers` as `[start, end]`, `[42, 42]` for the head, and per arm
  `outputs<arm>`, a list of `{output, file, shape}`) and `inputs<chunk>` (a list of
  `{input, file, shape}`). The files are round 0's outputs, `c<chunk>-<arm>-<name>.f16`,
  and the inputs, `in-c<chunk>-<name>.f16`, raw FP16 in C order.

### The recording

An all-process recording with the Core AI template plus Points of Interest and Activity
Monitor, started before the app is launched:

```sh
DEVICE=<UDID from xcrun xctrace list devices>
xcrun xctrace record --template 'Core AI' --instrument 'Points of Interest' \
  --instrument 'Activity Monitor' --device "$DEVICE" --all-processes \
  --time-limit 900s --output RUN/recording.trace
```

Once the recorder prints `Ctrl-C to stop the recording`, launch the workload. When the app
reports that it finished, wait about 3 seconds for trailing hardware rows, stop the
recorder with SIGINT, copy the app's output folder to `RUN/device/<folder>/` and terminate
the app. Then make two idle recordings with the app not running, each as long as the
workload took (at least 30 seconds), the idle control of C3:

```sh
for name in idle-1 idle-2; do
  xcrun xctrace record --template 'Core AI' --instrument 'Points of Interest' \
    --instrument 'Activity Monitor' --device "$DEVICE" --all-processes \
    --time-limit "${SECONDS_OF_WORKLOAD}s" --output "RUN/$name.trace"
done
```

List every file of each trace bundle with its size and SHA-256
(`within_core.verify.trace.bundle_manifest`), then export the tables the review reads:

```sh
mkdir RUN/exports
T='/trace-toc/run[@number="1"]/data/table'
X="xcrun xctrace export --input RUN/recording.trace"
$X --toc --output RUN/exports/recording-toc.xml
for schema in ane-hw-intervals metal-gpu-intervals activity-monitor-process-live; do
  $X --xpath "$T[@schema=\"$schema\"]" --output "RUN/exports/recording-$schema.xml"
done
# The Core AI table is CoreAIProfile in Xcode 27.2 and ODIEProfile in 27.0; the TOC says which.
$X --xpath "$T[@schema=\"CoreAIProfile\"]" --output RUN/exports/recording-coreai.xml
$X --xpath "$T[@schema=\"os-signpost\"][@category=\"PointsOfInterest\"][not(@dynamic-tracing-enabled-subsystems)]" \
  --output RUN/exports/recording-signposts.xml
for name in idle-1 idle-2; do
  xcrun xctrace export --input "RUN/$name.trace" --toc --output "RUN/exports/$name-toc.xml"
  xcrun xctrace export --input "RUN/$name.trace" --xpath "$T[@schema=\"ane-hw-intervals\"]" \
    --output "RUN/exports/$name-ane-hw-intervals.xml"
done
```

The Activity Monitor table is kept for inspection; the verdict does not read it. List the
bundles again: export must not change them. `RUN/result.json` then records the run:
`device` (the UDID, which P0 compares with the recording's target), `folder` (the device
folder's name), `pid` (the app's, from `report.json`), `coreAITable`, `traceManifests`
(the listings taken before export, by bundle: `recording`, `idle-1`, `idle-2`) and
`complete` (true only if the app reported `complete`, the recorder exited with status 0
and export left the bundles unchanged). The run folder is then:

```text
RUN/
  result.json
  recording.trace/  idle-1.trace/  idle-2.trace/
  device/<folder>/report.json, in-c*.f16, c*-<arm>-*.f16
  exports/recording-{toc,signposts,coreai,ane-hw-intervals,metal-gpu-intervals}.xml
  exports/idle-{1,2}-{toc,ane-hw-intervals}.xml
```

### The verdict

```sh
# FP32 recomputation of each chunk from the run's own inputs, one chunk per process.
for i in 0 1 2 3 4 5 6 7; do
  python -m within_core.verify reference RUN --checkpoint "$CKPT" --chunks "$i" \
    --output "RUN/numerics-c$i.json"
done
python -m within_core.verify review RUN --numerics RUN/numerics-c*.json \
  --subsystem <the app's signpost subsystem> --output RUN/review.json
```

`reference` rebuilds each chunk exactly as the conversion does (same graph, same
compressed weights), runs it in FP32 on the recorded input bytes and scores every arm's
outputs against it: PSNR (peak of the reference over the error, `null` when there is no
error), cosine, relative L2 and largest error; for the head, also top-1, KL divergence
and top-10 overlap over the 262144-token distribution joined from the 16 slices. It takes
about as long per chunk as conversion. `review` writes the verdict and exits with status
1 unless it passed. PASS = P0, C0, C1, C3 and C6; C2, C4 and C5 are recorded:

| Criterion | Rule |
|---|---|
| P0 | an all-process recording of the device in `result.json`, listing the app's PID; every call has exactly one completed `PartInputMaterialize`, `PartInference` and `PartReadback` interval, in that order; no unfinished or unexpected interval; trace bundles unchanged |
| Window | from the start of `PartInference` to the end of `PartReadback`; at least 90 per arm, the same number for each |
| C0 | the candidate has windows on every chunk |
| C1 | the app's own GPU intervals overlapping a window: candidate 0 of N, GPU control N of N |
| C3 | Neural Engine Prediction rows inside a window: candidate N of N; GPU control 0 of N (`decidedBy: gpuControlZero`) or a permutation p below 0.001 (`decidedBy: permutation`); each idle recording's prediction rate below 1% of the rate inside candidate windows |
| C4 | the app's Core AI `Function Execute` rows per window (recorded) |
| C5 | per arm, the share of window time covered by the app's GPU intervals (recorded) |
| C2 | needs a Time Profiler table, which this recording does not have (not recorded) |
| C6 | every candidate output: PSNR at least 35 dB and cosine at least 0.999 against the FP32 reference; the head: same top-1 token and KL at most 1e-3 nats |

The permutation test: each candidate and GPU-control window counts as 1 if it holds a
Neural Engine Prediction row; the statistic is the candidate's count minus the control's.
Under the hypothesis that the arm label does not matter, the labels are exchangeable; the
p-value is `(1 + hits) / (1 + 20000)`, where hits counts how many of 20000 random
relabelings (NumPy PCG64, seed 0) reach the observed statistic. It is always reported;
it decides C3 only when the GPU control has a prediction row in some window.

## Tests

```sh
python -m pytest
ruff format --check . && ruff check .
```

The tests compare the Neural Engine formulation with Transformers on toy-width models with
E4B's layer topology (norms, per-layer projection, feedforward, attention, and the chunks
driven by a host loop through prefill and decode with ring caches), check the chunk
contract and KV layout, the bounded head quantization and the manifest rules, and run the
whole conversion on a tiny checkpoint. The verification tests review small synthetic
recordings written the way `xctrace export` writes them (the pass, each way to fail, and
the permutation branch) and run the FP32 reference on a tiny checkpoint. They take about
45 seconds and execute no Core AI program.
