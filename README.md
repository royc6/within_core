# within_core

**within_core runs Google's Gemma 4 E4B language model on the Apple Neural Engine (ANE) of an iPhone, inside the
memory limit of a single app.** It is the on-device inference engine of within, a private AI chat app for iPhone, and
it is published here with the conversion pipeline, the verification tools, a demo app and a technical report.

- **Model:** Gemma 4 E4B-it by Google: 4.5 billion effective parameters (8 billion including embedding tables),
  42 decoder layers, converted into eight Core AI assets.
- **Device:** an iPhone with an A19 Pro chip or later and 12 GB of memory, iOS 27. All measurements below come
  from an iPhone 18 Pro with iOS 27.2.
- **Result:** every decoder chunk runs on the Neural Engine. Decode runs at 16.4 tokens/s with the KV cache compiled
  for 16,384 tokens (32-token prompt, 512 generated tokens; conditions under [Results](#results)), and the peak memory
  footprint is 5.73 GB against the 6.44 GB per-app limit.
- **Code:** a Swift package (the engine), Python (conversion and verification), and a SwiftUI demo app.
- **License:** Apache License 2.0 for this repository's code and documentation; the model itself is subject to the
  Gemma 4 terms (see [Others](#others)).

![One decode step: CPU work (tokenizer, embedding lookup, RoPE tables and masks, sampling, host-held KV cache) and the eight chunks on the Neural Engine](docs/figures/architecture.svg)

## Results

Measured on an iPhone 18 Pro with iOS 27.2 and the model package `lut6-c16384-20261001-01`. Every number comes from a
file in [`results/`](results/); conditions and methods are in the [technical report](REPORT.md).

| Question | Result |
| --- | --- |
| Does it run on the Neural Engine? | Yes. In a trace of all processes, 96 of 96 Neural Engine calls coincide with a Neural Engine prediction and 0 of 96 overlap the process's GPU work; the GPU and CPU control arms and two idle recordings behave as expected. |
| Are the Neural Engine's outputs numerically right? | 71 of 71 chunk outputs are within PSNR 35 dB and cosine 0.999 of an FP32 recomputation of the same quantized graph (worst 45.6 dB). |
| How much quality does quantization cost? | +0.0213 nats/token of negative log-likelihood over the FP32 original on a frozen set of 200 prompts (limit 0.05). |
| How fast is it? | 16.4 tokens/s decode with the KV cache compiled for 16,384 tokens<sup>1</sup>; 0.62–0.64 s to the first token of a 32-token prompt<sup>2</sup>. |
| How fast once the phone is warm? | About 9 tokens/s once the device warms and the engine paces itself (11-minute sustained run)<sup>3</sup>. |
| How much memory does it use? | A peak footprint of 5.73 GB of the 6.44 GB per-process limit over an 11-minute conversation<sup>4</sup>. |

1. Decode speed: a fixed 32-token prompt, no system prompt, thinking off; temperature 0.7, top-K 64, top-P 0.95;
   three warm-up generations of 64 tokens, then five runs of exactly 512 generated tokens (stop tokens suppressed),
   each started in the nominal thermal state; the median of the five runs. The capacity (16,384) is the number of
   tokens the KV cache is compiled for, not the length of the measured conversations, which reach 544 tokens. Every
   step binds the full cache, so speed depends on the capacity, not on the position: the same engine compiled for
   4,096 tokens decodes at 21.6 tokens/s (Debug build, plugged in). A speed means little without its capacity,
   prompt length and output length.
2. First token: from the request to the first sampled token, after warm-up; the medians for two exports of the same
   package. The first generation after loading the model takes 2.30–2.99 s.
3. Sustained run: one conversation of 20 turns of exactly 256 tokens, 5 s apart, 660 s in all, with the same
   sampling and the same 16,384-token capacity; airplane mode, screen at 50% brightness, unplugged. Once the phone
   leaves the nominal thermal state, the engine takes at least 110 ms per token.
4. Memory: `phys_footprint` sampled once a second. The limit is the one that applies with the increased memory limit
   entitlement.

All results: iPhone 18 Pro (iPhone19,2), iOS 27.2, Release build, all eight chunks on the Neural Engine. Google
publishes figures for Gemma 4 E4B on LiteRT-LM, the runtime of the Google AI Edge Gallery app, under different
conditions:

| Condition | This repository | Google's LiteRT-LM figures |
| --- | --- | --- |
| Device and compute unit | iPhone 18 Pro, Neural Engine | iPhone 17 Pro, GPU or CPU |
| KV capacity (context length) | 16,384 tokens | 2,048 tokens |
| Prompt and generated tokens | 32 and 512 | 1,024 and 256 |
| Weight precision | MLP 6-bit palettes, attention INT8, head INT4; embedding tables FP16 | not stated for E4B (2.24 GB of decoder weights, 0.67 GB of embedding parameters); Google's E2B card describes a mix of 2-, 4- and 8-bit weights |
| Model file | 10.5 GB | 3.66 GB |

Speed, time to first token and memory all depend on these conditions, so the two sets of figures cannot be compared
directly. Section 3.5 of the [technical report](REPORT.md) lists both with these caveats.

## How it works

1. **Partition.** The 42 decoder layers are exported as seven Core AI assets ("chunks") of six layers each, plus an
   eighth chunk for the vocabulary head, which returns its 262,144 logits as 16 slices. Each chunk is compiled for
   the Neural Engine.
2. **Host-held KV cache.** The app keeps the key-value (KV) caches in its own memory. Each layer reads its cache as an
   input and returns only the current step's keys and values, because the ANE compiler rejects KV kept as model
   state whenever the weights are compressed. Layers 24–41 reuse the KV of layers 22 and 23 (Gemma 4's KV sharing),
   so chunks 4–6 own no cache.
3. **Precision.** MLP weights are 6-bit palettized ("lut6"), attention and per-layer-embedding projections INT8, the
   vocabulary head INT4; activations are FP16. The 6.98 GB of embedding tables stay on the CPU as memory-mapped FP16
   lookups.
4. **Verification in four layers.** Placement (all-process traces with GPU and CPU control arms and idle recordings),
   per-call numerics (against an FP32 recomputation), whole-chain likelihood, and quality against the FP32 original.

The [technical report](REPORT.md) explains each choice, the platform constraints behind it, and the approaches that
failed.

## Requirements

- An iPhone with an A19 Pro chip or later and 12 GB of memory, running iOS 27.
- An app with the increased memory limit entitlement (`com.apple.developer.kernel.increased-memory-limit`).
- About 10.5 GB of storage for the model package.
- Xcode 27 to build.

## Quick start (Swift)

Add this package (`https://github.com/royc6/within_core`) to an iOS 27 app, put an installed model package folder on
the device (the demo shows a download that checks every file's SHA-256), then:

```swift
import WithinCore
import WithinCoreBase

let engine = try await Gemma4Engine(folder: packageFolder)
let options = Gemma4Engine.Options(temperature: 0.7, topK: 64, maxOutputTokens: 1024, thinking: false)
try await engine.generate(system: "", turns: [ChatTurn(role: .user, text: "Hello")], options: options) { event in
    if case .text(let replySoFar) = event { print(replySoFar) }  // the whole reply so far, once per token
}
```

The first load compiles every chunk for the Neural Engine and takes minutes; later loads come from the system's
model cache. Long generation makes any phone warm, and thinner models warm up and slow down sooner.

## Model package

The package is [royc6/within-gemma-4-e4b](https://huggingface.co/royc6/within-gemma-4-e4b) on Hugging Face at
revision `lut6-c16384-20261001-01`, converted from [Gemma 4 E4B-it](https://huggingface.co/google/gemma-4-E4B-it):
10.5 GB, of which 3.5 GB are chunk weights and 7.0 GB FP16 embedding tables. The engine never downloads anything; the
demo downloads the package only when the user starts it, with cellular data off unless allowed.

[`python/`](python/README.md) rebuilds the package offline from a local copy of the checkpoint. A rebuild has the
same program text and byte-identical weights; the files themselves are not byte-identical, because the converter
names subgraphs at random.

## Verify placement

[`python/`](python/README.md#verifying-placement) also contains `within_core.verify`, which decides from a device
recording whether the chunks ran on the Neural Engine and whether their outputs there are numerically right.

## Repository layout

| Path | Contents |
| --- | --- |
| [`Sources/WithinCore`](Sources/WithinCore) | Swift: the chunked decoder on Core AI with host-held KV, and `Gemma4Engine`, the generation loop |
| [`Sources/WithinCoreBase`](Sources/WithinCoreBase) | Swift, no Core AI: the package format and its checks, the chunk contract, rotary tables and attention masks, the prefill schedule, embedding lookup, the context budget |
| [`python/`](python/) | Conversion from a local checkpoint to the model package, and the placement and numerics verification |
| [`examples/WithinDemo`](examples/WithinDemo) | A minimal iPhone app: download the package, chat, and run the performance procedure |
| [`REPORT.md`](REPORT.md), [`results/`](results/), [`docs/figures/`](docs/figures/) | The technical report, its result summaries and figures |

## FAQ

**Does within_core use the GPU?** Not in the shipped path. In the traced run all eight chunks ran on the Neural
Engine; the GPU served only as a control when testing placement.

**Which iPhones are supported?** iPhones with an A19 Pro chip or later and 12 GB of memory, on iOS 27. Results were
measured on an iPhone 18 Pro with iOS 27.2.

**Does it work with other models?** The code targets Gemma 4 E4B. Section 2.9 of the [report](REPORT.md) lists which
steps carry over to another model or device and which must be redone.

**How do I cite it?** Use [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository").

## within

within is the chat app built on this engine; its interface, sync and storage are not part of this repository.

- [Privacy Policy](PRIVACY.md)
- [Support](SUPPORT.md)

Contact: within@royomni.com

## Others

- **Test conditions.** All tests and measurements here were made under specific conditions: the devices, OS and
  Xcode versions, model package, build settings, thermal and power states and workloads stated in the report.
  Results under other conditions may differ. Nothing here guarantees performance, Neural Engine placement, output
  quality or memory use on any other setup.
- **Context.** This repository documents one path that worked, and how it was verified, as a reference. Excerpts,
  quotes or single figures taken from it do not represent the author's complete meaning; read them together with the
  full report, including its conditions and limitations. Whether the approach suits your app, device or use is for
  you to judge, and to test before relying on it.
- **Model terms.** The model package is converted from Google's Gemma 4 E4B. Use of the model, and of anything it
  generates, is subject to the specific terms of the Gemma 4 license as published by Google ([Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)).
  This repository's Apache License 2.0 covers only its own code and documentation.
- **No warranty.** Everything here is provided "as is", without warranties or conditions of any kind (Apache
  License 2.0, sections 7 and 8).
- **Independence.** This is a personal project, not affiliated with or endorsed by Apple Inc. or Google LLC. Apple,
  iPhone and Apple Neural Engine are trademarks of Apple Inc.; Gemma is a trademark of Google LLC.

**中文**

- **测试条件。** 本仓库中的所有测试与测量均在特定条件下完成，即报告中注明的设备、系统与 Xcode 版本、模型包、编译设置、发热与供电状态以及工作负载。条件不同，结果可能不同。本仓库不保证在其他环境中的性能、Neural Engine 上的运行位置、输出质量或内存占用。
- **语境。** 本仓库记录了一条可行的路径及其验证过程，供参考。从中截取的片段、引述或单个数字不代表作者的完整意思，请结合完整报告（包括其条件与局限）阅读。这一方法是否适合你的应用、设备或用途，需要你自行判断，并在依赖之前自行测试。
- **模型条款。** 模型包由 Google 的 Gemma 4 E4B 转换而来。模型及其生成内容的使用，受 Google 发布的 Gemma 4 许可证具体条款约束（[Gemma 4 模型卡](https://ai.google.dev/gemma/docs/core/model_card_4)）。本仓库的 Apache License 2.0 仅适用于本仓库自身的代码与文档。
- **不提供担保。** 本仓库的所有内容均按“原样”提供，不附带任何形式的担保或条件（见 Apache License 2.0 第 7、8 条）。
- **独立性。** 这是个人项目，与 Apple Inc. 和 Google LLC 均无隶属或背书关系。Apple、iPhone 和 Apple Neural Engine 是 Apple Inc. 的商标；Gemma 是 Google LLC 的商标。

## License

Code and documentation in this repository are licensed under the [Apache License 2.0](LICENSE). The within name and
icon are not licensed. Cite this work with [`CITATION.cff`](CITATION.cff).
