# WithinDemo

A minimal iPhone app on WithinCore. It shows the whole path from a package on Hugging Face to text
on the screen, in four files, plus one for running it from a Mac:

| File | What it does |
| --- | --- |
| `Sources/ModelStore.swift` | Downloads the package named in `Resources/ModelCatalog.json` and checks every file's SHA-256 |
| `Sources/EngineHolder.swift` | Loads `Gemma4Engine` once and streams a reply |
| `Sources/Benchmark.swift` | The performance procedure behind the report's speed and first-token numbers |
| `Sources/Screens.swift` | Model, Chat and Benchmark screens |
| `Sources/LaunchCommand.swift` | The launch arguments below |

## Requirements

- An iPhone with an A19 Pro chip or later and 12 GB of memory, on iOS 27. Long runs make the phone warm, and thinner
  models warm up and slow down sooner, which lowers the benchmark's later runs.
- Xcode 27 and an Apple developer team: the app uses the increased memory limit entitlement.
- About 11 GB free on the iPhone. The package is 10.5 GB: 3.5 GB of chunk weights and 7.0 GB of
  FP16 embedding tables.

## Run

```sh
xcodegen generate   # or open the checked-in WithinDemo.xcodeproj
open WithinDemo.xcodeproj
```

Set your team and a bundle identifier of your own, then run on the device.

1. **Model**: tap Download. Cellular data is off unless you allow it. Keep the app open until it
   finishes.
2. **Chat**: the first message loads the model. The first load after installing compiles every
   chunk for the Neural Engine and takes a few minutes; later loads take seconds.
3. **Benchmark**: three 64-token warmups, then five 512-token runs, each starting at a nominal
   thermal state. It reports the median decode speed (1000 / median interval before tokens
   33–512), the median first-token latency and the peak memory footprint.

## Run from a Mac

With the package installed, two launch arguments run the app without taps and keep the screen on:

```sh
xcrun devicectl device process launch --device <device> --terminate-existing <bundle ID> -- -benchmark
xcrun devicectl device process launch --device <device> --terminate-existing <bundle ID> -- -chat "Hello" -max-tokens 32
```

`-benchmark` runs the Benchmark procedure; `-chat` sends one message (`-max-tokens` caps the reply, 1024 by
default). Each writes `Documents/benchmark.json` or `Documents/chat.json` after every step, with the thermal state
at each step and the engine's load and timing events, each with the memory footprint at that moment. Copy it back
with:

```sh
xcrun devicectl device copy from --device <device> --domain-type appDataContainer \
    --domain-identifier <bundle ID> --source Documents/benchmark.json --destination .
```
