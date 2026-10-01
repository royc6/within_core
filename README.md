# within_core

The on-device language-model engine behind **within**, a private AI chat app for iPhone, and a case study in fitting a
large language model onto heterogeneous mobile hardware.

This repository is being prepared. The conversion pipeline, the runtime and a technical report will be published here.
For now it hosts within's public pages:

- [Privacy Policy](PRIVACY.md)
- [Support](SUPPORT.md)

Contact: within@royomni.com

## Model package

within downloads [Gemma 4 E4B-it](https://huggingface.co/google/gemma-4-E4B-it), converted and quantized for iPhone,
from [royc6/within-gemma-4-e4b](https://huggingface.co/royc6/within-gemma-4-e4b) at revision `lut6-c16384-20261001-01`
(Apache License 2.0). The conversion pipeline that produces it will be published in this repository.

## License

Code and documentation in this repository are licensed under the [Apache License 2.0](LICENSE). The within name and
icon are not licensed.
