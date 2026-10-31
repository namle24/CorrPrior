# CorrPrior

**Correction-Supervised Spatial Prior for Real-World Image Super-Resolution**

Le Hoai Nam and Nguyen Hoang Ha · University of Science and Technology of Hanoi

CorrPrior learns a spatial prior from degraded–clean pairs using reconstruction
and correction supervision. The frozen encoder guides DRCT through residual SFT blocks.

![CorrPrior: prior learning in Stages 1–2, followed by frozen-encoder transfer to DRCT in Stage 3.](docs/assets/overview.png)

[Method](docs/method.md) · [Editable figure](docs/assets/architecture.drawio)

## Results

RealSR ×4, camera-balanced. Shared 100k-update SR budget; prior pretraining excluded.

| Method | PSNR (dB) ↑ | SSIM ↑ | DISTS ↓ | LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: |
| DRCT-Base | 27.2828 | 0.7807 | 0.196085 | **0.301174** |
| DRCT-RandPrior | 27.2985 | 0.7804 | 0.196111 | 0.302284 |
| **CorrPrior** | **27.4307** | **0.7831** | **0.195052** | 0.303456 |

[Full results and visual comparisons](docs/results.md)

## Getting started

[Installation](docs/reproducibility.md#installation) · [Data](docs/data.md) ·
[Training](docs/reproduction.md) · [Evaluation](docs/evaluation.md)

Pretrained checkpoints are not yet released.

## Citation and acknowledgements

[Citation](CITATION.cff) · [Licensing](THIRD_PARTY_NOTICES.md)

Built on [DRCT](https://github.com/ming053l/DRCT),
[BasicSR](https://github.com/XPixelGroup/BasicSR),
[MAE](https://github.com/facebookresearch/mae),
[timm](https://github.com/huggingface/pytorch-image-models) and
[SwinIR](https://github.com/JingyunLiang/SwinIR).
