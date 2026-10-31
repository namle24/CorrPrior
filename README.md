# CorrPrior

**Correction-Supervised Spatial Prior for Real-World Image Super-Resolution**

Le Hoai Nam and Nguyen Hoang Ha · University of Science and Technology of Hanoi

[Method](#method) · [Results](#results) · [Installation](#installation) · [Training and evaluation](#training-and-evaluation)

CorrPrior learns spatial features from degraded–clean image pairs and uses them
to guide real-world 4× super-resolution.

## Method

- **Stage 1 — Prior learning:** Train a ViT prior on partially degraded images using clean-image reconstruction and patch-level correction supervision.
- **Stage 2 — Prior adaptation:** Continue training the same encoder, decoder and correction head on clean and fully degraded images.
- **Stage 3 — Super-resolution:** Freeze the encoder and train the full DRCT backbone with three residual Spatial Feature Transform (SFT) blocks.
- **Correction head:** Predicts how much each patch differs from its clean reference. The head and decoder are used only during prior learning in the main model.

![CorrPrior: prior learning in Stages 1–2, followed by frozen-encoder transfer to DRCT in Stage 3.](docs/assets/overview.png)

[Method details](docs/method.md) · [Full architecture](docs/assets/architecture.png) · [Editable figure](docs/assets/architecture.drawio)

## Results

- **Evaluation:** Same LR inputs and evaluator; PSNR/SSIM use luminance with a four-pixel border crop.
- **External methods:** Each retains its own training history.
- **Bold values:** Best metric values in each table, including ties.

### RealSR ×4

Canon and Nikon are weighted equally.

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ |
| --- | ---: | ---: | ---: | ---: |
| BSRGAN | 26.5080 | 0.7750 | 0.2687 | 0.1759 |
| Real-ESRGAN | 25.8287 | 0.7733 | 0.2730 | 0.1688 |
| SwinIR-M | 26.0297 | 0.7807 | 0.2597 | 0.1610 |
| SwinIR-L | 26.4382 | 0.7869 | 0.2516 | **0.1545** |
| SRFormer-S | 25.7279 | 0.7768 | **0.2495** | 0.1601 |
| E2-RealSR | 26.4169 | 0.7821 | 0.2862 | 0.1889 |
| DAA-IMDN | **27.8407** | **0.8030** | 0.3484 | 0.2020 |
| AdcSR | 25.6082 | 0.7326 | 0.2989 | 0.1864 |
| DRCT-Base | 27.2828 | 0.7807 | 0.3012 | 0.1961 |
| **CorrPrior** | 27.4307 | 0.7831 | 0.3035 | 0.1951 |

### SVSR ×4

1,119 captures from 141 scenes.

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ |
| --- | ---: | ---: | ---: | ---: |
| BSRGAN | 26.8788 | 0.8234 | 0.2927 | 0.1886 |
| Real-ESRGAN | 26.0319 | 0.8087 | 0.2996 | 0.1813 |
| SwinIR-M | 26.4013 | 0.8200 | 0.2953 | 0.1789 |
| SwinIR-L | 26.6967 | 0.8250 | **0.2800** | **0.1715** |
| SRFormer-S | 25.7614 | 0.8036 | 0.2829 | 0.1743 |
| E2-RealSR | 27.2979 | 0.8037 | 0.3553 | 0.2292 |
| DAA-IMDN | 27.4485 | 0.8089 | 0.4538 | 0.2436 |
| AdcSR | 25.4297 | 0.7488 | 0.3439 | 0.2004 |
| DRCT-Base | 27.5431 | **0.8302** | 0.3033 | 0.2165 |
| **CorrPrior** | **27.5461** | **0.8302** | 0.3031 | 0.2141 |

The matched SVSR PSNR difference between CorrPrior and DRCT-Base is inconclusive;
see the [paired comparisons](docs/results.md#controlled-comparisons).

### Controlled comparisons on RealSR

- **Matched SR training:** Same initialization, data order, seed, loss and 100k updates. Prior pretraining is additional.
- **Random-prior control:** Same encoder architecture and SFT capacity as CorrPrior.

| Method | Prior | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| DRCT-Base | None | 27.2828 | 0.7807 | **0.301174** | 0.196085 |
| DRCT-RandPrior | Frozen random encoder | 27.2985 | 0.7804 | 0.302284 | 0.196111 |
| **CorrPrior** | Frozen learned encoder | **27.4307** | **0.7831** | 0.303456 | **0.195052** |
| CorrPrior-Gated | Learned encoder and score gating | 27.4290 | 0.7828 | 0.303573 | 0.195770 |

[Confidence intervals and diagnostics](docs/results.md#controlled-comparisons) ·
[Visual comparisons](docs/results.md#qualitative-comparison)

## Installation

- **Python:** 3.12.
- **PyTorch:** Install a build compatible with your hardware; see [environment notes](docs/reproducibility.md#environment).

```bash
git clone https://github.com/namle24/CorrPrior.git
cd CorrPrior
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1
python -m pip install -r requirements.txt
```

## Training and evaluation

- **Data:** Follow the [dataset layout and preparation guide](docs/data.md).
- **Training:** See the [three-stage training commands](docs/reproduction.md), including matched controls.
- **Evaluation:** Supply the Stage 2 prior and final SR checkpoints; example paths are shown below.
- **Checkpoints:** Public downloads are not yet available.

```bash
python scripts/make_sr_config.py --mode test --variant corrprior \
  --prior-checkpoint checkpoints/prior_stage2.pth \
  --sr-checkpoint checkpoints/corrprior_ema_100k.pth \
  --dataset realsr --name corrprior_realsr \
  --output runs/configs/corrprior_realsr.yml
python scripts/run.py DRCT/drct/test.py -opt runs/configs/corrprior_realsr.yml
```

[Metrics and evaluation settings](docs/evaluation.md) · [Code layout and documentation](docs/README.md)

## Citation and acknowledgements

- **Citation:** Use [CITATION.cff](CITATION.cff) to cite this repository.
- **Acknowledgements:** Built on [DRCT](https://github.com/ming053l/DRCT), [BasicSR](https://github.com/XPixelGroup/BasicSR), [MAE](https://github.com/facebookresearch/mae), [timm](https://github.com/huggingface/pytorch-image-models) and [SwinIR](https://github.com/JingyunLiang/SwinIR).
- **Licensing:** See [component licenses and attribution](THIRD_PARTY_NOTICES.md).
