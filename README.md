# CorrPrior

**Correction-Supervised Spatial Prior for Real-World Image Super-Resolution**

Le Hoai Nam and Nguyen Hoang Ha · University of Science and Technology of Hanoi

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

- **Benchmark:** RealSR ×4, with equal weighting of Canon and Nikon.
- **Protocol:** Shared SR initialization, data order, seed, loss and 100k updates. Prior pretraining is additional.

| Method | PSNR (dB) ↑ | SSIM ↑ | DISTS ↓ | LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: |
| DRCT-Base | 27.2828 | 0.7807 | 0.196085 | **0.301174** |
| DRCT-RandPrior | 27.2985 | 0.7804 | 0.196111 | 0.302284 |
| **CorrPrior** | **27.4307** | **0.7831** | **0.195052** | 0.303456 |

[SVSR, ablations, confidence intervals and visual comparisons](docs/results.md)

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
