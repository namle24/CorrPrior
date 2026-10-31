# Method

CorrPrior learns a spatial prior: a grid of encoder features that describes
image content and the local corrections needed to recover a clean image.
The encoder learns through two tasks on aligned degraded–clean pairs:
clean-image reconstruction and patch-level correction prediction.

![CorrPrior architecture and the DRCT and SFT blocks.](assets/architecture.png)

[Full-size image](assets/architecture.png) ·
[Editable draw.io source](assets/architecture.drawio)

## Training stages

| Stage | Input | Trainable components | Purpose |
| --- | --- | --- | --- |
| 1. Prior learning | Partially degraded images | ViT encoder, reconstruction decoder and correction head | Learn content and local correction patterns |
| 2. Prior adaptation | Clean and fully degraded images | The same encoder, decoder and head | Adapt the prior to the input conditions used for SR |
| 3. Super-resolution | LR images | Full DRCT backbone and three residual SFT blocks | Reconstruct HR images using frozen prior features |

Stages 1 and 2 retain all input patches. The decoder reconstructs the clean
reference, while the correction head predicts a continuous target derived
from the difference between each degraded patch and its aligned clean patch.
Both losses update the encoder. This supervision requires paired images,
without manual degradation labels.

Stage 3 transfers the selected Stage 2 encoder weights and freezes them.
The LR image is bicubic-upsampled for tiled prior encoding; the merged spatial
features are resized to the LR feature grid and passed to SFT blocks after
DRCT groups 2, 4 and 6. Their zero-initialized projections preserve the
pretrained backbone's initial output. The complete DRCT backbone and SFT
blocks are then trained together.

The main SR model uses the frozen encoder and the trained DRCT/SFT network.
It does not use the prior decoder or correction head. The head provides
supervision for learning the prior; direct correction-score gating is a
separate control.

## Controls and implementation

- **DRCT-Base:** the backbone without a prior branch.
- **DRCT-RandPrior:** the same prior branch and SFT capacity, with a frozen
  randomly initialized encoder.
- **CorrPrior:** the learned, frozen encoder with feature-based SFT.
- **CorrPrior-Gated:** the learned prior with direct correction-score gating.

The repository also includes an exploratory SwinIR integration. See
[results](results.md) for the scope of each comparison and
[training](reproduction.md) for the configurations.

The prior is implemented in
[`mae_arch.py`](../PD_MAE_SR/archs/mae_arch.py); DRCT and the residual SFT
integration are in [`DRCT_arch.py`](../DRCT/drct/archs/DRCT_arch.py).
