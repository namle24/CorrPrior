# Attribution and licensing scope

A project-wide license has not yet been assigned to CorrPrior's original code.
The component licenses and upstream notices below remain applicable.

## DRCT

`DRCT/drct` derives from [ming053l/DRCT](https://github.com/ming053l/DRCT).
The upstream MIT license and its copyright notice are preserved in
[`DRCT/LICENSE`](DRCT/LICENSE).

The working research implementation adds the frozen prior wrapper, residual SFT,
optional correction gating, tiling, matched-control settings and EMA auxiliary
synchronization. The extraction retains these research sources unchanged; new
packaging helpers are under `scripts/` and path changes are recorded in the
source manifest.

## MAE and timm

The prior follows the encoder/decoder organization of
[facebookresearch/mae](https://github.com/facebookresearch/mae), retains all
patches, and adds paired correction supervision. Its positional-embedding
utilities closely follow the MAE implementation. The upstream MAE license is
[CC BY-NC 4.0](https://github.com/facebookresearch/mae/blob/main/LICENSE), copied
to [`LICENSES/MAE-LICENSE.txt`](LICENSES/MAE-LICENSE.txt). Retain the attribution
to Meta Platforms, Inc. and affiliates for MAE-derived material. Changes include
the convolutional patch embedding, restoration path and correction head.

Transformer blocks are imported from
[timm](https://github.com/huggingface/pytorch-image-models), not vendored here.
DRCT also contains utility code attributed to timm in its source comments;
the accompanying Apache 2.0 license is preserved in
[`LICENSES/timm-LICENSE.txt`](LICENSES/timm-LICENSE.txt).
The source does not include a complete file-level licensing inventory.
The DRCT MIT license applies to that component, not to the entire repository.

## BasicSR and SwinIR

[BasicSR](https://github.com/XPixelGroup/BasicSR) is installed as a dependency.
`RNPDMAESwinIR_arch.py` subclasses its SwinIR implementation, originally from
[SwinIR](https://github.com/JingyunLiang/SwinIR), and reuses CorrPrior's SFT/prior
modules. Their upstream licenses remain applicable to their code. Neither the
complete BasicSR nor the SwinIR source tree is redistributed in this extraction.

## Datasets and pretrained weights

Dataset collections (DF2K, OST, RealSR and SVSR), pretrained backbones and
metric networks are not redistributed. Obtain them from their providers under
the applicable terms. Selected dataset-derived images appear in the
[documentation figures](docs/figures.md) and retain their original terms.
This notice does not grant permission to redistribute those assets.
