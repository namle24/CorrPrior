# Data layout and preparation

No datasets or example benchmark crops are distributed in this repository.
The directory names below are conventions used by the configuration generator;
`--data-root` can point to storage outside the clone.

```text
data/
  sources/DF2K/                       Full-resolution Stage 1 source images
  partial/DF2K/PD_90/{HR,PD_LQ,mask}/  Same layout for PD_80, PD_70, PD_60
  train/HR_sub/                       Aligned DF2K+OST clean crops
  train/LR_90ugly_sub/                 Same filenames; 4x smaller degraded crops
  train/LR_80ugly_sub/
  train/LR_70ugly_sub/
  train/LR_60ugly_sub/
  benchmarks/RealSR(V3)/Canon/{LR,HR}/
  benchmarks/RealSR(V3)/Nikon/{LR,HR}/
  benchmarks/SVSR/extracted/SVSR/{LR,HR}/
  benchmarks/SVSR/GT_X4_by_LR_stem/
```

## Stage 1

The recorded experiment uses 3,450 DF2K source images and one deterministic
256x256 crop per source at four degradation settings (13,800 image pairs).
The generator saves a manifest and uses deterministic seeds:

```bash
python scripts/run.py PD_MAE_SR/utils/generate_partial_restoration_dataset_v2.py \
  --dataset DF2K=data/sources/DF2K --output_root data/partial \
  --levels 60 70 80 90 --patch_size 256 --patches_per_image 1 --seed 276
```

Data generation uses seed 276; training and splitting use seed 0. Keep the
generator's source names and manifests. The loader groups filenames
ending in `_pNNN_lLEVEL_SEED` by the source stem. The recorded split contains
3,277 training sources and 173 validation sources. It is not a random split
of the four degradation versions.

## Stages 2 and 3

These stages consumed an existing aligned corpus of 150,877 crop names from
6,486 DF2K+OST sources. Each clean filename must be present in all four degraded
folders. A source's crop names use `source_sNNN` so the prior loader can separate
source images rather than individual crops. Stage 2 uses 143,295 training crops
from 6,154 sources and 7,582 validation crops from 332 sources. Stage 3 uses the
four-level Cartesian-product loader (603,508 logical LR/HR pairs).

The crop inventory and generated images are not bundled. Reproducing the exact
reported samples requires the original source list, crop inventory and
degradation artifacts from the authors. The degradation implementation is
provided, but independently regenerating stochastic full-degradation data is
not guaranteed to reproduce the original corpus byte for byte. Do not describe
a newly generated dataset as the original without comparing its manifest.

`tools/degradation/RealisticDegradationGenerator_RealESRGAN.py --help` documents
the generic full-degradation utility. Its q90/q80/q70/q60 settings affect several
degradations, not JPEG quality alone. The deterministic partial-data generator
uses this implementation and the complexity mask in `PD_MAE_SR/utils/pd_mae_utils.py`.

## Benchmarks

RealSR evaluation uses 50 Canon and 50 Nikon pairs. Keep both cameras separate
for camera-balanced statistics. SVSR uses 1,119 captures from 141 scenes.
Prepare SVSR targets and its scene manifest using:

```bash
python scripts/run.py DRCT/scripts/prepare_svsr_x4.py \
  --lr_root data/benchmarks/SVSR/extracted/SVSR/LR \
  --hr_root data/benchmarks/SVSR/extracted/SVSR/HR \
  --gt_output_root data/benchmarks/SVSR/GT_X4_by_LR_stem \
  --manifest runs/svsr_manifest.json
```

The preparation utility checks matching identifiers and geometry. Evaluation
must not silently skip missing predictions or unmatched targets. Neither
benchmark is used for prior checkpoint selection or SR early stopping.
