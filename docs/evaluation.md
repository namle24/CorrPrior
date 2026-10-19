# Evaluation

Use the final EMA checkpoint from each 100k-update SR run. Do not choose a
checkpoint from RealSR or SVSR metrics. The main model is ungated CorrPrior;
historical filenames containing `finetune` may refer to the gated control.
Use the configuration generator to select the intended variant explicitly.

## Inference

```bash
python scripts/make_sr_config.py --mode test --variant corrprior \
  --prior-checkpoint checkpoints/prior_stage2.pth \
  --sr-checkpoint checkpoints/corrprior_ema_100k.pth \
  --dataset realsr --name corrprior_realsr \
  --output runs/configs/corrprior_realsr.yml
python scripts/run.py DRCT/drct/test.py -opt runs/configs/corrprior_realsr.yml
```

For SVSR use `--dataset svsr` and a different output config/run name. For
DRCT-Base or DRCT-RandPrior, use `--variant baseline` or `--variant random`
and that variant's final EMA checkpoint. Those two do not need a learned
prior checkpoint. The constructor still requires the selected prior file for
CorrPrior even though the final SR state also contains encoder weights.

The retained test path uses SR tiles of 128 LR pixels with 16-pixel padding.
This is distinct from the prior encoder's 256-pixel tiles on upsampled input,
with 64-pixel overlap. Outputs from the command above are under
`runs/corrprior_realsr/visualization/RealSR_Canon` and `RealSR_Nikon`.
BasicSR appends `_corrprior_realsr` to output filenames.

## Standard metrics

Use the same evaluator, crop and color convention for every method. The
reported PSNR/SSIM use luminance and a four-pixel crop border. Example:

```bash
python scripts/run.py DRCT/scripts/evaluate_standard_sr_metrics.py \
  --method CorrPrior --dataset RealSR_Canon \
  --sr_root runs/corrprior_realsr/visualization/RealSR_Canon \
  --sr_suffix _corrprior_realsr \
  --gt_root 'data/benchmarks/RealSR(V3)/Canon/HR' \
  --lq_root 'data/benchmarks/RealSR(V3)/Canon/LR' \
  --checkpoint checkpoints/corrprior_ema_100k.pth \
  --crop_border 4 --scale 4 \
  --output_csv runs/metrics/corrprior_canon.csv \
  --output_json runs/metrics/corrprior_canon.json
```

Repeat for Nikon and each control. The evaluator rejects filename mismatches;
it should not silently reduce the test set to whatever outputs happen to exist.

## Perceptual metrics

`DRCT/scripts/evaluate_pyiqa_metrics.py` measures LPIPS, DISTS and MUSIQ and
supports additional secondary checks. Its model weights may require downloads.
Use PyIQA 0.1.16 in a **separate evaluation environment**, as in the research
workflow, rather than upgrading the training environment in place. The isolated
PyIQA environment is not part of the initial CPU validation of this extraction.

```bash
python3 -m venv .venv-metrics
source .venv-metrics/bin/activate
python -m pip install -r requirements-metrics.txt
python DRCT/scripts/evaluate_pyiqa_metrics.py --help
```

Keep software versions, model weight hashes and evaluator settings with each
report. Use identical inputs and settings for both sides of every comparison.
No perceptual metric establishes human preference by itself.

## Paired statistics

RealSR averages camera subsets equally. After generating the per-image CSVs:

```bash
python scripts/run.py DRCT/scripts/camera_balanced_metric_statistics.py \
  --camera Canon runs/metrics/random_canon.csv runs/metrics/corrprior_canon.csv \
  --camera Nikon runs/metrics/random_nikon.csv runs/metrics/corrprior_nikon.csv \
  --metrics psnr ssim niqe --bootstrap_samples 10000 --permutation_samples 10000 \
  --output runs/statistics/corrprior_vs_random_realsr.csv
```

For SVSR, use `DRCT/scripts/paired_metric_statistics.py` with `--baseline`,
`--candidate`, `--output` and `--cluster_manifest runs/svsr_manifest.json`.
The manifest must include the exact evaluated image stems and scene identifiers.
Captures from one scene must remain grouped during resampling.

Confidence intervals quantify test-sample uncertainty, not training-seed
variation. Baseline/candidate ordering determines the sign of an effect.
Interpret results using the recorded endpoint and [scope limits](results.md).

## Additional diagnostics

- Prior prediction: `PD_MAE_SR/experiments/evaluate_restoration_need.py`.
- Spatial-mixture construction and validation: `PD_MAE_SR/utils/generate_spatial_mixture_benchmark.py` and `verify_spatial_mixture_benchmark.py`.
- Prior localization: `DRCT/scripts/evaluate_prior_localization.py`.
- Post-hoc zero/shift features: `DRCT/scripts/run_prior_feature_counterfactual.py`.
- Cost profiling: `DRCT/scripts/profile_journal_model.py` and `summarize_efficiency_profile.py`.

Use `python scripts/run.py PATH --help` for each command. Distinguish newly
generated diagnostics from the paper's recorded results; do not run historical
endpoint utilities with an unrelated dataset or substitute a favorable
secondary metric for an inconclusive primary endpoint.
