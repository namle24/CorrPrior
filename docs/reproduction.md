# Training the three stages

Run commands from the repository root after installation. These are full
research jobs, not quick demos. They require the datasets in [data.md](data.md),
substantial GPU time and sufficient storage. No training job is launched by
installation or tests. Use only checkpoints from trusted sources.

## 1. Learn the prior

```bash
python scripts/run.py PD_MAE_SR/experiments/train_partial_restoration_need.py \
  --data_roots data/partial/DF2K/PD_60 data/partial/DF2K/PD_70 \
    data/partial/DF2K/PD_80 data/partial/DF2K/PD_90 \
  --output_dir runs/prior_stage1 --iters 200000 --batch_size 32 \
  --lr 1e-4 --weight_decay 0.05 --need_scale 10 --need_weight 0.2 \
  --reconstruction_floor 0.1 --val_fraction 0.05 --split_seed 0 \
  --seed 0 --save_freq 25000 --workers 4
```

Evaluate each saved checkpoint at 25k, 50k, ..., 200k on the same held-out
source subset. Example for the first checkpoint:

```bash
python scripts/run.py PD_MAE_SR/experiments/evaluate_partial_restoration_need.py \
  --data_roots data/partial/DF2K/PD_60 data/partial/DF2K/PD_70 \
    data/partial/DF2K/PD_80 data/partial/DF2K/PD_90 \
  --checkpoint runs/prior_stage1/rn_pd_mae_partial_iter25000.pth \
  --output_dir runs/stage1_eval_25000 --max_images 1000 \
  --val_fraction 0.05 --split_seed 0 --sample_seed 0 --batch_size 8 --workers 4
```

Select using all eight evaluations, not SR test metrics:

```bash
python scripts/run.py PD_MAE_SR/experiments/select_restoration_need_checkpoint.py \
  --mode partial \
  --evaluation 25000=runs/stage1_eval_25000 \
  --evaluation 50000=runs/stage1_eval_50000 \
  --evaluation 75000=runs/stage1_eval_75000 \
  --evaluation 100000=runs/stage1_eval_100000 \
  --evaluation 125000=runs/stage1_eval_125000 \
  --evaluation 150000=runs/stage1_eval_150000 \
  --evaluation 175000=runs/stage1_eval_175000 \
  --evaluation 200000=runs/stage1_eval_200000 \
  --checkpoint_pattern 'runs/prior_stage1/rn_pd_mae_partial_iter{iteration}.pth' \
  --expected_levels PD_60 PD_70 PD_80 PD_90 \
  --require_checkpoint --require_protocol --expected_split_unit source \
  --output runs/prior_stage1/selection.json
```

The selector minimizes macro reconstruction MSE, then correction MAE, then
iteration. The recorded run selected 25k; that is an observed outcome, not a
hard-coded instruction for a new run. Use `selected_checkpoint` from the JSON.

## 2. Adapt the prior

Set `CORRPRIOR_STAGE1` to the checkpoint selected above. Do not rename or move
the checkpoint without updating the selection record consistently.

```bash
CORRPRIOR_STAGE1="/absolute/path/to/selected_stage1_checkpoint.pth"
python scripts/run.py PD_MAE_SR/experiments/train_restoration_need.py \
  --hr_root data/train/HR_sub \
  --lq_roots data/train/HR_sub data/train/LR_90ugly_sub \
    data/train/LR_80ugly_sub data/train/LR_70ugly_sub data/train/LR_60ugly_sub \
  --pretrain "$CORRPRIOR_STAGE1" --pretrain_selection runs/prior_stage1/selection.json \
  --expected_pretrain_method partial_restoration_need_pretraining \
  --expected_selection_mode partial --output_dir runs/prior_stage2 \
  --iters 100000 --batch_size 32 --lr 1e-5 --weight_decay 0.05 \
  --need_scale 10 --need_weight 0.2 --reconstruction_floor 0.1 \
  --val_fraction 0.05 --split_seed 0 --split_unit source \
  --alignment_backend pytorch_bicubic --seed 0 --save_freq 25000
```

Evaluate 25k, 50k, 75k and 100k with
`PD_MAE_SR/experiments/evaluate_restoration_need.py`, using the same five
`--lq_roots` and `--hr_root`, `--split_unit source`,
`--alignment_backend pytorch_bicubic`, `--max_images 1000`, `--sample_seed 0`,
`--val_fraction 0.05`, `--split_seed 0`, `--batch_size 8` and `--workers 4`. Supply the relevant
`--checkpoint runs/prior_stage2/rn_pd_mae_iterITERATION.pth` and
`--output_dir runs/stage2_eval_ITERATION` for each checkpoint.

```bash
python scripts/run.py PD_MAE_SR/experiments/select_restoration_need_checkpoint.py \
  --mode calibration \
  --evaluation 25000=runs/stage2_eval_25000 \
  --evaluation 50000=runs/stage2_eval_50000 \
  --evaluation 75000=runs/stage2_eval_75000 \
  --evaluation 100000=runs/stage2_eval_100000 \
  --checkpoint_pattern 'runs/prior_stage2/rn_pd_mae_iter{iteration}.pth' \
  --expected_levels HR_sub LR_90ugly_sub LR_80ugly_sub LR_70ugly_sub LR_60ugly_sub \
  --identity_level HR_sub --identity_max 0.02 --calibration_tolerance 0.001 \
  --require_checkpoint --require_protocol --expected_split_unit source \
  --output runs/prior_stage2/selection.json
```

The recorded Stage 2 run selected 75k. The rule first requires mean correction
below 0.02 on clean inputs, then selects the earliest eligible checkpoint whose
degraded-input MAE is within 0.001 of the minimum. It does not consult SR results.

## 3. Train SR with matched controls

Provide the same 500k-update DRCT initialization used for every control. It is
not the final CorrPrior checkpoint and is not included in this repository.
Set `CORRPRIOR_STAGE2` to the selected Stage 2 checkpoint.

```bash
CORRPRIOR_STAGE2="/absolute/path/to/selected_stage2_checkpoint.pth"
python scripts/make_sr_config.py --mode train --variant corrprior \
  --init-checkpoint checkpoints/drct_base_500k.pth \
  --prior-checkpoint "$CORRPRIOR_STAGE2" \
  --output runs/configs/corrprior_train.yml
python scripts/make_sr_config.py --mode train --variant baseline \
  --init-checkpoint checkpoints/drct_base_500k.pth \
  --output runs/configs/baseline_train.yml
```

Before training, check the shared protocol and the selected prior:

```bash
python scripts/run.py scripts/audit_sr_protocol.py \
  --candidate runs/configs/corrprior_train.yml \
  --baseline runs/configs/baseline_train.yml --candidate-variant c2 \
  --prior-selection runs/prior_stage2/selection.json \
  --output runs/protocol_audit.json
python scripts/run.py DRCT/scripts/verify_journal_initialization.py \
  --candidate_config runs/configs/corrprior_train.yml \
  --baseline_config runs/configs/baseline_train.yml \
  --prior_selection runs/prior_stage2/selection.json --candidate_label C2 \
  --expected_prior_method restoration_need_calibration \
  --output runs/initialization_audit.json
python scripts/run.py DRCT/drct/train.py -opt runs/configs/corrprior_train.yml
```

The portable audit checks the configured prior path and SHA-256 against the
selection record, then applies the original matched-protocol checks. The
original checker expects a template with a null prior path; the adapter clears
that field on an in-memory copy only. It leaves the training config unchanged.
The initialization audit separately checks prior metadata, shared backbone
tensors and the initial SR outputs. Do not skip either audit. For the gated
variant use `--candidate-variant p0` and `--candidate_label P0`.

Train the baseline in a separate run. Generate the other configs using
`--variant random` and `--variant gated`, keeping the same initialization,
data, seed and SR budget. Random-prior training requires no learned prior.
Gated training requires the same selected prior as CorrPrior. Each generated
config has a separate experiment name and output directory. The generator
refuses to overwrite an existing config.

The protocol trains the entire DRCT and SFT blocks for 100k updates with batch
size four, Adam at 1e-4, the original milestones and EMA. The encoder stays
frozen. `mae_weight=0` disables the historical consistency loss. Evaluate only
the final `net_g_100000.pth` with `param_key_g=params_ema`.

For exploratory SwinIR transfer use `--backbone swinir` and the appropriate
shared SwinIR initialization. Its quality results are mixed; do not treat this
integration as evidence of backbone-independent gains.
