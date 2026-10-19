"""Evaluate RN-PD-MAE on known spatial degradation mixtures.

The benchmark blends aligned mild and hard LQ observations with a saved soft
coefficient field. The field is not treated as ground-truth restoration need;
it is used only as an external spatial-ordering signal. The primary alignment
diagnostic compares the predicted need to the method's actual training target,
computed from the paired bicubic-upsampled LQ and HR images. The evaluator also
reports coefficient alignment, a half-field-shift null, weighted contrast, and
pure hard-versus-mild ordering without selecting or training a checkpoint.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def average_ranks(values):
    """Return one-based average ranks with deterministic tie handling."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def correlation(first, second, rank=False):
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.size != second.size or first.size < 2:
        raise ValueError('Correlation arrays must have equal size of at least two.')
    if rank:
        first = average_ranks(first)
        second = average_ranks(second)
    if first.std() < 1e-12 or second.std() < 1e-12:
        return math.nan
    return float(np.corrcoef(first, second)[0, 1])


def roc_auc(scores, soft_labels):
    """Mann-Whitney ROC AUC after thresholding the known mixture at 0.5."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(soft_labels, dtype=np.float64).reshape(-1) >= 0.5
    if scores.size != labels.size:
        raise ValueError('AUC scores and labels must have equal size.')
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = average_ranks(scores)
    positive_rank_sum = float(ranks[labels].sum())
    u_statistic = positive_rank_sum - positives * (positives + 1) / 2.0
    return u_statistic / (positives * negatives)


def weighted_contrast(prediction, mask):
    prediction = np.asarray(prediction, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    if prediction.shape != mask.shape:
        raise ValueError('Prediction and mask must have the same shape.')
    hard_mean = float((prediction * mask).sum() / max(mask.sum(), 1e-12))
    mild_weight = 1.0 - mask
    mild_mean = float(
        (prediction * mild_weight).sum() / max(mild_weight.sum(), 1e-12)
    )
    return hard_mean, mild_mean, hard_mean - mild_mean


def localization_metrics(prediction, mask):
    prediction = np.asarray(prediction, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    if prediction.shape != mask.shape:
        raise ValueError('Prediction and mask must have the same shape.')
    shifted = np.roll(
        mask,
        shift=(max(1, mask.shape[0] // 2), max(1, mask.shape[1] // 2)),
        axis=(0, 1),
    )
    hard_mean, mild_mean, contrast = weighted_contrast(prediction, mask)
    pearson = correlation(prediction, mask)
    shifted_pearson = correlation(prediction, shifted)
    return {
        'pearson_aligned': pearson,
        'spearman_aligned': correlation(prediction, mask, rank=True),
        'auc_aligned': roc_auc(prediction, mask),
        'hard_weighted_need_mean': hard_mean,
        'mild_weighted_need_mean': mild_mean,
        'hard_minus_mild_weighted_need': contrast,
        'pearson_half_shift': shifted_pearson,
        'spearman_half_shift': correlation(prediction, shifted, rank=True),
        'aligned_minus_shift_pearson': pearson - shifted_pearson,
    }


def paired_correction_target(lq, gt, prediction_shape, input_upscale=4,
                             patch_size=8, scale=10.0):
    """Return the token-wise target used to train the restoration-need head.

    The frozen prior first bicubically upsamples an LR image and pads it by
    replication to a complete encoder-token grid. Repeating those operations
    here keeps the diagnostic aligned with both pretraining supervision and
    downstream prior extraction, including non-square images whose HR size is
    not divisible by the MAE patch size.
    """
    if lq.ndim != 4 or gt.ndim != 4 or lq.shape[0] != gt.shape[0]:
        raise ValueError('LQ and GT must be aligned BCHW batches.')
    if lq.shape[1] != gt.shape[1]:
        raise ValueError('LQ and GT must have the same channel count.')
    if input_upscale <= 0 or patch_size <= 0 or scale <= 0:
        raise ValueError('Upscale, patch size, and target scale must be positive.')
    if len(prediction_shape) != 2:
        raise ValueError('prediction_shape must contain token height and width.')

    observation = F.interpolate(
        lq,
        scale_factor=input_upscale,
        mode='bicubic',
        align_corners=False,
    ).clamp(0.0, 1.0)
    if observation.shape != gt.shape:
        raise ValueError(
            'Upsampled LQ and GT must have equal shapes, got '
            f'{tuple(observation.shape)} and {tuple(gt.shape)}.'
        )

    token_height, token_width = map(int, prediction_shape)
    padded_height = token_height * patch_size
    padded_width = token_width * patch_size
    height, width = observation.shape[-2:]
    if padded_height < height or padded_width < width:
        raise ValueError(
            'Prediction token grid is smaller than the aligned HR observation.'
        )
    padding = (0, padded_width - width, 0, padded_height - height)
    if padding[1] or padding[3]:
        observation = F.pad(observation, padding, mode='replicate')
        gt = F.pad(gt, padding, mode='replicate')

    batch, channels = observation.shape[:2]
    correction = (gt - observation).abs().reshape(
        batch,
        channels,
        token_height,
        patch_size,
        token_width,
        patch_size,
    ).mean(dim=(1, 3, 5))
    return 1.0 - torch.exp(-scale * correction)


def correction_alignment_metrics(prediction, target):
    """Measure prediction alignment with paired correction-demand targets."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError('Prediction and correction target must have equal shapes.')
    shifted = np.roll(
        target,
        shift=(max(1, target.shape[0] // 2), max(1, target.shape[1] // 2)),
        axis=(0, 1),
    )
    error = prediction - target
    pearson = correlation(prediction, target)
    shifted_pearson = correlation(prediction, shifted)
    return {
        'correction_target_mean': float(target.mean()),
        'correction_prediction_mean': float(prediction.mean()),
        'correction_mae': float(np.abs(error).mean()),
        'correction_rmse': float(np.sqrt(np.square(error).mean())),
        'correction_bias': float(error.mean()),
        'correction_pearson': pearson,
        'correction_spearman': correlation(prediction, target, rank=True),
        'correction_pearson_half_shift': shifted_pearson,
        'correction_spearman_half_shift': correlation(
            prediction, shifted, rank=True
        ),
        'correction_aligned_minus_shift_pearson': pearson - shifted_pearson,
    }


def finite_summary(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {'count': 0, 'mean': None, 'sample_standard_deviation': None}
    return {
        'count': int(finite.size),
        'mean': float(finite.mean()),
        'sample_standard_deviation': (
            float(finite.std(ddof=1)) if finite.size > 1 else None
        ),
        'median': float(np.median(finite)),
        'minimum': float(finite.min()),
        'maximum': float(finite.max()),
    }


def aggregate_rows(rows):
    metrics = (
        'pearson_aligned',
        'spearman_aligned',
        'auc_aligned',
        'hard_minus_mild_weighted_need',
        'pearson_half_shift',
        'spearman_half_shift',
        'aligned_minus_shift_pearson',
    )
    groups = defaultdict(list)
    for row in rows:
        groups[row['pattern']].append(row)
    result = {}
    for pattern, pattern_rows in sorted(groups.items()):
        result[pattern] = {
            'images': len(pattern_rows),
            **{
                metric: finite_summary([row[metric] for row in pattern_rows])
                for metric in metrics
            },
            'positive_weighted_contrast_fraction': float(np.mean([
                row['hard_minus_mild_weighted_need'] > 0.0
                for row in pattern_rows
            ])),
            'aligned_auc_above_half_fraction': float(np.mean([
                row['auc_aligned'] > 0.5 for row in pattern_rows
                if math.isfinite(row['auc_aligned'])
            ])),
        }
    return result


def aggregate_correction_rows(rows):
    metrics = (
        'correction_target_mean',
        'correction_prediction_mean',
        'correction_mae',
        'correction_rmse',
        'correction_bias',
        'correction_pearson',
        'correction_spearman',
        'correction_pearson_half_shift',
        'correction_spearman_half_shift',
        'correction_aligned_minus_shift_pearson',
        'target_vs_coefficient_pearson',
        'target_vs_coefficient_spearman',
    )
    groups = defaultdict(list)
    for row in rows:
        groups[row['pattern']].append(row)
    return {
        pattern: {
            'images': len(pattern_rows),
            **{
                metric: finite_summary([row[metric] for row in pattern_rows])
                for metric in metrics
            },
            'prediction_beats_shift_fraction': float(np.mean([
                row['correction_aligned_minus_shift_pearson'] > 0.0
                for row in pattern_rows
                if math.isfinite(
                    row['correction_aligned_minus_shift_pearson']
                )
            ])),
        }
        for pattern, pattern_rows in sorted(groups.items())
    }


def safe_member(root, relative):
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f'Path escapes benchmark root: {relative}') from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_rgb_tensor(path, device):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'Cannot read image: {path}')
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().div_(255.0)
    return tensor.unsqueeze(0).to(device), image


def load_selection(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding='utf-8') as file:
        selection = json.load(file)
    if selection.get('mode') != 'calibration':
        raise ValueError('Prior selection must have mode=calibration.')
    checkpoint = Path(selection['selected_checkpoint']).expanduser().resolve()
    expected = selection.get('selected_checkpoint_sha256')
    if not checkpoint.is_file() or file_sha256(checkpoint) != expected:
        raise ValueError('Selected prior checkpoint is missing or has wrong hash.')
    payload = torch.load(checkpoint, map_location='cpu')
    if payload.get('method') != 'restoration_need_calibration':
        raise ValueError('Selected checkpoint is not a calibrated need model.')
    if payload.get('iter') != selection.get('selected_iteration'):
        raise ValueError('Selected iteration disagrees with checkpoint metadata.')
    del payload
    return path, selection, checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark_root', type=Path, required=True)
    parser.add_argument('--prior_selection', type=Path, required=True)
    parser.add_argument('--output_csv', type=Path, required=True)
    parser.add_argument('--output_json', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--tile_batch_size', type=int, default=8)
    parser.add_argument('--tile_overlap', type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_csv.exists() or args.output_json.exists():
        raise RuntimeError('Refusing to overwrite localization output.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')
    benchmark_root = args.benchmark_root.expanduser().resolve()
    manifest_path = benchmark_root / 'manifest.csv'
    summary_path = benchmark_root / 'manifest_summary.json'
    protocol_path = benchmark_root / 'protocol.json'
    for path in (manifest_path, summary_path, protocol_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with summary_path.open(encoding='utf-8') as file:
        benchmark_summary = json.load(file)
    with protocol_path.open(encoding='utf-8') as file:
        benchmark_protocol = json.load(file)
    if benchmark_summary.get('manifest_sha256') != file_sha256(manifest_path):
        raise ValueError('Benchmark manifest hash disagrees with its summary.')
    if benchmark_summary.get('protocol_sha256') != file_sha256(protocol_path):
        raise ValueError('Benchmark protocol hash disagrees with its summary.')
    if benchmark_protocol.get('schema_version') != 2:
        raise ValueError('Unexpected spatial-mixture protocol schema.')
    if benchmark_protocol.get('scale') != 4:
        raise ValueError('Prior-localization evaluation is registered for x4.')
    if benchmark_protocol.get('mild_level') == benchmark_protocol.get('hard_level'):
        raise ValueError('Mild and hard benchmark levels must differ.')

    selection_path, selection, checkpoint = load_selection(
        args.prior_selection
    )
    from drct.archs.DRCT_arch import PD_MAE_Encoder_Wrapper

    device = torch.device(args.device)
    model = PD_MAE_Encoder_Wrapper(
        checkpoint_path=str(checkpoint),
        predict_restoration_need=True,
        input_upscale=4,
        tile_batch_size=args.tile_batch_size,
        tile_overlap=args.tile_overlap,
    ).eval().to(device)

    with manifest_path.open(newline='', encoding='utf-8') as file:
        manifest = list(csv.DictReader(file))
    if not manifest:
        raise ValueError('Spatial-mixture manifest is empty.')
    if benchmark_summary.get('records') != len(manifest):
        raise ValueError('Benchmark summary record count disagrees with manifest.')
    observed_patterns = sorted({row['pattern'] for row in manifest})
    expected_patterns = sorted(benchmark_protocol.get('patterns', []))
    if observed_patterns != expected_patterns:
        raise ValueError(
            'Manifest patterns disagree with benchmark protocol: '
            f'{observed_patterns} vs {expected_patterns}.'
        )
    seen = set()
    pure_cache = {}
    rows = []

    def predict_need(path):
        path_key = str(path)
        if path_key in pure_cache:
            return pure_cache[path_key]
        tensor, _ = read_rgb_tensor(path, device)
        with torch.inference_mode():
            _, need = model(tensor, return_need=True)
        result = need[0, 0].detach().cpu().numpy().astype(np.float64)
        pure_cache[path_key] = result
        return result

    for manifest_row in manifest:
        key = (
            manifest_row['dataset'], manifest_row['filename'],
            manifest_row['pattern'],
        )
        if key in seen:
            raise ValueError(f'Duplicate benchmark row: {key}')
        seen.add(key)
        lq_path = safe_member(benchmark_root, manifest_row['lq'])
        mask_path = safe_member(benchmark_root, manifest_row['mask'])
        mild_path = safe_member(benchmark_root, manifest_row['mild_lq'])
        hard_path = safe_member(benchmark_root, manifest_row['hard_lq'])
        gt_path = safe_member(benchmark_root, manifest_row['gt'])
        for path, hash_key in (
            (lq_path, 'lq_sha256'),
            (mask_path, 'mask_sha256'),
            (mild_path, 'mild_sha256'),
            (hard_path, 'hard_sha256'),
            (gt_path, 'gt_sha256'),
        ):
            if file_sha256(path) != manifest_row[hash_key]:
                raise ValueError(f'Hash mismatch for {path}.')

        prediction = predict_need(lq_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f'Cannot read mask: {mask_path}')
        mask = cv2.resize(
            mask.astype(np.float32) / 255.0,
            (prediction.shape[1], prediction.shape[0]),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float64)
        mild_need = predict_need(mild_path)
        hard_need = predict_need(hard_path)
        if mild_need.shape != prediction.shape or hard_need.shape != prediction.shape:
            raise ValueError(f'Pure/mixed need shape mismatch for {key}.')
        lq_tensor, _ = read_rgb_tensor(lq_path, device)
        gt_tensor, _ = read_rgb_tensor(gt_path, device)
        correction_target = paired_correction_target(
            lq_tensor,
            gt_tensor,
            prediction.shape,
            input_upscale=4,
            patch_size=model.mae.encoder.patch_size,
            scale=10.0,
        )[0].detach().cpu().numpy().astype(np.float64)
        row = {
            'dataset': manifest_row['dataset'],
            'image': Path(manifest_row['filename']).stem,
            'pattern': manifest_row['pattern'],
            'tokens': int(prediction.size),
            'mixture_need_mean': float(prediction.mean()),
            'mixture_coefficient_mean': float(mask.mean()),
            'pure_mild_need_mean': float(mild_need.mean()),
            'pure_hard_need_mean': float(hard_need.mean()),
            'pure_hard_minus_mild_need': float(
                hard_need.mean() - mild_need.mean()
            ),
            **localization_metrics(prediction, mask),
            **correction_alignment_metrics(prediction, correction_target),
            'target_vs_coefficient_pearson': correlation(
                correction_target, mask
            ),
            'target_vs_coefficient_spearman': correlation(
                correction_target, mask, rank=True
            ),
        }
        rows.append(row)

    pure_rows = {}
    for row in rows:
        key = (row['dataset'], row['image'])
        pure_rows.setdefault(key, row['pure_hard_minus_mild_need'])
    pure_effects = list(pure_rows.values())
    report = {
        'schema_version': 2,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'interpretation': (
            'Paired correction demand is the primary alignment target because '
            'it matches need-head training. The saved mixture coefficient is '
            'reported separately as an external severity-ordering signal and '
            'is not ground-truth restoration-need magnitude.'
        ),
        'evaluator_path': str(Path(__file__).resolve()),
        'evaluator_sha256': file_sha256(Path(__file__).resolve()),
        'benchmark': {
            'root': str(benchmark_root),
            'manifest': str(manifest_path),
            'manifest_sha256': file_sha256(manifest_path),
            'summary_sha256': file_sha256(summary_path),
            'protocol_sha256': file_sha256(protocol_path),
            'records': len(rows),
            'scale': benchmark_protocol['scale'],
            'mild_level': benchmark_protocol['mild_level'],
            'hard_level': benchmark_protocol['hard_level'],
            'patterns': observed_patterns,
        },
        'prior': {
            'selection': str(selection_path),
            'selection_sha256': file_sha256(selection_path),
            'selected_iteration': selection['selected_iteration'],
            'checkpoint': str(checkpoint),
            'checkpoint_sha256': file_sha256(checkpoint),
        },
        'inference': {
            'device': str(device),
            'torch_version': torch.__version__,
            'opencv_version': cv2.__version__,
            'numpy_version': np.__version__,
            'input_upscale': 4,
            'tile_batch_size': args.tile_batch_size,
            'tile_overlap': args.tile_overlap,
        },
        'per_pattern': aggregate_rows(rows),
        'per_pattern_correction_target': aggregate_correction_rows(rows),
        'pure_hard_minus_mild_need': {
            **finite_summary(pure_effects),
            'images': len(pure_effects),
            'hard_above_mild_fraction': float(np.mean([
                effect > 0.0 for effect in pure_effects
            ])),
        },
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with args.output_json.open('w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
        file.write('\n')
    print(json.dumps({
        'per_pattern': report['per_pattern'],
        'per_pattern_correction_target': report[
            'per_pattern_correction_target'
        ],
        'pure_hard_minus_mild_need': report['pure_hard_minus_mild_need'],
    }, indent=2))


if __name__ == '__main__':
    main()
