"""Evaluate restoration-need calibration on a deterministic held-out split."""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from PD_MAE_SR.archs.mae_arch import RestorationNeedPDMAE
from PD_MAE_SR.datasets.multilevel_restoration_dataset import MultiLevelRestorationDataset
from PD_MAE_SR.experiments.train_restoration_need import (
    patchify,
    repository_commit,
    restoration_need_target,
)


def pearson_numpy(prediction, target):
    prediction = prediction.reshape(-1).astype(np.float64)
    target = target.reshape(-1).astype(np.float64)
    if prediction.std() < 1e-12 or target.std() < 1e-12:
        return math.nan
    return float(np.corrcoef(prediction, target)[0, 1])


def spearman_numpy(prediction, target):
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return math.nan
    if np.std(prediction) < 1e-12 or np.std(target) < 1e-12:
        return math.nan
    return float(spearmanr(prediction.reshape(-1), target.reshape(-1)).statistic)


def expected_calibration_error(prediction, target, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = prediction.size
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            selected = (prediction >= lower) & (prediction <= upper)
        else:
            selected = (prediction >= lower) & (prediction < upper)
        if selected.any():
            error += selected.sum() / total * abs(
                prediction[selected].mean() - target[selected].mean()
            )
    return float(error)


def target_quartile_rows(level, prediction, target):
    """Stratify calibration error by target restoration-need quartile."""
    edges = np.quantile(target, [0.0, 0.25, 0.5, 0.75, 1.0])
    rows = []
    for index in range(4):
        lower, upper = float(edges[index]), float(edges[index + 1])
        if index == 3:
            selected = (target >= lower) & (target <= upper)
        else:
            selected = (target >= lower) & (target < upper)
        if not selected.any():
            continue
        rows.append({
            'level': level,
            'quartile': index + 1,
            'tokens': int(selected.sum()),
            'target_lower': lower,
            'target_upper': upper,
            'target_mean': float(target[selected].mean()),
            'prediction_mean': float(prediction[selected].mean()),
            'mae': float(np.mean(np.abs(prediction[selected] - target[selected]))),
            'pearson_token': pearson_numpy(prediction[selected], target[selected]),
        })
    return rows


def tensor_to_bgr(image):
    rgb = image.detach().cpu().permute(1, 2, 0).numpy()
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def heatmap(values, output_size):
    values = np.clip(values * 255.0, 0, 255).astype(np.uint8)
    values = cv2.resize(values, output_size, interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(values, cv2.COLORMAP_TURBO)


def save_visualization(path, lq, hr, target, prediction):
    height, width = hr.shape[-2:]
    panels = [
        tensor_to_bgr(lq),
        tensor_to_bgr(hr),
        heatmap(target, (width, height)),
        heatmap(prediction, (width, height)),
    ]
    labels = ('LQ', 'HR', 'target need', 'predicted need')
    for panel, label in zip(panels, labels):
        cv2.rectangle(panel, (0, 0), (width, 24), (0, 0, 0), thickness=-1)
        cv2.putText(panel, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, np.concatenate(panels, axis=1))


def evaluate_level(model, args, lq_root, output_dir):
    dataset = MultiLevelRestorationDataset(
        args.hr_root,
        [lq_root],
        patch_size=args.patch_size,
        augment=False,
        deterministic_level=True,
        split='val',
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        deterministic_crop=True,
        split_unit=args.split_unit,
        alignment_backend=args.alignment_backend,
    )
    if args.max_images > 0 and args.max_images < len(dataset):
        rng = np.random.default_rng(args.sample_seed)
        indices = np.sort(
            rng.choice(len(dataset), size=args.max_images, replace=False)
        ).tolist()
        dataset = Subset(dataset, indices)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    predictions = []
    targets = []
    per_image_pearson = []
    per_image_rows = []
    visual_count = 0
    level_name = os.path.basename(os.path.normpath(lq_root))
    level_vis_dir = os.path.join(output_dir, level_name)
    os.makedirs(level_vis_dir, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            lq = batch['lq'].cuda(non_blocking=True)
            hr = batch['hr'].cuda(non_blocking=True)
            lq_patches = patchify(lq, args.mae_patch_size)
            hr_patches = patchify(hr, args.mae_patch_size)
            target = restoration_need_target(
                lq_patches, hr_patches, scale=args.need_scale
            )
            latent = model.encoder(lq)
            prediction = torch.sigmoid(
                model.need_head(latent[:, 1:, :]).squeeze(-1)
            )

            target_np = target.cpu().numpy()
            prediction_np = prediction.cpu().numpy()
            predictions.append(prediction_np.reshape(-1))
            targets.append(target_np.reshape(-1))
            for item, (pred_item, target_item) in enumerate(
                    zip(prediction_np, target_np)):
                correlation = pearson_numpy(pred_item, target_item)
                if not math.isnan(correlation):
                    per_image_pearson.append(correlation)
                per_image_rows.append({
                    'level': level_name,
                    'image': os.path.splitext(batch['filename'][item])[0],
                    'source_group': batch['source_group'][item],
                    'target_mean': float(target_item.mean()),
                    'prediction_mean': float(pred_item.mean()),
                    'mae': float(np.mean(np.abs(pred_item - target_item))),
                    'pearson_tokens': correlation,
                })

            grid = int(round(math.sqrt(target.shape[1])))
            for item in range(lq.shape[0]):
                if visual_count >= args.num_visuals:
                    break
                filename = os.path.splitext(batch['filename'][item])[0]
                save_visualization(
                    os.path.join(level_vis_dir, f'{filename}.png'),
                    lq[item],
                    hr[item],
                    target_np[item].reshape(grid, grid),
                    prediction_np[item].reshape(grid, grid),
                )
                visual_count += 1

    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    summary = {
        'level': level_name,
        'images': len(dataset),
        'tokens': target.size,
        'target_mean': float(target.mean()),
        'prediction_mean': float(prediction.mean()),
        'mae': float(np.mean(np.abs(prediction - target))),
        'rmse': float(np.sqrt(np.mean((prediction - target) ** 2))),
        'pearson_token': pearson_numpy(prediction, target),
        'spearman_token': spearman_numpy(prediction, target),
        'pearson_image_mean': (
            float(np.mean(per_image_pearson)) if per_image_pearson else math.nan
        ),
        'calibration_error': expected_calibration_error(prediction, target),
        'prediction_gt_0.1_fraction': float(np.mean(prediction > 0.1)),
        'prediction_gt_0.2_fraction': float(np.mean(prediction > 0.2)),
    }
    return summary, per_image_rows, target_quartile_rows(
        level_name, prediction, target
    )


def cross_level_metrics(per_image_rows):
    grouped = defaultdict(list)
    for row in per_image_rows:
        grouped[row['image']].append(row)

    pearsons = []
    spearmans = []
    correct_pairs = 0
    comparable_pairs = 0
    for rows in grouped.values():
        targets = np.asarray([row['target_mean'] for row in rows])
        predictions = np.asarray([row['prediction_mean'] for row in rows])
        pearson = pearson_numpy(predictions, targets)
        spearman = spearman_numpy(predictions, targets)
        if not math.isnan(pearson):
            pearsons.append(pearson)
        if not math.isnan(spearman):
            spearmans.append(spearman)
        for first in range(len(rows)):
            for second in range(first + 1, len(rows)):
                target_delta = targets[first] - targets[second]
                if abs(target_delta) < 1e-8:
                    continue
                prediction_delta = predictions[first] - predictions[second]
                correct_pairs += int(target_delta * prediction_delta > 0)
                comparable_pairs += 1

    return {
        'images': len(grouped),
        'image_level_groups_with_pearson': len(pearsons),
        'image_level_groups_with_spearman': len(spearmans),
        'pearson_across_levels_mean': (
            float(np.mean(pearsons)) if pearsons else math.nan
        ),
        'spearman_across_levels_mean': (
            float(np.mean(spearmans)) if spearmans else math.nan
        ),
        'pairwise_ordering_accuracy': (
            correct_pairs / comparable_pairs if comparable_pairs else math.nan
        ),
        'correct_level_pairs': correct_pairs,
        'comparable_level_pairs': comparable_pairs,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Held-out restoration-need evaluation')
    parser.add_argument('--hr_root', required=True)
    parser.add_argument('--lq_roots', nargs='+', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--mae_patch_size', type=int, default=8)
    parser.add_argument('--need_scale', type=float, default=10.0)
    parser.add_argument('--val_fraction', type=float, default=0.05)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument(
        '--split_unit', choices=['filename', 'source'], default='filename'
    )
    parser.add_argument(
        '--alignment_backend',
        choices=['opencv_bicubic', 'pytorch_bicubic'],
        default='opencv_bicubic',
    )
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max_images', type=int, default=1000)
    parser.add_argument('--sample_seed', type=int, default=0)
    parser.add_argument('--num_visuals', type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state = checkpoint.get('model_state_dict', checkpoint.get('params', checkpoint))
    model = RestorationNeedPDMAE(
        img_size=args.patch_size,
        patch_size=args.mae_patch_size,
    )
    model.load_state_dict(state, strict=True)
    model.eval().cuda()

    rows = []
    per_image_rows = []
    quartile_rows = []
    for lq_root in args.lq_roots:
        summary, level_rows, level_quartiles = evaluate_level(
            model, args, lq_root, args.output_dir
        )
        rows.append(summary)
        per_image_rows.extend(level_rows)
        quartile_rows.extend(level_quartiles)
    output_csv = os.path.join(args.output_dir, 'restoration_need_metrics.csv')
    with open(output_csv, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    per_image_csv = os.path.join(
        args.output_dir, 'restoration_need_per_image.csv'
    )
    with open(per_image_csv, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)

    quartile_csv = os.path.join(
        args.output_dir, 'restoration_need_quartiles.csv'
    )
    with open(quartile_csv, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(quartile_rows[0].keys()))
        writer.writeheader()
        writer.writerows(quartile_rows)

    cross_level = cross_level_metrics(per_image_rows)
    cross_level_path = os.path.join(
        args.output_dir, 'restoration_need_cross_level.json'
    )
    with open(cross_level_path, 'w', encoding='utf-8') as file:
        json.dump(cross_level, file, indent=2)
        file.write('\n')

    protocol = {
        'arguments': vars(args),
        'git_commit': repository_commit(),
        'checkpoint_iter': checkpoint.get('iter'),
        'checkpoint_method': checkpoint.get('method'),
        'checkpoint_config': checkpoint.get('config'),
        'checkpoint_pretrain_provenance': checkpoint.get(
            'pretrain_provenance'
        ),
        'checkpoint_selection_provenance': checkpoint.get(
            'selection_provenance'
        ),
        'evaluated_unique_images': len({row['image'] for row in per_image_rows}),
        'evaluated_source_groups': len({
            row['source_group'] for row in per_image_rows
        }),
        'split_unit': args.split_unit,
        'input_levels': [
            os.path.basename(os.path.normpath(path)) for path in args.lq_roots
        ],
    }
    with open(
        os.path.join(args.output_dir, 'evaluation_protocol.json'),
        'w', encoding='utf-8'
    ) as file:
        json.dump(protocol, file, indent=2)
        file.write('\n')

    for row in rows:
        print(row)
    print(cross_level)
    print(f'Saved metrics to {output_csv}')


if __name__ == '__main__':
    main()
