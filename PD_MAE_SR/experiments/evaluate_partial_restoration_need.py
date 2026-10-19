"""Evaluate corrected partial-degradation pretraining on held-out sources."""

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from PD_MAE_SR.archs.mae_arch import RestorationNeedPDMAE
from PD_MAE_SR.datasets.partial_restoration_dataset import PartialRestorationDataset
from PD_MAE_SR.experiments.evaluate_restoration_need import (
    cross_level_metrics,
    expected_calibration_error,
    pearson_numpy,
    save_visualization,
    spearman_numpy,
    target_quartile_rows,
)
from PD_MAE_SR.experiments.train_restoration_need import (
    patchify,
    repository_commit,
    restoration_need_target,
)


def psnr_from_mse(mse):
    return float(-10.0 * np.log10(max(float(mse), 1e-12)))


def source_group_digest(source_groups):
    digest = hashlib.sha256()
    for source_group in sorted(source_groups):
        digest.update(source_group.encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as source_file:
        for chunk in iter(lambda: source_file.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def joint_validation_source_groups(args):
    """Reproduce the training split jointly across all degradation roots."""
    dataset = PartialRestorationDataset(
        args.data_roots,
        patch_size=args.patch_size,
        augment=False,
        split='val',
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        deterministic_crop=True,
    )
    source_groups = list(dataset.source_groups)
    if args.max_images > 0 and args.max_images < len(source_groups):
        rng = np.random.default_rng(args.sample_seed)
        indices = np.sort(
            rng.choice(len(source_groups), size=args.max_images, replace=False)
        )
        source_groups = [source_groups[index] for index in indices]
    return source_groups


def dataset_for_source_groups(args, data_root, source_groups):
    """Select the same precomputed validation sources from one level root."""
    dataset = PartialRestorationDataset(
        [data_root],
        patch_size=args.patch_size,
        augment=False,
        split='all',
        deterministic_crop=True,
    )
    wanted = set(source_groups)
    grouped_indices = {}
    for index, sample in enumerate(dataset.samples):
        if sample['source_group'] in wanted:
            grouped_indices.setdefault(sample['source_group'], []).append(index)
    missing = sorted(wanted - set(grouped_indices))
    ambiguous = sorted(
        group for group, indices in grouped_indices.items() if len(indices) != 1
    )
    if missing or ambiguous:
        raise RuntimeError(
            f'Level {data_root} does not have exactly one image for every '
            f'selected source: missing={missing[:5]}, ambiguous={ambiguous[:5]}.'
        )
    indices = [grouped_indices[group][0] for group in sorted(wanted)]
    return Subset(dataset, indices)


def evaluate_root(model, args, data_root, output_dir, source_groups):
    dataset = dataset_for_source_groups(args, data_root, source_groups)
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
    input_mses = []
    reconstruction_mses = []
    visual_count = 0
    level_name = os.path.basename(os.path.normpath(data_root))
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
            reconstruction, need_logits = model(lq, return_need=True)
            prediction = torch.sigmoid(need_logits)

            input_mse = (lq_patches - hr_patches).square().mean(dim=(1, 2))
            reconstruction_mse = (
                reconstruction - hr_patches
            ).square().mean(dim=(1, 2))
            input_mses.extend(input_mse.cpu().numpy().tolist())
            reconstruction_mses.extend(
                reconstruction_mse.cpu().numpy().tolist()
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
                    'image': batch['source_group'][item],
                    'filename': batch['filename'][item],
                    'target_mean': float(target_item.mean()),
                    'prediction_mean': float(pred_item.mean()),
                    'mae': float(np.mean(np.abs(pred_item - target_item))),
                    'pearson_tokens': correlation,
                    'input_psnr': psnr_from_mse(input_mse[item].item()),
                    'reconstruction_psnr': psnr_from_mse(
                        reconstruction_mse[item].item()
                    ),
                })

            grid = int(round(math.sqrt(target.shape[1])))
            for item in range(lq.shape[0]):
                if visual_count >= args.num_visuals:
                    break
                save_visualization(
                    os.path.join(
                        level_vis_dir, f'{batch["source_group"][item]}.png'
                    ),
                    lq[item],
                    hr[item],
                    target_np[item].reshape(grid, grid),
                    prediction_np[item].reshape(grid, grid),
                )
                visual_count += 1

    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    input_mses = np.asarray(input_mses, dtype=np.float64)
    reconstruction_mses = np.asarray(reconstruction_mses, dtype=np.float64)
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
        'input_mse_mean': float(input_mses.mean()),
        'reconstruction_mse_mean': float(reconstruction_mses.mean()),
        'input_psnr_mean': float(np.mean([
            psnr_from_mse(value) for value in input_mses
        ])),
        'reconstruction_psnr_mean': float(np.mean([
            psnr_from_mse(value) for value in reconstruction_mses
        ])),
        'reconstruction_gain_db': float(np.mean([
            psnr_from_mse(reconstruction) - psnr_from_mse(observation)
            for observation, reconstruction in zip(
                input_mses, reconstruction_mses
            )
        ])),
    }
    return summary, per_image_rows, target_quartile_rows(
        level_name, prediction, target
    )


def write_rows(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_roots', nargs='+', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--mae_patch_size', type=int, default=8)
    parser.add_argument('--need_scale', type=float, default=10.0)
    parser.add_argument('--val_fraction', type=float, default=0.05)
    parser.add_argument('--split_seed', type=int, default=0)
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

    summaries = []
    per_image_rows = []
    quartile_rows = []
    validation_source_groups = joint_validation_source_groups(args)
    for data_root in args.data_roots:
        summary, image_rows, root_quartiles = evaluate_root(
            model,
            args,
            data_root,
            args.output_dir,
            validation_source_groups,
        )
        summaries.append(summary)
        per_image_rows.extend(image_rows)
        quartile_rows.extend(root_quartiles)

    write_rows(
        os.path.join(args.output_dir, 'partial_restoration_need_metrics.csv'),
        summaries,
    )
    write_rows(
        os.path.join(args.output_dir, 'partial_restoration_need_per_image.csv'),
        per_image_rows,
    )
    write_rows(
        os.path.join(args.output_dir, 'partial_restoration_need_quartiles.csv'),
        quartile_rows,
    )
    cross_level = cross_level_metrics(per_image_rows)
    with open(
        os.path.join(args.output_dir, 'partial_restoration_need_cross_level.json'),
        'w', encoding='utf-8'
    ) as file:
        json.dump(cross_level, file, indent=2)
        file.write('\n')

    protocol = {
        'arguments': vars(args),
        'git_commit': repository_commit(),
        'checkpoint_iter': checkpoint.get('iter'),
        'checkpoint_method': checkpoint.get('method'),
        'checkpoint_config': checkpoint.get('config'),
        'split_unit': 'source',
        'evaluated_source_groups': len(validation_source_groups),
        'evaluated_source_group_sha256': source_group_digest(
            validation_source_groups
        ),
        'joint_split_across_roots': True,
        'implementation_sha256': {
            'evaluator': file_sha256(__file__),
            'dataset': file_sha256(inspect.getfile(PartialRestorationDataset)),
            'model': file_sha256(inspect.getfile(RestorationNeedPDMAE)),
            'need_target': file_sha256(inspect.getfile(restoration_need_target)),
            'metric_helpers': file_sha256(inspect.getfile(cross_level_metrics)),
        },
    }
    with open(
        os.path.join(args.output_dir, 'evaluation_protocol.json'),
        'w', encoding='utf-8'
    ) as file:
        json.dump(protocol, file, indent=2)
        file.write('\n')

    for summary in summaries:
        print(summary)
    print(cross_level)
    print(f'Saved metrics to {args.output_dir}.')


if __name__ == '__main__':
    main()
