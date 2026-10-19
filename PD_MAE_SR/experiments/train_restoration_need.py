"""Calibrate PD-MAE to predict token-wise restoration need.

The target is derived from the paired LQ/HR correction magnitude.  This avoids
relying on a recomputed binary mask and remains valid for both partial and
full-image degradations.
"""

import argparse
import hashlib
import json
import logging
import os
import random
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from PD_MAE_SR.archs.mae_arch import RestorationNeedPDMAE
from PD_MAE_SR.datasets.multilevel_restoration_dataset import MultiLevelRestorationDataset


def patchify(images, patch_size=8):
    """Convert ``B,C,H,W`` images into ``B,L,C*p*p`` patch vectors."""
    batch, channels, height, width = images.shape
    if height != width or height % patch_size:
        raise ValueError(
            f'Expected square images divisible by {patch_size}, got {height}x{width}.'
        )
    grid = height // patch_size
    patches = images.reshape(batch, channels, grid, patch_size, grid, patch_size)
    patches = torch.einsum('nchpwq->nhwpqc', patches)
    return patches.reshape(batch, grid * grid, patch_size * patch_size * channels)


def restoration_need_target(lq_patches, hr_patches, scale=10.0):
    """Map patch correction magnitudes to a bounded restoration-need target."""
    correction = (hr_patches - lq_patches).abs().mean(dim=-1)
    return 1.0 - torch.exp(-scale * correction)


def pearson_correlation(prediction, target, eps=1e-8):
    prediction = prediction.flatten()
    target = target.flatten()
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    denominator = prediction.square().sum().sqrt() * target.square().sum().sqrt()
    return (prediction * target).sum() / denominator.clamp_min(eps)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_pretrain(model, checkpoint_path, expected_method=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('params', checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=False)
    expected_missing = {'need_head.0.weight', 'need_head.0.bias',
                        'need_head.1.weight', 'need_head.1.bias'}
    missing = set(incompatible.missing_keys)
    # A corrected partial-degradation checkpoint already contains a calibrated
    # need head, while a legacy Stage-1 checkpoint does not. Both are valid
    # initializations; every other mismatch is an error.
    if missing not in (set(), expected_missing) or incompatible.unexpected_keys:
        raise RuntimeError(
            'Unexpected Stage-1 checkpoint mismatch. '
            f'Missing={sorted(missing)}, unexpected={incompatible.unexpected_keys}'
        )
    method = checkpoint.get('method')
    if expected_method is not None and method != expected_method:
        raise RuntimeError(
            f'Expected pretrain method {expected_method!r}, found {method!r} '
            f'in {checkpoint_path}.'
        )
    return {
        'path': os.path.abspath(os.path.expanduser(checkpoint_path)),
        'sha256': sha256_file(checkpoint_path),
        'iteration': checkpoint.get('iter'),
        'method': method,
        'config': checkpoint.get('config'),
    }


def load_selection_provenance(selection_path, pretrain_provenance,
                              expected_mode=None):
    if not selection_path:
        return None
    selection_path = os.path.abspath(os.path.expanduser(selection_path))
    with open(selection_path, encoding='utf-8') as selection_file:
        selection = json.load(selection_file)
    if expected_mode is not None and selection.get('mode') != expected_mode:
        raise RuntimeError(
            f'Expected selection mode {expected_mode!r}, found '
            f'{selection.get("mode")!r} in {selection_path}.'
        )
    selected_checkpoint = os.path.abspath(os.path.expanduser(
        selection.get('selected_checkpoint', '')
    ))
    if selected_checkpoint != pretrain_provenance['path']:
        raise RuntimeError(
            'The selection artifact points to a different pretrain: '
            f'{selected_checkpoint!r} != {pretrain_provenance["path"]!r}.'
        )
    selected_sha256 = selection.get('selected_checkpoint_sha256')
    if expected_mode is not None and selected_sha256 is None:
        raise RuntimeError(
            'The required selection artifact does not contain a selected '
            'checkpoint SHA-256.'
        )
    if (selected_sha256 is not None
            and selected_sha256 != pretrain_provenance['sha256']):
        raise RuntimeError(
            'The selected pretrain checkpoint hash does not match the '
            'selection artifact.'
        )
    if selection.get('selected_iteration') != pretrain_provenance['iteration']:
        raise RuntimeError(
            'The selected iteration does not match the pretrain checkpoint '
            'metadata.'
        )
    return {
        'path': selection_path,
        'sha256': sha256_file(selection_path),
        'mode': selection.get('mode'),
        'selected_iteration': selection.get('selected_iteration'),
        'selected_checkpoint': selected_checkpoint,
        'selected_checkpoint_sha256': selected_sha256,
        'selector_git_commit': selection.get('selector_git_commit'),
    }


def save_checkpoint(path, model, optimizer, scheduler, iteration, metrics,
                    config, pretrain_provenance, selection_provenance):
    torch.save({
        'iter': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'method': 'restoration_need_calibration',
        'config': config,
        'pretrain_provenance': pretrain_provenance,
        'selection_provenance': selection_provenance,
    }, path)


def repository_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_run_manifest(output_dir, args, dataset, pretrain_provenance,
                       selection_provenance):
    manifest = {
        'command': [sys.executable, *sys.argv],
        'arguments': vars(args),
        'git_commit': repository_commit(),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'cuda_device': torch.cuda.get_device_name(0),
        'train_images': len(dataset),
        'train_source_groups': len(dataset.source_groups),
        'split_unit': args.split_unit,
        'input_levels': len(args.lq_roots),
        'input_level_roots': args.lq_roots,
        'dataloader_generator_seed': args.seed,
        'pretrain_provenance': pretrain_provenance,
        'selection_provenance': selection_provenance,
    }
    manifest_path = os.path.join(output_dir, 'run_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write('\n')
    split_path = os.path.join(output_dir, 'train_filenames.txt')
    with open(split_path, 'w', encoding='utf-8') as split_file:
        for path in dataset.hr_paths:
            split_file.write(os.path.basename(path) + '\n')
    group_path = os.path.join(output_dir, 'train_source_groups.txt')
    with open(group_path, 'w', encoding='utf-8') as group_file:
        for group in dataset.source_groups:
            group_file.write(group + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Restoration-need calibration for PD-MAE')
    parser.add_argument('--hr_root', required=True)
    parser.add_argument('--lq_roots', nargs='+', required=True,
                        help='Aligned inputs, optionally including HR itself as an identity control.')
    parser.add_argument(
        '--pretrain', required=True,
        help='PD-MAE pretraining checkpoint used to initialize calibration.',
    )
    parser.add_argument(
        '--pretrain_selection', default=None,
        help=(
            'Optional checkpoint-selection JSON. When supplied, its selected '
            'path and hash must match --pretrain.'
        ),
    )
    parser.add_argument(
        '--expected_pretrain_method', default=None,
        help='Optional fail-closed check for the checkpoint method metadata.',
    )
    parser.add_argument(
        '--expected_selection_mode', default=None,
        help='Optional fail-closed check for --pretrain_selection mode.',
    )
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--iters', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--mae_patch_size', type=int, default=8)
    parser.add_argument('--need_scale', type=float, default=10.0)
    parser.add_argument('--need_weight', type=float, default=0.2)
    parser.add_argument('--reconstruction_floor', type=float, default=0.1)
    parser.add_argument(
        '--train_scope', choices=['all', 'head'], default='all',
        help=(
            'Update the complete PD-MAE or only the restoration-need head. '
            'The head-only setting is a representation-preservation control.'
        ),
    )
    parser.add_argument('--save_freq', type=int, default=25000)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--val_fraction', type=float, default=0.05,
                        help='Fraction reserved for held-out need-field evaluation.')
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument(
        '--split_unit', choices=['filename', 'source'], default='filename',
        help=(
            'Split individual crop filenames or keep all crops from one '
            'source image in the same partition.'
        ),
    )
    parser.add_argument(
        '--alignment_backend',
        choices=['opencv_bicubic', 'pytorch_bicubic'],
        default='opencv_bicubic',
        help=(
            'Resize LQ to the HR calibration grid. Final journal calibration '
            'uses pytorch_bicubic to match frozen-prior inference.'
        ),
    )
    return parser.parse_args()


def train():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'train.log')),
            logging.StreamHandler(),
        ],
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = MultiLevelRestorationDataset(
        args.hr_root,
        args.lq_roots,
        patch_size=args.patch_size,
        augment=True,
        split='train',
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        split_unit=args.split_unit,
        alignment_backend=args.alignment_backend,
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        generator=loader_generator,
    )

    model = RestorationNeedPDMAE(
        img_size=args.patch_size,
        patch_size=args.mae_patch_size,
    ).cuda()
    pretrain_provenance = load_pretrain(
        model, args.pretrain, expected_method=args.expected_pretrain_method
    )
    selection_provenance = load_selection_provenance(
        args.pretrain_selection, pretrain_provenance,
        expected_mode=args.expected_selection_mode,
    )
    write_run_manifest(
        args.output_dir, args, dataset, pretrain_provenance,
        selection_provenance,
    )
    logging.info(
        'Loaded pretrain %s at iteration %s (sha256=%s).',
        args.pretrain, pretrain_provenance['iteration'],
        pretrain_provenance['sha256'],
    )

    if args.train_scope == 'head':
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith('need_head.')
        logging.info('Training only the restoration-need head.')

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = AdamW(
        trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-6)
    current_iter = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if checkpoint.get('pretrain_provenance') != pretrain_provenance:
            raise RuntimeError(
                'Resume checkpoint was created from a different pretrain.'
            )
        if checkpoint.get('selection_provenance') != selection_provenance:
            raise RuntimeError(
                'Resume checkpoint was created from a different selection '
                'artifact.'
            )
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        current_iter = int(checkpoint['iter'])
        logging.info('Resumed restoration-need calibration at iteration %d', current_iter)

    logging.info(
        'Training on %d HR images and %d input levels for %d iterations.',
        len(dataset), len(args.lq_roots), args.iters,
    )
    progress = tqdm(total=args.iters, initial=current_iter)
    running = {'loss': 0.0, 'recon': 0.0, 'need': 0.0, 'corr': 0.0}
    level_names = [
        os.path.basename(os.path.normpath(path)) for path in args.lq_roots
    ]
    level_counts = {name: 0 for name in level_names}
    if args.resume:
        saved_counts = checkpoint.get('metrics', {}).get(
            'training_level_counts', {}
        )
        if saved_counts:
            if set(saved_counts) != set(level_counts):
                raise RuntimeError(
                    'Resume checkpoint level-count keys do not match the '
                    'current input roots.'
                )
            level_counts.update({
                name: int(saved_counts[name]) for name in level_names
            })
    window_count = 0

    while current_iter < args.iters:
        for batch in loader:
            if current_iter >= args.iters:
                break

            lq = batch['lq'].cuda(non_blocking=True)
            hr = batch['hr'].cuda(non_blocking=True)
            for level_index in batch['level_index'].tolist():
                level_counts[level_names[level_index]] += 1
            lq_patches = patchify(lq, args.mae_patch_size)
            hr_patches = patchify(hr, args.mae_patch_size)
            need_target = restoration_need_target(
                lq_patches, hr_patches, scale=args.need_scale
            )

            prediction, need_logits = model(lq, return_need=True)
            predicted_need = torch.sigmoid(need_logits)

            patch_error = (prediction - hr_patches).square().mean(dim=-1)
            weights = args.reconstruction_floor + (
                1.0 - args.reconstruction_floor
            ) * need_target
            reconstruction_loss = (patch_error * weights).sum() / weights.sum().clamp_min(1e-8)
            need_loss = F.smooth_l1_loss(predicted_need, need_target)
            loss = reconstruction_loss + args.need_weight * need_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            current_iter += 1
            window_count += 1
            correlation = pearson_correlation(predicted_need.detach(), need_target)
            running['loss'] += loss.item()
            running['recon'] += reconstruction_loss.item()
            running['need'] += need_loss.item()
            running['corr'] += correlation.item()
            progress.update(1)
            progress.set_description(
                f'loss={loss.item():.4f} need-r={correlation.item():.3f}'
            )

            if current_iter % args.log_freq == 0:
                averages = {key: value / window_count for key, value in running.items()}
                logging.info(
                    'Iter %d/%d loss=%.6f recon=%.6f need=%.6f need_r=%.4f lr=%.2e',
                    current_iter, args.iters, averages['loss'], averages['recon'],
                    averages['need'], averages['corr'], scheduler.get_last_lr()[0],
                )
                running = {key: 0.0 for key in running}
                window_count = 0

            if current_iter % args.save_freq == 0 or current_iter == args.iters:
                metrics = {
                    'loss': loss.item(),
                    'reconstruction_loss': reconstruction_loss.item(),
                    'need_loss': need_loss.item(),
                    'need_correlation': correlation.item(),
                    'need_scale': args.need_scale,
                    'training_level_counts': level_counts.copy(),
                    'training_samples_seen': sum(level_counts.values()),
                }
                checkpoint_path = os.path.join(
                    args.output_dir, f'rn_pd_mae_iter{current_iter}.pth'
                )
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    current_iter,
                    metrics,
                    vars(args),
                    pretrain_provenance,
                    selection_provenance,
                )
                logging.info('Saved checkpoint: %s', checkpoint_path)

    progress.close()
    summary_path = os.path.join(args.output_dir, 'training_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        json.dump({
            'iterations': current_iter,
            'level_counts': level_counts,
            'samples_seen': sum(level_counts.values()),
            'pretrain_provenance': pretrain_provenance,
            'selection_provenance': selection_provenance,
        }, summary_file, indent=2)
        summary_file.write('\n')
    logging.info('Restoration-need calibration complete.')


if __name__ == '__main__':
    train()
