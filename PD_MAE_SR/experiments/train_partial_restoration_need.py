"""Corrected partial-degradation pretraining for RN-PD-MAE.

The legacy Stage-1 objective recomputed a binary mask that did not exactly
match the feathered region used to generate its input. This script instead
derives both reconstruction weights and token-wise restoration need from the
actual paired partial-degradation/HR correction.
"""

import argparse
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from PD_MAE_SR.archs.mae_arch import RestorationNeedPDMAE
from PD_MAE_SR.datasets.partial_restoration_dataset import PartialRestorationDataset
from PD_MAE_SR.experiments.train_restoration_need import (
    patchify,
    pearson_correlation,
    restoration_need_target,
)


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


def load_optional_pretrain(model, path):
    if not path:
        return None
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('model_state_dict', checkpoint.get('params', checkpoint))
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = {
        'need_head.0.weight', 'need_head.0.bias',
        'need_head.1.weight', 'need_head.1.bias',
    }
    if (set(incompatible.missing_keys) not in (set(), allowed_missing)
            or incompatible.unexpected_keys):
        raise RuntimeError(
            f'Unexpected checkpoint mismatch: missing={incompatible.missing_keys}, '
            f'unexpected={incompatible.unexpected_keys}'
        )
    return checkpoint.get('iter', 'unknown')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_roots', nargs='+', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--pretrain', default=None)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--iters', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--mae_patch_size', type=int, default=8)
    parser.add_argument('--need_scale', type=float, default=10.0)
    parser.add_argument('--need_weight', type=float, default=0.2)
    parser.add_argument('--reconstruction_floor', type=float, default=0.1)
    parser.add_argument('--val_fraction', type=float, default=0.05)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=25000)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument(
        '--sampling', choices=['uniform_roots', 'uniform_samples'],
        default='uniform_roots',
        help='Balance degradation roots or sample the concatenated data directly.',
    )
    return parser.parse_args()


def save_checkpoint(path, model, optimizer, scheduler, iteration, metrics, args):
    torch.save({
        'iter': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'method': 'partial_restoration_need_pretraining',
        'config': vars(args),
    }, path)


def main():
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

    dataset = PartialRestorationDataset(
        args.data_roots,
        patch_size=args.patch_size,
        augment=True,
        split='train',
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
    )
    sampler = None
    if args.sampling == 'uniform_roots':
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(args.seed)
        sampler = WeightedRandomSampler(
            dataset.root_balancing_weights(),
            num_samples=len(dataset),
            replacement=True,
            generator=sampler_generator,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    manifest = {
        'command': [sys.executable, *sys.argv],
        'arguments': vars(args),
        'git_commit': repository_commit(),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'cuda_device': torch.cuda.get_device_name(0),
        'train_images': len(dataset),
        'train_source_groups': len(dataset.source_groups),
        'samples_per_root': dict(sorted(dataset.root_counts.items())),
        'sampling': args.sampling,
        'split_unit': 'source',
    }
    with open(os.path.join(args.output_dir, 'run_manifest.json'), 'w', encoding='utf-8') as file:
        json.dump(manifest, file, indent=2)
        file.write('\n')
    with open(os.path.join(args.output_dir, 'train_source_groups.txt'), 'w', encoding='utf-8') as file:
        for group in dataset.source_groups:
            file.write(group + '\n')

    model = RestorationNeedPDMAE(
        img_size=args.patch_size, patch_size=args.mae_patch_size
    ).cuda()
    loaded_iter = load_optional_pretrain(model, args.pretrain)
    if loaded_iter is not None:
        logging.info('Loaded pretrain %s at iteration %s.', args.pretrain, loaded_iter)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-6)
    current_iter = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        current_iter = int(checkpoint['iter'])

    progress = tqdm(total=args.iters, initial=current_iter)
    running = {'loss': 0.0, 'recon': 0.0, 'need': 0.0, 'corr': 0.0}
    window_count = 0
    while current_iter < args.iters:
        for batch in loader:
            if current_iter >= args.iters:
                break
            lq = batch['lq'].cuda(non_blocking=True)
            hr = batch['hr'].cuda(non_blocking=True)
            lq_patches = patchify(lq, args.mae_patch_size)
            hr_patches = patchify(hr, args.mae_patch_size)
            target_need = restoration_need_target(
                lq_patches, hr_patches, scale=args.need_scale
            )
            prediction, need_logits = model(lq, return_need=True)
            predicted_need = torch.sigmoid(need_logits)
            patch_error = (prediction - hr_patches).square().mean(dim=-1)
            weights = args.reconstruction_floor + (
                1.0 - args.reconstruction_floor
            ) * target_need
            reconstruction_loss = (patch_error * weights).sum() / weights.sum().clamp_min(1e-8)
            need_loss = F.smooth_l1_loss(predicted_need, target_need)
            loss = reconstruction_loss + args.need_weight * need_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            current_iter += 1
            window_count += 1
            correlation = pearson_correlation(predicted_need.detach(), target_need)
            values = (loss.item(), reconstruction_loss.item(),
                      need_loss.item(), correlation.item())
            for key, value in zip(running, values):
                running[key] += value
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
                path = os.path.join(
                    args.output_dir, f'rn_pd_mae_partial_iter{current_iter}.pth'
                )
                save_checkpoint(path, model, optimizer, scheduler, current_iter, {
                    'loss': loss.item(),
                    'reconstruction_loss': reconstruction_loss.item(),
                    'need_loss': need_loss.item(),
                    'need_correlation': correlation.item(),
                    'need_scale': args.need_scale,
                }, args)
                logging.info('Saved checkpoint: %s', path)
    progress.close()


if __name__ == '__main__':
    main()
