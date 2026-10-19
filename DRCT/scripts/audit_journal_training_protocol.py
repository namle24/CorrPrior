"""Audit the matched journal training protocol before launching SR runs.

The candidate and baseline must differ only in the proposed frozen-prior/SFT
branch and in the checkpoint strictness required by that additional branch.
This script also verifies the aligned multi-level corpus and records the exact
BasicSR sampler prefix consumed by the fixed iteration budget.
"""

import argparse
import hashlib
import inspect
import json
import os
import platform
import re
import struct
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
CANDIDATE_NETWORK_KEYS = {
    'allow_random_prior',
    'mae_checkpoint',
    'mae_input_upscale',
    'mae_tile_batch_size',
    'mae_tile_overlap',
    'restoration_need_gate_mode',
    'use_restoration_need_gate',
    'use_sft',
    'zero_init_sft',
}


class ProtocolError(RuntimeError):
    """Raised when the compared runs are not a matched experiment."""


def apply_manual_seed_override(candidate, baseline, manual_seed):
    """Apply the same non-negative runtime seed to a matched pair."""
    if manual_seed is None:
        return candidate, baseline
    if manual_seed < 0:
        raise ProtocolError('--manual_seed_override must be non-negative.')
    candidate['manual_seed'] = manual_seed
    baseline['manual_seed'] = manual_seed
    return candidate, baseline


def parse_args():
    parser = argparse.ArgumentParser(
        description='Audit matched candidate/baseline journal configurations.'
    )
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected_gt_images', type=int)
    parser.add_argument('--expected_source_groups', type=int)
    parser.add_argument('--expected_level_order', nargs='+')
    parser.add_argument(
        '--candidate_variant', choices=('p0', 'c2'), default='p0',
        help='Expected candidate treatment: spatially gated P0 or ungated C2.',
    )
    parser.add_argument(
        '--manual_seed_override', type=int,
        help=(
            'Audit a matched runtime seed applied to both configurations. '
            'The base files remain hashed unchanged.'
        ),
    )
    return parser.parse_args()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _without(mapping, keys):
    return {key: value for key, value in mapping.items() if key not in keys}


def validate_matched_configs(candidate, baseline, candidate_variant='p0'):
    """Validate all intentional and shared configuration differences."""
    if candidate_variant not in {'p0', 'c2'}:
        raise ProtocolError(f'Unknown candidate variant: {candidate_variant!r}.')
    candidate = deepcopy(candidate)
    baseline = deepcopy(baseline)

    if candidate.get('name') == baseline.get('name'):
        raise ProtocolError('Candidate and baseline experiment names must differ.')
    if float(candidate.get('mae_weight', 0.0)) != 0.0:
        raise ProtocolError('The primary candidate must set mae_weight to 0.')
    if float(baseline.get('mae_weight', 0.0)) != 0.0:
        raise ProtocolError('The matched baseline must not use a MAE loss.')

    candidate_network = candidate.get('network_g', {})
    baseline_network = baseline.get('network_g', {})
    expected_candidate = {
        'use_sft': True,
        'use_restoration_need_gate': candidate_variant == 'p0',
        'restoration_need_gate_mode': 'spatial',
        'zero_init_sft': True,
        'allow_random_prior': False,
        'mae_checkpoint': None,
        'mae_input_upscale': 4,
        'mae_tile_batch_size': 8,
        'mae_tile_overlap': 64,
    }
    observed_candidate = {
        key: candidate_network.get(key) for key in expected_candidate
    }
    if observed_candidate != expected_candidate:
        raise ProtocolError(
            f'Unexpected {candidate_variant.upper()} auxiliary configuration: '
            f'expected={expected_candidate}, observed={observed_candidate}'
        )
    if baseline_network.get('use_sft') is not False:
        raise ProtocolError('The matched baseline must set use_sft=false.')
    if baseline_network.get('mae_checkpoint') is not None:
        raise ProtocolError('The source baseline must contain a null MAE path.')
    forbidden_baseline = (
        CANDIDATE_NETWORK_KEYS - {'use_sft', 'mae_checkpoint'}
    ) & set(baseline_network)
    if forbidden_baseline:
        raise ProtocolError(
            'Candidate-only keys leaked into the baseline: '
            f'{sorted(forbidden_baseline)}'
        )

    candidate_path = candidate.get('path', {})
    baseline_path = baseline.get('path', {})
    if candidate_path.get('strict_load_g') is not False:
        raise ProtocolError(
            f'{candidate_variant.upper()} must use strict_load_g=false for added branches.'
        )
    if baseline_path.get('strict_load_g') is not True:
        raise ProtocolError('B0 must use strict_load_g=true.')

    candidate_shared = _without(candidate, {'name', 'mae_weight'})
    baseline_shared = _without(baseline, {'name', 'mae_weight'})
    candidate_shared['network_g'] = _without(
        candidate_network, CANDIDATE_NETWORK_KEYS
    )
    baseline_shared['network_g'] = _without(
        baseline_network, CANDIDATE_NETWORK_KEYS
    )
    candidate_shared['path'] = _without(candidate_path, {'strict_load_g'})
    baseline_shared['path'] = _without(baseline_path, {'strict_load_g'})
    if candidate_shared != baseline_shared:
        differing = sorted(
            key for key in set(candidate_shared) | set(baseline_shared)
            if candidate_shared.get(key) != baseline_shared.get(key)
        )
        raise ProtocolError(
            'Candidate and baseline differ outside the allowed treatment: '
            f'{differing}'
        )

    train = candidate.get('datasets', {}).get('train', {})
    if train.get('type') != 'MultiLevelPairedImageDataset':
        raise ProtocolError(
            'Journal runs must use MultiLevelPairedImageDataset.'
        )
    if int(candidate.get('num_gpu', 0)) != 1:
        raise ProtocolError('The audited sampler protocol currently requires one GPU.')
    if int(train.get('dataset_enlarge_ratio', 1)) != 1:
        raise ProtocolError('The audited protocol requires dataset_enlarge_ratio=1.')
    if int(train.get('batch_size_per_gpu', 0)) <= 0:
        raise ProtocolError('batch_size_per_gpu must be positive.')
    if int(candidate.get('train', {}).get('total_iter', 0)) <= 0:
        raise ProtocolError('total_iter must be positive.')

    return {
        'candidate_name': candidate['name'],
        'baseline_name': baseline['name'],
        'candidate_variant': candidate_variant,
        'allowed_treatment': {
            'candidate_top_level': {'mae_weight': 0.0},
            'candidate_network_g': expected_candidate,
            'candidate_strict_load_g': False,
            'baseline_network_g': {
                'use_sft': False,
                'mae_checkpoint': None,
            },
            'baseline_strict_load_g': True,
        },
        'shared_fields_verified': sorted(candidate_shared),
    }


def image_names(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ProtocolError(f'Dataset root does not exist: {root}')
    names = sorted(
        entry.name for entry in root.iterdir()
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not names:
        raise ProtocolError(f'No supported images found in {root}')
    return root, names


def source_group(filename):
    return re.sub(r'_s\d+$', '', Path(filename).stem)


def audit_dataset(train_opt, expected_images=None, expected_sources=None,
                  expected_level_order=None):
    gt_root, gt_names = image_names(train_opt['dataroot_gt'])
    raw_lq_roots = train_opt.get('dataroot_lq_levels')
    if raw_lq_roots is None:
        raw_lq_roots = train_opt['dataroot_lq']
    if isinstance(raw_lq_roots, str):
        raw_lq_roots = [raw_lq_roots]
    lq_roots = []
    for raw_root in raw_lq_roots:
        root, names = image_names(raw_root)
        if names != gt_names:
            gt_set = set(gt_names)
            names_set = set(names)
            raise ProtocolError(
                f'Unaligned LQ root {root}: '
                f'{len(gt_set - names_set)} missing and '
                f'{len(names_set - gt_set)} unexpected filenames.'
            )
        lq_roots.append(root)

    level_order = [root.name for root in lq_roots]
    if expected_level_order and level_order != list(expected_level_order):
        raise ProtocolError(
            f'Expected level order {expected_level_order}, got {level_order}.'
        )
    if expected_images is not None and len(gt_names) != expected_images:
        raise ProtocolError(
            f'Expected {expected_images} GT images, got {len(gt_names)}.'
        )
    sources = sorted({source_group(name) for name in gt_names})
    if expected_sources is not None and len(sources) != expected_sources:
        raise ProtocolError(
            f'Expected {expected_sources} source groups, got {len(sources)}.'
        )

    names_payload = ('\n'.join(gt_names) + '\n').encode('utf-8')
    sources_payload = ('\n'.join(sources) + '\n').encode('utf-8')
    return {
        'gt_root': str(gt_root),
        'lq_roots': [str(root) for root in lq_roots],
        'level_order': level_order,
        'gt_images': len(gt_names),
        'source_groups': len(sources),
        'filename_list_sha256': sha256_bytes(names_payload),
        'source_group_list_sha256': sha256_bytes(sources_payload),
        'logical_samples': len(gt_names) * len(lq_roots),
    }


class _SizedDataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


def sampler_prefix(dataset_size, level_count, total_iter, batch_size):
    """Return the exact single-rank BasicSR sampler prefix and diagnostics."""
    try:
        import torch
        from basicsr.data.data_sampler import EnlargedSampler
    except ImportError as error:
        raise ProtocolError(
            'Run this audit in the same environment used for DRCT training.'
        ) from error

    target = total_iter * batch_size
    selected = []
    epoch = 0
    while len(selected) < target:
        sampler = EnlargedSampler(_SizedDataset(dataset_size), 1, 0, ratio=1)
        sampler.set_epoch(epoch)
        selected.extend(iter(sampler))
        epoch += 1
    selected = selected[:target]

    digest = hashlib.sha256()
    level_counts = [0] * level_count
    crop_multiplicity = [0] * (dataset_size // level_count)
    for index in selected:
        digest.update(struct.pack('<q', index))
        level_counts[index % level_count] += 1
        crop_multiplicity[index // level_count] += 1
    multiplicity_histogram = {
        str(count): crop_multiplicity.count(count)
        for count in range(max(crop_multiplicity) + 1)
    }
    sampler_source = inspect.getsource(EnlargedSampler).encode('utf-8')
    return {
        'sampler': 'basicsr.data.data_sampler.EnlargedSampler',
        'sampler_source_sha256': sha256_bytes(sampler_source),
        'torch_version': torch.__version__,
        'total_iterations': total_iter,
        'batch_size_per_gpu': batch_size,
        'consumed_samples': target,
        'sampler_epochs_touched': epoch,
        'unique_logical_indices': len(set(selected)),
        'duplicate_logical_indices': target - len(set(selected)),
        'level_counts': level_counts,
        'source_crop_multiplicity_histogram': multiplicity_histogram,
        'first_ten_indices': selected[:10],
        'sampled_index_prefix_sha256_int64_le': digest.hexdigest(),
    }


def deterministic_worker_seeds(num_workers, seed, rank=0):
    if num_workers <= 0:
        raise ProtocolError(
            'The paired data-randomness audit requires positive num_workers.'
        )
    return [
        num_workers * rank + worker_id + seed
        for worker_id in range(num_workers)
    ]


def data_randomness_protocol(num_workers, seed, rank=0):
    """Bind crop/augmentation RNG sources and their deterministic worker seeds."""
    try:
        from basicsr.data import worker_init_fn
        from basicsr.data.transforms import augment, paired_random_crop
    except ImportError as error:
        raise ProtocolError(
            'Run this audit in the same BasicSR environment used for training.'
        ) from error
    worker_seeds = deterministic_worker_seeds(num_workers, seed, rank)
    return {
        'manual_seed': seed,
        'rank': rank,
        'num_workers': num_workers,
        'worker_seeds': worker_seeds,
        'worker_init_fn_source_sha256': sha256_bytes(
            inspect.getsource(worker_init_fn).encode('utf-8')
        ),
        'paired_random_crop_source_sha256': sha256_bytes(
            inspect.getsource(paired_random_crop).encode('utf-8')
        ),
        'augment_source_sha256': sha256_bytes(
            inspect.getsource(augment).encode('utf-8')
        ),
        'paired_guarantee': (
            'Candidate and baseline use the same fixed worker seeds; crop and augmentation '
            'draws therefore do not depend on candidate model initialization.'
        ),
    }


def main():
    args = parse_args()
    try:
        import yaml
    except ImportError as error:
        raise ProtocolError(
            'PyYAML is required; run the audit in the DRCT training environment.'
        ) from error

    candidate_path = Path(args.candidate).expanduser().resolve()
    baseline_path = Path(args.baseline).expanduser().resolve()
    with candidate_path.open(encoding='utf-8') as input_file:
        candidate = yaml.safe_load(input_file)
    with baseline_path.open(encoding='utf-8') as input_file:
        baseline = yaml.safe_load(input_file)

    candidate, baseline = apply_manual_seed_override(
        candidate, baseline, args.manual_seed_override
    )

    comparison = validate_matched_configs(
        candidate, baseline, candidate_variant=args.candidate_variant
    )
    train_opt = candidate['datasets']['train']
    dataset = audit_dataset(
        train_opt,
        expected_images=args.expected_gt_images,
        expected_sources=args.expected_source_groups,
        expected_level_order=args.expected_level_order,
    )
    sampler = sampler_prefix(
        dataset['logical_samples'],
        len(dataset['level_order']),
        int(candidate['train']['total_iter']),
        int(train_opt['batch_size_per_gpu']),
    )
    sampler['level_counts_by_name'] = dict(
        zip(dataset['level_order'], sampler['level_counts'])
    )
    data_randomness = data_randomness_protocol(
        int(train_opt['num_worker_per_gpu']),
        int(candidate['manual_seed']),
    )

    result = {
        'status': 'passed',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': (
            f'Pre-run audit of the matched {args.candidate_variant.upper()} '
            'candidate and B0 baseline; '
            'this is not an evaluation result.'
        ),
        'configs': {
            'candidate': str(candidate_path),
            'candidate_sha256': sha256_file(candidate_path),
            'baseline': str(baseline_path),
            'baseline_sha256': sha256_file(baseline_path),
        },
        'runtime_overrides': {
            'manual_seed': args.manual_seed_override,
            'candidate_variant': args.candidate_variant,
        },
        'comparison': comparison,
        'dataset': dataset,
        'sampler_prefix': sampler,
        'data_randomness': data_randomness,
        'environment': {
            'python': sys.version,
            'platform': platform.platform(),
            'working_directory': os.getcwd(),
        },
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
