"""Profile a DRCT configuration with a fixed input and reproducible settings."""

import argparse
import json
import os
import sys
import time
from copy import deepcopy

import torch
import yaml


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import drct.archs  # noqa: E402,F401
from basicsr.archs import build_network  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='Profile a journal DRCT model.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument(
        '--prior_checkpoint', default=None,
        help='Optional override for network_g.mae_checkpoint.',
    )
    parser.add_argument('--param_key', default='params_ema')
    parser.add_argument(
        '--strict_load', choices=['config', 'true', 'false'], default='config',
        help=(
            'Checkpoint loading policy. Use false only when profiling a '
            'candidate initialized from a baseline checkpoint; missing keys '
            'are then restricted to the prior and SFT branches.'
        ),
    )
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.height, args.width, args.warmup, args.repeats) <= 0:
        raise ValueError('Input dimensions, warmup, and repeats must be positive.')

    with open(args.config, encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    if args.prior_checkpoint:
        config['network_g']['mae_checkpoint'] = os.path.expanduser(
            args.prior_checkpoint
        )
    model = build_network(deepcopy(config['network_g']))

    checkpoint_path = args.checkpoint or config.get('path', {}).get('pretrain_network_g')
    missing_keys = []
    unexpected_keys = []
    if checkpoint_path:
        checkpoint = torch.load(os.path.expanduser(checkpoint_path), map_location='cpu')
        if args.param_key not in checkpoint:
            raise KeyError(
                f'Parameter key {args.param_key!r} not found in {checkpoint_path}; '
                f'available keys are {sorted(checkpoint)}.'
            )
        if args.strict_load == 'config':
            strict_load = bool(config.get('path', {}).get('strict_load_g', True))
        else:
            strict_load = args.strict_load == 'true'
        incompatible = model.load_state_dict(
            checkpoint[args.param_key], strict=strict_load
        )
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)
        if not strict_load:
            allowed_prefixes = ('sft_layers.', 'mae_wrapper.')
            invalid_missing = [
                key for key in missing_keys
                if not key.startswith(allowed_prefixes)
            ]
            if invalid_missing or unexpected_keys:
                raise RuntimeError(
                    'Unsafe partial checkpoint load: '
                    f'invalid_missing={invalid_missing}, '
                    f'unexpected={unexpected_keys}'
                )

    model.eval().cuda()
    image_generator = torch.Generator(device='cuda').manual_seed(args.seed)
    image = torch.rand(
        1, 3, args.height, args.width, device='cuda', generator=image_generator
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(image)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        wall_start = time.perf_counter()
        for _ in range(args.repeats):
            output = model(image)
        end_event.record()
        torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - wall_start
        elapsed_ms = start_event.elapsed_time(end_event)

    result = {
        'config': os.path.abspath(args.config),
        'checkpoint': os.path.abspath(os.path.expanduser(checkpoint_path))
        if checkpoint_path else None,
        'prior_checkpoint': (
            os.path.abspath(os.path.expanduser(args.prior_checkpoint))
            if args.prior_checkpoint else
            config.get('network_g', {}).get('mae_checkpoint')
        ),
        'checkpoint_strict_load': strict_load if checkpoint_path else None,
        'checkpoint_size_mib': (
            os.path.getsize(os.path.expanduser(checkpoint_path)) / (1024.0 ** 2)
            if checkpoint_path else None
        ),
        'prior_checkpoint_size_mib': (
            os.path.getsize(os.path.expanduser(args.prior_checkpoint)) /
            (1024.0 ** 2)
            if args.prior_checkpoint else None
        ),
        'checkpoint_missing_keys': missing_keys,
        'checkpoint_unexpected_keys': unexpected_keys,
        'input_shape': list(image.shape),
        'output_shape': list(output.shape),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'latency_ms_mean_cuda': elapsed_ms / args.repeats,
        'latency_ms_mean_wall': wall_seconds * 1000.0 / args.repeats,
        'peak_memory_mib': torch.cuda.max_memory_allocated() / (1024.0 ** 2),
        'warmup': args.warmup,
        'repeats': args.repeats,
        'seed': args.seed,
        'torch_version': torch.__version__,
        'torch_cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'cuda_device': torch.cuda.get_device_name(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as output_file:
        json.dump(result, output_file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
