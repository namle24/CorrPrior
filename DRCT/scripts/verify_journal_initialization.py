"""Verify that zero-initialized RN-PD-MAE starts as the matched DRCT model."""

import argparse
import gc
import hashlib
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import drct.archs  # noqa: E402,F401
from basicsr.archs import build_network  # noqa: E402


class InitializationError(RuntimeError):
    """Raised when P0 and B0 do not share the declared initialization."""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path):
    with open(path, encoding='utf-8') as input_file:
        return yaml.safe_load(input_file)


def selected_prior(
        selection_path, expected_mode='calibration', expected_method=None):
    selection_path = os.path.abspath(os.path.expanduser(selection_path))
    with open(selection_path, encoding='utf-8') as selection_file:
        selection = json.load(selection_file)
    if selection.get('mode') != expected_mode:
        raise InitializationError(
            f'Prior selection must have mode={expected_mode}.'
        )
    if expected_method is None:
        expected_method = {
            'calibration': 'restoration_need_calibration',
            'partial': 'partial_restoration_need_pretraining',
        }[expected_mode]
    checkpoint_path = os.path.abspath(os.path.expanduser(
        selection.get('selected_checkpoint', '')
    ))
    expected_hash = selection.get('selected_checkpoint_sha256')
    if not expected_hash:
        raise InitializationError(
            'Final prior selection has no selected checkpoint SHA-256.'
        )
    if not os.path.isfile(checkpoint_path):
        raise InitializationError(
            f'Selected prior checkpoint does not exist: {checkpoint_path}'
        )
    observed_hash = sha256_file(checkpoint_path)
    if observed_hash != expected_hash:
        raise InitializationError(
            'Selected prior checkpoint hash differs from the selection JSON.'
        )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if checkpoint.get('method') != expected_method:
        raise InitializationError(
            f'Selected prior method must be {expected_method!r}, found '
            f'{checkpoint.get("method")!r}.'
        )
    if checkpoint.get('iter') != selection.get('selected_iteration'):
        raise InitializationError(
            'Selected prior iteration differs from the selection JSON.'
        )
    checkpoint_metadata = {
        'path': checkpoint_path,
        'sha256': observed_hash,
        'iteration': checkpoint.get('iter'),
        'method': checkpoint.get('method'),
        'pretrain_provenance': checkpoint.get('pretrain_provenance'),
        'selection_provenance': checkpoint.get('selection_provenance'),
    }
    del checkpoint
    gc.collect()
    return selection, checkpoint_metadata


def validate_candidate_missing_keys(missing_keys, unexpected_keys):
    allowed_prefixes = ('sft_layers.', 'mae_wrapper.')
    invalid_missing = [
        key for key in missing_keys if not key.startswith(allowed_prefixes)
    ]
    if invalid_missing or unexpected_keys:
        raise InitializationError(
            'Unsafe candidate checkpoint load: '
            f'invalid_missing={invalid_missing}, unexpected={unexpected_keys}'
        )


def maximum_parameter_magnitude(module):
    maxima = [
        parameter.detach().abs().max().item()
        for parameter in module.parameters()
    ]
    return max(maxima, default=0.0)


def compare_shared_state(candidate, baseline):
    candidate_state = candidate.state_dict()
    baseline_state = baseline.state_dict()
    missing = sorted(set(baseline_state) - set(candidate_state))
    unequal = []
    for key, baseline_value in baseline_state.items():
        candidate_value = candidate_state.get(key)
        if candidate_value is not None and not torch.equal(
                candidate_value.cpu(), baseline_value.cpu()):
            unequal.append(key)
    if missing or unequal:
        raise InitializationError(
            f'Shared backbone mismatch: missing={missing}, unequal={unequal}'
        )
    return len(baseline_state)


def load_backbone_checkpoint(candidate, baseline, candidate_opt, baseline_opt):
    candidate_path = os.path.abspath(os.path.expanduser(
        candidate_opt['path']['pretrain_network_g']
    ))
    baseline_path = os.path.abspath(os.path.expanduser(
        baseline_opt['path']['pretrain_network_g']
    ))
    if candidate_path != baseline_path:
        raise InitializationError(
            'Candidate and baseline do not use the same backbone checkpoint.'
        )
    checkpoint = torch.load(candidate_path, map_location='cpu')
    candidate_key = candidate_opt['path'].get('param_key_g', 'params')
    baseline_key = baseline_opt['path'].get('param_key_g', 'params')
    if candidate_key != baseline_key or candidate_key not in checkpoint:
        raise InitializationError(
            'Candidate and baseline checkpoint parameter keys do not match.'
        )
    state = checkpoint[candidate_key]
    baseline_incompatible = baseline.load_state_dict(state, strict=True)
    if (baseline_incompatible.missing_keys
            or baseline_incompatible.unexpected_keys):
        raise InitializationError('Strict baseline checkpoint load failed.')
    candidate_incompatible = candidate.load_state_dict(state, strict=False)
    validate_candidate_missing_keys(
        candidate_incompatible.missing_keys,
        candidate_incompatible.unexpected_keys,
    )
    return {
        'path': candidate_path,
        'sha256': sha256_file(candidate_path),
        'param_key': candidate_key,
        'candidate_missing_keys': sorted(candidate_incompatible.missing_keys),
        'candidate_unexpected_keys': sorted(
            candidate_incompatible.unexpected_keys
        ),
    }


def parameter_counts(model):
    return {
        'total': sum(parameter.numel() for parameter in model.parameters()),
        'trainable': sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def parse_shape(value):
    try:
        height_text, width_text = value.lower().split('x', 1)
        height, width = int(height_text), int(width_text)
    except (AttributeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f'Expected HEIGHTxWIDTH, received {value!r}.'
        ) from error
    if min(height, width) <= 0:
        raise argparse.ArgumentTypeError('Shape dimensions must be positive.')
    return height, width


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate_config', required=True)
    parser.add_argument('--baseline_config', required=True)
    parser.add_argument('--prior_selection', required=True)
    parser.add_argument(
        '--expected_selection_mode', choices=['calibration', 'partial'],
        default='calibration',
    )
    parser.add_argument('--expected_prior_method', default=None)
    parser.add_argument('--candidate_label', choices=['P0', 'C1', 'C2'])
    parser.add_argument(
        '--shape', action='append', type=parse_shape,
        default=None, help='LR test shape HEIGHTxWIDTH; may be repeated.',
    )
    parser.add_argument('--atol', type=float, default=1e-7)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.atol < 0:
        raise ValueError('--atol must be nonnegative.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise InitializationError('CUDA was requested but is unavailable.')
    device = torch.device(args.device)
    shapes = args.shape or [(64, 64), (64, 96)]

    candidate_config_path = Path(args.candidate_config).expanduser().resolve()
    baseline_config_path = Path(args.baseline_config).expanduser().resolve()
    candidate_opt = load_yaml(candidate_config_path)
    baseline_opt = load_yaml(baseline_config_path)
    selection, prior = selected_prior(
        args.prior_selection,
        expected_mode=args.expected_selection_mode,
        expected_method=args.expected_prior_method,
    )

    candidate_network_opt = deepcopy(candidate_opt['network_g'])
    candidate_network_opt['mae_checkpoint'] = prior['path']
    candidate = build_network(candidate_network_opt)
    baseline = build_network(deepcopy(baseline_opt['network_g']))
    backbone = load_backbone_checkpoint(
        candidate, baseline, candidate_opt, baseline_opt
    )
    shared_tensors = compare_shared_state(candidate, baseline)
    sft_max_abs = maximum_parameter_magnitude(candidate.sft_layers)
    if sft_max_abs != 0.0:
        raise InitializationError(
            f'SFT parameters are not exactly zero: max_abs={sft_max_abs}.'
        )

    candidate.eval().to(device)
    baseline.eval().to(device)
    generator = torch.Generator(device='cpu')
    generator.manual_seed(0)
    comparisons = []
    with torch.inference_mode():
        for height, width in shapes:
            image = torch.rand(
                1, 3, height, width, generator=generator
            ).to(device)
            baseline_output = baseline(image)
            candidate_output = candidate(image)
            difference = (candidate_output - baseline_output).abs()
            max_abs = difference.max().item()
            mean_abs = difference.mean().item()
            if max_abs > args.atol:
                raise InitializationError(
                    f'Initialization mismatch for {height}x{width}: '
                    f'max_abs={max_abs} > atol={args.atol}.'
                )
            comparisons.append({
                'input_shape': list(image.shape),
                'output_shape': list(candidate_output.shape),
                'max_abs_difference': max_abs,
                'mean_abs_difference': mean_abs,
                'atol': args.atol,
            })

    candidate_label = args.candidate_label or (
        'C2' if candidate_opt['network_g'].get('use_sft') is True
        and candidate_opt['network_g'].get('use_restoration_need_gate') is False
        else 'P0'
    )
    result = {
        'status': 'passed',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': f'Verify matched {candidate_label}/B0 function initialization.',
        'candidate_label': candidate_label,
        'candidate_config': str(candidate_config_path),
        'candidate_config_sha256': sha256_file(candidate_config_path),
        'baseline_config': str(baseline_config_path),
        'baseline_config_sha256': sha256_file(baseline_config_path),
        'prior_selection': str(Path(args.prior_selection).expanduser().resolve()),
        'prior_selection_sha256': sha256_file(args.prior_selection),
        'prior_selection_mode': selection['mode'],
        'prior': prior,
        'backbone': backbone,
        'shared_backbone_tensors_verified': shared_tensors,
        'sft_parameter_max_abs': sft_max_abs,
        'candidate_parameters': parameter_counts(candidate),
        'baseline_parameters': parameter_counts(baseline),
        'device': str(device),
        'torch_version': torch.__version__,
        'comparisons': comparisons,
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
