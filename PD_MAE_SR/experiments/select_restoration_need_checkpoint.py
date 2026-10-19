"""Select RN-PD-MAE checkpoints using predeclared validation rules.

The selector consumes machine-readable evaluator outputs and never reads
downstream SR metrics.  It supports the two checkpoint decisions in the final
journal lineage:

* partial pretraining: minimize macro reconstruction MSE, then need MAE;
* full-degradation calibration: enforce the identity false-positive threshold,
  minimize degraded-input need MAE, and prefer the earlier checkpoint when the
  difference is within the declared tolerance.
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import subprocess


METRIC_FILES = {
    'partial': 'partial_restoration_need_metrics.csv',
    'calibration': 'restoration_need_metrics.csv',
}

PROTOCOL_ARGUMENT_KEYS = {
    'partial': (
        'data_roots', 'patch_size', 'mae_patch_size', 'need_scale',
        'val_fraction', 'split_seed', 'max_images', 'sample_seed',
    ),
    'calibration': (
        'hr_root', 'lq_roots', 'patch_size', 'mae_patch_size', 'need_scale',
        'val_fraction', 'split_seed', 'split_unit', 'alignment_backend',
        'max_images', 'sample_seed',
    ),
}


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


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_evaluation_spec(spec):
    try:
        iteration_text, directory = spec.split('=', 1)
        iteration = int(iteration_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            f'Expected ITERATION=DIRECTORY, received {spec!r}.'
        ) from error
    if iteration <= 0 or not directory:
        raise argparse.ArgumentTypeError(
            f'Expected a positive iteration and nonempty directory, received {spec!r}.'
        )
    return iteration, os.path.abspath(os.path.expanduser(directory))


def read_rows(path):
    with open(path, newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError(f'No metric rows found in {path}.')
    if 'level' not in rows[0]:
        raise RuntimeError(f'Metric file has no level column: {path}.')
    return rows


def finite_float(row, field, path):
    if field not in row:
        raise RuntimeError(f'Metric file {path} has no {field} column.')
    try:
        value = float(row[field])
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f'Invalid {field} value {row[field]!r} in {path}.'
        ) from error
    if not math.isfinite(value):
        raise RuntimeError(f'Non-finite {field} value in {path}.')
    return value


def macro_mean(rows, field, path):
    return sum(finite_float(row, field, path) for row in rows) / len(rows)


def evaluation_provenance(directory):
    protocol_path = os.path.join(directory, 'evaluation_protocol.json')
    if not os.path.isfile(protocol_path):
        return {'protocol_path': protocol_path, 'protocol': None}
    with open(protocol_path, encoding='utf-8') as file:
        protocol = json.load(file)
    return {'protocol_path': protocol_path, 'protocol': protocol}


def summarize_partial(iteration, directory, expected_levels=None):
    path = os.path.join(directory, METRIC_FILES['partial'])
    rows = read_rows(path)
    levels = [row['level'] for row in rows]
    if len(levels) != len(set(levels)):
        raise RuntimeError(f'Duplicate level rows in {path}: {levels}.')
    if expected_levels and set(levels) != set(expected_levels):
        raise RuntimeError(
            f'Level mismatch in {path}: expected {sorted(expected_levels)}, '
            f'found {sorted(levels)}.'
        )
    return {
        'iteration': iteration,
        'evaluation_dir': directory,
        'metric_file': path,
        'levels': levels,
        'macro_reconstruction_mse': macro_mean(
            rows, 'reconstruction_mse_mean', path
        ),
        'macro_need_mae': macro_mean(rows, 'mae', path),
        'provenance': evaluation_provenance(directory),
    }


def summarize_calibration(
        iteration, directory, identity_level, identity_max, expected_levels=None):
    path = os.path.join(directory, METRIC_FILES['calibration'])
    rows = read_rows(path)
    levels = [row['level'] for row in rows]
    if len(levels) != len(set(levels)):
        raise RuntimeError(f'Duplicate level rows in {path}: {levels}.')
    if expected_levels and set(levels) != set(expected_levels):
        raise RuntimeError(
            f'Level mismatch in {path}: expected {sorted(expected_levels)}, '
            f'found {sorted(levels)}.'
        )
    identity_rows = [row for row in rows if row['level'] == identity_level]
    if len(identity_rows) != 1:
        raise RuntimeError(
            f'Expected one identity row named {identity_level!r} in {path}, '
            f'found {len(identity_rows)}.'
        )
    degraded_rows = [row for row in rows if row['level'] != identity_level]
    if not degraded_rows:
        raise RuntimeError(f'No degraded-input rows found in {path}.')
    identity_prediction_mean = finite_float(
        identity_rows[0], 'prediction_mean', path
    )
    return {
        'iteration': iteration,
        'evaluation_dir': directory,
        'metric_file': path,
        'levels': levels,
        'identity_level': identity_level,
        'identity_prediction_mean': identity_prediction_mean,
        'identity_eligible': identity_prediction_mean < identity_max,
        'macro_need_mae': macro_mean(degraded_rows, 'mae', path),
        'provenance': evaluation_provenance(directory),
    }


def select_partial(candidates):
    return min(
        candidates,
        key=lambda item: (
            item['macro_reconstruction_mse'],
            item['macro_need_mae'],
            item['iteration'],
        ),
    )


def select_calibration(candidates, tolerance):
    eligible = [item for item in candidates if item['identity_eligible']]
    if not eligible:
        raise RuntimeError('No checkpoint passes the identity false-positive gate.')
    best_mae = min(item['macro_need_mae'] for item in eligible)
    statistically_tied = [
        item for item in eligible
        if item['macro_need_mae'] <= best_mae + tolerance
    ]
    return min(statistically_tied, key=lambda item: item['iteration'])


def validate_protocols(
        candidates, mode, require_protocol=False, expected_split_unit=None,
        checkpoint_pattern=None):
    """Reject stale or incomparable evaluator outputs before selection."""
    reference_signature = None
    for candidate in candidates:
        provenance = candidate['provenance']
        protocol = provenance['protocol']
        if protocol is None:
            if require_protocol:
                raise RuntimeError(
                    f'Missing evaluation protocol: {provenance["protocol_path"]}'
                )
            continue

        checkpoint_iter = protocol.get('checkpoint_iter')
        if checkpoint_iter is None or int(checkpoint_iter) != candidate['iteration']:
            raise RuntimeError(
                'Evaluator/checkpoint iteration mismatch in '
                f'{provenance["protocol_path"]}: expected '
                f'{candidate["iteration"]}, found {checkpoint_iter!r}.'
            )
        if (expected_split_unit is not None
                and protocol.get('split_unit') != expected_split_unit):
            raise RuntimeError(
                f'Expected split_unit={expected_split_unit!r} in '
                f'{provenance["protocol_path"]}, found '
                f'{protocol.get("split_unit")!r}.'
            )
        if mode == 'partial' and protocol.get('joint_split_across_roots') is not True:
            raise RuntimeError(
                'Partial-pretraining evaluation must reproduce one joint '
                f'source split across all roots: {provenance["protocol_path"]}.'
            )
        if mode == 'partial' and not protocol.get('evaluated_source_group_sha256'):
            raise RuntimeError(
                'Partial-pretraining evaluation has no source-group digest: '
                f'{provenance["protocol_path"]}.'
            )
        if mode == 'partial' and not protocol.get('implementation_sha256'):
            raise RuntimeError(
                'Partial-pretraining evaluation has no implementation hashes: '
                f'{provenance["protocol_path"]}.'
            )

        arguments = protocol.get('arguments')
        if not isinstance(arguments, dict):
            raise RuntimeError(
                f'Protocol has no argument mapping: {provenance["protocol_path"]}'
            )
        if checkpoint_pattern is not None:
            expected_checkpoint = os.path.abspath(os.path.expanduser(
                checkpoint_pattern.format(iteration=candidate['iteration'])
            ))
            observed_checkpoint = os.path.abspath(os.path.expanduser(
                arguments.get('checkpoint', '')
            ))
            if observed_checkpoint != expected_checkpoint:
                raise RuntimeError(
                    'Evaluator used a different checkpoint in '
                    f'{provenance["protocol_path"]}: expected '
                    f'{expected_checkpoint}, found {observed_checkpoint}.'
                )

        signature = {
            'arguments': {
                key: arguments.get(key)
                for key in PROTOCOL_ARGUMENT_KEYS[mode]
            },
            'checkpoint_method': protocol.get('checkpoint_method'),
            'checkpoint_config': protocol.get('checkpoint_config'),
            'checkpoint_pretrain_provenance': protocol.get(
                'checkpoint_pretrain_provenance'
            ),
            'checkpoint_selection_provenance': protocol.get(
                'checkpoint_selection_provenance'
            ),
            'evaluated_source_groups': protocol.get('evaluated_source_groups'),
            'evaluated_source_group_sha256': protocol.get(
                'evaluated_source_group_sha256'
            ),
            'joint_split_across_roots': protocol.get(
                'joint_split_across_roots'
            ),
            'implementation_sha256': protocol.get('implementation_sha256'),
        }
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise RuntimeError(
                'Evaluator protocols are not comparable. Mismatch found in '
                f'{provenance["protocol_path"]}.'
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=sorted(METRIC_FILES), required=True)
    parser.add_argument(
        '--evaluation', action='append', type=parse_evaluation_spec, required=True,
        metavar='ITERATION=DIRECTORY',
        help='Evaluator output. Repeat once for every scheduled checkpoint.',
    )
    parser.add_argument(
        '--checkpoint_pattern', required=True,
        help='Checkpoint path containing a {iteration} replacement field.',
    )
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected_levels', nargs='+', default=None)
    parser.add_argument('--identity_level', default='HR_sub')
    parser.add_argument('--identity_max', type=float, default=0.02)
    parser.add_argument('--calibration_tolerance', type=float, default=0.001)
    parser.add_argument('--require_checkpoint', action='store_true')
    parser.add_argument('--require_protocol', action='store_true')
    parser.add_argument('--expected_split_unit', choices=['filename', 'source'])
    return parser.parse_args()


def main():
    args = parse_args()
    evaluations = sorted(args.evaluation)
    iterations = [iteration for iteration, _ in evaluations]
    if len(iterations) != len(set(iterations)):
        raise RuntimeError(f'Duplicate checkpoint iterations: {iterations}.')
    if '{iteration}' not in args.checkpoint_pattern:
        raise RuntimeError('--checkpoint_pattern must contain {iteration}.')

    if args.mode == 'partial':
        candidates = [
            summarize_partial(iteration, directory, args.expected_levels)
            for iteration, directory in evaluations
        ]
    else:
        candidates = [
            summarize_calibration(
                iteration, directory, args.identity_level, args.identity_max,
                args.expected_levels,
            )
            for iteration, directory in evaluations
        ]

    validate_protocols(
        candidates,
        args.mode,
        require_protocol=args.require_protocol,
        expected_split_unit=args.expected_split_unit,
        checkpoint_pattern=args.checkpoint_pattern,
    )

    if args.mode == 'partial':
        selected = select_partial(candidates)
        rule = {
            'primary': 'lowest macro-average reconstruction_mse_mean',
            'tie_breaker': 'lowest macro-average need MAE, then earlier iteration',
        }
    else:
        selected = select_calibration(candidates, args.calibration_tolerance)
        rule = {
            'eligibility': (
                f'{args.identity_level} prediction mean < {args.identity_max}'
            ),
            'primary': 'lowest macro-average degraded-input need MAE',
            'near_tie': (
                'earlier iteration when macro MAE is within '
                f'{args.calibration_tolerance} of the minimum'
            ),
        }

    checkpoint = os.path.abspath(os.path.expanduser(
        args.checkpoint_pattern.format(iteration=selected['iteration'])
    ))
    if args.require_checkpoint and not os.path.isfile(checkpoint):
        raise RuntimeError(f'Selected checkpoint does not exist: {checkpoint}.')
    checkpoint_sha256 = (
        sha256_file(checkpoint) if os.path.isfile(checkpoint) else None
    )

    output = {
        'schema_version': 1,
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'selector_git_commit': repository_commit(),
        'mode': args.mode,
        'selection_rule': rule,
        'candidates': candidates,
        'selected_iteration': selected['iteration'],
        'selected_checkpoint': checkpoint,
        'selected_checkpoint_sha256': checkpoint_sha256,
    }
    output_path = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(output, file, indent=2)
        file.write('\n')
    print(checkpoint)


if __name__ == '__main__':
    main()
