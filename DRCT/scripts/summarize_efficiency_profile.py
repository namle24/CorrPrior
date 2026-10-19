"""Validate and summarize repeated B0/C2 efficiency profiles."""

import argparse
import json
import os
import statistics
from pathlib import Path


MODELS = ('B0', 'C2')
SIZES = (64, 128)
TRIALS = tuple(range(1, 6))
REQUIRED_STABLE_FIELDS = (
    'total_parameters',
    'trainable_parameters',
    'checkpoint_size_mib',
    'input_shape',
    'output_shape',
    'cuda_device',
    'torch_version',
    'torch_cuda_version',
    'cudnn_version',
    'warmup',
    'repeats',
    'seed',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize repeated final-model efficiency profiles.'
    )
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def _load_profiles(input_dir):
    records = {}
    for model in MODELS:
        for size in SIZES:
            key = (model, size)
            records[key] = []
            for trial in TRIALS:
                path = input_dir / f'{model}_{size}x{size}_trial{trial}.json'
                if not path.is_file():
                    raise FileNotFoundError(f'Missing profile trial: {path}')
                with path.open(encoding='utf-8') as profile_file:
                    record = json.load(profile_file)
                record['_source'] = str(path.resolve())
                records[key].append(record)
    return records


def _assert_stable(records, model, size):
    reference = records[0]
    for trial_index, record in enumerate(records[1:], start=2):
        for field in REQUIRED_STABLE_FIELDS:
            if record.get(field) != reference.get(field):
                raise ValueError(
                    f'{model} {size}x{size} field {field!r} differs in '
                    f'trial {trial_index}: {record.get(field)!r} != '
                    f'{reference.get(field)!r}'
                )


def _describe(values):
    return {
        'values': values,
        'mean': statistics.fmean(values),
        'sample_std': statistics.stdev(values),
        'median': statistics.median(values),
        'minimum': min(values),
        'maximum': max(values),
    }


def summarize(input_dir):
    records = _load_profiles(input_dir)
    summary = {
        'schema': 'rn_pd_mae_efficiency_profile_v1',
        'trial_count': len(TRIALS),
        'sizes': {},
    }
    for size in SIZES:
        size_summary = {'models': {}}
        for model in MODELS:
            group = records[(model, size)]
            _assert_stable(group, model, size)
            first = group[0]
            size_summary['models'][model] = {
                field: first.get(field) for field in REQUIRED_STABLE_FIELDS
            }
            size_summary['models'][model].update({
                'latency_ms_mean_cuda': _describe([
                    float(record['latency_ms_mean_cuda']) for record in group
                ]),
                'latency_ms_mean_wall': _describe([
                    float(record['latency_ms_mean_wall']) for record in group
                ]),
                'peak_memory_mib': _describe([
                    float(record['peak_memory_mib']) for record in group
                ]),
                'source_files': [record['_source'] for record in group],
            })

        baseline = size_summary['models']['B0']
        candidate = size_summary['models']['C2']
        b0_latency = baseline['latency_ms_mean_cuda']['median']
        c2_latency = candidate['latency_ms_mean_cuda']['median']
        b0_memory = baseline['peak_memory_mib']['median']
        c2_memory = candidate['peak_memory_mib']['median']
        size_summary['C2_minus_B0'] = {
            'total_parameters': (
                candidate['total_parameters'] - baseline['total_parameters']
            ),
            'trainable_parameters': (
                candidate['trainable_parameters'] -
                baseline['trainable_parameters']
            ),
            'checkpoint_size_mib': (
                candidate['checkpoint_size_mib'] -
                baseline['checkpoint_size_mib']
            ),
            'latency_ms_median': c2_latency - b0_latency,
            'latency_ratio_median': c2_latency / b0_latency,
            'peak_memory_mib_median': c2_memory - b0_memory,
            'peak_memory_ratio_median': c2_memory / b0_memory,
        }
        summary['sizes'][f'{size}x{size}'] = size_summary
    return summary


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    result = summarize(input_dir)
    output_path = Path(os.path.abspath(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as output_file:
        json.dump(result, output_file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
