"""Evaluate post-hoc RealSR prior-feature counterfactuals."""

import argparse
import datetime
import json
from pathlib import Path

try:
    from scripts.evaluate_fidelity_mechanism_controls import file_sha256
    from scripts.evaluate_prior_stage_controls import evaluate
except ModuleNotFoundError:  # Direct execution from DRCT/scripts.
    from evaluate_fidelity_mechanism_controls import file_sha256
    from evaluate_prior_stage_controls import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument(
        '--camera', action='append', nargs=6, required=True,
        metavar=(
            'COMPARISON', 'CAMERA', 'BASE_STD', 'CAND_STD',
            'BASE_IQA', 'CAND_IQA',
        ),
    )
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(
            f'Refusing to overwrite prior-feature-control report: {output}'
        )
    protocol_path = args.protocol.expanduser().resolve()
    with protocol_path.open(encoding='utf-8') as protocol_file:
        protocol = json.load(protocol_file)
    if protocol.get('schema_version') != 1:
        raise ValueError('Unexpected prior-feature-control protocol schema.')
    decision = evaluate(args.camera, protocol)
    report = {
        'schema_version': 1,
        'created_utc': datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        'disclosure': protocol['disclosure'],
        'protocol': str(protocol_path),
        'protocol_sha256': file_sha256(protocol_path),
        'implementation_sha256': file_sha256(Path(__file__).resolve()),
        'dependency_sha256': file_sha256(
            Path(__file__).with_name('evaluate_prior_stage_controls.py')
        ),
        **decision,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as output_file:
        json.dump(report, output_file, indent=2)
        output_file.write('\n')
    print(json.dumps(report['claims'], indent=2))


if __name__ == '__main__':
    main()
