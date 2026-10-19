"""Run BasicSR inference after a deterministic intervention on the frozen prior.

This entrypoint is reserved for post-hoc mechanism diagnostics.  It changes
only the feature tensor returned by ``PD_MAE_Encoder_Wrapper`` at inference:
``zero`` removes image-conditioned features while retaining trained SFT
biases, and ``shifted`` preserves the feature values but breaks their spatial
correspondence with a half-field circular shift.  The trained checkpoint is
never modified.
"""

import argparse
import datetime
import hashlib
import json
import os.path as osp
from pathlib import Path
import subprocess
import sys


CONTROLS = ('zero', 'shifted')


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(chunk_size), b''):
            digest.update(block)
    return digest.hexdigest()


def repository_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_control_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--prior_feature_control', required=True, choices=CONTROLS
    )
    parser.add_argument('--prior_feature_record', type=Path, required=True)
    args, remaining = parser.parse_known_args(argv[1:])
    return args, [argv[0], *remaining]


def transform_feature(feature, control):
    if control == 'zero':
        return feature.new_zeros(feature.shape)
    if control == 'shifted':
        return feature.roll(
            shifts=(
                max(1, feature.shape[-2] // 2),
                max(1, feature.shape[-1] // 2),
            ),
            dims=(-2, -1),
        )
    raise ValueError(f'Unknown prior-feature control: {control!r}.')


def install_control(control):
    from drct.archs.DRCT_arch import PD_MAE_Encoder_Wrapper

    original_forward = PD_MAE_Encoder_Wrapper.forward

    def controlled_forward(wrapper, *args, **kwargs):
        output = original_forward(wrapper, *args, **kwargs)
        if isinstance(output, tuple):
            feature, *remainder = output
            return (transform_feature(feature, control), *remainder)
        return transform_feature(output, control)

    PD_MAE_Encoder_Wrapper.forward = controlled_forward
    return original_forward


def named_force_paths(argv):
    artifacts = {}
    for value in argv:
        for field in ('path:pretrain_network_g', 'network_g:mae_checkpoint'):
            prefix = f'{field}='
            if value.startswith(prefix):
                path = Path(value[len(prefix):]).expanduser().resolve()
                if path.is_file():
                    artifacts[field] = {
                        'path': str(path),
                        'bytes': path.stat().st_size,
                        'sha256': file_sha256(path),
                    }
    return artifacts


def configuration_metadata(argv):
    if '-opt' not in argv:
        return None
    index = argv.index('-opt') + 1
    if index >= len(argv):
        return None
    path = Path(argv[index]).expanduser().resolve()
    if not path.is_file():
        return None
    return {
        'path': str(path),
        'bytes': path.stat().st_size,
        'sha256': file_sha256(path),
    }


def main():
    control_args, downstream_argv = parse_control_args(sys.argv)
    record = control_args.prior_feature_record.expanduser().resolve()
    if record.exists():
        raise RuntimeError(
            f'Refusing to overwrite prior-feature record: {record}'
        )

    sys.argv = downstream_argv
    install_control(control_args.prior_feature_control)

    # Importing drct.test registers the project modules and installs the
    # repository's output-root and post-override seed handling before BasicSR
    # parses the remaining arguments.
    import basicsr.test as basicsr_test
    import drct.test  # noqa: F401

    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    basicsr_test.test_pipeline(root_path)

    report = {
        'schema_version': 1,
        'created_utc': datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        'status': 'completed',
        'analysis_scope': 'post-hoc inference-only mechanism diagnostic',
        'prior_feature_control': control_args.prior_feature_control,
        'control_definition': {
            'zero': (
                'replace the frozen prior feature field by zeros while '
                'retaining the trained SFT parameters'
            ),
            'shifted': (
                'circularly shift the frozen prior feature field by half of '
                'each spatial axis while preserving all feature values'
            ),
        }[control_args.prior_feature_control],
        'repository_commit': repository_commit(),
        'entrypoint': str(Path(__file__).resolve()),
        'entrypoint_sha256': file_sha256(Path(__file__).resolve()),
        'command': downstream_argv,
        'configuration': configuration_metadata(downstream_argv),
        'artifacts': named_force_paths(downstream_argv),
    }
    record.parent.mkdir(parents=True, exist_ok=True)
    with record.open('x', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
        file.write('\n')
    print(f'Prior-feature control complete: {record}')


if __name__ == '__main__':
    main()
