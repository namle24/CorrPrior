"""Verify every provenance record in a spatial-mixture benchmark."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


class VerificationError(RuntimeError):
    """Raised when a benchmark artifact fails provenance verification."""


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolved_child(root, relative_path):
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise VerificationError(
            f'Artifact path escapes the benchmark root: {relative_path}'
        ) from error
    return path


def require_hash(path, expected_hash, label):
    if not path.is_file():
        raise VerificationError(f'Missing {label}: {path}')
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise VerificationError(
            f'{label} hash mismatch for {path}: '
            f'expected {expected_hash}, got {observed_hash}.'
        )


def verify_benchmark(root, expected_degradation_sha=None,
                     expected_records=None, expected_sources=None,
                     expected_patterns=None):
    root = Path(root).expanduser().resolve()
    protocol_path = root / 'protocol.json'
    manifest_path = root / 'manifest.csv'
    summary_path = root / 'manifest_summary.json'
    for path in (protocol_path, manifest_path, summary_path):
        if not path.is_file():
            raise VerificationError(f'Missing benchmark metadata: {path}')

    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if protocol.get('schema_version') != 2:
        raise VerificationError('Expected spatial-mixture protocol schema 2.')
    degradation_sha = protocol.get('degradation_source_sha256')
    if (expected_degradation_sha is not None
            and degradation_sha != expected_degradation_sha):
        raise VerificationError(
            f'Expected degradation hash {expected_degradation_sha}, '
            f'got {degradation_sha}.'
        )
    require_hash(
        protocol_path, summary.get('protocol_sha256'), 'protocol file'
    )
    require_hash(
        manifest_path, summary.get('manifest_sha256'), 'manifest file'
    )

    with manifest_path.open(newline='', encoding='utf-8') as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    if not rows:
        raise VerificationError('The benchmark manifest is empty.')
    if len(rows) != int(summary.get('records', -1)):
        raise VerificationError('Manifest record count differs from its summary.')
    if expected_records is not None and len(rows) != expected_records:
        raise VerificationError(
            f'Expected {expected_records} records, got {len(rows)}.'
        )

    datasets = protocol.get('datasets', {})
    seen_records = set()
    seen_sources = set()
    patterns = set()
    artifact_fields = {
        'gt': 'gt_sha256',
        'mild_lq': 'mild_sha256',
        'hard_lq': 'hard_sha256',
        'lq': 'lq_sha256',
        'mask': 'mask_sha256',
    }
    verified_artifacts = set()
    for row in rows:
        dataset = row['dataset']
        pattern = row['pattern']
        source_name = row['source_name']
        record_key = (dataset, source_name, pattern)
        if record_key in seen_records:
            raise VerificationError(f'Duplicate manifest record: {record_key}')
        seen_records.add(record_key)
        seen_sources.add((dataset, source_name))
        patterns.add(pattern)
        if dataset not in datasets:
            raise VerificationError(f'Unknown dataset in manifest: {dataset}')

        source_path = (Path(datasets[dataset]).expanduser() / source_name).resolve()
        require_hash(source_path, row['source_sha256'], 'source image')
        for path_field, hash_field in artifact_fields.items():
            artifact_path = resolved_child(root, row[path_field])
            artifact_key = (str(artifact_path), row[hash_field])
            if artifact_key not in verified_artifacts:
                require_hash(artifact_path, row[hash_field], path_field)
                verified_artifacts.add(artifact_key)
        height = int(row['gt_height'])
        width = int(row['gt_width'])
        if min(height, width) < 4 or height % 4 or width % 4:
            raise VerificationError(
                f'Invalid x4-aligned GT shape in {record_key}: {height}x{width}.'
            )

    summary_sources = int(summary.get('unique_source_images', -1))
    if len(seen_sources) != summary_sources:
        raise VerificationError(
            'Unique-source count differs from the manifest summary.'
        )
    if expected_sources is not None and len(seen_sources) != expected_sources:
        raise VerificationError(
            f'Expected {expected_sources} sources, got {len(seen_sources)}.'
        )
    summary_patterns = set(summary.get('patterns', []))
    if patterns != summary_patterns:
        raise VerificationError('Pattern set differs from the manifest summary.')
    if expected_patterns is not None and patterns != set(expected_patterns):
        raise VerificationError(
            f'Expected patterns {sorted(expected_patterns)}, '
            f'got {sorted(patterns)}.'
        )

    return {
        'status': 'passed',
        'root': str(root),
        'repository_commit': protocol.get('repository_commit'),
        'degradation_source_sha256': degradation_sha,
        'generator_source_sha256': protocol.get('generator_source_sha256'),
        'protocol_sha256': summary['protocol_sha256'],
        'manifest_sha256': summary['manifest_sha256'],
        'records': len(rows),
        'unique_source_images': len(seen_sources),
        'patterns': sorted(patterns),
        'unique_artifacts_verified': len(verified_artifacts),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--expected_degradation_sha')
    parser.add_argument('--expected_records', type=int)
    parser.add_argument('--expected_sources', type=int)
    parser.add_argument('--expected_patterns', nargs='+')
    return parser.parse_args()


def main():
    args = parse_args()
    result = verify_benchmark(
        args.root,
        expected_degradation_sha=args.expected_degradation_sha,
        expected_records=args.expected_records,
        expected_sources=args.expected_sources,
        expected_patterns=args.expected_patterns,
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
