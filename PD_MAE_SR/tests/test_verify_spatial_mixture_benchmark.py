import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PD_MAE_SR.utils.verify_spatial_mixture_benchmark import (
    VerificationError,
    sha256_file,
    verify_benchmark,
)


class SpatialMixtureVerificationTest(unittest.TestCase):
    def make_fixture(self, root):
        root = Path(root)
        source_root = root / 'sources'
        benchmark = root / 'benchmark'
        source_root.mkdir()
        benchmark.mkdir()
        source = source_root / 'a.png'
        source.write_bytes(b'source')
        artifact_paths = {}
        for name in ('gt', 'mild', 'hard', 'mix', 'mask'):
            path = benchmark / f'{name}.png'
            path.write_bytes(name.encode('utf-8'))
            artifact_paths[name] = path

        protocol_path = benchmark / 'protocol.json'
        protocol_path.write_text(json.dumps({
            'schema_version': 2,
            'repository_commit': 'abc123',
            'degradation_source_sha256': 'degradation-hash',
            'generator_source_sha256': 'generator-hash',
            'datasets': {'tiny': str(source_root)},
        }) + '\n', encoding='utf-8')
        row = {
            'dataset': 'tiny',
            'source_name': 'a.png',
            'filename': 'a.png',
            'pattern': 'linear',
            'seed': '1',
            'gt': 'gt.png',
            'mild_lq': 'mild.png',
            'hard_lq': 'hard.png',
            'lq': 'mix.png',
            'mask': 'mask.png',
            'source_sha256': sha256_file(source),
            'gt_sha256': sha256_file(artifact_paths['gt']),
            'mild_sha256': sha256_file(artifact_paths['mild']),
            'hard_sha256': sha256_file(artifact_paths['hard']),
            'lq_sha256': sha256_file(artifact_paths['mix']),
            'mask_sha256': sha256_file(artifact_paths['mask']),
            'gt_height': '8',
            'gt_width': '12',
        }
        manifest_path = benchmark / 'manifest.csv'
        with manifest_path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        (benchmark / 'manifest_summary.json').write_text(json.dumps({
            'schema_version': 1,
            'manifest': 'manifest.csv',
            'manifest_sha256': sha256_file(manifest_path),
            'records': 1,
            'unique_source_images': 1,
            'patterns': ['linear'],
            'protocol_sha256': sha256_file(protocol_path),
        }) + '\n', encoding='utf-8')
        return benchmark, artifact_paths

    def test_complete_fixture_passes(self):
        with tempfile.TemporaryDirectory() as root:
            benchmark, _ = self.make_fixture(root)
            result = verify_benchmark(
                benchmark,
                expected_degradation_sha='degradation-hash',
                expected_records=1,
                expected_sources=1,
                expected_patterns=['linear'],
            )
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['unique_artifacts_verified'], 5)

    def test_tampered_artifact_fails(self):
        with tempfile.TemporaryDirectory() as root:
            benchmark, artifacts = self.make_fixture(root)
            artifacts['mix'].write_bytes(b'tampered')
            with self.assertRaisesRegex(VerificationError, 'hash mismatch'):
                verify_benchmark(benchmark)


if __name__ == '__main__':
    unittest.main()
