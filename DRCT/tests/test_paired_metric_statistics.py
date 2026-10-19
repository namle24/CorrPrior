import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.paired_metric_statistics import (
    metric_preference,
    paired_statistics,
    read_cluster_manifest,
    read_metrics,
)


class PairedMetricStatisticsTest(unittest.TestCase):
    def test_metric_directions_include_secondary_iqa(self):
        self.assertEqual(metric_preference('dists'), 'lower')
        self.assertEqual(metric_preference('topiq_fr-pipal'), 'higher')
        self.assertEqual(metric_preference('clipiqa+'), 'higher')

    def test_ignores_provenance_path_columns(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'metrics.csv'
            with path.open('w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=['image', 'sr_path', 'gt_path', 'lpips'],
                )
                writer.writeheader()
                writer.writerow({
                    'image': 'a',
                    'sr_path': '/tmp/sr/a.png',
                    'gt_path': '/tmp/gt/a.png',
                    'lpips': '0.125',
                })

            metrics = read_metrics(path)

            self.assertEqual(metrics, {'a': {'lpips': 0.125}})

    def test_reads_exact_cluster_assignments(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'manifest.json'
            path.write_text(json.dumps({'pairs': [
                {'stem': 'a', 'scene': 'scene_1'},
                {'stem': 'b', 'scene': 'scene_1'},
                {'stem': 'c', 'scene': 'scene_2'},
            ]}))
            clusters = read_cluster_manifest(path, {'a', 'b', 'c'})
            self.assertEqual(clusters, {
                'a': 'scene_1', 'b': 'scene_1', 'c': 'scene_2'
            })

    def test_rejects_incomplete_cluster_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'manifest.json'
            path.write_text(json.dumps({'pairs': [
                {'stem': 'a', 'scene': 'scene_1'},
            ]}))
            with self.assertRaisesRegex(ValueError, 'exactly the evaluated images'):
                read_cluster_manifest(path, {'a', 'b'})

    def test_clustered_statistics_use_scene_as_resampling_unit(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest('numpy is not installed in the lightweight test environment')
        baseline = {
            'a': {'psnr': 1.0}, 'b': {'psnr': 2.0}, 'c': {'psnr': 3.0}
        }
        candidate = {
            'a': {'psnr': 2.0}, 'b': {'psnr': 3.0}, 'c': {'psnr': 4.0}
        }
        row = paired_statistics(
            baseline,
            candidate,
            'psnr',
            np.random.default_rng(0),
            bootstrap_samples=100,
            permutation_samples=100,
            confidence=0.95,
            clusters={'a': 'scene_1', 'b': 'scene_1', 'c': 'scene_2'},
        )
        self.assertEqual(row['images'], 3)
        self.assertEqual(row['clusters'], 2)
        self.assertEqual(row['resampling_unit'], 'cluster')
        self.assertEqual(row['better'], 'higher')
        self.assertAlmostEqual(row['candidate_minus_baseline'], 1.0)


if __name__ == '__main__':
    unittest.main()
