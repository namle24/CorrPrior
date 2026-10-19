import csv
import json
import os
import tempfile
import unittest

from PD_MAE_SR.experiments.select_restoration_need_checkpoint import (
    select_calibration,
    select_partial,
    summarize_calibration,
    summarize_partial,
    validate_protocols,
)


def write_metrics(directory, filename, rows):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_protocol(directory, iteration, checkpoint, split_unit='source',
                   data_roots=None, pretrain_sha='shared-pretrain'):
    protocol = {
        'arguments': {
            'data_roots': data_roots or ['/data/pd60', '/data/pd90'],
            'checkpoint': checkpoint,
            'output_dir': directory,
            'patch_size': 256,
            'mae_patch_size': 8,
            'need_scale': 10.0,
            'val_fraction': 0.05,
            'split_seed': 0,
            'max_images': 1000,
            'sample_seed': 0,
        },
        'checkpoint_iter': iteration,
        'checkpoint_method': 'partial_restoration_need_pretraining',
        'checkpoint_config': {'seed': 0},
        'checkpoint_pretrain_provenance': {
            'path': '/data/pretrain.pth',
            'sha256': pretrain_sha,
        },
        'checkpoint_selection_provenance': {
            'path': '/data/selection.json',
            'sha256': 'shared-selection',
        },
        'split_unit': split_unit,
        'evaluated_source_groups': 10,
        'evaluated_source_group_sha256': 'shared-source-groups',
        'joint_split_across_roots': True,
        'implementation_sha256': {
            'evaluator': 'shared-evaluator',
            'dataset': 'shared-dataset',
            'model': 'shared-model',
            'need_target': 'shared-target',
            'metric_helpers': 'shared-metrics',
        },
    }
    with open(os.path.join(directory, 'evaluation_protocol.json'), 'w',
              encoding='utf-8') as file:
        json.dump(protocol, file)


class CheckpointSelectionTest(unittest.TestCase):
    def test_partial_prefers_reconstruction_mse_before_need_mae(self):
        with tempfile.TemporaryDirectory() as root:
            candidates = []
            for iteration, reconstruction_mse, need_mae in (
                    (25000, 0.02, 0.01), (50000, 0.01, 0.02)):
                directory = os.path.join(root, str(iteration))
                write_metrics(
                    directory, 'partial_restoration_need_metrics.csv',
                    [
                        {
                            'level': 'PD_60',
                            'reconstruction_mse_mean': reconstruction_mse,
                            'mae': need_mae,
                        },
                        {
                            'level': 'PD_90',
                            'reconstruction_mse_mean': reconstruction_mse,
                            'mae': need_mae,
                        },
                    ],
                )
                candidates.append(summarize_partial(iteration, directory))
            self.assertEqual(select_partial(candidates)['iteration'], 50000)

    def test_calibration_applies_identity_gate_and_earlier_tie_rule(self):
        with tempfile.TemporaryDirectory() as root:
            candidates = []
            for iteration, identity_mean, need_mae in (
                    (25000, 0.01, 0.1005),
                    (50000, 0.01, 0.1000),
                    (75000, 0.03, 0.0500)):
                directory = os.path.join(root, str(iteration))
                write_metrics(
                    directory, 'restoration_need_metrics.csv',
                    [
                        {
                            'level': 'HR_sub',
                            'prediction_mean': identity_mean,
                            'mae': identity_mean,
                        },
                        {
                            'level': 'q60',
                            'prediction_mean': 0.4,
                            'mae': need_mae,
                        },
                        {
                            'level': 'q90',
                            'prediction_mean': 0.3,
                            'mae': need_mae,
                        },
                    ],
                )
                candidates.append(summarize_calibration(
                    iteration, directory, 'HR_sub', 0.02
                ))
            selected = select_calibration(candidates, tolerance=0.001)
            self.assertEqual(selected['iteration'], 25000)

    def test_protocol_validation_rejects_stale_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            iteration = 25000
            directory = os.path.join(root, str(iteration))
            checkpoint_pattern = os.path.join(root, 'model_{iteration}.pth')
            write_metrics(
                directory, 'partial_restoration_need_metrics.csv',
                [{
                    'level': 'PD_60',
                    'reconstruction_mse_mean': 0.01,
                    'mae': 0.02,
                }],
            )
            write_protocol(
                directory,
                iteration=50000,
                checkpoint=checkpoint_pattern.format(iteration=iteration),
            )
            candidate = summarize_partial(iteration, directory)
            with self.assertRaisesRegex(RuntimeError, 'iteration mismatch'):
                validate_protocols(
                    [candidate], 'partial', require_protocol=True,
                    expected_split_unit='source',
                    checkpoint_pattern=checkpoint_pattern,
                )

    def test_protocol_validation_rejects_incomparable_data_roots(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint_pattern = os.path.join(root, 'model_{iteration}.pth')
            candidates = []
            for iteration, data_roots in (
                    (25000, ['/data/pd60']),
                    (50000, ['/data/pd90'])):
                directory = os.path.join(root, str(iteration))
                write_metrics(
                    directory, 'partial_restoration_need_metrics.csv',
                    [{
                        'level': 'PD_60',
                        'reconstruction_mse_mean': 0.01,
                        'mae': 0.02,
                    }],
                )
                write_protocol(
                    directory, iteration,
                    checkpoint_pattern.format(iteration=iteration),
                    data_roots=data_roots,
                )
                candidates.append(summarize_partial(iteration, directory))
            with self.assertRaisesRegex(RuntimeError, 'not comparable'):
                validate_protocols(
                    candidates, 'partial', require_protocol=True,
                    expected_split_unit='source',
                    checkpoint_pattern=checkpoint_pattern,
                )

    def test_protocol_validation_rejects_different_pretrain_lineage(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint_pattern = os.path.join(root, 'model_{iteration}.pth')
            candidates = []
            for iteration, pretrain_sha in (
                    (25000, 'pretrain-a'), (50000, 'pretrain-b')):
                directory = os.path.join(root, str(iteration))
                write_metrics(
                    directory, 'partial_restoration_need_metrics.csv',
                    [{
                        'level': 'PD_60',
                        'reconstruction_mse_mean': 0.01,
                        'mae': 0.02,
                    }],
                )
                write_protocol(
                    directory, iteration,
                    checkpoint_pattern.format(iteration=iteration),
                    pretrain_sha=pretrain_sha,
                )
                candidates.append(summarize_partial(iteration, directory))
            with self.assertRaisesRegex(RuntimeError, 'not comparable'):
                validate_protocols(
                    candidates, 'partial', require_protocol=True,
                    expected_split_unit='source',
                    checkpoint_pattern=checkpoint_pattern,
                )


if __name__ == '__main__':
    unittest.main()
