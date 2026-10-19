import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.audit_journal_training_protocol import (
    ProtocolError,
    apply_manual_seed_override,
    audit_dataset,
    deterministic_worker_seeds,
    sampler_prefix,
    validate_matched_configs,
)


class _FakeEnlargedSampler:
    def __init__(self, dataset, world_size, rank, ratio=1):
        self.size = len(dataset)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        return iter(range(self.size))


def matched_configs():
    shared_network = {
        'type': 'DRCT',
        'upscale': 4,
        'depths': [6, 6, 6, 6, 6, 6],
    }
    shared = {
        'model_type': 'DRCTModel',
        'scale': 4,
        'num_gpu': 1,
        'manual_seed': 0,
        'output_root': '/var/tmp/runs',
        'datasets': {
            'train': {
                'type': 'MultiLevelPairedImageDataset',
                'dataroot_gt': '/gt',
                'dataroot_lq': ['/q90', '/q60'],
                'batch_size_per_gpu': 4,
                'dataset_enlarge_ratio': 1,
            }
        },
        'train': {'total_iter': 100000, 'optim_g': {'type': 'Adam'}},
        'logger': {'print_freq': 100},
    }
    candidate = {
        **shared,
        'name': 'candidate',
        'mae_weight': 0.0,
        'network_g': {
            **shared_network,
            'use_sft': True,
            'use_restoration_need_gate': True,
            'restoration_need_gate_mode': 'spatial',
            'zero_init_sft': True,
            'allow_random_prior': False,
            'mae_checkpoint': None,
            'mae_input_upscale': 4,
            'mae_tile_batch_size': 8,
            'mae_tile_overlap': 64,
        },
        'path': {'pretrain_network_g': '/model.pth', 'strict_load_g': False},
    }
    baseline = {
        **shared,
        'name': 'baseline',
        'network_g': {
            **shared_network,
            'use_sft': False,
            'mae_checkpoint': None,
        },
        'path': {'pretrain_network_g': '/model.pth', 'strict_load_g': True},
    }
    return candidate, baseline


class JournalProtocolAuditTest(unittest.TestCase):
    def test_accepts_only_the_declared_treatment_difference(self):
        candidate, baseline = matched_configs()
        result = validate_matched_configs(candidate, baseline)
        self.assertEqual(result['candidate_name'], 'candidate')

    def test_second_backbone_configs_are_matched(self):
        repository = Path(__file__).resolve().parents[1]
        candidate_path = (
            repository / 'options/train/journal'
            / 'train_RN_PD_MAE_SwinIR_x4_finetune_100k.yml'
        )
        baseline_path = (
            repository / 'options/train/journal'
            / 'train_SwinIR_x4_baseline_continue_100k.yml'
        )
        with candidate_path.open(encoding='utf-8') as input_file:
            candidate = yaml.safe_load(input_file)
        with baseline_path.open(encoding='utf-8') as input_file:
            baseline = yaml.safe_load(input_file)
        result = validate_matched_configs(candidate, baseline)
        self.assertEqual(
            result['candidate_name'],
            'journal_RN_PD_MAE_SwinIR_x4_finetune_100k',
        )

    def test_primary_backbone_configs_are_matched_and_parser_compatible(self):
        repository = Path(__file__).resolve().parents[1]
        option_root = repository / 'options/train/journal'
        paths = [
            option_root / 'train_RN_PD_MAE_DRCT_x4_finetune_100k.yml',
            option_root / 'train_DRCT_x4_baseline_continue_100k.yml',
        ]
        configs = []
        for path in paths:
            with path.open(encoding='utf-8') as input_file:
                config = yaml.safe_load(input_file)
            train = config['datasets']['train']
            self.assertIsNone(train['dataroot_lq'])
            self.assertEqual(len(train['dataroot_lq_levels']), 4)
            configs.append(config)
        result = validate_matched_configs(*configs)
        self.assertEqual(
            result['candidate_name'],
            'journal_RN_PD_MAE_DRCT_x4_finetune_100k',
        )

    def test_ungated_primary_candidate_requires_explicit_c2_variant(self):
        repository = Path(__file__).resolve().parents[1]
        option_root = repository / 'options/train/journal'
        with (
            option_root / 'train_RN_PD_MAE_DRCT_x4_ungated_100k.yml'
        ).open(encoding='utf-8') as input_file:
            candidate = yaml.safe_load(input_file)
        with (
            option_root / 'train_DRCT_x4_baseline_continue_100k.yml'
        ).open(encoding='utf-8') as input_file:
            baseline = yaml.safe_load(input_file)
        with self.assertRaisesRegex(ProtocolError, 'Unexpected P0'):
            validate_matched_configs(candidate, baseline)
        result = validate_matched_configs(
            candidate, baseline, candidate_variant='c2'
        )
        self.assertEqual(result['candidate_variant'], 'c2')

    def test_ungated_second_backbone_configs_are_matched(self):
        repository = Path(__file__).resolve().parents[1]
        option_root = repository / 'options/train/journal'
        with (
            option_root / 'train_RN_PD_MAE_SwinIR_x4_ungated_100k.yml'
        ).open(encoding='utf-8') as input_file:
            candidate = yaml.safe_load(input_file)
        with (
            option_root / 'train_SwinIR_x4_baseline_continue_100k.yml'
        ).open(encoding='utf-8') as input_file:
            baseline = yaml.safe_load(input_file)
        result = validate_matched_configs(
            candidate, baseline, candidate_variant='c2'
        )
        self.assertEqual(result['candidate_variant'], 'c2')

    def test_second_backbone_configs_use_parser_compatible_level_key(self):
        repository = Path(__file__).resolve().parents[1]
        option_root = repository / 'options/train/journal'
        for name in (
            'train_RN_PD_MAE_SwinIR_x4_finetune_100k.yml',
            'train_SwinIR_x4_baseline_continue_100k.yml',
        ):
            with (option_root / name).open(encoding='utf-8') as input_file:
                train = yaml.safe_load(input_file)['datasets']['train']
            self.assertIsNone(train['dataroot_lq'])
            self.assertEqual(len(train['dataroot_lq_levels']), 4)

    def test_rejects_a_training_budget_difference(self):
        candidate, baseline = matched_configs()
        baseline['train'] = {'total_iter': 50000, 'optim_g': {'type': 'Adam'}}
        with self.assertRaisesRegex(ProtocolError, 'outside the allowed'):
            validate_matched_configs(candidate, baseline)

    def test_rejects_a_nonzero_primary_consistency_loss(self):
        candidate, baseline = matched_configs()
        candidate['mae_weight'] = 0.05
        with self.assertRaisesRegex(ProtocolError, 'mae_weight'):
            validate_matched_configs(candidate, baseline)

    def test_dataset_audit_checks_alignment_and_source_groups(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            gt = root / 'GT'
            q90 = root / 'q90'
            q60 = root / 'q60'
            for folder in (gt, q90, q60):
                folder.mkdir()
                for name in ('a_s001.png', 'a_s002.png', 'b_s001.png'):
                    (folder / name).touch()
            result = audit_dataset(
                {
                    'dataroot_gt': str(gt),
                    'dataroot_lq': [str(q90), str(q60)],
                },
                expected_images=3,
                expected_sources=2,
                expected_level_order=['q90', 'q60'],
            )
            self.assertEqual(result['logical_samples'], 6)
            self.assertEqual(result['source_groups'], 2)

    def test_dataset_audit_rejects_an_unaligned_level(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            gt = root / 'GT'
            q90 = root / 'q90'
            gt.mkdir()
            q90.mkdir()
            (gt / 'a.png').touch()
            (q90 / 'b.png').touch()
            with self.assertRaisesRegex(ProtocolError, 'Unaligned'):
                audit_dataset({
                    'dataroot_gt': str(gt),
                    'dataroot_lq': [str(q90)],
                })

    def test_dataset_audit_accepts_parser_compatible_level_key(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            gt = root / 'GT'
            q90 = root / 'q90'
            for folder in (gt, q90):
                folder.mkdir()
                (folder / 'a.png').touch()
            result = audit_dataset({
                'dataroot_gt': str(gt),
                'dataroot_lq': None,
                'dataroot_lq_levels': [str(q90)],
            })
            self.assertEqual(result['level_order'], ['q90'])

    def test_data_randomness_records_fixed_worker_seeds(self):
        self.assertEqual(
            deterministic_worker_seeds(num_workers=3, seed=7, rank=0),
            [7, 8, 9],
        )

    def test_matched_nonzero_seed_remains_a_valid_comparison(self):
        candidate, baseline = matched_configs()
        candidate['manual_seed'] = 2
        baseline['manual_seed'] = 2
        result = validate_matched_configs(candidate, baseline)
        self.assertEqual(result['candidate_name'], 'candidate')

    def test_rejects_a_seed_difference(self):
        candidate, baseline = matched_configs()
        candidate['manual_seed'] = 1
        baseline['manual_seed'] = 2
        with self.assertRaisesRegex(ProtocolError, 'outside the allowed'):
            validate_matched_configs(candidate, baseline)

    def test_runtime_seed_override_is_matched_and_rejects_negative_seed(self):
        candidate, baseline = matched_configs()
        apply_manual_seed_override(candidate, baseline, 2)
        self.assertEqual(candidate['manual_seed'], 2)
        self.assertEqual(baseline['manual_seed'], 2)
        with self.assertRaisesRegex(ProtocolError, 'non-negative'):
            apply_manual_seed_override(candidate, baseline, -1)

    def test_sampler_prefix_returns_complete_diagnostics(self):
        torch_stub = types.ModuleType('torch')
        torch_stub.__version__ = 'test'
        basicsr_stub = types.ModuleType('basicsr')
        data_stub = types.ModuleType('basicsr.data')
        sampler_stub = types.ModuleType('basicsr.data.data_sampler')
        sampler_stub.EnlargedSampler = _FakeEnlargedSampler
        with patch.dict(sys.modules, {
            'torch': torch_stub,
            'basicsr': basicsr_stub,
            'basicsr.data': data_stub,
            'basicsr.data.data_sampler': sampler_stub,
        }):
            result = sampler_prefix(
                dataset_size=4, level_count=2, total_iter=1, batch_size=2
            )
        self.assertEqual(result['consumed_samples'], 2)
        self.assertEqual(result['level_counts'], [1, 1])
        self.assertEqual(len(result['sampled_index_prefix_sha256_int64_le']), 64)


if __name__ == '__main__':
    unittest.main()
