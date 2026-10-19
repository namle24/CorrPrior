import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_journal_initialization import (
    InitializationError,
    parse_shape,
    selected_prior,
    validate_candidate_missing_keys,
)


class JournalInitializationVerificationTest(unittest.TestCase):
    def test_candidate_missing_keys_are_restricted_to_new_branches(self):
        validate_candidate_missing_keys(
            ['sft_layers.1.weight', 'mae_wrapper.need_head.0.weight'], []
        )
        with self.assertRaisesRegex(InitializationError, 'Unsafe'):
            validate_candidate_missing_keys(['conv_first.weight'], [])

    def test_shape_parser(self):
        self.assertEqual(parse_shape('64x96'), (64, 96))

    def test_selected_prior_checks_hash_method_and_iteration(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            checkpoint_path = root / 'prior.pth'
            import torch
            torch.save({
                'iter': 50000,
                'method': 'restoration_need_calibration',
                'model_state_dict': {},
                'pretrain_provenance': {'sha256': 'phase-a'},
                'selection_provenance': {'mode': 'partial'},
            }, checkpoint_path)
            checkpoint_hash = hashlib.sha256(
                checkpoint_path.read_bytes()
            ).hexdigest()
            selection_path = root / 'selection.json'
            selection_path.write_text(json.dumps({
                'mode': 'calibration',
                'selected_iteration': 50000,
                'selected_checkpoint': str(checkpoint_path),
                'selected_checkpoint_sha256': checkpoint_hash,
            }), encoding='utf-8')
            _, prior = selected_prior(selection_path)
            self.assertEqual(prior['sha256'], checkpoint_hash)
            self.assertEqual(prior['iteration'], 50000)

    def test_selected_partial_prior_uses_explicit_lineage(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            checkpoint_path = root / 'partial.pth'
            import torch
            torch.save({
                'iter': 25000,
                'method': 'partial_restoration_need_pretraining',
                'model_state_dict': {},
            }, checkpoint_path)
            checkpoint_hash = hashlib.sha256(
                checkpoint_path.read_bytes()
            ).hexdigest()
            selection_path = root / 'selection.json'
            selection_path.write_text(json.dumps({
                'mode': 'partial',
                'selected_iteration': 25000,
                'selected_checkpoint': str(checkpoint_path),
                'selected_checkpoint_sha256': checkpoint_hash,
            }), encoding='utf-8')

            _, prior = selected_prior(
                selection_path, expected_mode='partial'
            )

            self.assertEqual(prior['method'], 'partial_restoration_need_pretraining')
            self.assertEqual(prior['iteration'], 25000)


if __name__ == '__main__':
    unittest.main()
