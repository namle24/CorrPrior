import hashlib
import json
import os
import tempfile
import unittest

import torch

from PD_MAE_SR.experiments.train_restoration_need import (
    load_pretrain,
    load_selection_provenance,
)


class RestorationNeedLineageTest(unittest.TestCase):
    def test_pretrain_method_and_hash_are_recorded(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, 'partial.pth')
            source = torch.nn.Linear(2, 2)
            torch.save({
                'iter': 25000,
                'method': 'partial_restoration_need_pretraining',
                'config': {'seed': 0},
                'model_state_dict': source.state_dict(),
            }, path)
            target = torch.nn.Linear(2, 2)
            provenance = load_pretrain(
                target, path,
                expected_method='partial_restoration_need_pretraining',
            )
            with open(path, 'rb') as checkpoint_file:
                expected_hash = hashlib.sha256(checkpoint_file.read()).hexdigest()
            self.assertEqual(provenance['sha256'], expected_hash)
            self.assertEqual(provenance['iteration'], 25000)

    def test_selection_must_match_path_iteration_mode_and_hash(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint_path = os.path.join(root, 'partial.pth')
            with open(checkpoint_path, 'wb') as checkpoint_file:
                checkpoint_file.write(b'checkpoint')
            checkpoint_hash = hashlib.sha256(b'checkpoint').hexdigest()
            selection_path = os.path.join(root, 'selection.json')
            with open(selection_path, 'w', encoding='utf-8') as selection_file:
                json.dump({
                    'mode': 'partial',
                    'selected_iteration': 25000,
                    'selected_checkpoint': checkpoint_path,
                    'selected_checkpoint_sha256': checkpoint_hash,
                    'selector_git_commit': 'abc123',
                }, selection_file)
            result = load_selection_provenance(
                selection_path,
                {
                    'path': checkpoint_path,
                    'sha256': checkpoint_hash,
                    'iteration': 25000,
                },
                expected_mode='partial',
            )
            self.assertEqual(result['selected_checkpoint_sha256'], checkpoint_hash)

    def test_selection_rejects_a_different_checkpoint_hash(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint_path = os.path.join(root, 'partial.pth')
            selection_path = os.path.join(root, 'selection.json')
            with open(selection_path, 'w', encoding='utf-8') as selection_file:
                json.dump({
                    'mode': 'partial',
                    'selected_iteration': 25000,
                    'selected_checkpoint': checkpoint_path,
                    'selected_checkpoint_sha256': 'wrong',
                }, selection_file)
            with self.assertRaisesRegex(RuntimeError, 'hash does not match'):
                load_selection_provenance(
                    selection_path,
                    {
                        'path': checkpoint_path,
                        'sha256': 'correct',
                        'iteration': 25000,
                    },
                    expected_mode='partial',
                )


if __name__ == '__main__':
    unittest.main()
