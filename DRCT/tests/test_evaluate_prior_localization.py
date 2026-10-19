import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

try:
    import cv2  # noqa: F401
    import numpy as np
    import torch  # noqa: F401
except ImportError as error:
    raise unittest.SkipTest(
        'Localization metric tests require NumPy, OpenCV, and Torch.'
    ) from error

from scripts.evaluate_prior_localization import (
    average_ranks,
    correction_alignment_metrics,
    load_selection,
    localization_metrics,
    paired_correction_target,
    roc_auc,
)


class PriorLocalizationMetricsTest(unittest.TestCase):
    def test_average_ranks_handles_ties(self):
        np.testing.assert_allclose(
            average_ranks([3.0, 1.0, 1.0, 2.0]),
            [4.0, 1.5, 1.5, 3.0],
        )

    def test_auc_is_one_for_perfect_spatial_order(self):
        self.assertEqual(
            roc_auc([0.1, 0.2, 0.8, 0.9], [0.0, 0.2, 0.8, 1.0]),
            1.0,
        )

    def test_localization_metrics_reward_aligned_field(self):
        mask = np.tile(np.linspace(0.0, 1.0, 8), (8, 1))
        result = localization_metrics(mask.copy(), mask)
        self.assertAlmostEqual(result['pearson_aligned'], 1.0)
        self.assertAlmostEqual(result['spearman_aligned'], 1.0)
        self.assertEqual(result['auc_aligned'], 1.0)
        self.assertGreater(result['hard_minus_mild_weighted_need'], 0.0)
        self.assertGreater(result['aligned_minus_shift_pearson'], 0.0)

    def test_constant_prediction_is_visible_as_undefined_correlation(self):
        mask = np.tile(np.linspace(0.0, 1.0, 8), (8, 1))
        result = localization_metrics(np.ones((8, 8)), mask)
        self.assertTrue(math.isnan(result['pearson_aligned']))
        self.assertAlmostEqual(result['auc_aligned'], 0.5)

    def test_paired_correction_target_matches_training_definition(self):
        lq = torch.zeros(1, 3, 2, 3)
        identity_gt = torch.zeros(1, 3, 8, 12)
        identity = paired_correction_target(
            lq, identity_gt, prediction_shape=(1, 2)
        )
        self.assertEqual(tuple(identity.shape), (1, 1, 2))
        torch.testing.assert_close(identity, torch.zeros_like(identity))

        changed_gt = torch.full((1, 3, 8, 12), 0.1)
        changed = paired_correction_target(
            lq, changed_gt, prediction_shape=(1, 2)
        )
        expected = 1.0 - math.exp(-1.0)
        torch.testing.assert_close(
            changed, torch.full_like(changed, expected), rtol=1e-6, atol=1e-6
        )

    def test_paired_correction_target_rejects_wrong_scale_alignment(self):
        with self.assertRaisesRegex(ValueError, 'equal shapes'):
            paired_correction_target(
                torch.zeros(1, 3, 2, 2),
                torch.zeros(1, 3, 7, 8),
                prediction_shape=(1, 1),
            )

    def test_correction_alignment_rewards_the_true_spatial_target(self):
        target = np.tile(np.linspace(0.05, 0.95, 8), (8, 1))
        result = correction_alignment_metrics(target.copy(), target)
        self.assertAlmostEqual(result['correction_mae'], 0.0)
        self.assertAlmostEqual(result['correction_rmse'], 0.0)
        self.assertAlmostEqual(result['correction_pearson'], 1.0)
        self.assertAlmostEqual(result['correction_spearman'], 1.0)
        self.assertGreater(
            result['correction_aligned_minus_shift_pearson'], 0.0
        )

    def test_selection_is_bound_to_calibrated_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            checkpoint = root / 'prior.pth'
            torch.save({
                'method': 'restoration_need_calibration',
                'iter': 25000,
                'model_state_dict': {},
            }, checkpoint)
            checkpoint_hash = hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest()
            selection = root / 'selection.json'
            selection.write_text(json.dumps({
                'mode': 'calibration',
                'selected_iteration': 25000,
                'selected_checkpoint': str(checkpoint),
                'selected_checkpoint_sha256': checkpoint_hash,
            }), encoding='utf-8')
            _, payload, selected = load_selection(selection)
            self.assertEqual(payload['selected_iteration'], 25000)
            self.assertEqual(selected, checkpoint)


if __name__ == '__main__':
    unittest.main()
