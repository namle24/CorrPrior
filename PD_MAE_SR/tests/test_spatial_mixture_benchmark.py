import unittest

import numpy as np

from PD_MAE_SR.utils.generate_spatial_mixture_benchmark import (
    spatial_mask,
    stable_seed,
)


class SpatialMixtureBenchmarkTest(unittest.TestCase):
    def test_stable_seed_depends_on_dataset_and_filename(self):
        first = stable_seed(276, 'DIV2K100', '0801.png')
        self.assertEqual(first, stable_seed(276, 'DIV2K100', '0801.png'))
        self.assertNotEqual(first, stable_seed(276, 'DIV2K100', '0802.png'))
        self.assertNotEqual(first, stable_seed(276, 'Other', '0801.png'))

    def test_linear_mask_spans_the_full_mild_to_hard_range(self):
        mask = spatial_mask('linear', 5, 7, seed=0)
        self.assertEqual(mask.shape, (5, 7))
        self.assertTrue(np.allclose(mask[:, 0], 0.0))
        self.assertTrue(np.allclose(mask[:, -1], 1.0))

    def test_random_field_is_deterministic_and_bounded(self):
        first = spatial_mask('random_field', 32, 48, seed=17)
        second = spatial_mask('random_field', 32, 48, seed=17)
        self.assertTrue(np.array_equal(first, second))
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)
        self.assertGreater(float(first.std()), 0.0)


if __name__ == '__main__':
    unittest.main()
