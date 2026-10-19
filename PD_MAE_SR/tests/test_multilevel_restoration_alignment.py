import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from PD_MAE_SR.datasets.multilevel_restoration_dataset import (
    MultiLevelRestorationDataset,
    align_lq_to_hr,
)


class MultiLevelRestorationAlignmentTest(unittest.TestCase):
    def test_pytorch_alignment_matches_inference_operator(self):
        lq = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
        observed = align_lq_to_hr(
            lq, height=12, width=16, backend='pytorch_bicubic'
        )
        tensor = torch.from_numpy(lq).permute(2, 0, 1).unsqueeze(0)
        expected = F.interpolate(
            tensor.float().div(255.0),
            size=(12, 16),
            mode='bicubic',
            align_corners=False,
        ).clamp(0.0, 1.0)
        expected = expected.squeeze(0).permute(1, 2, 0).numpy()
        self.assertTrue(np.array_equal(observed, expected))

    def test_pytorch_alignment_is_bounded(self):
        lq = np.zeros((2, 2, 3), dtype=np.uint8)
        lq[0, 0] = 255
        observed = align_lq_to_hr(
            lq, height=8, width=8, backend='pytorch_bicubic'
        )
        self.assertGreaterEqual(float(observed.min()), 0.0)
        self.assertLessEqual(float(observed.max()), 1.0)

    def test_unknown_backend_is_rejected(self):
        lq = np.zeros((2, 2, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            align_lq_to_hr(lq, 4, 4, backend='unknown')

    def test_same_size_integer_input_is_scaled_even_when_very_dark(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            hr_root = root / 'hr'
            lq_root = root / 'lq'
            hr_root.mkdir()
            lq_root.mkdir()
            image = np.ones((4, 4, 3), dtype=np.uint8)
            cv2.imwrite(str(hr_root / 'a.png'), image)
            cv2.imwrite(str(lq_root / 'a.png'), image)
            dataset = MultiLevelRestorationDataset(
                hr_root,
                [lq_root],
                patch_size=4,
                augment=False,
                deterministic_level=True,
                alignment_backend='pytorch_bicubic',
            )
            sample = dataset[0]
            self.assertAlmostEqual(
                sample['lq'].max().item(), 1.0 / 255.0, places=7
            )


if __name__ == '__main__':
    unittest.main()
