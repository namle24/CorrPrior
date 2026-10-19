import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_standard_sr_metrics import (
    aligned_triplets,
    image_set_digest,
    validate_metric_row,
    verify_shapes,
)


class StandardSRMetricEvaluationTest(unittest.TestCase):
    @staticmethod
    def write(path, value=b'image'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)

    def test_aligns_three_folders_with_explicit_suffixes(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for stem in ('a', 'b'):
                self.write(root / 'gt' / f'{stem}.png')
                self.write(root / 'sr' / f'{stem}_ours.png')
                self.write(root / 'lq' / f'{stem}_x4.png')

            pairs = aligned_triplets(
                root / 'sr',
                root / 'gt',
                root / 'lq',
                sr_suffix='_ours',
                lq_suffix='_x4',
            )

            self.assertEqual([item[0] for item in pairs], ['a', 'b'])

    def test_rejects_incomplete_output_set(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            self.write(root / 'gt' / 'a.png')
            self.write(root / 'gt' / 'b.png')
            self.write(root / 'sr' / 'a.png')

            with self.assertRaisesRegex(RuntimeError, 'not identical'):
                aligned_triplets(root / 'sr', root / 'gt')

    def test_rejects_wrong_scale_even_when_sr_matches_gt(self):
        with self.assertRaisesRegex(RuntimeError, 'scale mismatch'):
            verify_shapes('sample', (64, 64, 3), (64, 64, 3), (20, 16, 3), 4)

    def test_image_set_digest_changes_with_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'a.png'
            self.write(path, b'first')
            first = image_set_digest([('a', path)])
            self.write(path, b'second')
            second = image_set_digest([('a', path)])
            self.assertNotEqual(first, second)

    def test_rejects_non_finite_metric_at_named_image(self):
        with self.assertRaisesRegex(
                RuntimeError, "Non-finite metric output for image 'sample'"):
            validate_metric_row(
                'sample', {'psnr': 25.0, 'ssim': 0.8, 'niqe': float('nan')}
            )


if __name__ == '__main__':
    unittest.main()
