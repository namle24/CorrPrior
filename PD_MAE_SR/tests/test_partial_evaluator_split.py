import argparse
from pathlib import Path
import tempfile
import unittest

from PD_MAE_SR.experiments.evaluate_partial_restoration_need import (
    dataset_for_source_groups,
    joint_validation_source_groups,
    source_group_digest,
)


class PartialEvaluatorSplitTest(unittest.TestCase):
    def make_roots(self, root, source_count=10):
        roots = []
        for level in (60, 70, 80, 90):
            level_root = Path(root) / f'PD_{level}'
            lq_root = level_root / 'PD_LQ'
            hr_root = level_root / 'HR'
            lq_root.mkdir(parents=True)
            hr_root.mkdir(parents=True)
            for source in range(source_count):
                filename = f'{source:06d}_p000_l{level}_{source:010d}.png'
                (lq_root / filename).touch()
                (hr_root / filename).touch()
            roots.append(str(level_root))
        return roots

    @staticmethod
    def arguments(roots):
        return argparse.Namespace(
            data_roots=roots,
            patch_size=256,
            val_fraction=0.25,
            split_seed=0,
            max_images=1000,
            sample_seed=0,
        )

    def test_joint_split_is_identical_for_every_root(self):
        with tempfile.TemporaryDirectory() as root:
            roots = self.make_roots(root)
            args = self.arguments(roots)
            groups = joint_validation_source_groups(args)
            self.assertEqual(len(groups), 3)
            for data_root in roots:
                dataset = dataset_for_source_groups(args, data_root, groups)
                observed = {
                    dataset.dataset.samples[index]['source_group']
                    for index in dataset.indices
                }
                self.assertEqual(observed, set(groups))

    def test_digest_is_order_independent(self):
        self.assertEqual(
            source_group_digest(['a', 'b']), source_group_digest(['b', 'a'])
        )

    def test_missing_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            roots = self.make_roots(root, source_count=2)
            args = self.arguments(roots)
            with self.assertRaisesRegex(RuntimeError, 'exactly one image'):
                dataset_for_source_groups(args, roots[0], ['not_present'])


if __name__ == '__main__':
    unittest.main()
