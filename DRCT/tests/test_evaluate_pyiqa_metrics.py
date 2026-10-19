import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_pyiqa_metrics import (
    METRIC_PROTOCOL,
    aligned_pairs,
    evaluate_metrics,
    image_set_digest,
    summarize,
)


class PyIQAEvaluationTest(unittest.TestCase):
    @staticmethod
    def touch(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def test_aligns_an_explicit_sr_suffix(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            gt_root = root / 'gt'
            sr_root = root / 'sr'
            self.touch(gt_root / 'a.png')
            self.touch(gt_root / 'b.png')
            self.touch(sr_root / 'a_method.png')
            self.touch(sr_root / 'b_method.png')

            pairs = aligned_pairs(sr_root, gt_root, sr_suffix='_method')

            self.assertEqual([item[0] for item in pairs], ['a', 'b'])

    def test_rejects_an_incomplete_image_set(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            gt_root = root / 'gt'
            sr_root = root / 'sr'
            self.touch(gt_root / 'a.png')
            self.touch(gt_root / 'b.png')
            self.touch(sr_root / 'a.png')

            with self.assertRaisesRegex(RuntimeError, 'not identical'):
                aligned_pairs(sr_root, gt_root)

    def test_summary_uses_sample_standard_deviation(self):
        result = summarize([1.0, 2.0, 3.0])
        self.assertEqual(result['images'], 3)
        self.assertEqual(result['mean'], 2.0)
        self.assertEqual(result['sample_std'], 1.0)

    def test_image_set_digest_is_independent_of_parent_root(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            first = root / 'one' / 'sample.png'
            second = root / 'two' / 'sample.png'
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b'same')
            second.write_bytes(b'same')
            self.assertEqual(
                image_set_digest([('sample', first)]),
                image_set_digest([('sample', second)]),
            )

    def test_secondary_metric_protocol_has_the_correct_direction(self):
        self.assertEqual(METRIC_PROTOCOL['topiq_fr-pipal'], {
            'reference': True,
            'better': 'higher',
        })
        self.assertEqual(METRIC_PROTOCOL['clipiqa+'], {
            'reference': False,
            'better': 'higher',
        })

    def test_metrics_are_created_and_released_sequentially(self):
        class FakeScore:
            def detach(self):
                return self

            def cpu(self):
                return self

            def reshape(self, *_shape):
                return self

            def mean(self):
                return 0.25

        events = []

        class FakeMetric:
            def __init__(self, name):
                self.name = name

            def __call__(self, *_args):
                events.append(('score', self.name))
                return FakeScore()

        class FakePyIQA:
            @staticmethod
            def create_metric(name, device):
                events.append(('create', name, device))
                return FakeMetric(name)

        class FakeNoGrad:
            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return False

        class FakeCuda:
            @staticmethod
            def empty_cache():
                events.append(('empty_cache',))

        class FakeTorch:
            cuda = FakeCuda()

            @staticmethod
            def no_grad():
                return FakeNoGrad()

        class FakeTensor:
            shape = (1, 3, 8, 8)

        import scripts.evaluate_pyiqa_metrics as evaluator
        original_loader = evaluator.load_rgb_tensor
        evaluator.load_rgb_tensor = lambda *_args, **_kwargs: FakeTensor()
        try:
            rows = evaluate_metrics(
                [('a', Path('/sr/a.png'), Path('/gt/a.png'))],
                ['lpips', 'clipiqa+'],
                FakePyIQA(),
                FakeTorch(),
                'cuda',
                4,
            )
        finally:
            evaluator.load_rgb_tensor = original_loader

        self.assertEqual(rows[0]['lpips'], 0.25)
        self.assertEqual(rows[0]['clipiqa+'], 0.25)
        self.assertEqual(events, [
            ('create', 'lpips', 'cuda'),
            ('score', 'lpips'),
            ('empty_cache',),
            ('create', 'clipiqa+', 'cuda'),
            ('score', 'clipiqa+'),
            ('empty_cache',),
        ])


if __name__ == '__main__':
    unittest.main()
