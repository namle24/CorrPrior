import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / 'drct/runtime_options.py'
SPEC = importlib.util.spec_from_file_location('runtime_options', MODULE_PATH)
RUNTIME_OPTIONS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME_OPTIONS)
reseed_after_option_overrides = RUNTIME_OPTIONS.reseed_after_option_overrides
configure_exact_resume_skip_batches = (
    RUNTIME_OPTIONS.configure_exact_resume_skip_batches
)
build_exact_resume_cpu_prefetcher = (
    RUNTIME_OPTIONS.build_exact_resume_cpu_prefetcher
)


class RuntimeOptionsTest(unittest.TestCase):
    def test_reseeds_with_final_forced_seed_and_rank(self):
        observed = []
        result = reseed_after_option_overrides(
            {'manual_seed': 2, 'rank': 3}, observed.append
        )
        self.assertEqual(result, 5)
        self.assertEqual(observed, [5])

    def test_missing_seed_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'manual_seed'):
            reseed_after_option_overrides({'rank': 0}, lambda seed: None)

    def _resume_opt(self, state='25000.state'):
        return {
            'num_gpu': 1,
            'path': {'resume_state': f'/tmp/run/training_states/{state}'},
            'datasets': {'train': {'prefetch_mode': None}},
        }

    def test_exact_resume_is_disabled_without_explicit_environment(self):
        opt = self._resume_opt()
        result = configure_exact_resume_skip_batches(opt, {})
        self.assertEqual(result, 0)
        self.assertEqual(opt['exact_resume_skip_batches'], 0)

    def test_exact_resume_validates_state_iteration(self):
        opt = self._resume_opt()
        result = configure_exact_resume_skip_batches(
            opt, {'RN_PD_MAE_EXACT_RESUME_SKIP_BATCHES': '25000'}
        )
        self.assertEqual(result, 25000)
        self.assertEqual(opt['exact_resume_skip_batches'], 25000)
        with self.assertRaisesRegex(ValueError, 'filename iteration'):
            configure_exact_resume_skip_batches(
                self._resume_opt('20000.state'),
                {'RN_PD_MAE_EXACT_RESUME_SKIP_BATCHES': '25000'},
            )

    def test_exact_resume_requires_resume_state_and_cpu_prefetch(self):
        opt = self._resume_opt()
        opt['path']['resume_state'] = None
        with self.assertRaisesRegex(ValueError, 'path.resume_state'):
            configure_exact_resume_skip_batches(
                opt, {'RN_PD_MAE_EXACT_RESUME_SKIP_BATCHES': '25000'}
            )
        opt = self._resume_opt()
        opt['datasets']['train']['prefetch_mode'] = 'cuda'
        with self.assertRaisesRegex(ValueError, 'CPU prefetching'):
            configure_exact_resume_skip_batches(
                opt, {'RN_PD_MAE_EXACT_RESUME_SKIP_BATCHES': '25000'}
            )

    def test_exact_resume_prefetcher_skips_only_the_first_reset(self):
        events = []

        class BasePrefetcher:
            def __init__(self, loader):
                self.ori_loader = loader
                self.loader = iter(loader)

            def reset(self):
                self.loader = iter(self.ori_loader)

            def next(self):
                return next(self.loader, None)

        wrapper = build_exact_resume_cpu_prefetcher(
            BasePrefetcher, 3, events.append
        )
        prefetcher = wrapper(list(range(6)))
        prefetcher.reset()
        self.assertEqual(prefetcher.next(), 3)
        self.assertEqual(events, [
            'Exact-resume dataloader replay consumed 3 batches.'
        ])
        prefetcher.reset()
        self.assertEqual(prefetcher.next(), 0)
        self.assertEqual(len(events), 1)

    def test_exact_resume_prefetcher_fails_on_epoch_overrun(self):
        class BasePrefetcher:
            def __init__(self, loader):
                self.ori_loader = loader
                self.loader = iter(loader)

            def reset(self):
                self.loader = iter(self.ori_loader)

        wrapper = build_exact_resume_cpu_prefetcher(BasePrefetcher, 4)
        with self.assertRaisesRegex(RuntimeError, 'exceeds'):
            wrapper([0, 1, 2]).reset()


if __name__ == '__main__':
    unittest.main()
