import csv
from pathlib import Path
import tempfile
import unittest

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    NUMPY_AVAILABLE = False
else:
    NUMPY_AVAILABLE = True
    from scripts.evaluate_fidelity_mechanism_controls import evaluate


def write_metrics(path, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def camera_files(root, comparison, camera, psnr=0.1, ssim=0.001,
                 dists=0.001, lpips=0.002, images=12):
    paths = [root / f'{comparison}_{camera}_{suffix}.csv' for suffix in (
        'base_std', 'cand_std', 'base_iqa', 'cand_iqa'
    )]
    stems = [f'{camera}_{index}' for index in range(images)]
    write_metrics(paths[0], [
        {'image': stem, 'psnr': 30.0, 'ssim': 0.8} for stem in stems
    ])
    write_metrics(paths[1], [
        {'image': stem, 'psnr': 30.0 + psnr, 'ssim': 0.8 + ssim}
        for stem in stems
    ])
    write_metrics(paths[2], [
        {'image': stem, 'dists': 0.2, 'lpips': 0.3} for stem in stems
    ])
    write_metrics(paths[3], [
        {'image': stem, 'dists': 0.2 + dists, 'lpips': 0.3 + lpips}
        for stem in stems
    ])
    return [comparison, camera, *(str(path) for path in paths)]


def protocol():
    return {'mechanism_decision': {
        'comparisons': {
            'C2_vs_C0': 'prior',
            'P0_vs_C2': 'gate',
        },
        'confidence': 0.95,
        'bootstrap_samples': 500,
        'permutation_samples': 1000,
        'permutation_alpha': 0.05,
        'seed': 276,
        'guardrails': {
            'ssim_candidate_minus_baseline_minimum': 0.0,
            'dists_candidate_minus_baseline_maximum': 0.002,
            'lpips_candidate_minus_baseline_maximum': 0.005,
        },
    }}


@unittest.skipUnless(NUMPY_AVAILABLE, 'numpy is not installed')
class FidelityMechanismControlsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def specifications(self, overrides=None):
        overrides = overrides or {}
        rows = []
        for comparison in ('C2_vs_C0', 'P0_vs_C2'):
            for camera in ('Canon', 'Nikon'):
                values = overrides.get((comparison, camera), {})
                rows.append(camera_files(
                    self.root, comparison, camera, **values
                ))
        return rows

    def test_both_mechanism_claims_pass_clear_effects(self):
        report = evaluate(self.specifications(), protocol())
        self.assertTrue(report['claims']['learned_prior_supported'])
        self.assertTrue(report['claims']['spatial_gate_supported'])

    def test_per_camera_guardrail_blocks_spatial_claim(self):
        rows = self.specifications({
            ('P0_vs_C2', 'Nikon'): {'dists': 0.003},
        })
        report = evaluate(rows, protocol())
        self.assertTrue(report['comparisons']['P0_vs_C2']['primary']['pass'])
        self.assertFalse(report['claims']['spatial_gate_supported'])

    def test_missing_comparison_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Comparison set'):
            evaluate(self.specifications()[:2], protocol())


if __name__ == '__main__':
    unittest.main()
