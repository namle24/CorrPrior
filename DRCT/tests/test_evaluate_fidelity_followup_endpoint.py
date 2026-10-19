import unittest
from pathlib import Path

from scripts.evaluate_fidelity_followup_endpoint import evaluate


def test_endpoint_implementation_is_hashable():
    from scripts.evaluate_fidelity_followup_endpoint import file_sha256

    path = Path(__file__).resolve().parents[1] / 'scripts' / \
        'evaluate_fidelity_followup_endpoint.py'
    assert len(file_sha256(path)) == 64


def metric(delta, ci_low=-0.1, ci_high=0.1, p_value=0.5):
    return {
        'baseline_mean': 1.0,
        'candidate_mean': 1.0 + delta,
        'candidate_minus_baseline': delta,
        'ci_low_raw_difference': ci_low,
        'ci_high_raw_difference': ci_high,
        'confidence': 0.95,
        'permutation_p_value': p_value,
    }


class FidelityFollowupEndpointTest(unittest.TestCase):
    def setUp(self):
        self.protocol = {
            'unseen_confirmation': {
                'permutation_alpha': 0.05,
                'guardrails': {
                    'ssim_candidate_minus_baseline_minimum': 0.0,
                    'dists_candidate_minus_baseline_maximum': 0.002,
                    'lpips_candidate_minus_baseline_maximum': 0.005,
                },
            },
        }
        self.standard = {
            'psnr': metric(0.1, ci_low=0.05, ci_high=0.15, p_value=0.001),
            'ssim': metric(0.001),
            'niqe': metric(0.0),
        }
        self.pyiqa = {
            'dists': metric(0.001),
            'lpips': metric(0.003),
            'musiq': metric(0.0),
        }

    def test_complete_pass(self):
        result = evaluate(self.protocol, self.standard, self.pyiqa)
        self.assertTrue(result['primary']['pass'])
        self.assertTrue(result['guardrails_pass'])
        self.assertTrue(result['confirmation_pass'])

    def test_ci_crossing_zero_fails_primary(self):
        self.standard['psnr'] = metric(
            0.1, ci_low=-0.01, ci_high=0.2, p_value=0.001
        )
        result = evaluate(self.protocol, self.standard, self.pyiqa)
        self.assertFalse(result['primary']['pass'])
        self.assertFalse(result['confirmation_pass'])

    def test_perceptual_guardrail_can_fail_positive_psnr(self):
        self.pyiqa['dists'] = metric(0.003)
        result = evaluate(self.protocol, self.standard, self.pyiqa)
        self.assertTrue(result['primary']['pass'])
        self.assertFalse(result['guardrails']['dists']['pass'])
        self.assertFalse(result['confirmation_pass'])

    def test_can_select_separately_named_frozen_rule(self):
        self.protocol['ungated_pivot_confirmation'] = dict(
            self.protocol['unseen_confirmation']
        )
        result = evaluate(
            self.protocol,
            self.standard,
            self.pyiqa,
            rule_key='ungated_pivot_confirmation',
        )
        self.assertTrue(result['confirmation_pass'])

    def test_revised_transfer_and_replication_rules_share_endpoint_logic(self):
        for rule_key in ('matched_transfer', 'training_seed_replication'):
            self.protocol[rule_key] = dict(self.protocol['unseen_confirmation'])
            result = evaluate(
                self.protocol,
                self.standard,
                self.pyiqa,
                rule_key=rule_key,
            )
            self.assertTrue(result['confirmation_pass'])


if __name__ == '__main__':
    unittest.main()
