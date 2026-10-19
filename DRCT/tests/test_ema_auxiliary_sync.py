import unittest

import torch
from torch import nn

from drct.models.drct_model import DRCTModel


class ToyGenerator(nn.Module):
    def __init__(self, prior_value, sft_value, include_prior=True):
        super().__init__()
        self.backbone = nn.Linear(2, 2, bias=False)
        if include_prior:
            self.mae_wrapper = nn.Linear(2, 2, bias=False)
            nn.init.constant_(self.mae_wrapper.weight, prior_value)
        self.sft_layers = nn.ModuleDict({
            '1': nn.Conv2d(1, 1, 1, bias=False),
        })
        nn.init.constant_(self.sft_layers['1'].weight, sft_value)


class EMAAuxiliarySyncTest(unittest.TestCase):
    def test_copies_auxiliary_modules_without_touching_backbone(self):
        training = ToyGenerator(prior_value=1.0, sft_value=2.0)
        ema = ToyGenerator(prior_value=3.0, sft_value=4.0)
        nn.init.constant_(training.backbone.weight, 5.0)
        nn.init.constant_(ema.backbone.weight, 6.0)

        names = DRCTModel._synchronize_auxiliary_modules(training, ema)

        self.assertEqual(names, ['mae_wrapper', 'sft_layers'])
        self.assertTrue(torch.equal(
            training.mae_wrapper.weight, ema.mae_wrapper.weight
        ))
        self.assertTrue(torch.equal(
            training.sft_layers['1'].weight, ema.sft_layers['1'].weight
        ))
        self.assertFalse(torch.equal(
            training.backbone.weight, ema.backbone.weight
        ))

    def test_rejects_asymmetric_auxiliary_module_layout(self):
        training = ToyGenerator(
            prior_value=1.0, sft_value=2.0, include_prior=True
        )
        ema = ToyGenerator(
            prior_value=3.0, sft_value=4.0, include_prior=False
        )

        with self.assertRaisesRegex(RuntimeError, 'mae_wrapper'):
            DRCTModel._synchronize_auxiliary_modules(training, ema)


if __name__ == '__main__':
    unittest.main()
