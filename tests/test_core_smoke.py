"""Small tensor tests, not evaluations of restoration quality."""
import pytest
import torch

from PD_MAE_SR.archs.mae_arch import RestorationNeedPDMAE
from PD_MAE_SR.experiments.train_restoration_need import patchify, restoration_need_target
from drct.archs.DRCT_arch import SFT_Layer, PD_MAE_Encoder_Wrapper


def test_prior_produces_reconstruction_and_patch_scores():
    model = RestorationNeedPDMAE(img_size=32).eval()
    with torch.no_grad():
        rgb, logits = model(torch.rand(1, 3, 32, 32), return_need=True)
    assert rgb.shape == (1, 16, 192)
    assert logits.shape == (1, 16)
    assert torch.isfinite(rgb).all()
    assert torch.equal(logits, torch.zeros_like(logits))


def test_known_paired_difference_target():
    clean = patchify(torch.zeros(1, 3, 16, 16))
    assert torch.equal(restoration_need_target(clean, clean), torch.zeros(1, 4))
    degraded = patchify(torch.full((1, 3, 16, 16), 0.1))
    assert torch.allclose(
        restoration_need_target(degraded, clean),
        torch.full((1, 4), 1.0 - torch.exp(torch.tensor(-1.0)).item()),
    )


def test_zero_sft_is_exact_identity_and_receives_gradients():
    layer = SFT_Layer(drct_channels=6, mae_channels=8)
    features = torch.rand(1, 6, 7, 9, requires_grad=True)
    prior = torch.rand(1, 8, 3, 4)
    output = layer(features, prior)
    assert torch.equal(output, features)
    output.sum().backward()
    assert layer.shift_conv.weight.grad.abs().sum() > 0


def test_learned_prior_fails_closed_without_checkpoint():
    with pytest.raises(FileNotFoundError, match="checkpoint is required"):
        PD_MAE_Encoder_Wrapper(checkpoint_path=None, img_size=32)


def test_explicit_random_prior_is_frozen_and_stays_in_eval():
    prior = PD_MAE_Encoder_Wrapper(allow_random_prior=True, img_size=32)
    prior.train(True)
    assert not prior.training
    assert all(not param.requires_grad for param in prior.parameters())
    assert not hasattr(prior.mae, "decoder")
    assert not hasattr(prior, "need_head")
