"""SwinIR with optional restoration-need-gated RN-PD-MAE conditioning.

The baseline and candidate use this same registered architecture.  Setting
``use_sft=False`` recovers the upstream SwinIR computation and state-dict
layout exactly; setting it to ``True`` adds the frozen prior and zero-initialized
residual SFT layers used by the matched second-backbone experiment.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from basicsr.archs.swinir_arch import SwinIR
from basicsr.utils.registry import ARCH_REGISTRY

from .DRCT_arch import PD_MAE_Encoder_Wrapper, SFT_Layer


@ARCH_REGISTRY.register()
class RNPDMAESwinIR(SwinIR):
    """Apply the RN-PD-MAE intervention to an otherwise unchanged SwinIR."""

    def __init__(self, *args, use_sft=False, mae_checkpoint=None,
                 use_restoration_need_gate=False,
                 restoration_need_gate_mode='spatial', zero_init_sft=True,
                 allow_random_prior=False, mae_input_upscale=4,
                 mae_tile_batch_size=8, mae_tile_overlap=0, **kwargs):
        super().__init__(*args, **kwargs)

        valid_gate_modes = {
            'spatial', 'global', 'shifted', 'permuted', 'gradient'
        }
        if restoration_need_gate_mode not in valid_gate_modes:
            raise ValueError(
                'Unknown restoration_need_gate_mode='
                f'{restoration_need_gate_mode}. Expected one of '
                f'{sorted(valid_gate_modes)}.'
            )
        if use_restoration_need_gate and not use_sft:
            raise ValueError(
                'Restoration-need gating requires use_sft=True.'
            )

        self.use_sft = use_sft
        self.use_restoration_need_gate = use_restoration_need_gate
        self.restoration_need_gate_mode = restoration_need_gate_mode

        if self.use_sft:
            # Match the three relative intervention depths used in the six-RDG
            # DRCT experiment.  The official SwinIR-M real-SR model also has six
            # residual groups, yielding indices 1, 3, and 5.
            sft_indices = sorted({
                max(0, self.num_layers // 3 - 1),
                max(0, 2 * self.num_layers // 3 - 1),
                self.num_layers - 1,
            })
            self.sft_indices = tuple(sft_indices)
            self.sft_layers = nn.ModuleDict({
                str(index): SFT_Layer(
                    drct_channels=self.embed_dim,
                    mae_channels=384,
                    zero_init=zero_init_sft,
                )
                for index in self.sft_indices
            })
            self.mae_wrapper = PD_MAE_Encoder_Wrapper(
                checkpoint_path=mae_checkpoint,
                img_size=256,
                predict_restoration_need=self.use_restoration_need_gate,
                allow_random_prior=allow_random_prior,
                input_upscale=mae_input_upscale,
                tile_batch_size=mae_tile_batch_size,
                tile_overlap=mae_tile_overlap,
            )

    @staticmethod
    def _gradient_gate(image, eps=1e-6):
        if image.shape[1] == 3:
            gray = (
                0.299 * image[:, 0:1]
                + 0.587 * image[:, 1:2]
                + 0.114 * image[:, 2:3]
            )
        else:
            gray = image.mean(dim=1, keepdim=True)
        horizontal = F.pad(
            gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0)
        )
        vertical = F.pad(
            gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1)
        )
        magnitude = torch.sqrt(horizontal.square() + vertical.square())
        scale = magnitude.flatten(1).mean(dim=1).view(-1, 1, 1, 1)
        scale = scale + 2.0 * magnitude.flatten(1).std(
            dim=1, unbiased=False
        ).view(-1, 1, 1, 1)
        return (magnitude / scale.clamp_min(eps)).clamp_(0.0, 1.0)

    def _prior_fields(self, raw_input):
        mae_feat = None
        restoration_need = None
        if not self.use_sft:
            return mae_feat, restoration_need

        if self.use_restoration_need_gate:
            if self.restoration_need_gate_mode == 'gradient':
                mae_feat = self.mae_wrapper(raw_input)
                restoration_need = self._gradient_gate(raw_input)
            else:
                mae_feat, restoration_need = self.mae_wrapper(
                    raw_input, return_need=True
                )
            if self.restoration_need_gate_mode == 'global':
                restoration_need = restoration_need.mean(
                    dim=(-2, -1), keepdim=True
                )
            elif self.restoration_need_gate_mode in {'shifted', 'permuted'}:
                restoration_need = torch.roll(
                    restoration_need,
                    shifts=(
                        max(1, restoration_need.shape[-2] // 2),
                        max(1, restoration_need.shape[-1] // 2),
                    ),
                    dims=(-2, -1),
                )
        else:
            mae_feat = self.mae_wrapper(raw_input)
        return mae_feat, restoration_need

    def forward_features(self, x, mae_feat=None, restoration_need=None):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for index, layer in enumerate(self.layers):
            x = layer(x, x_size)
            if (self.use_sft and str(index) in self.sft_layers
                    and mae_feat is not None):
                x_2d = self.patch_unembed(x, x_size)
                x_2d = self.sft_layers[str(index)](
                    x_2d, mae_feat, gate=restoration_need
                )
                # Avoid applying PatchEmbed's LayerNorm twice.  At zero SFT
                # initialization this conversion is exactly the identity.
                x = x_2d.flatten(2).transpose(1, 2)

        x = self.norm(x)
        return self.patch_unembed(x, x_size)

    def forward(self, x):
        raw_input = x
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        mae_feat, restoration_need = self._prior_fields(raw_input)

        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)
            x = self.conv_after_body(
                self.forward_features(x, mae_feat, restoration_need)
            ) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            x = self.conv_first(x)
            x = self.conv_after_body(
                self.forward_features(x, mae_feat, restoration_need)
            ) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            x = self.conv_first(x)
            x = self.conv_after_body(
                self.forward_features(x, mae_feat, restoration_need)
            ) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(F.interpolate(
                x, scale_factor=2, mode='nearest'
            )))
            x = self.lrelu(self.conv_up2(F.interpolate(
                x, scale_factor=2, mode='nearest'
            )))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            x_first = self.conv_first(x)
            residual = self.conv_after_body(
                self.forward_features(x_first, mae_feat, restoration_need)
            ) + x_first
            x = x + self.conv_last(residual)

        return x / self.img_range + self.mean
