import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.nn import functional as F
import importlib.util
import os

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_

from einops import rearrange

# Load PDMAE by absolute file path (avoids `archs` name clashes on sys.path).
_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
_pd_mae_mae_arch = os.path.join(_proj_root, 'PD_MAE_SR', 'archs', 'mae_arch.py')

_pdmae_import_err = None
PDMAE = None
MAEEncoder = None
if os.path.isfile(_pd_mae_mae_arch):
    try:
        _spec = importlib.util.spec_from_file_location('pd_mae_sr_mae_arch', _pd_mae_mae_arch)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        PDMAE = _mod.PDMAE
        MAEEncoder = _mod.MAEEncoder
    except Exception as e:
        _pdmae_import_err = e
else:
    _pdmae_import_err = FileNotFoundError(_pd_mae_mae_arch)

if MAEEncoder is None:
    print(f'Warning: Could not import PDMAE from {_pd_mae_mae_arch}: {_pdmae_import_err}')


class SFT_Layer(nn.Module):
    def __init__(self, drct_channels=180, mae_channels=384, zero_init=True):
        super().__init__()
        self.scale_conv = nn.Conv2d(mae_channels, drct_channels, 1)
        self.shift_conv = nn.Conv2d(mae_channels, drct_channels, 1)
        if zero_init:
            nn.init.zeros_(self.scale_conv.weight)
            nn.init.zeros_(self.scale_conv.bias)
            nn.init.zeros_(self.shift_conv.weight)
            nn.init.zeros_(self.shift_conv.bias)

    def forward(self, drct_feat, mae_feat, gate=None):
        mae_resized = F.interpolate(mae_feat,
                                    size=drct_feat.shape[-2:],
                                    mode='bilinear',
                                    align_corners=False)
        scale = self.scale_conv(mae_resized)
        shift = self.shift_conv(mae_resized)
        modulation = drct_feat * scale + shift
        if gate is not None:
            gate = F.interpolate(gate, size=drct_feat.shape[-2:],
                                 mode='bilinear', align_corners=False)
            modulation = modulation * gate
        return drct_feat + modulation


class PD_MAE_Encoder_Wrapper(nn.Module):
    """Frozen PD-MAE encoder with an optional restoration-need head."""

    def __init__(self, checkpoint_path=None, img_size=256, patch_size=8,
                 predict_restoration_need=False, allow_random_prior=False,
                 input_upscale=4, tile_batch_size=8, tile_overlap=0):
        super().__init__()
        if MAEEncoder is None:
            raise ImportError(
                f'MAEEncoder is not available. Expected mae_arch at: {_pd_mae_mae_arch}. '
                f'Import error: {_pdmae_import_err}'
            ) from _pdmae_import_err

        # Keep the historical ``mae.encoder`` key layout while omitting the
        # unused decoder from DRCT checkpoints.
        self.mae = nn.Module()
        self.mae.encoder = MAEEncoder(img_size=img_size, patch_size=patch_size)
        self.encoder_img_size = img_size
        self.input_upscale = input_upscale
        self.tile_batch_size = tile_batch_size
        self.tile_overlap = tile_overlap
        self.predict_restoration_need = predict_restoration_need
        if self.predict_restoration_need:
            self.need_head = nn.Sequential(
                nn.LayerNorm(384),
                nn.Linear(384, 1),
            )

        if checkpoint_path and os.path.isfile(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        elif not allow_random_prior:
            raise FileNotFoundError(
                f'PD-MAE checkpoint is required but was not found: {checkpoint_path}'
            )

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    @staticmethod
    def _checkpoint_state(checkpoint):
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        if 'params' in checkpoint:
            return checkpoint['params']
        return checkpoint

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state = self._checkpoint_state(checkpoint)
        encoder_state = {
            key[len('encoder.'):]: value
            for key, value in state.items()
            if key.startswith('encoder.')
        }
        if not encoder_state:
            encoder_state = {
                key[len('mae.encoder.'):]: value
                for key, value in state.items()
                if key.startswith('mae.encoder.')
            }
        if not encoder_state:
            raise RuntimeError(
                f'No encoder weights found in PD-MAE checkpoint: {checkpoint_path}'
            )
        self.mae.encoder.load_state_dict(encoder_state, strict=True)

        if self.predict_restoration_need:
            need_state = {
                key[len('need_head.'):]: value
                for key, value in state.items()
                if key.startswith('need_head.')
            }
            if not need_state:
                raise RuntimeError(
                    'Restoration-need gating requires a calibrated checkpoint '
                    f'with need_head weights: {checkpoint_path}'
                )
            self.need_head.load_state_dict(need_state, strict=True)

    def train(self, mode=True):
        """Keep the frozen prior deterministic when the parent model trains."""
        super().train(False)
        return self

    @staticmethod
    def _tile_starts(length, tile_size, overlap):
        """Return patch-aligned starts that cover an axis without hard gaps."""
        if length < tile_size:
            raise ValueError(
                f'Padded axis length {length} is smaller than tile size {tile_size}.'
            )
        if not 0 <= overlap < tile_size:
            raise ValueError(
                f'tile_overlap must be in [0, {tile_size}), got {overlap}.'
            )
        if length == tile_size:
            return [0]
        stride = tile_size - overlap
        starts = list(range(0, length - tile_size + 1, stride))
        final_start = length - tile_size
        if starts[-1] != final_start:
            starts.append(final_start)
        return starts

    @staticmethod
    def _blend_window(tile_grid, overlap_tokens, reference):
        """Create a positive separable window for overlap-normalized stitching."""
        if overlap_tokens == 0:
            return reference.new_ones(1, 1, tile_grid, tile_grid)
        axis = torch.hann_window(
            tile_grid, periodic=False, dtype=reference.dtype,
            device=reference.device,
        ).clamp_min_(1e-3)
        return (axis[:, None] * axis[None, :]).view(
            1, 1, tile_grid, tile_grid
        )

    def _encode_tiles(self, x, input_upscale=None):
        """Encode an LR image at its native aspect ratio using HR-scale tiles.

        PD-MAE is calibrated on 256-pixel HR-aligned crops, whereas DRCT sees
        their 64-pixel LR counterparts.  Resizing a complete test image to a
        single square crop would destroy both aspect ratio and spatial
        correspondence.  We instead bicubically upscale by the training scale,
        pad to complete 256-pixel tiles, encode each tile independently, and
        stitch the token grids back into one aligned feature field.
        """
        input_upscale = self.input_upscale if input_upscale is None else input_upscale
        if input_upscale <= 0:
            raise ValueError(f'input_upscale must be positive, got {input_upscale}.')
        if self.tile_batch_size <= 0:
            raise ValueError(
                f'tile_batch_size must be positive, got {self.tile_batch_size}.'
            )

        x = F.interpolate(
            x,
            scale_factor=input_upscale,
            mode='bicubic',
            align_corners=False,
        ).clamp_(0.0, 1.0)
        batch, channels, height, width = x.shape
        tile_size = self.encoder_img_size
        patch_size = self.mae.encoder.patch_size
        if tile_size % patch_size:
            raise ValueError(
                f'Encoder tile size {tile_size} must be divisible by patch size {patch_size}.'
            )

        tile_overlap = self.tile_overlap
        if not 0 <= tile_overlap < tile_size:
            raise ValueError(
                f'tile_overlap must be in [0, {tile_size}), got {tile_overlap}.'
            )
        if tile_overlap % patch_size:
            raise ValueError(
                f'tile_overlap {tile_overlap} must be divisible by patch size '
                f'{patch_size}.'
            )

        if tile_overlap == 0:
            # Preserve the original non-overlapping path exactly when requested.
            padded_height = math.ceil(height / tile_size) * tile_size
            padded_width = math.ceil(width / tile_size) * tile_size
        else:
            # Overlapping starts and the final crop must stay on the token grid.
            padded_height = max(
                tile_size, math.ceil(height / patch_size) * patch_size
            )
            padded_width = max(
                tile_size, math.ceil(width / patch_size) * patch_size
            )
        pad_height = padded_height - height
        pad_width = padded_width - width
        if pad_height or pad_width:
            # Reflection padding fails when a requested pad is not smaller than
            # the corresponding input dimension, which can occur for tiny LR
            # images. Replication is well-defined for every valid image size.
            x = F.pad(x, (0, pad_width, 0, pad_height), mode='replicate')

        tiles = []
        locations = []
        top_starts = self._tile_starts(
            padded_height, tile_size, tile_overlap
        )
        left_starts = self._tile_starts(
            padded_width, tile_size, tile_overlap
        )
        for top in top_starts:
            for left in left_starts:
                tiles.append(x[:, :, top:top + tile_size, left:left + tile_size])
                locations.append((top // patch_size, left // patch_size))

        tile_grid = tile_size // patch_size
        encoded = []
        need_logits = []
        # Stack tile groups along the batch axis.  This preserves the original
        # batch ordering within every tile and bounds inference-time memory.
        for start in range(0, len(tiles), self.tile_batch_size):
            tile_group = torch.cat(tiles[start:start + self.tile_batch_size], dim=0)
            latent = self.mae.encoder(tile_group)
            tokens = latent[:, 1:, :]
            encoded.append(tokens)
            if self.predict_restoration_need:
                need_logits.append(self.need_head(tokens).squeeze(-1))

        encoded = torch.cat(encoded, dim=0)
        if self.predict_restoration_need:
            need_logits = torch.cat(need_logits, dim=0)

        feature = x.new_zeros(
            batch,
            encoded.shape[-1],
            padded_height // patch_size,
            padded_width // patch_size,
        )
        blend = self._blend_window(
            tile_grid, tile_overlap // patch_size, feature
        )
        blend_sum = x.new_zeros(
            1, 1, padded_height // patch_size, padded_width // patch_size
        )
        need = None
        if self.predict_restoration_need:
            need = x.new_zeros(
                batch,
                1,
                padded_height // patch_size,
                padded_width // patch_size,
            )

        # Each tile tensor contains the complete input batch, so tile t occupies
        # rows [t*B:(t+1)*B] in the concatenated encoder output.
        for tile_index, (top, left) in enumerate(locations):
            begin = tile_index * batch
            end = begin + batch
            tile_feature = encoded[begin:end].transpose(1, 2).reshape(
                batch, encoded.shape[-1], tile_grid, tile_grid
            )
            feature[:, :, top:top + tile_grid, left:left + tile_grid].add_(
                tile_feature * blend
            )
            blend_sum[:, :, top:top + tile_grid, left:left + tile_grid].add_(
                blend
            )
            if need is not None:
                tile_need = torch.sigmoid(need_logits[begin:end]).reshape(
                    batch, 1, tile_grid, tile_grid
                )
                need[:, :, top:top + tile_grid, left:left + tile_grid].add_(
                    tile_need * blend
                )

        feature = feature / blend_sum.clamp_min_(1e-12)
        if need is not None:
            need = need / blend_sum.clamp_min_(1e-12)

        valid_height = math.ceil(height / patch_size)
        valid_width = math.ceil(width / patch_size)
        feature = feature[:, :, :valid_height, :valid_width]
        if need is not None:
            need = need[:, :, :valid_height, :valid_width]
        return feature, need

    def forward(self, x, return_need=False, input_upscale=None):
        feat, need_map = self._encode_tiles(x, input_upscale=input_upscale)
        if not return_need:
            return feat

        if not self.predict_restoration_need:
            raise RuntimeError('This wrapper was created without a restoration-need head.')
        return feat, need_map


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops



class RDG(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, shift_size, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer, gc, patch_size, img_size):
        super(RDG, self).__init__()

        self.swin1 = SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                          num_heads=num_heads, window_size=window_size,
                                          shift_size=0,  # For first block
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust1 = nn.Conv2d(dim, gc, 1) 
        
        self.swin2 = SwinTransformerBlock(dim + gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + gc)%num_heads), window_size=window_size,
                                          shift_size=window_size//2,  # For first block
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust2 = nn.Conv2d(dim+gc, gc, 1) 
        
        self.swin3 = SwinTransformerBlock(dim + 2 * gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + 2 * gc)%num_heads), window_size=window_size,
                                          shift_size=0,  # For first block
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust3 = nn.Conv2d(dim+gc*2, gc, 1) 
        
        self.swin4 = SwinTransformerBlock(dim + 3 * gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + 3 * gc)%num_heads), window_size=window_size,
                                          shift_size=window_size//2,  # For first block
                                          mlp_ratio=1,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust4 = nn.Conv2d(dim+gc*3, gc, 1) 
        
        self.swin5 = SwinTransformerBlock(dim + 4 * gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + 4 * gc)%num_heads), window_size=window_size,
                                          shift_size=0,  # For first block
                                          mlp_ratio=1,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.adjust5 = nn.Conv2d(dim+gc*4, dim, 1) 
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.pe = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.pue = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)
        
       

    def forward(self, x, xsize):
        x1 = self.pe(self.lrelu(self.adjust1(self.pue(self.swin1(x,xsize), xsize))))
        x2 = self.pe(self.lrelu(self.adjust2(self.pue(self.swin2(torch.cat((x, x1), -1), xsize), xsize))))
        x3 = self.pe(self.lrelu(self.adjust3(self.pue(self.swin3(torch.cat((x, x1, x2), -1), xsize), xsize))))
        x4 = self.pe(self.lrelu(self.adjust4(self.pue(self.swin4(torch.cat((x, x1, x2, x3), -1), xsize), xsize))))
        x5 = self.pe(           self.adjust5(self.pue(self.swin5(torch.cat((x, x1, x2, x3, x4), -1), xsize), xsize)))
        

        return x5 * 0.2 + x
    
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x





class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # 结构为 [B, num_patches, C]
        if self.norm is not None:
            x = self.norm(x)  # 归一化
        return x


    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    输入:
        img_size (int): 图像的大小，默认为 224*224.
        patch_size (int): Patch token 的大小，默认为 4*4.
        in_chans (int): 输入图像的通道数，默认为 3.
        embed_dim (int): 线性 projection 输出的通道数，默认为 96.
        norm_layer (nn.Module, optional): 归一化层， 默认为N None.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)  # 图像的大小，默认为 224*224
        patch_size = to_2tuple(patch_size)  # Patch token 的大小，默认为 4*4
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # patch 的分辨率
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # patch 的个数，num_patches

        self.in_chans = in_chans  # 输入图像的通道数
        self.embed_dim = embed_dim  # 线性 projection 输出的通道数

    def forward(self, x, x_size):
        B, HW, C = x.shape  # 输入 x 的结构
        x = x.transpose(1, 2).view(B, -1, x_size[0], x_size[1])  # 输出结构为 [B, Ph*Pw, C]
        return x



class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


@ARCH_REGISTRY.register()
class DRCT(nn.Module):

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=7,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 overlap_ratio=0.5,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 gc = 32,
                 use_sft=False,
                 mae_checkpoint=None,
                 use_restoration_need_gate=False,
                 restoration_need_gate_mode='spatial',
                 zero_init_sft=True,
                 allow_random_prior=False,
                 mae_input_upscale=4,
                 mae_tile_batch_size=8,
                 mae_tile_overlap=0,
                 **kwargs):
        super(DRCT, self).__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.overlap_ratio = overlap_ratio

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            
            layer = RDG(dim=embed_dim, input_resolution=(patches_resolution[0], patches_resolution[1]),
                                 num_heads= num_heads[i_layer], window_size=window_size, depth=0,
                                 shift_size= window_size//2, mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,  drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], 
                                             norm_layer=norm_layer,gc=gc, img_size=img_size, patch_size=patch_size)

            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        self.use_sft = use_sft
        self.use_restoration_need_gate = use_restoration_need_gate
        valid_gate_modes = {
            'spatial', 'global', 'shifted', 'permuted', 'gradient'
        }
        if restoration_need_gate_mode not in valid_gate_modes:
            raise ValueError(
                f'Unknown restoration_need_gate_mode={restoration_need_gate_mode}. '
                f'Expected one of {sorted(valid_gate_modes)}.'
            )
        self.restoration_need_gate_mode = restoration_need_gate_mode
        if self.use_sft:
            sft_indices = [1, 3, 5] if len(depths) >= 6 else [1, 3]
            self.sft_layers = nn.ModuleDict({
                str(i): SFT_Layer(
                    drct_channels=embed_dim,
                    mae_channels=384,
                    zero_init=zero_init_sft,
                )
                for i in sft_indices
            })

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv_after_body = nn.Identity()

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # Initialize the SR backbone and modulation heads before loading the
        # frozen prior.  Loading first would let this recursive call overwrite
        # every Linear and LayerNorm weight in the pretrained MAE encoder.
        self.apply(self._init_weights)

        if self.use_sft:
            # ``self.apply`` initializes Linear/LayerNorm modules, so restore
            # the baseline-preserving zero initialization afterward.
            if zero_init_sft:
                for sft_layer in self.sft_layers.values():
                    nn.init.zeros_(sft_layer.scale_conv.weight)
                    nn.init.zeros_(sft_layer.scale_conv.bias)
                    nn.init.zeros_(sft_layer.shift_conv.weight)
                    nn.init.zeros_(sft_layer.shift_conv.bias)
            self.mae_wrapper = PD_MAE_Encoder_Wrapper(
                checkpoint_path=mae_checkpoint,
                img_size=256,
                predict_restoration_need=self.use_restoration_need_gate,
                allow_random_prior=allow_random_prior,
                input_upscale=mae_input_upscale,
                tile_batch_size=mae_tile_batch_size,
                tile_overlap=mae_tile_overlap,
            )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

   

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, mae_feat=None, restoration_need=None):
        x_size = (x.shape[2], x.shape[3])

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for i, layer in enumerate(self.layers):
            x = layer(x, x_size)
            if self.use_sft and str(i) in self.sft_layers and mae_feat is not None:
                # 1. Un-embed to 2D
                x_2d = self.patch_unembed(x, x_size)  # [B, C, H, W]
                # 2. Apply SFT
                x_2d = self.sft_layers[str(i)](
                    x_2d, mae_feat, gate=restoration_need
                )
                # 3. Return to the token layout without applying PatchEmbed's
                # LayerNorm a second time.  With zero modulation this path is
                # now exactly the identity mapping.
                x = x_2d.flatten(2).transpose(1, 2)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    @staticmethod
    def _gradient_gate(image, eps=1e-6):
        """Return a parameter-free edge-strength gate for an ablation control.

        The gate is normalized independently for every image. It deliberately
        lacks an estimate of global degradation severity, so it tests whether
        the calibrated need predictor contributes more than a generic edge
        detector while keeping the same PD-MAE features and SFT capacity.
        """
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

    def forward(self, x):
        raw_input = x
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        mae_feat = None
        restoration_need = None
        if self.use_sft:
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
                    # A half-field circular shift preserves the gate histogram
                    # but breaks its spatial correspondence. Unlike a flip, it
                    # also misaligns radially symmetric degradation fields.
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

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(
                self.forward_features(x, mae_feat, restoration_need)
            ) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean

        return x
