import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    grid = grid.flatten()  # (M,)
    out = np.einsum('m,d->md', grid, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class MAEEncoder(nn.Module):
    def __init__(self, img_size=256, patch_size=8, in_chans=3,
                 embed_dim=384, depth=12, num_heads=6,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.patch_embed.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)

    def forward(self, x, mask_indices=None):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, 1:, :]
        
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

class MAEDecoder(nn.Module):
    def __init__(self, num_patches, patch_size=8, in_chans=3,
                 embed_dim=192, depth=8, num_heads=6,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, encoder_dim=384):
        super().__init__()

        self.num_patches = num_patches
        self.patch_size = patch_size
        
        self.decoder_embed = nn.Linear(encoder_dim, embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_size ** 2 * in_chans, bias=True)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Sine-cosine positional embedding
        pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.xavier_uniform_(self.decoder_embed.weight)
        torch.nn.init.normal_(self.decoder_pred.weight, std=.02)

    def forward(self, x, mask_indices=None):
        # PD-MAE: Decoder simply processes the full sequence from encoder
        x = self.decoder_embed(x)

        # Add pos embed (cls token is at index 0)
        x = x + self.decoder_pos_embed

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Predict pixels
        x = self.decoder_pred(x)
        x = x[:, 1:, :] # Remove cls token

        return x

class PDMAE(nn.Module):
    def __init__(self, img_size=256, patch_size=8, encoder_dim=384, decoder_dim=192):
        super().__init__()
        self.encoder = MAEEncoder(img_size=img_size, patch_size=patch_size, embed_dim=encoder_dim)
        self.decoder = MAEDecoder(num_patches=self.encoder.num_patches, patch_size=patch_size, 
                                  embed_dim=decoder_dim, encoder_dim=encoder_dim)

    def forward(self, x, mask_indices=None):
        # mask_indices not used in architecture anymore, only in Loss
        latent = self.encoder(x)
        pred = self.decoder(latent)
        return pred


class RestorationNeedPDMAE(PDMAE):
    """PD-MAE with a token-wise restoration-need prediction head.

    The head predicts a scalar in ``[0, 1]`` for every image token.  Its
    training target is computed from the paired LQ/HR correction magnitude,
    so it stays aligned with the actual corruption even when a generated
    partial-degradation mask has feathered boundaries.
    """

    def __init__(self, img_size=256, patch_size=8, encoder_dim=384,
                 decoder_dim=192):
        super().__init__(img_size=img_size, patch_size=patch_size,
                         encoder_dim=encoder_dim, decoder_dim=decoder_dim)
        self.need_head = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, 1),
        )
        nn.init.zeros_(self.need_head[-1].weight)
        nn.init.zeros_(self.need_head[-1].bias)

    def forward(self, x, mask_indices=None, return_need=False):
        latent = self.encoder(x)
        pred = self.decoder(latent)
        if not return_need:
            return pred

        need_logits = self.need_head(latent[:, 1:, :]).squeeze(-1)
        return pred, need_logits
