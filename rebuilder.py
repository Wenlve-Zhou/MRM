import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from mmseg.ops import resize
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
import numpy as np
from functools import partial
from timm.layers import LayerNorm2d
import torch.nn.functional as F

def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
            
def generate_mask(shape, device, mask_ratio=0.40):
    B, _, H, W = shape
    mshape = B, 1, H, W
    input_mask = torch.rand(mshape, device=device)
    input_mask = (input_mask > mask_ratio).float()
    return input_mask

class Rebuilder(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        print(cfg)
        self.decoder_type = cfg["single_scale_head"] if "single_scale_head" in cfg.keys() else cfg["type"]
        self.in_channels = cfg["in_channels"]
        self.in_index = cfg["in_index"]
        if self.decoder_type == "DAFormerHead":
            self.in_spatial = [128, 64, 32, 16]
            self.scaled_size = 16
            self.channel_dim = self.in_channels[-1]
        elif self.decoder_type == "DLV2Head":
            self.in_spatial = 64
            self.scaled_size = 16
            self.channel_dim = self.in_channels

        self.embed_dim = 512
        self.mask_ratio = 0.40
        self.stride = [2, 2, 1, 1]

        self.embedding = Embedding(self.channel_dim, self.embed_dim, self.scaled_size)
        if self.decoder_type == "DAFormerHead":
            self.transformer = Transformer(spatial_dim=self.scaled_size,embed_dim=self.embed_dim)
            self.projector = {}
            for i in range(len(self.in_spatial)):
                self.projector["projector" + str(i+1)] = Projector(in_channels=self.embed_dim, out_channels=self.in_channels[i],scale=self.in_spatial[i] // self.scaled_size)
            self.projector = nn.ModuleDict(self.projector)

        elif self.decoder_type == "DLV2Head":
            self.transformer = Transformer(spatial_dim=self.scaled_size, embed_dim=self.embed_dim,fpn_style="single-scale")
            self.projector = Projector(in_channels=self.embed_dim, out_channels=self.in_channels,scale=self.in_spatial // self.scaled_size)

    def forward(self, inputs):
        outputs = []
        embedding = self.embedding(inputs[-1])
        mask = generate_mask(shape=embedding.shape,device=embedding.device, mask_ratio=self.mask_ratio)
        transformer_out = self.transformer(embedding, mask)
        if self.decoder_type =="DAFormerHead":
            for i in range(len(self.in_channels)):
                out = self.projector["projector" + str(i+1)](transformer_out, inputs[i], mask)
                outputs.append(out)
        else:
            out = self.projector(transformer_out, inputs[-1], mask)
            outputs = [out] * 4
        return outputs

    def extra_repr(self):
        return f'mask_ratio={round(self.mask_ratio,3):0.3f}'

class Projector(nn.Module):
    def __init__(self,  in_channels, out_channels, scale=1.0, mid_channels=256):
        super().__init__()
        if scale == 1.0:
            self.fpn = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),)
            
        elif scale == 2.0:
            self.fpn = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels//2, 4, 2, 1),
                nn.Conv2d(in_channels//2, out_channels, 1),
            )

        elif scale == 4.0:
            self.fpn = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels//2, 4, 2, 1),
                LayerNorm2d(in_channels//2),
                nn.GELU(),
                nn.ConvTranspose2d(in_channels//2, in_channels//4, 4, 2, 1),
                nn.Conv2d(in_channels//4, out_channels, 1),
            )

        elif scale == 8.0:
            self.fpn = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels//2, 4, 2, 1),
                LayerNorm2d(in_channels//2),
                nn.GELU(),
                nn.ConvTranspose2d(in_channels//2, in_channels//4, 4, 2, 1),
                LayerNorm2d(in_channels//4),
                nn.GELU(),
                nn.ConvTranspose2d(in_channels//4, in_channels//4, 4, 2, 1),
                nn.Conv2d(in_channels//4, out_channels, 1),

            )
        self.norm = LayerNorm2d(out_channels)

    def forward(self, attn, src, mask):
        mask = resize(
            input=mask.float(),
            size=src.shape[2:],
            mode='nearest')
        offset = self.pred_offset(attn)
        x = src * mask + offset * (1-mask)
        return x

    def pred_offset(self, x):
        x = self.fpn(x)
        x = self.norm(x)
        return x

class Transformer(nn.Module):
    def __init__(self, spatial_dim, embed_dim=512, num_heads=16, mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm,eps=1e-5), depth=2, fpn_style="multi-scale", fpn_dim=256):
        super().__init__()
        # Patch Embedding & Blocks Construction
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim, 1, 1), requires_grad=True)
        nn.init.normal_(self.mask_token,std=.02)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.input_proj = nn.Identity()
        self.norm = norm_layer(embed_dim)
        self.apply(_init_weights)

        self.fpn_style = fpn_style

        # Position Embedding
        self.num_patches = spatial_dim ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            int(self.num_patches ** .5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x, mask):
        # Remove the Source Feature in Mixture and Add Learnable Token in the Position
        B, C, H, W = x.size()
        mask_token = self.mask_token.repeat(B, 1, H, W)
        x = x * mask + (1 - mask) * mask_token
        x = x.flatten(2).transpose(1, 2)

        # Contextual Aggregation using Attention Mechanism
        x = x + self.pos_embed
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        B, N, C = x.size()
        H = W = int(N ** 0.5)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class Embedding(nn.Module):
    def __init__(self, channel_dim, embed_dim, scaled_size):
        super().__init__()
        self.linear_embed = nn.Linear(channel_dim,embed_dim)
        self.scaled_size = (scaled_size,scaled_size)
        self.apply(_init_weights)

    def forward(self,x):
        B, C, H, W = x.size()
        x = self.linear_embed(x.flatten(2).transpose(1, 2).contiguous())
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        if (H, W) != self.scaled_size:
            x = resize(
                input=x,
                size=self.scaled_size,
                mode='bilinear',
                align_corners=False)
        return x

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    returns:
        pos_embed: [grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)  # order of meshgrid is very important for indexing as [h, w]

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)  # (H*W, D/2)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    embed_dim: output dimension for each position
    grid_size: int of the grid length
    returns:
        pos_embed: [grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size, embed_dim] (w/ cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    returns: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
