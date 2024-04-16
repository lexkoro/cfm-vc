# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math

import torch
from einops import rearrange
from torch import nn


class ConditionalGroupNorm(nn.Module):
    def __init__(self, groups, normalized_shape, context_dim):
        super().__init__()
        self.norm = nn.GroupNorm(groups, normalized_shape, affine=False)
        self.context_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(context_dim, 2 * normalized_shape)
        )
        self.context_mlp[1].weight.data.zero_()
        self.context_mlp[1].bias.data.zero_()

    def forward(self, x, context):
        context = self.context_mlp(context)
        ndims = " 1" * len(x.shape[2:])
        context = rearrange(context, f"b c -> b c{ndims}")

        scale, shift = context.chunk(2, dim=1)
        x = self.norm(x) * (scale + 1.0) + shift
        return x


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(nn.Module):
    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)  # kernel=3, stride=2, padding=1.

    def forward(self, x):
        return self.conv(x)


class Rezero(nn.Module):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) * self.g


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, context_dim=None):
        super(Block, self).__init__()
        self.conv2d = torch.nn.Conv2d(dim, dim_out, 3, padding=1)
        if context_dim is None:
            self.norm = torch.nn.GroupNorm(groups, dim_out)
        else:
            self.norm = ConditionalGroupNorm(groups, dim_out, context_dim)
        self.mish = Mish()

    def forward(self, x, mask, utt_emb=None):
        output = self.conv2d(x * mask)
        if utt_emb is not None:
            output = self.norm(output, utt_emb)
        else:
            output = self.norm(output)
        output = self.mish(output)
        return output * mask


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8, context_dim=512):
        super(ResnetBlock, self).__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block(dim, dim_out, groups=groups, context_dim=context_dim)
        self.block2 = Block(dim_out, dim_out, groups=groups, context_dim=context_dim)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask, time_emb, utt_emb=None):
        h = self.block1(x, mask, utt_emb)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask, utt_emb)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(
            dim, hidden_dim * 3, 1, bias=False
        )  # NOTE: 1x1 conv
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class FiLM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Convolution layers for gain and bias
        self.cond_conv = nn.Conv2d(
            in_channels, out_channels * 2, kernel_size=3, padding=1
        )

    def forward(self, x, condition):
        # Assuming x and condition have the same shape [batch, channels, height, width]
        gain, bias = self.cond_conv(condition).chunk(2, dim=1)

        return x * gain + bias


class ConditionalLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, cond_emb_dim=512):
        super(ConditionalLinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_query = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_cond = nn.Conv2d(
            cond_emb_dim, hidden_dim * 2, 1, bias=False
        )  # Speaker embedding transformation
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x, cond_emb=None):
        b, c, h, w = x.shape

        # Transform input and speaker embedding
        q = self.to_query(x)
        speaker_embedding = self.to_cond(cond_emb)

        # Separate q, k, v
        q = rearrange(q, "b (heads c) h w -> b heads c (h w)", heads=self.heads)

        # Modify k and v using speaker embedding
        k_utt_emb, v_utt_emb = rearrange(
            speaker_embedding,
            "b (kv heads c) h w -> kv b heads c (h w)",
            heads=self.heads,
            kv=2,
        )

        k_utt_emb = k_utt_emb.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k_utt_emb, v_utt_emb)
        out = torch.einsum("bhde,bhdn->bhen", context, q)

        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GradLogPEstimator2d(nn.Module):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4),
        groups=8,
        spk_emb_dim=64,
        pe_scale=1000,
    ):
        super(GradLogPEstimator2d, self).__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.spk_emb_dim = spk_emb_dim
        self.pe_scale = pe_scale

        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4), Mish(), torch.nn.Linear(dim * 4, dim)
        )

        dims = [2, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                torch.nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                        ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                        Residual(Rezero(LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else torch.nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(
                torch.nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                        Residual(Rezero(LinearAttention(dim_in))),
                        Upsample(dim_in),
                    ]
                )
            )

        self.final_block = Block(dim, dim)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        # x, mu: [B, 80, L], t: [B, ], mask: [B, 1, L]

        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)  # [B, 64]

        # stack mu and x
        x = torch.stack([mu, x], 1)  # [B, 2, 80, L]
        mask = mask.unsqueeze(1)  # [B, 1, 1, L]

        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t, spks)  # [B, 64, 80, L]
            x = resnet2(x, mask_down, t, spks)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t, spks)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t, spks)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t, spks)
            x = resnet2(x, mask_up, t, spks)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return (output * mask).squeeze(1)
