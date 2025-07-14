# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
# and https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py
# and https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py
# and https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/resampler.py

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from ccdiff_pp.layers import PositionNet, GatedSelfAttentionDense

class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs=64, temperature=100):
        super().__init__()

        self.num_freqs = num_freqs
        self.temperature = temperature

        freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)
        freq_bands = freq_bands[None, None]
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def __call__(self, x):
        x = self.freq_bands * x.unsqueeze(-1)
        return torch.stack((x.sin(), x.cos()), dim=-1).permute(0, 2, 3, 1).reshape(x.shape[0], -1)

# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class SelfAttentionLayer(nn.Module):
    def __init__(self, channels, nhead, dropout=0.0):
        super().__init__() 
        self.norm1 = nn.LayerNorm(channels)
        self.self_attn = nn.MultiheadAttention(channels, nhead, dropout=dropout)

        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                input,
                mask = None,):
        h = self.norm1(input)
        h1 = self.self_attn(query=h, key=h, value=h, attn_mask=mask)[0]
        h = h + self.dropout(h1)
        h = self.norm2(h)
        return h
        

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, use_mask=False):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        n_q = latents.shape[-2]
        n_kv = x.shape[-2]
        
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, d = latents.shape

        q = self.to_q(latents.view(1, -1, d))
        kv_input = torch.cat((x, latents[:-1]), dim=1).view(1, -1, d)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        
        if use_mask:
            block = torch.ones((n_q, n_q + n_kv))
            blocks = [block] * x.size(0)
            mask = torch.block_diag(*blocks)
            additional_rows = torch.ones((n_q, mask.size(1)))
            mask = torch.cat((mask, additional_rows), dim=0).unsqueeze(0)
            # mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(weight.dtype).max
            # mask = repeat(mask, 'b j -> (b h) () j', h=h)
            mask = mask.repeat(self.heads, 1, 1).unsqueeze(0).to(torch.bool).to(weight.device)
            weight.masked_fill_(~mask, max_neg_value)
        
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class CrossAttentionLayer(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.perceiver_fg = PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads)
        self.perceiver_bg = PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads)
    def forward(self, x, latents):
        x_fg, x_bg = x
        latents_fg, latents_bg = latents
        out_fg = self.perceiver_fg(x_fg, latents_fg)
        out_bg = self.perceiver_bg(x_bg, latents_bg)
        return out_fg, out_bg

class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len: int = 257,  # CLIP tokens + CLS token
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        # self.dumm_queries = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        # SelfAttentionLayer(channels=dim, nhead=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x, coherent_queries=None):
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            x = x + pos_emb

        # latents = self.latents.repeat(x.size(0), 1, 1)
        latents = self.latents.repeat(x.size(0), 1, 1) if coherent_queries is None else \
                  torch.cat([self.latents, coherent_queries], dim=1).repeat(x.size(0), 1, 1)

        x = self.proj_in(x)
        # if serial_input is not None:
        #     x = torch.cat([x, serial_input], dim=1)
        
        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        
        # inter_latents = latents.detach()
        latents = self.proj_out(latents)
        # return self.norm_out(latents), inter_latents
        return self.norm_out(latents)

 
class CoSampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len: int = 257,  # CLIP tokens + CLS token
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None

        self.latents = nn.Parameter(torch.randn(3, num_queries, dim) / dim**0.5)    # fg / bg / rel tokens
        # self.dumm_queries = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.point_net = PositionNet(in_dim=embedding_dim, out_dim=embedding_dim)
        
    def forward(self, x_objs, obboxes, x_bg):
        B = x_bg.shape[0]
        obboxes = torch.from_numpy(np.array([obbox[::2] + obbox[1::2] for obbox in obboxes[0]])).float().to(x_objs.device)
        embed_obboxes = self.point_net(obboxes).unsqueeze(1)
        
        x = torch.cat([(x_objs + embed_obboxes), x_bg], dim=0)
        
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            x = x + pos_emb
        
        assert x.size(0) > 1
        fg_latents = self.latents[0].unsqueeze(0).repeat(x.size(0) - 1, 1, 1)
        latents = torch.cat([fg_latents, self.latents[1:]], dim=0)
        
        # latents = self.latents[0].repeat(x.size(0) - 1, 1, 1)

        x = self.proj_in(x)
        
        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x, latents, use_mask=True) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)
        

def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)
