import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, num_groups=32):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()

        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        else:
            self.time_mlp = None

    def forward(self, x, t_emb=None):
        h = self.conv1(self.act(self.norm1(x)))

        if self.time_mlp is not None and t_emb is not None:
            time_emb = self.time_mlp(t_emb)
            h = h + time_emb.unsqueeze(-1).unsqueeze(-1)

        h = self.conv2(self.act(self.norm2(h)))

        return h + self.skip_connection(x)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = heads * dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None):
        b, seq_len, _ = x.shape
        h = self.heads
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: t.view(b, -1, h, t.shape[-1] // h).transpose(1, 2), (q, k, v))
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(b, seq_len, -1)
        return self.to_out(out)

class AttentionBlock(nn.Module):
    def __init__(self, channels, context_dim, heads=8, dim_head=64, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.self_attn = CrossAttention(channels, channels, heads=heads, dim_head=dim_head)
        self.cross_attn = CrossAttention(channels, context_dim, heads=heads, dim_head=dim_head)

    def forward(self, x, context=None):
        residual = x

        x = self.norm(x)
        b, c, h, w = x.shape
        x = x.view(b, c, -1).transpose(1, 2)

        x = self.self_attn(x) + x

        if context is not None:
            x = self.cross_attn(x, context=context) + x

        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x + residual

class CustomUNet(nn.Module):
    def __init__(self, in_channels=4, model_channels=416, out_channels=4, context_dim=1024, num_groups=32):
        super().__init__()

        time_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        self.down_block1 = nn.ModuleList([
            ResidualBlock(model_channels, model_channels, time_emb_dim=time_dim, num_groups=num_groups),
            AttentionBlock(model_channels, context_dim=context_dim, num_groups=num_groups)
        ])
        self.down_block2 = nn.ModuleList([
            nn.MaxPool2d(2),
            ResidualBlock(model_channels, model_channels * 2, time_emb_dim=time_dim, num_groups=num_groups),
            AttentionBlock(model_channels * 2, context_dim=context_dim, num_groups=num_groups)
        ])
        self.down_block3 = nn.ModuleList([
            nn.MaxPool2d(2),
            ResidualBlock(model_channels * 2, model_channels * 4, time_emb_dim=time_dim, num_groups=num_groups),
            AttentionBlock(model_channels * 4, context_dim=context_dim, num_groups=num_groups)
        ])
        self.down_block4 = nn.ModuleList([
            nn.MaxPool2d(2),
            ResidualBlock(model_channels * 4, model_channels * 8, time_emb_dim=time_dim, num_groups=num_groups),
            AttentionBlock(model_channels * 8, context_dim=context_dim, num_groups=num_groups)
        ])

        self.mid_block = nn.ModuleList([
            ResidualBlock(model_channels * 8, model_channels * 8, time_emb_dim=time_dim, num_groups=num_groups),
            AttentionBlock(model_channels * 8, context_dim=context_dim, num_groups=num_groups),
            ResidualBlock(model_channels * 8, model_channels * 8, time_emb_dim=time_dim, num_groups=num_groups)
        ])

        self.upsample_op = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.up_block1 = nn.ModuleList([
            ResidualBlock(12 * model_channels, 4 * model_channels, time_emb_dim=time_dim, num_groups=num_groups),
            AttentionBlock(4 * model_channels, context_dim=context_dim, num_groups=num_groups)
        ])
        self.up_block2 = nn.ModuleList([
            ResidualBlock(6 * model_channels, 2 * model_channels, time_emb_dim=time_dim, num_groups=num_groups),
            AttentionBlock(2 * model_channels, context_dim=context_dim, num_groups=num_groups)
        ])
        self.up_block3 = nn.ModuleList([
            ResidualBlock(3 * model_channels, model_channels, time_emb_dim=time_dim, num_groups=num_groups),
            AttentionBlock(model_channels, context_dim=context_dim, num_groups=num_groups)
        ])
        self.up_block4 = nn.ModuleList([
            ResidualBlock(2 * model_channels, model_channels, time_emb_dim=time_dim, num_groups=num_groups),
            AttentionBlock(model_channels, context_dim=context_dim, num_groups=num_groups)
        ])

        self.final_norm = nn.GroupNorm(num_groups, model_channels)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(model_channels, out_channels, kernel_size=1)

    def forward(self, x, timestep, context):
        t = self.time_mlp(timestep)

        h = self.conv_in(x)
        skips = [h]

        h = self.down_block1[0](h, t)
        h = self.down_block1[1](h, context)
        skips.append(h)

        h = self.down_block2[0](h)
        h = self.down_block2[1](h, t)
        h = self.down_block2[2](h, context)
        skips.append(h)

        h = self.down_block3[0](h)
        h = self.down_block3[1](h, t)
        h = self.down_block3[2](h, context)
        skips.append(h)

        h = self.down_block4[0](h)
        h = self.down_block4[1](h, t)
        h = self.down_block4[2](h, context)

        h = self.mid_block[0](h, t)
        h = self.mid_block[1](h, context)
        h = self.mid_block[2](h, t)

        h = self.upsample_op(h)
        skip_d3 = skips.pop()
        h = torch.cat([h, skip_d3], dim=1)
        h = self.up_block1[0](h, t)
        h = self.up_block1[1](h, context)

        h = self.upsample_op(h)
        skip_d2 = skips.pop()
        h = torch.cat([h, skip_d2], dim=1)
        h = self.up_block2[0](h, t)
        h = self.up_block2[1](h, context)

        h = self.upsample_op(h)
        skip_d1 = skips.pop()
        h = torch.cat([h, skip_d1], dim=1)
        h = self.up_block3[0](h, t)
        h = self.up_block3[1](h, context)

        skip_initial = skips.pop()
        h = torch.cat([h, skip_initial], dim=1)
        h = self.up_block4[0](h, t)
        h = self.up_block4[1](h, context)

        h = self.final_act(self.final_norm(h))
        return self.final_conv(h)