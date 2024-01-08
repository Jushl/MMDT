import torch.nn as nn
import torch
from .layer import LayerNorm, LayerScale
from .drop import DropPath
import math
from typing import Tuple
import torch.nn.functional as F

try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message


def window_partition(x, window_size: Tuple[int, int]):
    B, H, W, C = x.shape
    _assert(H % window_size[0] == 0, f'height ({H}) must be divisible by window ({window_size[0]})')
    _assert(W % window_size[1] == 0, f'width ({W}) must be divisible by window ({window_size[1]})')
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size: Tuple[int, int], img_size: Tuple[int, int]):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


def grid_partition(x, grid_size: Tuple[int, int]):
    B, H, W, C = x.shape
    _assert(H % grid_size[0] == 0, f'height {H} must be divisible by grid {grid_size[0]}')
    _assert(W % grid_size[1] == 0, f'width {W} must be divisible by grid {grid_size[1]}')
    x = x.view(B, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return windows


def grid_reverse(windows, grid_size: Tuple[int, int], img_size: Tuple[int, int]):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, H, W, C)
    return x


class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, channel_last, bias: bool = True):
        super().__init__()

        proj_out_dim = dim_out * 2
        self.proj = nn.Linear(dim_in, proj_out_dim, bias=bias) if channel_last else \
            nn.Conv2d(dim_in, proj_out_dim, kernel_size=1, stride=1, bias=bias)

        self.channel_dim = -1 if channel_last else 1

        self.act_layer = nn.GELU()

    def forward(self, x: torch.Tensor):
        x, gate = torch.tensor_split(self.proj(x), 2, dim=self.channel_dim)
        return x * self.act_layer(gate)


class MLP(nn.Module):
    def __init__(self, dim, channel_last, expansion_ratio, gated=True, bias=True, drop_prob: float = 0.):
        super().__init__()

        inner_dim = int(dim * expansion_ratio)
        if gated:
            inner_dim = math.floor(inner_dim * 2 / 3 / 32) * 32
            proj_in = GLU(dim_in=dim, dim_out=inner_dim, channel_last=channel_last, bias=bias)
        else:
            proj_in = nn.Sequential(
                nn.Linear(in_features=dim, out_features=inner_dim, bias=bias),
                nn.GELU(),
            )
        self.net = nn.Sequential(
            proj_in,
            nn.Dropout(p=drop_prob),
            nn.Linear(in_features=inner_dim, out_features=dim, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, dim_head=32, bias=True):
        super().__init__()
        self.num_heads = dim // dim_head
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        restore_shape = x.shape[:-1]

        q, k, v = self.qkv(x).view(B, -1, self.num_heads, self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(restore_shape + (-1,))
        x = self.proj(x)
        return x


class Attn_(nn.Module):
    def __init__(self, dim, partition_type, skip_first_norm, ls_init_value=1e-5, dim_head=32, drop_path=0.0,
                 partition_size=(8, 10)):
        super().__init__()

        self.partition_type = partition_type
        self.partition_size = partition_size

        partition_size = tuple(partition_size)
        assert len(partition_size) == 2

        self.norm1 = nn.Identity() if skip_first_norm else nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, dim_head=dim_head, bias=True)
        self.ls1 = LayerScale(dim=dim, init_values=ls_init_value) if ls_init_value > 0 else nn.Identity()
        self.drop_path1 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.ls2 = LayerScale(dim=dim, init_values=ls_init_value) if ls_init_value > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()

        self.mlp = MLP(dim=dim,
                       channel_last=True,
                       expansion_ratio=4,
                       gated=False,
                       bias=True,
                       drop_prob=0.0)

    def _partition_attn(self, x):
        img_size = x.shape[1:3]

        H, W = img_size
        H_win, W_win = self.partition_size
        pad_l = pad_t = 0
        pad_r = (W_win - W % W_win) % W_win
        pad_b = (H_win - H % H_win) % H_win
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.partition_type == 'WINDOW':
            partitioned = window_partition(x, self.partition_size)
        elif self.partition_type == 'GRID':
            partitioned = grid_partition(x, self.partition_size)
        else:
            partitioned = None
            assert self.partition_type in ['WINDOW', 'GRID']

        partitioned = self.self_attn(partitioned)

        if self.partition_type == 'WINDOW':
            x = window_reverse(partitioned, self.partition_size, (Hp, Wp))
        elif self.partition_type == 'GRID':
            x = grid_reverse(partitioned, self.partition_size, (Hp, Wp))
        else:
            assert self.partition_type in ['WINDOW', 'GRID']

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        return x

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self._partition_attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class Attention_Block(nn.Module):
    def __init__(self, dim, skip_first_norm):
        super().__init__()

        self.att_window = Attn_(dim=dim, partition_type='WINDOW', skip_first_norm=skip_first_norm)
        self.att_grid = Attn_(dim=dim, partition_type='GRID', skip_first_norm=False)

    def forward(self, x):
        x = self.att_window(x)
        x = self.att_grid(x)
        return x

