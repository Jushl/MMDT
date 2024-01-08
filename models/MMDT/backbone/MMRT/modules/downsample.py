import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models._efficientnet_blocks import SqueezeExcite


def NHWC_2_NCHW(x: torch.Tensor):
    assert x.ndim == 4
    return x.permute(0, 3, 1, 2)


def NCHW_2_NHWC(x: torch.Tensor):
    assert x.ndim == 4
    return x.permute(0, 2, 3, 1)


class PatchMerging(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        assert 2 * dim == dim_out, "dim in scale mismatch dim out"

        # self.SE_block = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(1, 1)),
        #                               SqueezeExcite(in_chs=dim, rd_ratio=0.25),
        #                               nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(1, 1))
        #                               )

    def forward(self, x):
        # x = self.SE_block(x)

        B, C, H, W = x.shape
        pad_l = pad_t = 0
        pad_r = (2 - W % 2) % 2
        pad_b = (2 - H % 2) % 2

        x = NCHW_2_NHWC(x)
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.reduction(x)
        x = x.view(B, Hp // 2, Wp // 2, 2 * C)

        x = NHWC_2_NCHW(x)
        return x


class DownsampleBase(nn.Module):
    def __init__(self):
        super().__init__()


class ConvDownsampling(DownsampleBase):
    def __init__(self, dim_in, dim_out, downsample_factor, downsample_type, overlap=True):
        super().__init__()
        if overlap:
            kernel_size = (downsample_factor - 1) * 2 + 1
            padding = kernel_size // 2
        else:
            kernel_size = downsample_factor
            padding = 0

        if downsample_factor == 4:
            self.conv = nn.Conv2d(in_channels=dim_in,
                                        out_channels=dim_out,
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        stride=downsample_factor,
                                        bias=False)
        elif downsample_factor == 2:
            if downsample_type == 'CONV':
                self.conv = nn.Conv2d(in_channels=dim_in,
                                            out_channels=dim_out,
                                            kernel_size=kernel_size,
                                            padding=padding,
                                            stride=downsample_factor,
                                            bias=False)
            if downsample_type == 'PATCH':
                self.conv = PatchMerging(dim=dim_in, dim_out=dim_out)

        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x: torch.Tensor):

        x = self.conv(x)
        x = NCHW_2_NHWC(x)
        x = self.norm(x)
        return x


def get_downsample_layer(dim_in, dim_out, downsample_factor, downsample_type) -> DownsampleBase:
    return ConvDownsampling(dim_in=dim_in, dim_out=dim_out, downsample_factor=downsample_factor, downsample_type=downsample_type)
