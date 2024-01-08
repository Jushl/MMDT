import torch.nn as nn
import torch.nn.functional as F
import torch


class LayerNorm(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float=1e-5, inplace: bool=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma
        return x.mul_(gamma) if self.inplace else x * gamma



