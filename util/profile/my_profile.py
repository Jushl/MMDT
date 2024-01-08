from util.profile.my_basic_hooks import *
import torch.nn as nn
import torch

default_dtype = torch.float64

register_hooks = {
    nn.Conv2d: count_convNd,
    nn.BatchNorm2d: count_normalization,
    nn.LayerNorm: count_normalization,
    nn.Softmax: count_softmax,
    nn.ReLU: zero_ops,
    nn.MaxPool2d: zero_ops,
    nn.Linear: count_linear,
    nn.Dropout: zero_ops,
    nn.Sequential: zero_ops,
}