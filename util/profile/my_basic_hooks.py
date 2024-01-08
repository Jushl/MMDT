import logging
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
'''
float32——MC：3.7pJ
       ——AC：0.9pJ
int32——MC：3.1pJ
     ——AC：0.1pJ
'''

def l_prod(in_list):
    res = 1
    for _ in in_list:
        res *= _
    return res


def l_sum(in_list):
    res = 0
    for _ in in_list:
        res += _
    return res


def zero_ops(m, x, y):
    flops = torch.DoubleTensor([int(0)])
    m.total_ops += flops


def count_convNd(m: _ConvNd, x, y: torch.Tensor):
    x = x[0]
    input_size = list(x.shape)
    output_size = list(y.shape)
    kernel_size = list(m.weight.shape)
    groups = m.groups

    in_c = input_size[1]
    g = groups  # 一直是1

    output_times = l_prod(output_size)  # 1*32*240*304
    kernel_times = l_prod(kernel_size[2:])  # 3*3

    flops = output_times * (in_c // g) * kernel_times  # l_prod与numel()效果相同

    m.total_ops += flops



def count_normalization(m: nn.modules.batchnorm._BatchNorm, x, y):
    x = x[0]
    input_size = x.numel()  # x.shape的乘积，比如[1,32,240,304].numel()=1*32*240*304
    flops = torch.DoubleTensor([2 * input_size])
    if (getattr(m, 'affine', False) or getattr(m, 'elementwise_affine', False)):
        flops *= 2

    m.total_ops += flops



def count_softmax(m, x, y):
    x = x[0]
    nfeatures = x.size()[m.dim]
    batch_size = x.numel() // nfeatures

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)
    flops = torch.DoubleTensor([int(total_ops)])


    m.total_ops += flops



def count_linear(m, x, y):
    in_feature = m.in_features
    num_elements = y.numel()
    flops = torch.DoubleTensor([int(in_feature * num_elements)])

    m.total_ops += flops



def count_avgpool(m, x, y):
    input_size = y.numel()
    flops = torch.DoubleTensor([int(input_size)])

    m.total_ops += flops



