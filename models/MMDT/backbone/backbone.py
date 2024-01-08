from torch import nn
from typing import List
from util.misc import NestedTensor
from models.MMDT.backbone.position_encoding import build_position_encoding
from models.MMDT.backbone.MMRT.mmrt import build_mmrt
import torch.nn.functional as F
import torch
from collections import OrderedDict


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, option):
        super().__init__(backbone, position_embedding)
        self.option = option

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        B, C_, H, W = x.shape
        x = x.view(B, -1, 3, H, W).permute(0, 2, 3, 4, 1).permute(4, 0, 1, 2, 3)
        prev_states = None

        for i in range(len(x)):
            input = x[i]
            outs, states = self[0](x=input, prev_states=prev_states)
            prev_states = states

        if self.option == [0, 1, 2, 3]:
            outs = OrderedDict(outs)
        elif self.option == [1, 2, 3]:
            outs = {k: v for k, v in list(outs.items())[1:]}
            outs = OrderedDict(outs)
        elif self.option == [2, 3]:
            outs = {k: v for k, v in list(outs.items())[2:]}
            outs = OrderedDict(outs)
        else:
            outs = None
            assert outs in [[2, 3], [1, 2, 3], [0, 1, 2, 3]]

        outs_dict = {}
        for idx, out_i in outs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=out_i.shape[-2:]).to(torch.bool)[0]
            outs_dict[idx] = NestedTensor(out_i, mask)
        xs = outs_dict

        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)

    return_interm_indices = args.return_interm_indices
    assert return_interm_indices in [[2, 3], [1, 2, 3], [0, 1, 2, 3]]

    if args.backbone in ['MMRT_B', 'MMRT_S', 'MMRT_T']:
        backbone = build_mmrt(args)
        bb_num_channels = backbone.stage_dims[4 - len(return_interm_indices):]

    else:
        raise NotImplementedError("Unknown backbone {}".format(args.backbone))

    model = Joiner(backbone, position_embedding, args.return_interm_indices)
    model.num_channels = bb_num_channels 
    assert isinstance(bb_num_channels, List), "bb_num_channels is expected to be a List but {}".format(type(bb_num_channels))
    return model


if __name__ == '__main__':
    import argparse
    def parser():
        parser = argparse.ArgumentParser(description='Multi Model Recurrent Transformer Net', add_help=False)
        parser.add_argument('--embed_dim', default=64, type=int)
        parser.add_argument('--name', default='MMRT')
        parser.add_argument('--num_blocks', default=[1, 1, 1, 1])
        parser.add_argument('--dim_multiplier', default=[4, 8, 16, 32])
        parser.add_argument('--enable_masking', default=False)
        parser.add_argument('--hidden_dim', default=256)
        parser.add_argument('--position_embedding', default='sine')
        parser.add_argument('--backbone', default='MMRT_B')
        parser.add_argument('--return_interm_indices', default=[0, 1, 2, 3])
        parser.add_argument('--pe_temperatureH', default=20)
        parser.add_argument('--pe_temperatureW', default=20)
        return parser

    parser = argparse.ArgumentParser('Multi Model Recurrent Transformer Net', parents=[parser()])
    config = parser.parse_args()
    backbone = build_backbone(config)
    x = torch.randn((4, 6, 256, 341))
    prev_states = None
    xs, pos = backbone(x)
