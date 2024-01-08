import torch.nn as nn
import torch
from models.MMDT.backbone.MMRT.modules.downsample import get_downsample_layer
from models.MMDT.backbone.MMRT.modules.lstm import DWSLSTM


def NHWC_2_NCHW(x: torch.Tensor):
    assert x.ndim == 4
    return x.permute(0, 3, 1, 2)


class MMRT_Stage(nn.Module):
    def __init__(self, dim_in, stage_dim, spatial_downsample_factor, num_blocks, attn_type, downsamp_type):
        super().__init__()

        if downsamp_type == 'CONV':
            self.downsample = get_downsample_layer(dim_in=dim_in,
                                                   dim_out=stage_dim,
                                                   downsample_factor=spatial_downsample_factor,
                                                   downsample_type=downsamp_type)
        elif downsamp_type == 'PATCH':
            self.downsample = get_downsample_layer(dim_in=dim_in,
                                                   dim_out=stage_dim,
                                                   downsample_factor=spatial_downsample_factor,
                                                   downsample_type=downsamp_type)

        if attn_type == 'SWIN':
            from models.MMDT.backbone.MMRT.modules.attention_block.swin_attention.attention import Attention_Block
        elif attn_type == 'GRID':
            from models.MMDT.backbone.MMRT.modules.attention_block.grid_attention.attention import Attention_Block
        else:
            Attention_Block = None

        blocks = [Attention_Block(dim=stage_dim, skip_first_norm=i == 0) for i in range(num_blocks)]

        self.att_blocks = nn.ModuleList(blocks)

        self.lstm = DWSLSTM(dim=stage_dim,
                            dws_conv=False,
                            dws_conv_only_hidden=True,
                            dws_conv_kernel_size=3,
                            cell_update_dropout=0)

        self.mask_token = None

    def forward(self, x, h_and_c_previous, token_mask):
        x = self.downsample(x)
        if token_mask is not None:
            assert self.mask_token is not None, 'No mask token present in this stage'
            x[token_mask] = self.mask_token

        for blk in self.att_blocks:
            x = blk(x)
        x = NHWC_2_NCHW(x)
        h_c_tuple = self.lstm(x, h_and_c_previous)
        x = h_c_tuple[0]
        return x, h_c_tuple