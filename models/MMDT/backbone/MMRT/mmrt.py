import torch.nn as nn
from typing import Tuple
from models.MMDT.backbone.MMRT.mmrt_stage import MMRT_Stage


class BaseDetector(nn.Module):
    def get_stage_dims(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        raise NotImplementedError

    def get_strides(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        raise NotImplementedError


class MMRT_Backbone(BaseDetector):
    def __init__(self, config):
        super().__init__()

        embed_dim = config.embed_dim
        dim_multiplier_per_stage = tuple(config.dim_multiplier)
        num_blocks_per_stage = tuple(config.num_blocks)
        attn_type = config.attention_type
        downsamp_type = config.downsamp_type
        num_stages = len(num_blocks_per_stage)

        input_dim = 3
        stride = 1

        self.stage_dims = [embed_dim * x for x in dim_multiplier_per_stage]

        self.stages = nn.ModuleList()
        self.strides = []

        for stage_idx, num_blocks in enumerate(num_blocks_per_stage):
            spatial_downsample_factor = 4 if stage_idx == 0 else 2
            stage_dim = self.stage_dims[stage_idx]

            stage = MMRT_Stage(dim_in=input_dim,
                               stage_dim=stage_dim,
                               spatial_downsample_factor=spatial_downsample_factor,
                               num_blocks=num_blocks,
                               attn_type=attn_type,
                               downsamp_type=downsamp_type)

            stride = stride * spatial_downsample_factor
            self.strides.append(stride)

            input_dim = stage_dim
            self.stages.append(stage)

        self.num_stages = num_stages

    def forward(self, x, prev_states=None, token_mask=None):
        if prev_states is None:
            prev_states = [None] * self.num_stages
        states = list()
        output = {}
        for stage_idx, stage in enumerate(self.stages):
            x, state = stage(x, prev_states[stage_idx], token_mask if stage_idx == 0 else None)

            states.append(state)
            stage_number = stage_idx
            output[stage_number] = x

        return output, states


def build_mmrt(config):
    if config.backbone in ['MMRT_B', 'MMRT_S', 'MMRT_T']:
        return MMRT_Backbone(config)
