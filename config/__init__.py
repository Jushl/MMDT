import argparse
from util.slconfig import DictAction


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, default='config/MMDT/MMDT_B.py')
    parser.add_argument('--backbone', default='MMRT_B')
    parser.add_argument('--output_dir', default='logs/MMDT/', help='path where to save, empty for no saving')
    parser.add_argument('--attention_type', default='GRID')  # SWIN or GRID
    parser.add_argument('--downsamp_type', default='CONV')  # CONV or PATCH

    parser.add_argument('--dataset_file', default='SOD')
    parser.add_argument('--sod_path', type=str, default='your SOD path')
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')
    parser.add_argument('--cda', default=False)
    parser.add_argument('--options', nargs='+', action=DictAction, help='')

    parser.add_argument('--pretrain_model_path', help='resume from checkpoint')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--note', default='', help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int, help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true', help="Train with mixed precision")

    return parser
