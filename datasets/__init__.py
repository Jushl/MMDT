import torch.utils.data
from .SOD import build as build_sod


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'SOD':
        return build_sod(image_set, args)

    raise ValueError(f'dataset {args.dataset_file} not supported')
