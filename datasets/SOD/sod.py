from pathlib import Path
from datasets.coco import CocoDetection
import datasets.SOD.sod_transforms as T


def make_sod_transforms(setting, backbone, timestep, ts_v):
    scales = [256, 264, 272, 280, 288, 296, 304, 312, 320, 328, 336, 344, 352]
    max_size = 633
    scales2_resize = [280, 300, 350]
    scales2_crop = [256, 350]

    if backbone in ['MMRT_B', 'MMRT_S', 'MMRT_T']:
        ts = timestep
    else:
        ts = -1
        assert backbone in ['MMRT_B', 'MMRT_S', 'MMRT_T']

    if setting == 'train':
        return T.Compose([
            T.Drop(),
            T.Fuse(time_steps=ts, delta=110, steps_v=ts_v),
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([T.RandomResize(scales2_resize),
                           T.RandomSizeCrop(*scales2_crop),
                           T.RandomResize(scales, max_size=max_size),
                           ])),
            T.Normalize([0.4589], [0.1005])
        ])
    if setting == 'val' or 'test':
        return T.Compose([
            T.Fuse(time_steps=ts, delta=110, steps_v=ts_v),
            T.RandomResize([max(scales)], max_size=max_size),
            T.Normalize([0.4589], [0.1005])
        ])
    raise ValueError(f'unknown {setting}')
    

def build(setting, args):
    root = Path(args.sod_path)
    assert root.exists(), f'provided path {root} does not exist'
    PATHS = {
        "train": (root / 'train_motion_blur' / 'events', root / 'train_motion_blur' / 'images', root / 'train_motion_blur' / 'annotations' / 'train_annotations.json'),
        "val": (root / 'val_motion_blur' / 'events', root / 'val_motion_blur' / 'images', root / 'val_motion_blur' / 'annotations' / 'val_annotations.json'),
        "test": (root / 'test_motion_blur' / 'events', root / 'test_motion_blur' / 'images', root / 'test_motion_blur' / 'annotations' / 'test_annotations.json'),
    }
    event_folder, image_folder, ann_file = PATHS[setting]

    dataset = CocoDetection(event_folder, image_folder, ann_file,
                            transforms=make_sod_transforms(setting, args.backbone, args.STEPS, args.STEPS_VALID),
                            return_masks=args.masks,
                            dataset_file=args.dataset_file)
    return dataset