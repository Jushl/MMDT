from torchvision.datasets.vision import VisionDataset
import os
import os.path
from typing import Any, Callable, Optional, Tuple, List


class SodDetection(VisionDataset):
    def __init__(self, root_evt, root_img: str, annFile: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None,):
        super().__init__(root_img, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.evt = root_evt

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        event_path = os.path.join(self.root, path)
        return event_path

    def _load_target(self, id) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        event_name = os.path.join(os.path.basename(image).split('.')[0] + '.npy')
        event = os.path.join(self.evt, event_name)
        # if self.transforms is not None:
        #     image, target = self.transforms(image, target)

        return image, event, target

    def __len__(self) -> int:
        return len(self.ids)
