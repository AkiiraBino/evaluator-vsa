import os
import pickle

import cv2
from cv2.typing import MatLike
from loguru import logger
from torch import HalfTensor, Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2

from src.utils import get_file_content


class YoloDetectorDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        transform: v2.Transform | None = None,
    ) -> None:
        self.dataset_dir: str = dataset_dir
        self.imgs_dir, self.lbls_dir = (
            os.path.join(dataset_dir, i)
            for i in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, i))
        )

        self.img_pths: list[str] = [
            os.path.join(self.imgs_dir, _dir)
            for _dir in os.listdir(self.imgs_dir)
        ]

        self.lbl_pths: list[str] = [
            os.path.join(self.lbls_dir, _dir)
            for _dir in os.listdir(self.lbls_dir)
        ]

        self.transform: v2.Transform | None = transform
        logger.info(
            f"images dir {self.imgs_dir} | label dir {self.lbls_dir} OK"
        )

    def __len__(self) -> int:
        return len(self.img_pths)

    def __getitem__(
        self, index
    ) -> tuple[Tensor, bytes | None]:
        img: MatLike = cv2.imread(self._get_full_pth(index))
        img = cv2.resize(img, (640,640))
        img_tensor: Tensor = HalfTensor(img).permute(2, 0, 1) / 255
        if self.transform:
            img_tensor = self.transform(img_tensor)

        file_content = None
        if bool(self.lbl_pths):
            file_content = get_file_content(self.lbl_pths[index])
        file_content_pickle: bytes = pickle.dumps(file_content)
        logger.info(f"get {index} | OK")
        return (img_tensor, file_content_pickle)

    def _get_full_pth(self, idx) -> str:
        return os.path.join(self.dataset_dir, self.img_pths[idx])

