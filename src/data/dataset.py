from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from src.utils.image import load_image


class COCODataset(Dataset):
    """
    Lightweight wrapper around a directory of JPEG images from MS COCO.

    Args:
        root:       Path to the image directory (e.g. train2017/).
        image_size: Images are resized to (image_size x image_size).
    """

    def __init__(self, root: str | Path, image_size: int = 256) -> None:
        self.root = Path(root)
        self.image_size = image_size

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.paths = sorted(
            f for f in self.root.iterdir() if f.suffix.lower() in extensions
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load, transform, and return a single image tensor.

        Args:
            idx: Integer index.

        Returns:
            Tensor of shape (3, image_size, image_size), normalised.
        """
        return load_image(self.paths[idx], self.image_size)


def build_dataloader(
    root: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Convenience factory that wires COCODataset into a DataLoader."""
    dataset = COCODataset(root=root, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # when using BatchNorm2d, batch size should be preferably even
        persistent_workers=num_workers > 0,
    )
