from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from src.utils.image import load_image, load_image_h_w


class COCODataset(Dataset):
    """
    Lightweight wrapper around a directory of JPEG images from MS COCO.

    Args:
        root:       Path to the image directory (e.g. train2017/).
        image_size: Images are resized to (image_size x image_size).
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int | None,
        image_h: int | None = None,
        image_w: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size

        self.image_h = image_h
        self.image_w = image_w

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
        if self.image_h and self.image_w:
            img = load_image_h_w(self.paths[idx], self.image_h, self.image_w)
        else:
            if not self.image_size:
                raise ValueError(
                    "image_size must be specified if image_h and image_w are not specified"
                )
            img = load_image(self.paths[idx], self.image_size)

        return img


def build_dataloader(
    root: str | Path,
    image_size: int | None,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    distributed: bool = False,
    image_h: int | None = None,
    image_w: int | None = None,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Convenience factory that wires COCODataset into a DataLoader.

    Returns:
        A tuple of (DataLoader, sampler). The sampler is non-None only when
        ``distributed=True``; callers must call ``sampler.set_epoch(epoch)``
        each epoch to ensure proper shuffling.
    """
    if image_h and image_w:
        dataset = COCODataset(root=root, image_h=image_h, image_w=image_w)
    else:
        if not image_size:
            raise ValueError(
                "image_size must be specified if image_h and image_w are not specified"
            )
        dataset = COCODataset(root=root, image_size=image_size)

    sampler: DistributedSampler | None = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # when using BatchNorm2d, batch size should be preferably even
        persistent_workers=num_workers > 0,
    ), sampler
