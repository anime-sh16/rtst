from pathlib import Path
import logging
import os
import torch

from PIL import Image, ImageOps
from torchvision.transforms import v2
from torchvision.utils import save_image as tv_save_image

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# These tensors lives on CPU, never moved
IMAGENET_MEAN_RESHAPED = torch.tensor(IMAGENET_MEAN).reshape(-1, 1, 1)
IMAGENET_STD_RESHAPED = torch.tensor(IMAGENET_STD).reshape(-1, 1, 1)

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)


def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transform(image_size: int | None) -> v2.Compose:
    """
    Standard content-image transform: resize to square, convert to tensor, normalise.

    Args:
        image_size: Target spatial resolution (both H and W).

    Returns:
        A composed torchvision transform.
    """
    if image_size:
        preprocess = v2.Compose(
            [
                v2.Resize(image_size),
                v2.CenterCrop(image_size),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        preprocess = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return preprocess


def build_transform_h_w(h: int, w: int) -> v2.Compose:
    """
    Content-image transform: resize shortest edge to max(h, w), center crop to hxw,
    convert to tensor, and normalise.

    Args:
        h: Target height.
        w: Target width.

    Returns:
        A composed torchvision transform.
    """

    preprocess = v2.Compose(
        [
            v2.Resize(max(h, w)),
            v2.CenterCrop((h, w)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return preprocess


def _round_down_to_multiple(value: int, multiple: int) -> int:
    """Round down to the nearest multiple."""
    return (value // multiple) * multiple


def load_image(
    path: str | Path,
    size: int | None = None,
    keep_aspect: bool = False,
) -> torch.Tensor:
    """
    Load an image from disk into a normalised tensor (C, H, W).

    Args:
        path: Path to the image file.
        size: Target resolution. Behavior depends on keep_aspect:
              - keep_aspect=False (default): resize shorter edge then centre-crop to square.
              - keep_aspect=True: resize shorter edge to `size`, keep aspect ratio,
                crop both dims to nearest multiple of 4.
        keep_aspect: If True, preserve the original aspect ratio.

    Returns:
        Float tensor of shape (3, H, W), normalised with ImageNet stats.
    """
    img = ImageOps.exif_transpose(Image.open(path).convert("RGB"))

    if keep_aspect and size is not None:
        # Resize shorter edge to `size`, preserving aspect ratio
        w, h = img.size
        shorter_edge = min(w, h)
        if size > shorter_edge:
            logger.warning(
                "Requested size %d > shorter edge %d for %s — using original size",
                size,
                shorter_edge,
                path,
            )
            size = shorter_edge
        if h < w:
            new_h = size
            new_w = int(round(w * size / h))
        else:
            new_w = size
            new_h = int(round(h * size / w))
        # Round both dims down to multiple of 4
        new_h = _round_down_to_multiple(new_h, 4)
        new_w = _round_down_to_multiple(new_w, 4)
        preprocess = v2.Compose(
            [
                v2.Resize((new_h, new_w)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    elif keep_aspect and size is None:
        # Keep original size, just round to multiple of 4
        w, h = img.size
        new_h = _round_down_to_multiple(h, 4)
        new_w = _round_down_to_multiple(w, 4)
        preprocess = v2.Compose(
            [
                v2.CenterCrop((new_h, new_w)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        preprocess = build_transform(size)

    return preprocess(img)


def load_image_h_w(
    path: str | Path,
    h: int,
    w: int,
) -> torch.Tensor:
    """
    Load an image from disk into a normalised tensor (C, H, W).

    Args:
        path: Path to the image file.
        h: Target height
        w: Target width

    Returns:
        Float tensor of shape (3, H, W), normalised with ImageNet stats.
    """
    img = ImageOps.exif_transpose(Image.open(path).convert("RGB"))

    preprocess = build_transform_h_w(h, w)

    return preprocess(img)


def save_image(tensor: torch.Tensor, path: str | Path, exist_ok: bool = True) -> None:
    """
    Save a tensor as an image file.

    Args:
        tensor: Float tensor of shape (3, H, W) or (1, 3, H, W).
                Values should be in [0, 1] (i.e. already denormalised).
        path:   Destination file path (extension determines format).
    """
    if isinstance(path, str):
        path = Path(path)
    os.makedirs(path.parent, exist_ok=exist_ok)
    tv_save_image(tensor.clamp(0, 1), path)


def normalize(
    tensor: torch.Tensor,
    mean: torch.Tensor = IMAGENET_MEAN_RESHAPED,
    std: torch.Tensor = IMAGENET_STD_RESHAPED,
) -> torch.Tensor:
    """
    Apply ImageNet normalisation to a tensor of shape (..., 3, H, W).
    NOTE: By default, this will apply normalisation on CPU!!
    To apply normalisation on the GPU, use GPU tensors for mean and std.

    Args:
        tensor: Tensor of shape (..., 3, H, W).

    Returns:
        Normalised tensor, same shape.
    """
    return (tensor - mean) / std


def denormalize(
    tensor: torch.Tensor,
    mean: torch.Tensor = IMAGENET_MEAN_RESHAPED,
    std: torch.Tensor = IMAGENET_STD_RESHAPED,
) -> torch.Tensor:
    """
    Reverse ImageNet normalisation so values are back in [0, 1].
    NOTE: By default, this will apply normalisation on CPU!!
    To apply normalisation on the GPU, use GPU tensors for mean and std.

    Args:
        tensor: Tensor of shape (..., 3, H, W) normalised with IMAGENET_MEAN/STD.

    Returns:
        Denormalised tensor, same shape, clamped to [0, 1].
    """
    tensor = tensor * std + mean
    return tensor.clamp(0, 1)
