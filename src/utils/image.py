from pathlib import Path
import os
import torch
from PIL import Image, ImageOps
from torchvision.transforms import v2
from torchvision.utils import save_image as tv_save_image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# These tensors lives on CPU, never moved
IMAGENET_MEAN_RESHAPED = torch.tensor(IMAGENET_MEAN).reshape(-1, 1, 1)
IMAGENET_STD_RESHAPED = torch.tensor(IMAGENET_STD).reshape(-1, 1, 1)


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


def load_image(
    path: str | Path,
    size: int | None = None,
) -> torch.Tensor:
    """
    Load an image from disk into a normalised tensor (C, H, W).
    Applies `build_transform(size)` to the image.

    Args:
        path: Path to the image file.
        size: If provided, resize the shorter edge to `size` and centre-crop.

    Returns:
        Float tensor of shape (3, H, W), normalised with ImageNet stats.
    """
    img = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    preprocess = build_transform(size)
    img_tensor = preprocess(img)
    return img_tensor


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
