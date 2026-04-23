import struct
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path


def read_flo(path: str | Path) -> torch.Tensor:
    """
    Read a .flo optical flow file (Middlebury format).

    Args:
        path: Path to the .flo file.

    Returns:
        Tensor of shape (2, H, W) containing (u, v) flow components.
    """
    with open(path, "rb") as f:
        magic = struct.unpack("f", f.read(4))[0]
        assert abs(magic - 202021.25) < 1e-5, f"Bad .flo magic: {magic}"
        w = struct.unpack("i", f.read(4))[0]
        h = struct.unpack("i", f.read(4))[0]
        data = np.frombuffer(f.read(h * w * 2 * 4), dtype=np.float32)

    flow = torch.from_numpy(data.copy().reshape(h, w, 2))
    return flow.permute(2, 0, 1)  # (2, H, W)


def backward_warp(
    image: torch.Tensor,
    flow: torch.Tensor,
) -> torch.Tensor:
    """
    Warp an image using backward warping with optical flow.

    Given flow F_{t→t+1}, this warps image_t to align with image_{t+1}.
    Uses bilinear interpolation via grid_sample.

    Args:
        image: Tensor of shape (B, C, H, W) — the source image to warp.
        flow:  Tensor of shape (B, 2, H, W) — optical flow (u, v) in pixels.

    Returns:
        Warped image of shape (B, C, H, W).
    """
    B, _, H, W = image.shape

    y_coords = torch.arange(H, device=image.device, dtype=image.dtype)
    x_coords = torch.arange(W, device=image.device, dtype=image.dtype)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    base_grid = torch.stack((grid_x, grid_y), dim=-1)
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)

    flow_permuted = flow.permute(0, 2, 3, 1)  # (B, H, W, 2)
    target_grid = base_grid + flow_permuted

    target_grid[..., 0] = 2.0 * target_grid[..., 0] / (W - 1) - 1.0
    target_grid[..., 1] = 2.0 * target_grid[..., 1] / (H - 1) - 1.0

    warped_image = F.grid_sample(
        image, target_grid, mode="bilinear", padding_mode="border", align_corners=True
    )

    return warped_image
