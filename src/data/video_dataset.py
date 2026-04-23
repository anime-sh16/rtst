from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from src.utils.image import load_image, read_occlusion
from src.utils.warp import read_flo


@dataclass
class FramePair:
    """A consecutive frame pair with optical flow and occlusion mask."""

    frame_t: torch.Tensor  # (3, H, W)
    frame_t1: torch.Tensor  # (3, H, W)
    flow: torch.Tensor  # (2, H, W) — forward flow from t to t+1
    occlusion_mask: torch.Tensor  # (1, H, W) — 1 = occluded, 0 = visible


class SintelFramePairDataset(Dataset):
    """
    Dataset that yields consecutive frame pairs from MPI-Sintel
    along with optical flow and occlusion masks.

    Expected Sintel layout:
        root/training/
            clean/
                scene_name/
                    frame_0001.png
                    frame_0002.png
                    ...
            flow/
                scene_name/
                    frame_0001.flo   (maps frame_0001 → frame_0002)
                    ...
            occlusions/
                scene_name/
                    frame_0001.png
                    ...

    Each sample is a pair: (frame_N, frame_N+1, flow_N, occlusion_N).
    The last frame of each scene is skipped (no flow available).

    Frames are loaded at their original Sintel resolution (1024x436) and
    only ImageNet-normalized — no resize or crop is applied. Flow and
    occlusion masks are returned as-is so they stay aligned with the frames.

    Args:
        root:         Path to Sintel root directory (contains training/).
        render_pass:  Which render pass to use ("clean" or "final").
    """

    def __init__(
        self,
        root: str | Path,
        render_pass: str = "clean",
    ) -> None:
        self.root = Path(root)

        frames_dir = self.root / "training" / render_pass
        flow_dir = self.root / "training" / "flow"
        occlusions_dir = self.root / "training" / "occlusions"

        self.pairs: list[tuple[Path, Path, Path, Path]] = []

        for scene_dir in sorted(d for d in frames_dir.iterdir() if d.is_dir()):
            scene_name = scene_dir.name
            frames = sorted(f for f in scene_dir.iterdir() if f.suffix == ".png")

            for frame_t, frame_t1 in zip(frames[:-1], frames[1:]):
                flo_path = flow_dir / scene_name / f"{frame_t.stem}.flo"
                occ_path = occlusions_dir / scene_name / f"{frame_t.stem}.png"

                if flo_path.exists() and occ_path.exists():
                    self.pairs.append((frame_t, frame_t1, flo_path, occ_path))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> FramePair:
        frame_t_path, frame_t1_path, flow_path, occ_path = self.pairs[idx]

        frame_t = load_image(frame_t_path, keep_aspect=True)
        frame_t1 = load_image(frame_t1_path, keep_aspect=True)
        flow = read_flo(flow_path)
        occ = read_occlusion(occ_path)

        return FramePair(frame_t, frame_t1, flow, occ)


def build_video_dataloader(
    root: str | Path,
    batch_size: int,
    render_pass: str = "clean",
    num_workers: int = 4,
    shuffle: bool = True,
    distributed: bool = False,
) -> tuple[DataLoader, DistributedSampler | None]:
    """
    Build a DataLoader for Sintel frame pairs.

    Returns:
        (DataLoader, optional DistributedSampler)
    """
    dataset = SintelFramePairDataset(
        root=root,
        render_pass=render_pass,
    )

    sampler: DistributedSampler | None = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False

    def collate_frame_pairs(batch: list[FramePair]) -> FramePair:
        return FramePair(
            frame_t=torch.stack([b.frame_t for b in batch]),
            frame_t1=torch.stack([b.frame_t1 for b in batch]),
            flow=torch.stack([b.flow for b in batch]),
            occlusion_mask=torch.stack([b.occlusion_mask for b in batch]),
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_frame_pairs,
    ), sampler
