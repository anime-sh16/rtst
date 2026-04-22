# ruff: noqa: F841

"""
Fine-tune a pretrained style transfer model with temporal consistency loss.

Usage:
    uv run python src/finetune.py --config configs/finetune_config.yaml --weights models/fast-nst.pth
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import wandb
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

from src.data.video_dataset import build_video_dataloader
from src.models.loss_net import LossNetwork
from src.models.trans_net_v2 import TransformationNetworkV2
from src.utils.image import (
    IMAGENET_MEAN_RESHAPED,
    IMAGENET_STD_RESHAPED,
)

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune style transfer model with temporal loss."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to finetune YAML config."
    )
    parser.add_argument(
        "--weights", type=Path, required=True, help="Path to pretrained model weights."
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to finetune checkpoint to resume.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def finetune(
    config: dict[str, Any],
    weights_path: Path,
    resume_path: Path | None = None,
) -> None:
    """
    Fine-tuning loop: loads pretrained weights, trains with
    content + style + TV + temporal losses on video frame pairs.
    """
    distributed = dist.is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    img_mean = IMAGENET_MEAN_RESHAPED.to(device)
    img_std = IMAGENET_STD_RESHAPED.to(device)

    logger.info("Device: %s (rank %d, distributed=%s)", device, local_rank, distributed)

    # --- Model setup ---
    se_attention = config["model"].get("se_attention", False)
    trans_net = TransformationNetworkV2(se_attention_bool=se_attention).to(device)

    # Load pretrained weights
    pretrained_state = torch.load(weights_path, map_location=device)
    trans_net.load_state_dict(pretrained_state)
    logger.info("Loaded pretrained weights from %s", weights_path)

    # Loss network (VGG16, frozen)
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    loss_net = LossNetwork(model=vgg16).to(device)

    # TODO: Precompute style gram targets (same as train.py)

    # --- Video DataLoader ---
    dataloader, sampler = build_video_dataloader(
        root=config["data"]["sintel_root"],
        image_size=config["data"]["image_size"],
        batch_size=config["data"]["batch_size"],
        render_pass=config["data"].get("render_pass", "clean"),
        num_workers=config["data"]["num_workers"],
        distributed=distributed,
    )

    # --- Optimizer (lower LR for fine-tuning) ---
    optimizer = Adam(
        params=trans_net.parameters(),
        lr=config["training"]["learning_rate"],
    )

    num_steps = len(dataloader) * config["training"]["epochs"]
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_steps,
        eta_min=config["training"]["scheduler"]["eta_min"],
    )

    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume from finetune checkpoint
    start_epoch = 0
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        trans_net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    if distributed:
        trans_net = DDP(trans_net, device_ids=[local_rank])

    # W&B
    if is_main:
        wandb.init(
            project=config["wandb"]["project"],
            name=config["wandb"]["run_name"],
            config=config,
        )

    # --- Training loop ---
    # TODO: Implement the fine-tuning loop

    raise NotImplementedError

    # Save final weights
    if is_main:
        model_to_save = trans_net.module if isinstance(trans_net, DDP) else trans_net
        torch.save(model_to_save.state_dict(), config["training"]["final_model_path"])
        wandb.finish()

    if distributed:
        dist.destroy_process_group()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)

    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")

    finetune(config, weights_path=args.weights, resume_path=args.resume)


if __name__ == "__main__":
    main()
