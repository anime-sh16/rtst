import os
import argparse
import logging
import time
from pathlib import Path
from typing import Any
from src.models.trans_net import TransformationNetwork
from src.models.loss_net import LossNetwork, LossFeatures
from src.utils.image import (
    load_image,
    normalize,
    denormalize,
    get_device,
    IMAGENET_MEAN_RESHAPED,
    IMAGENET_STD_RESHAPED,
)
from src.utils.gram import gram_matrix
from src.utils.loss import compute_content_loss, compute_style_loss, compute_tv_loss
from src.data.dataset import build_dataloader
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn as nn
from torchvision import models
import wandb
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure root logger with a readable format for terminal output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the NST Transformation Network."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default="configs/train_config.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint .pth to resume from.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(config: dict[str, Any], resume_path: Path | None = None) -> None:
    """
    Main training loop.

    Args:
        config:      Parsed YAML config dictionary.
        resume_path: Optional path to a checkpoint .pth file to resume training from.
    """
    device = get_device()
    img_mean = IMAGENET_MEAN_RESHAPED.to(device)
    img_std = IMAGENET_STD_RESHAPED.to(device)

    logger.info("Device: %s", device)
    logger.info(
        "Training: epochs=%d, batch_size=%d, lr=%s",
        config["training"]["epochs"],
        config["data"]["batch_size"],
        config["training"]["learning_rate"],
    )
    logger.info("Style image: %s", config["data"]["style_path"])

    # W&B setup
    wandb.init(
        project=config["wandb"]["project"],
        name=config["wandb"]["run_name"],
        config=config,
    )

    # Model setup
    norm_type = (
        nn.BatchNorm2d if config["model"]["norm_type"] == "batch" else nn.InstanceNorm2d
    )
    trans_net = TransformationNetwork(norm_type).to(device)
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    loss_net = LossNetwork(model=vgg16).to(device)  # Always frozen in LossNetwork

    # Style target precomputation
    style_target = (
        load_image(config["data"]["style_path"], config["data"]["image_size"])
        .unsqueeze(0)
        .to(device)
    )
    with loss_net as extractor:
        style_target_features = extractor(style_target)
    style_target_features = style_target_features.style
    # compute gram matrix for each style layer
    for key, value in style_target_features.items():
        style_target_features.update({key: gram_matrix(value.detach())})

    # DataLoader
    dataloader: DataLoader = build_dataloader(
        root=config["data"]["content_dir"],
        image_size=config["data"]["image_size"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    # Optimizer
    optimizer = Adam(
        params=trans_net.parameters(), lr=config["training"]["learning_rate"]
    )

    # Numer of steps
    num_steps = len(dataloader) * config["training"]["epochs"]

    # Scheduler
    if config["training"]["scheduler"]["type"] == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_steps,
            eta_min=config["training"]["scheduler"]["eta_min"],
        )

    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume from checkpoint
    start_epoch = 0
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        trans_net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    # Wrap with DataParallel after loading checkpoint (clean keys)
    if torch.cuda.device_count() > 1:
        logger.info("Using %d GPUs with DataParallel", torch.cuda.device_count())
        trans_net = nn.DataParallel(trans_net)

    # Pre-load the validation image samples for logging
    val_images = [
        load_image(path, config["data"]["image_size"]).unsqueeze(0).to(device)
        for path in sorted(Path(config["data"]["validation_dir"]).iterdir())
        if path.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]

    def log_val_images(global_step: int) -> None:
        """Run validation images through the model and log to W&B."""
        # Use unwrapped model for single-image inference (avoids NCCL errors with DataParallel)
        raw_model = (
            trans_net.module if isinstance(trans_net, nn.DataParallel) else trans_net
        )
        raw_model.eval()
        with torch.no_grad():
            for val_idx, val_image in enumerate(val_images):
                gen_val = raw_model(val_image)
                wandb.log(
                    {
                        f"val/content_{val_idx}": wandb.Image(
                            denormalize(val_image[0].cpu())
                        ),
                        f"val/generated_{val_idx}": wandb.Image(gen_val[0].cpu()),
                    },
                    step=global_step,
                )
        raw_model.train()

    # Training loop
    image_log_every_n_steps = max(1, num_steps // 10)
    with loss_net as extractor:
        for epoch in range(start_epoch, config["training"]["epochs"]):
            pbar = tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                desc=f"Epoch {epoch + 1}/{config['training']['epochs']}",
            )
            for step, content_images in pbar:
                step_start = time.monotonic()
                optimizer.zero_grad(set_to_none=True)

                content_images = content_images.to(device)

                generated = trans_net(content_images)
                # Normalise
                generated = normalize(generated, img_mean, img_std)

                generated_features: LossFeatures = extractor(generated)
                with torch.no_grad():
                    content_image_features: LossFeatures = extractor(content_images)

                content_loss = compute_content_loss(
                    generated_features.content, content_image_features.content
                )

                style_loss = compute_style_loss(
                    generated_features.style, style_target_features
                )
                tv_loss = compute_tv_loss(generated)

                total_loss = (
                    config["training"]["content_weight"] * content_loss
                    + config["training"]["style_weight"] * style_loss
                    + config["training"]["tv_weight"] * tv_loss
                )

                total_loss.backward()
                # max_norm=inf: no clipping, just computing norm for monitoring
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trans_net.parameters(), max_norm=float("inf")
                )
                optimizer.step()
                scheduler.step()
                batch_time = time.monotonic() - step_start

                # Logging
                global_step = epoch * len(dataloader) + step
                w_content = config["training"]["content_weight"] * content_loss.item()
                w_style = config["training"]["style_weight"] * style_loss.item()
                w_tv = config["training"]["tv_weight"] * tv_loss.item()

                pbar.set_postfix(
                    total=f"{total_loss.item():.4f}",
                    content=f"{w_content:.4f}",
                    style=f"{w_style:.4f}",
                    tv=f"{w_tv:.4f}",
                )

                if step % config["wandb"]["log_every_n_steps"] == 0:
                    wandb.log(
                        {
                            "loss/total": total_loss.item(),
                            "loss/content": w_content,
                            "loss/style": w_style,
                            "loss/tv": w_tv,
                            "loss/raw_content": content_loss.item(),
                            "loss/raw_style": style_loss.item(),
                            "loss/raw_tv": tv_loss.item(),
                            "training/learning_rate": optimizer.param_groups[0]["lr"],
                            "training/global_step": global_step,
                            "training/grad_norm": grad_norm.item(),
                            "training/batch_time": batch_time,
                            "weights/tv_loss": config["training"]["tv_weight"],
                            "weights/content_loss": config["training"][
                                "content_weight"
                            ],
                            "weights/style_loss": config["training"]["style_weight"],
                        },
                        step=global_step,
                    )

                if (global_step + 1) % image_log_every_n_steps == 0:
                    log_val_images(global_step + 1)

            # Checkpoint — unwrap DataParallel if present
            model_to_save = (
                trans_net.module
                if isinstance(trans_net, nn.DataParallel)
                else trans_net
            )
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": total_loss.item(),
            }
            torch.save(checkpoint, checkpoint_dir / f"checkpoint_{epoch}.pth")
            logger.info(
                "Epoch %d complete | loss=%.4f | saved to %s",
                epoch + 1,
                total_loss.item(),
                checkpoint_dir / f"checkpoint_{epoch}.pth",
            )

    # Log final validation images after training completes
    log_val_images(num_steps)

    model_to_save = (
        trans_net.module if isinstance(trans_net, nn.DataParallel) else trans_net
    )
    weights_final = model_to_save.state_dict()
    torch.save(weights_final, config["training"]["final_model_path"])

    wandb.finish()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    train(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
