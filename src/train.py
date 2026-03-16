import os
import argparse
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
import torch
import torch.nn as nn
from torchvision import models
import wandb
import yaml


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
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    # Optimizer
    optimizer = Adam(
        params=trans_net.parameters(), lr=config["training"]["learning_rate"]
    )
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume from checkpoint
    start_epoch = 0
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        trans_net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    # Training loop
    for epoch in range(start_epoch, config["training"]["epochs"]):
        for step, content_images in enumerate(dataloader):
            optimizer.zero_grad()

            content_images = content_images.to(device)

            generated = trans_net(content_images)
            # Normalise
            generated = normalize(generated, img_mean, img_std)
            content_images = normalize(content_images, img_mean, img_std)

            with loss_net as extractor:
                generated_features: LossFeatures = extractor(generated)
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
            optimizer.step()

            # Logging
            global_step = epoch * len(dataloader) + step
            if step % config["wandb"]["log_every_n_steps"] == 0:
                wandb.log(
                    {
                        "loss/total": total_loss.item(),
                        "loss/content": content_loss.item(),
                        "loss/style": style_loss.item(),
                        "loss/tv": tv_loss.item(),
                    },
                    step=global_step,
                )

            if step % config["wandb"]["image_log_every_n_steps"] == 0:
                content_sample = denormalize(content_images[0].cpu())
                generated_sample = denormalize(generated[0].detach().cpu())
                style_sample = denormalize(style_target.cpu())
                wandb.log(
                    {
                        "samples/content": wandb.Image(content_sample),
                        "samples/style": wandb.Image(style_sample),
                        "samples/generated": wandb.Image(generated_sample),
                    },
                    step=global_step,
                )

        # Checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": trans_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": total_loss.item(),
        }
        torch.save(checkpoint, checkpoint_dir / f"checkpoint_{epoch}.pth")

    weights_final = trans_net.state_dict()
    torch.save(weights_final, config["training"]["final_model_path"])

    wandb.finish()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
