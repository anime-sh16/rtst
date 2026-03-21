import argparse
import logging
from pathlib import Path
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
        description="Test and try hyperparametes for the NST Transformation Network."
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=800,
        help="Epochs.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--content_w",
        type=float,
        default=1.0e-1,
        help="Weight for content loss.",
    )
    parser.add_argument(
        "--style_w",
        type=float,
        default=1.0e4,
        help="Weight for style loss.",
    )
    parser.add_argument(
        "--tv_w",
        type=float,
        default=0.0,
        help="Weight for tv loss.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="try-content-1",
        help="W&B run name.",
    )
    return parser.parse_args()


IMAGE_SIZE = 256
PROJECT_NAME = "configure-nst"
LOG_EVERY_N_STEPS = 5
IMAGE_LOG_EVERY_N_STEPS = 24
STYLE_IMAGE_PATH = Path("data/styles/mosaic.jpg")
DATA_PATH = "data/validation"


def train(
    epochs: int,
    batch_size: int,
    lr: float,
    content_w: float,
    style_w: float,
    tv_w: float,
    run_name: str,
) -> None:
    """
    Training loop.

    Args:
        batch_size:      Batch size.
        content_w:       Content loss weight.
        style_w:         Style loss weight.
        tv_w:            Total variation loss weight.
        run_name:        W&B run name.
    """
    device = get_device()
    img_mean = IMAGENET_MEAN_RESHAPED.to(device)
    img_std = IMAGENET_STD_RESHAPED.to(device)

    logger.info("Device: %s", device)
    logger.info(
        "Training: epochs=%d, batch_size=%d, lr=%s",
        epochs,
        batch_size,
        lr,
    )
    logger.info("Style image: %s", STYLE_IMAGE_PATH)

    config = {
        "training": {
            "epochs": epochs,
            "learning_rate": lr,
            "content_weight": content_w,
            "style_weight": style_w,
            "tv_weight": tv_w,
        },
        "data": {
            "content_dir": DATA_PATH,
            "style_path": STYLE_IMAGE_PATH,
            "image_size": IMAGE_SIZE,
            "batch_size": batch_size,
            "num_workers": 4,
        },
        "wandb": {
            "project": PROJECT_NAME,
            "run_name": run_name,
            "log_every_n_steps": LOG_EVERY_N_STEPS,
            "image_log_every_n_steps": IMAGE_LOG_EVERY_N_STEPS,
        },
    }

    # W&B setup
    wandb.init(
        project=PROJECT_NAME,
        name=run_name,
        config=config,
    )

    # Model setup
    norm_type = nn.InstanceNorm2d
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

    # Resume from checkpoint
    start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, config["training"]["epochs"]):
        for step, content_images in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)

            content_images = content_images.to(device)

            generated = trans_net(content_images)
            # Normalise
            generated = normalize(generated, img_mean, img_std)
            content_images_loss = normalize(content_images, img_mean, img_std)

            with loss_net as extractor:
                generated_features: LossFeatures = extractor(generated)
                with torch.no_grad():
                    content_image_features: LossFeatures = extractor(
                        content_images_loss
                    )

            content_loss = compute_content_loss(
                generated_features.content, content_image_features.content
            )

            style_loss = compute_style_loss(
                generated_features.style, style_target_features
            )
            if config["training"]["tv_weight"] != 0:
                tv_loss = compute_tv_loss(generated)
            else:
                tv_loss = torch.tensor(0.0).to(device)

            total_loss = (
                config["training"]["content_weight"] * content_loss
                + config["training"]["style_weight"] * style_loss
            )
            if config["training"]["tv_weight"] != 0:
                total_loss += config["training"]["tv_weight"] * tv_loss

            total_loss.backward()
            optimizer.step()

            # Logging
            global_step = epoch * len(dataloader) + step
            if step % config["wandb"]["log_every_n_steps"] == 0:
                w_content = config["training"]["content_weight"] * content_loss.item()
                w_style = config["training"]["style_weight"] * style_loss.item()
                w_tv = (
                    config["training"]["tv_weight"] * tv_loss.item()
                    if config["training"]["tv_weight"] != 0
                    else 0
                )
                logger.info(
                    "Epoch [%d/%d] Step [%d/%d] | total=%.4f content=%.4f style=%.4f tv=%.4f",
                    epoch + 1,
                    config["training"]["epochs"],
                    step,
                    len(dataloader),
                    total_loss.item(),
                    w_content,
                    w_style,
                    w_tv,
                )
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
                        "weights/tv_loss": config["training"]["tv_weight"],
                        "weights/content_loss": config["training"]["content_weight"],
                        "weights/style_loss": config["training"]["style_weight"],
                    },
                    step=global_step,
                )

            if step % config["wandb"]["image_log_every_n_steps"] == 0:
                # Run the validation set images and log them to wandb
                for i, (content, generated) in enumerate(
                    zip(content_images, generated)
                ):
                    wandb.log(
                        {
                            f"image/content_{i}": wandb.Image(
                                denormalize(content.cpu())
                            ),
                            f"image/generated_{i}": wandb.Image(
                                denormalize(generated.cpu())
                            ),
                        },
                        step=global_step,
                    )
        logger.info(
            "Epoch %d complete | loss=%.4f",
            epoch + 1,
            total_loss.item(),
        )

    wandb.finish()


def main() -> None:
    setup_logging()
    args = parse_args()
    epoch = args.epoch
    batch_size = args.batch
    lr = args.lr
    content_w = args.content_w
    style_w = args.style_w
    tv_w = args.tv_w
    run_name = args.run_name
    train(epoch, batch_size, lr, content_w, style_w, tv_w, run_name)


if __name__ == "__main__":
    main()
