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
        description="Test and try hyperparametes for the NST Transformation Network."
    )
    parser.add_argument(
        "--content_path",
        type=Path,
        default="data/train1000",
        help="Path to content images subset.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=10,
        help="Epochs.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers.",
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
        default=1.0,
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
STYLE_IMAGE_PATH = Path("data/styles/mosaic.jpg")
VAL_DATA_PATH = "data/validation"


def train(
    content_path: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
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

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    logger.info("Device: %s", device)
    logger.info(
        "Training: epochs=%d, batch_size=%d, lr=%s",
        epochs,
        batch_size,
        lr,
    )
    logger.info("Style image: %s", STYLE_IMAGE_PATH)

    IMAGE_LOG_EVERY_N_STEPS = epochs // 5
    config = {
        "training": {
            "epochs": epochs,
            "learning_rate": lr,
            "content_weight": content_w,
            "style_weight": style_w,
            "tv_weight": tv_w,
        },
        "data": {
            "content_dir": content_path,
            "validation_dir": VAL_DATA_PATH,
            "style_path": STYLE_IMAGE_PATH,
            "image_size": IMAGE_SIZE,
            "batch_size": batch_size,
            "num_workers": num_workers,
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
        shuffle=True,
    )

    # Optimizer
    optimizer = Adam(
        params=trans_net.parameters(), lr=config["training"]["learning_rate"]
    )

    # Resume from checkpoint
    start_epoch = 0

    # Pre-load validation images for logging
    val_images = [
        load_image(path, config["data"]["image_size"]).unsqueeze(0).to(device)
        for path in sorted(Path(config["data"]["validation_dir"]).iterdir())
        if path.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]

    # Training loop
    total_steps = config["training"]["epochs"] * len(dataloader)
    image_log_every_n_steps = max(1, total_steps // 10)
    for epoch in range(start_epoch, config["training"]["epochs"]):
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch + 1}/{config['training']['epochs']}",
        )
        for step, content_images in pbar:
            optimizer.zero_grad(set_to_none=True)

            content_images = content_images.to(device)

            generated = trans_net(content_images)
            # Normalise
            generated = normalize(generated, img_mean, img_std)

            with loss_net as extractor:
                generated_features: LossFeatures = extractor(generated)
                with torch.no_grad():
                    content_image_features: LossFeatures = extractor(content_images)

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
            w_content = config["training"]["content_weight"] * content_loss.item()
            w_style = config["training"]["style_weight"] * style_loss.item()
            w_tv = (
                config["training"]["tv_weight"] * tv_loss.item()
                if config["training"]["tv_weight"] != 0
                else 0
            )

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
                        "weights/tv_loss": config["training"]["tv_weight"],
                        "weights/content_loss": config["training"]["content_weight"],
                        "weights/style_loss": config["training"]["style_weight"],
                    },
                    step=global_step,
                )

            if (global_step + 1) % image_log_every_n_steps == 0:
                # Log fixed validation images to W&B
                trans_net.eval()
                with torch.no_grad():
                    for val_idx, val_image in enumerate(val_images):
                        gen_val = trans_net(val_image)
                        wandb.log(
                            {
                                f"val/content_{val_idx}": wandb.Image(
                                    denormalize(val_image[0].cpu())
                                ),
                                f"val/generated_{val_idx}": wandb.Image(
                                    gen_val[0].cpu()
                                ),
                            },
                            step=global_step,
                        )
                trans_net.train()

    wandb.finish()


def main() -> None:
    setup_logging()
    args = parse_args()
    content_path = args.content_path
    epoch = args.epoch
    batch_size = args.batch
    num_workers = args.num_workers
    lr = args.lr
    content_w = args.content_w
    style_w = args.style_w
    tv_w = args.tv_w
    run_name = args.run_name
    train(
        content_path,
        epoch,
        batch_size,
        num_workers,
        lr,
        content_w,
        style_w,
        tv_w,
        run_name,
    )


if __name__ == "__main__":
    main()
