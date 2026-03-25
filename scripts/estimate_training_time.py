import argparse
import time
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models
import yaml

from src.models.trans_net import TransformationNetwork
from src.models.loss_net import LossNetwork
from src.utils.image import (
    load_image,
    normalize,
    get_device,
    IMAGENET_MEAN_RESHAPED,
    IMAGENET_STD_RESHAPED,
)
from src.utils.gram import gram_matrix
from src.utils.loss import compute_content_loss, compute_style_loss, compute_tv_loss
from src.data.dataset import build_dataloader

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate training time per epoch.")
    parser.add_argument(
        "--config",
        type=Path,
        default="configs/train_config.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=50,
        help="Number of batches to time (after warm-up).",
    )
    parser.add_argument(
        "--warmup_batches",
        type=int,
        default=5,
        help="Number of warm-up batches (not timed).",
    )
    return parser.parse_args()


def benchmark(config: dict[str, Any], num_batches: int, warmup_batches: int) -> None:
    device = get_device()
    logger.info("Device: %s", device)

    # --- Model setup (same as train.py) ---
    norm_type = (
        nn.BatchNorm2d if config["model"]["norm_type"] == "batch" else nn.InstanceNorm2d
    )
    trans_net = TransformationNetwork(norm_type).to(device)
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    loss_net = LossNetwork(model=vgg16).to(device)
    optimizer = torch.optim.Adam(
        trans_net.parameters(), lr=config["training"]["learning_rate"]
    )

    img_mean = IMAGENET_MEAN_RESHAPED.to(device)
    img_std = IMAGENET_STD_RESHAPED.to(device)

    # Style target
    style_target = (
        load_image(config["data"]["style_path"], config["data"]["image_size"])
        .unsqueeze(0)
        .to(device)
    )
    with loss_net as extractor:
        style_target_features = extractor(style_target)
    style_target_features = style_target_features.style
    for key, value in style_target_features.items():
        style_target_features[key] = gram_matrix(value.detach())

    # DataLoader
    dataloader = build_dataloader(
        root=config["data"]["content_dir"],
        image_size=config["data"]["image_size"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    total_batches_in_epoch = len(dataloader)
    total_needed = warmup_batches + num_batches
    logger.info(
        "Dataset: %d batches/epoch (batch_size=%d)",
        total_batches_in_epoch,
        config["data"]["batch_size"],
    )
    logger.info("Running %d warm-up + %d timed batches...", warmup_batches, num_batches)

    if total_needed > total_batches_in_epoch:
        logger.warning(
            "Requested %d batches but dataset only has %d. Reduce --num_batches.",
            total_needed,
            total_batches_in_epoch,
        )
        num_batches = total_batches_in_epoch - warmup_batches
        if num_batches <= 0:
            logger.error("Not enough data for even warm-up. Exiting.")
            return

    batch_times: list[float] = []
    for i, content_images in enumerate(dataloader):
        if i >= warmup_batches + num_batches:
            break

        content_images = content_images.to(device)

        # Sync before start (GPU ops are async)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        # Forward + backward (same as train.py)
        optimizer.zero_grad()
        generated = trans_net(content_images)
        generated = normalize(generated, img_mean, img_std)
        content_images = normalize(content_images, img_mean, img_std)

        with loss_net as extractor:
            generated_features = extractor(generated)
            content_image_features = extractor(content_images)

        content_loss = compute_content_loss(
            generated_features.content, content_image_features.content
        )
        style_loss = compute_style_loss(generated_features.style, style_target_features)
        tv_loss = compute_tv_loss(generated)

        total_loss = (
            config["training"]["content_weight"] * content_loss
            + config["training"]["style_weight"] * style_loss
            + config["training"]["tv_weight"] * tv_loss
        )
        total_loss.backward()
        optimizer.step()

        # Sync before end
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        if i < warmup_batches:
            logger.info("  warm-up batch %d: %.3fs", i, elapsed)
        else:
            batch_times.append(elapsed)
            if (i - warmup_batches) % 10 == 0:
                logger.info("  timed batch %d: %.3fs", i - warmup_batches, elapsed)

    avg_time = sum(batch_times) / len(batch_times)
    epoch_estimate = avg_time * total_batches_in_epoch

    logger.info("--- Results ---")
    logger.info("Avg batch time:    %.4fs", avg_time)
    logger.info("Batches per epoch: %d", total_batches_in_epoch)
    logger.info(
        "Estimated epoch:   %.1f seconds (%.1f minutes)",
        epoch_estimate,
        epoch_estimate / 60,
    )
    logger.info("Images/sec:        %.1f", config["data"]["batch_size"] / avg_time)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    benchmark(config, args.num_batches, args.warmup_batches)
