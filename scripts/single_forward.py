import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from src.models.loss_net import LossNetwork
from src.models.trans_net import TransformationNetwork
from src.utils.gram import gram_matrix
from src.utils.image import (
    IMAGENET_MEAN_RESHAPED,
    IMAGENET_STD_RESHAPED,
    get_device,
    load_image,
    normalize,
)
from src.utils.loss import compute_content_loss, compute_style_loss, compute_tv_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single forward pass diagnostic.")
    parser.add_argument(
        "--content",
        type=Path,
        default="data/coco/test1000",
        help="Path to a directory of content images (picks the first one).",
    )
    parser.add_argument(
        "--style",
        type=Path,
        default="data/styles/mosaic.jpg",
        help="Path to the style image.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image size to use (default: 256).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"\nDevice: {device}")

    img_mean = IMAGENET_MEAN_RESHAPED.to(device)
    img_std = IMAGENET_STD_RESHAPED.to(device)

    # --- Load one content image from the directory ---
    image_files = sorted(args.content.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"No .jpg images found in {args.content}")
    content_path = image_files[0]
    print(f"Content image: {content_path.name}")

    content_image = load_image(content_path, args.image_size).unsqueeze(0).to(device)

    # --- Load style image ---
    style_image = load_image(args.style, args.image_size).unsqueeze(0).to(device)
    print(f"Style image:   {args.style.name}\n")

    # --- Build models ---
    norm_type = nn.InstanceNorm2d
    trans_net = TransformationNetwork(norm_type).to(device)
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    loss_net = LossNetwork(model=vgg16).to(device)

    # --- Precompute style gram targets ---
    with loss_net as extractor:
        style_features = extractor(style_image)
    style_gram_targets = {
        k: gram_matrix(v.detach()) for k, v in style_features.style.items()
    }

    # --- Single forward pass ---
    with torch.no_grad():
        generated = trans_net(content_image)
        generated_norm = normalize(generated, img_mean, img_std)

        with loss_net as extractor:
            gen_features = extractor(generated_norm)
            content_features = extractor(content_image)

        content_loss = compute_content_loss(
            gen_features.content, content_features.content
        )
        style_loss = compute_style_loss(gen_features.style, style_gram_targets)
        tv_loss = compute_tv_loss(generated)

    # --- Print results ---
    print("=" * 50)
    print("RAW LOSS VALUES (unweighted, untrained model)")
    print("=" * 50)
    print(f"  content_loss : {content_loss.item():.4e}")
    print(f"  style_loss   : {style_loss.item():.4e}")
    print(f"  tv_loss      : {tv_loss.item():.4e}")
    print("=" * 50)

    # --- TODO: Add your weight guidance logic here ---
    # Hint: you want all three weighted losses (weight * raw_loss) to be
    # in the same order of magnitude so none dominates.
    # e.g. if content=1e4, style=1e10, tv=1e-3:
    #   → content_weight=1.0, style_weight=1e-5 (or 1e-4), tv_weight=1e3
    # Print suggested starting weights based on the ratio to content_loss.

    print("\nSuggested starting point (equal weighting):")
    print("=" * 50)
    print("  content_weight : 1.0")
    print(f"  style_weight   : {content_loss.item() / style_loss.item():.4e}")
    print(f"  tv_weight      : {content_loss.item() / tv_loss.item():.4e}")


if __name__ == "__main__":
    main()
