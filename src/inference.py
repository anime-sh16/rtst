import argparse
from pathlib import Path
from typing import Any

import logging
import yaml
import torch
import torch.nn as nn

from src.utils.image import load_image, save_image, get_device
from src.models.trans_net import TransformationNetwork

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NST inference on images.")
    parser.add_argument(
        "--image", type=Path, required=True, help="Path to content image or directory."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/fast-nst.pth"),
        help="Path to .pth weights file.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/results"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/train_config.yaml")
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--keep-aspect",
        action="store_true",
        help="Preserve aspect ratio; dims rounded to multiple of 4.",
    )
    return parser.parse_args()


def build_model(
    config_path: Path, weights_path: Path, device: torch.device
) -> TransformationNetwork:
    """Load config, build TransformationNetwork, load weights, return in eval mode."""
    config = load_config(config_path)
    norm_type = (
        nn.InstanceNorm2d
        if config["model"]["norm_type"] == "instance"
        else nn.BatchNorm2d
    )
    trans_net = TransformationNetwork(norm_type)
    state_dict = torch.load(weights_path, map_location=device)
    trans_net.load_state_dict(state_dict)
    trans_net.to(device)
    trans_net.eval()
    logger.info("Loaded weights from %s", weights_path)
    logger.info("Device: %s", device)
    return trans_net


def styled_name(path: Path) -> str:
    """Return filename as '<stem>_style<suffix>'."""
    return f"{path.stem}_style{path.suffix}"


def collect_images(directory: Path) -> list[Path]:
    """Return sorted list of image paths in a directory."""
    return sorted(
        p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def stylise_single(
    image_path: Path,
    model: TransformationNetwork,
    output_path: Path,
    image_size: int | None,
    device: torch.device,
    keep_aspect: bool = False,
) -> None:
    """Stylise a single image and save it."""
    logger.info("Stylising: %s", image_path)
    image = load_image(image_path, image_size, keep_aspect=keep_aspect)
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
    save_image(output.squeeze(0), output_path)
    logger.info("Saved to %s", output_path)


def stylise_batch(
    image_paths: list[Path],
    model: TransformationNetwork,
    output_dir: Path,
    image_size: int,
    batch_size: int,
    device: torch.device,
) -> None:
    """Stylise images in batches (all resized to image_size) and save them."""
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch = torch.stack([load_image(p, image_size) for p in batch_paths]).to(device)
        logger.info("Batch %d-%d / %d", i + 1, i + len(batch_paths), len(image_paths))
        with torch.no_grad():
            outputs = model(batch)
        for path, output in zip(batch_paths, outputs):
            save_image(output, output_dir / styled_name(path))
    logger.info("Done. Results saved to %s", output_dir)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    args = parse_args()
    device = get_device()
    output_dir = args.output_dir

    model = build_model(args.config, args.model, device)

    if args.image.is_dir():
        image_paths = collect_images(args.image)
        logger.info("Found %d images in %s", len(image_paths), args.image)

        if args.image_size is not None and not args.keep_aspect:
            # Batched: all images resized to same square size
            stylise_batch(
                image_paths, model, output_dir, args.image_size, args.batch_size, device
            )
        else:
            # Sequential: different sizes can't be batched
            for path in image_paths:
                out_path = output_dir / styled_name(path)
                stylise_single(
                    path, model, out_path, args.image_size, device, args.keep_aspect
                )
    else:
        # Single image
        out_path = output_dir / styled_name(args.image)
        stylise_single(
            args.image, model, out_path, args.image_size, device, args.keep_aspect
        )


if __name__ == "__main__":
    main()
