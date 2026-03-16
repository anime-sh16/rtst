import argparse
from pathlib import Path
from src.utils.image import load_image, save_image, get_device
from src.models.trans_net import TransformationNetwork

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NST inference on a single image.")
    parser.add_argument(
        "--image", type=Path, required=True, help="Path to content image."
    )
    parser.add_argument(
        "--model", type=Path, required=True, help="Path to .pth weights file."
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/output.jpg"))
    parser.add_argument("--image-size", type=int, default=512)
    return parser.parse_args()


def stylise(
    image_path: Path,
    weights_path: Path,
    output_path: Path,
    image_size: int,
    device: torch.device,
) -> None:
    """
    Load a trained TransformationNetwork and apply it to a single content image.

    Args:
        image_path:   Path to the input content image.
        weights_path: Path to saved TransformationNetwork state dict.
        output_path:  Where to save the stylised image.
        image_size:   Resize the input image to this resolution before inference.
        device:       CPU or CUDA device.
    """
    image = load_image(image_path, image_size)
    trans_net = TransformationNetwork()
    state_dict = torch.load(weights_path, map_location=device)
    trans_net.load_state_dict(state_dict)
    trans_net.to(device)
    trans_net.eval()
    with torch.no_grad():
        output = trans_net(image.unsqueeze(0).to(device))

    save_image(output.squeeze(0), output_path)


def main() -> None:
    args = parse_args()
    device = get_device()
    stylise(args.image, args.model, args.output, args.image_size, device)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
