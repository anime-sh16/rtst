"""
Quantization-Aware Training (QAT) for the BN-variant TransformationNetwork.

Uses PT2E flow: torch.export → prepare_qat_pt2e → train → convert_pt2e.
Distillation-only loss (cosine similarity + L2) against the float teacher model.
No perceptual loss needed — the model already learned style; QAT only recovers
quality lost from int8 quantization.

Usage:
    uv run python src/qat.py --config configs/qat_config.yaml
"""

import argparse
import logging
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchao
import yaml
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

from src.models.trans_net import TransformationNetwork
from src.models.trans_net_v2 import TransformationNetworkV2
from src.utils.image import load_image_h_w, denormalize
from src.data.dataset import build_dataloader
from src.export_pipeline import analyze_delegation, compute_checkpoint_hash

logger = logging.getLogger(__name__)


# Boilerplate: logging, args, config


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a clean, readable logger with color-coded levels."""

    class ColorFormatter(logging.Formatter):
        COLORS = {
            logging.DEBUG: "\033[90m",  # gray
            logging.INFO: "\033[36m",  # cyan
            logging.WARNING: "\033[33m",  # yellow
            logging.ERROR: "\033[31m",  # red
            logging.CRITICAL: "\033[1;31m",  # bold red
        }
        RESET = "\033[0m"
        BOLD = "\033[1m"

        def format(self, record: logging.LogRecord) -> str:
            color = self.COLORS.get(record.levelno, self.RESET)
            ts = time.strftime("%H:%M:%S", time.localtime(record.created))
            ms = f"{record.created % 1:.3f}"[1:]
            level = record.levelname.ljust(8)
            name = record.name.split(".")[-1]
            msg = record.getMessage()

            return (
                f"{self.BOLD}{color}{ts}{ms}{self.RESET}  "
                f"{color}{level}{self.RESET}  "
                f"\033[90m{name:<12}{self.RESET}  "
                f"{msg}"
            )

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())
    logging.root.handlers.clear()
    logging.root.addHandler(handler)
    logging.root.setLevel(level)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QAT for TransformationNetwork.")
    parser.add_argument(
        "--config",
        type=Path,
        default="configs/qat_config.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a QAT checkpoint to resume from.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


# Model setup helpers


def load_teacher_model(
    config: dict[str, Any], device: torch.device
) -> TransformationNetwork | TransformationNetworkV2:
    """Load the pretrained float model (teacher) and freeze it."""
    if config["model"]["model_type"] == "mobilenet":
        teacher = TransformationNetworkV2(
            se_attention_bool=config["model"]["se_attention"],
            export_mode=config["model"]["export_mode"],
        )
    else:
        norm_type = (
            nn.BatchNorm2d
            if config["model"]["norm_type"] == "bn"
            else nn.InstanceNorm2d
        )
        teacher = TransformationNetwork(
            norm_layer_type=norm_type,
            export_mode=config["model"]["export_mode"],
        )
    checkpoint = torch.load(config["model"]["checkpoint_path"], map_location=device)
    # Handle both raw state_dict and checkpoint dict formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        teacher.load_state_dict(checkpoint["model_state_dict"])
    else:
        teacher.load_state_dict(checkpoint)

    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def build_qat_model(
    config: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """Export → prepare_qat_pt2e: returns the QAT-prepared model with fake-quant nodes."""
    image_h = config["data"]["image_h"]
    image_w = config["data"]["image_w"]

    if config["model"]["model_type"] == "mobilenet":
        student = TransformationNetworkV2(
            se_attention_bool=config["model"]["se_attention"],
            export_mode=config["model"]["export_mode"],
        )
    else:
        norm_type = (
            nn.BatchNorm2d
            if config["model"]["norm_type"] == "bn"
            else nn.InstanceNorm2d
        )
        student = TransformationNetwork(
            norm_layer_type=norm_type,
            export_mode=config["model"]["export_mode"],
        )
    # Load same pretrained weights into student before QAT preparation
    checkpoint = torch.load(config["model"]["checkpoint_path"], map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        student.load_state_dict(checkpoint["model_state_dict"])
    else:
        student.load_state_dict(checkpoint)

    # PT2E: export → prepare_qat_pt2e
    example_inputs = (torch.randn(2, 3, image_h, image_w),)
    dynamic_shapes = ({0: torch.export.Dim("batch", min=1)},)
    exported = torch.export.export(
        student, example_inputs, dynamic_shapes=dynamic_shapes
    ).module()

    quantizer = XNNPACKQuantizer()
    quantizer.set_global(
        get_symmetric_quantization_config(
            is_per_channel=config["quantization"]["is_per_channel"],
            is_qat=True,
            is_dynamic=config["quantization"]["is_dynamic"],
        )
    )
    prepared = prepare_qat_pt2e(exported, quantizer)
    prepared.to(device)
    torchao.quantization.pt2e.move_exported_model_to_train(prepared)
    return prepared


# Distillation loss


def compute_distillation_loss(
    teacher_out: torch.Tensor,
    student_out: torch.Tensor,
    cosine_weight: float,
    l2_weight: float,
) -> tuple[torch.Tensor, float, float]:
    """
    Compute distillation loss between teacher (float) and student (QAT) outputs.

    Returns:
        (total_loss, cosine_similarity_value, l2_value)
    """

    cosine_sim = torch.nn.functional.cosine_similarity(
        teacher_out.flatten(1), student_out.flatten(1), dim=1
    ).mean()
    l2 = torch.nn.functional.mse_loss(teacher_out, student_out)
    total_loss = (1 - cosine_sim) * cosine_weight + l2 * l2_weight
    return total_loss, cosine_sim.item(), l2.item()


# Validation image logging


def log_val_images(
    teacher: nn.Module,
    student: nn.Module,
    val_images: list[torch.Tensor],
    global_step: int,
) -> None:
    """Run validation images through both models and log comparison to W&B."""
    torchao.quantization.pt2e.move_exported_model_to_eval(student)
    with torch.no_grad():
        for idx, val_image in enumerate(val_images):
            teacher_out = teacher(val_image)
            student_out = student(val_image)
            wandb.log(
                {
                    f"val/content_{idx}": wandb.Image(denormalize(val_image[0].cpu())),
                    f"val/teacher_{idx}": wandb.Image(teacher_out[0].cpu()),
                    f"val/student_qat_{idx}": wandb.Image(student_out[0].cpu()),
                },
                step=global_step,
            )
    torchao.quantization.pt2e.move_exported_model_to_train(student)


# Main QAT training loop


def qat_train(config: dict[str, Any], resume_path: Path | None = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # W&B
    wandb.init(
        project=config["wandb"]["project"],
        name=config["wandb"]["run_name"],
        config=config,
    )

    # Models
    teacher = load_teacher_model(config, device)
    student = build_qat_model(config, device)

    # Data
    dataloader, _ = build_dataloader(
        root=config["data"]["content_dir"],
        image_size=None,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        image_h=config["data"]["image_h"],
        image_w=config["data"]["image_w"],
    )

    # Optimizer & scheduler
    optimizer = Adam(
        params=student.parameters(), lr=config["training"]["learning_rate"]
    )
    num_epochs = config["training"]["epochs"]
    num_steps = len(dataloader) * num_epochs
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_steps,
        eta_min=config["training"]["scheduler"]["eta_min"],
    )

    # Resume
    start_epoch = 0
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device)
        student.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        logger.info("Resumed from epoch %d", start_epoch)

    # Validation images
    val_images = [
        load_image_h_w(path, config["data"]["image_h"], config["data"]["image_w"])
        .unsqueeze(0)
        .to(device)
        for path in sorted(Path(config["data"]["validation_dir"]).iterdir())
        if path.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]

    # Config shorthands
    cosine_weight = config["training"]["cosine_weight"]
    l2_weight = config["training"]["l2_weight"]
    max_grad_norm = config["training"]["max_grad_norm"]
    log_every = config["wandb"]["log_every_n_steps"]
    eval_interval = max(num_steps // 10, 1)
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    total_loss = None
    for epoch in range(start_epoch, num_epochs):
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
        )
        for idx, images in pbar:
            step_start = time.monotonic()
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)

            student_out = student(images)
            with torch.no_grad():
                teacher_out = teacher(images)

            total_loss, cos_sim, l2_val = compute_distillation_loss(
                teacher_out, student_out, cosine_weight, l2_weight
            )

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(), max_norm=max_grad_norm
            )
            optimizer.step()
            scheduler.step()

            batch_time = time.monotonic() - step_start
            global_step = epoch * len(dataloader) + idx

            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                cos_sim=f"{cos_sim:.4f}",
                l2=f"{l2_val:.4f}",
            )

            if (global_step) % log_every == 0:
                wandb.log(
                    {
                        "loss/total": total_loss.item(),
                        "loss/cosine": (1 - cos_sim) * cosine_weight,
                        "loss/l2": l2_val * l2_weight,
                        "training/cos_sim": cos_sim,
                        "training/l2": l2_val,
                        "training/learning_rate": optimizer.param_groups[0]["lr"],
                        "training/grad_norm": grad_norm.item(),
                        "training/batch_time": batch_time,
                    },
                    step=global_step,
                )

            if (global_step) % eval_interval == 0:
                log_val_images(teacher, student, val_images, global_step + 1)

        # Checkpoint after each epoch
        if total_loss is None:
            logger.warning("Epoch %d had no batches — skipping checkpoint.", epoch + 1)
            continue
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": total_loss.item(),
        }
        torch.save(checkpoint, checkpoint_dir / f"qat_checkpoint_{epoch}.pth")
        logger.info(
            "Epoch %d complete | loss=%.4f | saved to %s",
            epoch + 1,
            total_loss.item(),
            checkpoint_dir / f"qat_checkpoint_{epoch}.pth",
        )

    # Final: log val images + convert to true int8 + save
    log_val_images(teacher, student, val_images, num_steps)

    # Convert QAT model → truly quantized int8
    quantized_model = convert_pt2e(student)
    logger.info("Quantized model graph:\n%s", quantized_model.graph)

    torchao.quantization.pt2e.move_exported_model_to_eval(quantized_model)

    image_h = config["data"]["image_h"]
    image_w = config["data"]["image_w"]

    # Move to CPU for ExecuTorch export (XNNPACK targets CPU)
    quantized_model = quantized_model.cpu()
    example_inputs = (torch.randn(1, 3, image_h, image_w),)

    # Save as ExportedProgram (.pt2)
    final_path = Path(config["training"]["final_model_path"])
    os.makedirs(final_path.parent, exist_ok=True)
    quantized_ep = torch.export.export(quantized_model, example_inputs)
    pt2_path = final_path.with_suffix(".pt2")
    torch.export.save(quantized_ep, str(pt2_path))
    logger.info("Saved ExportedProgram to %s", pt2_path)
    # Lower to ExecuTorch (.pte)
    # QAT convert_pt2e already produced real quantized ops — skip PTQ, go straight to lowering
    logger.info("Lowering to ExecuTorch...")
    try:
        edge = to_edge_transform_and_lower(
            quantized_ep,
            partitioner=[XnnpackPartitioner()],
        )
    except Exception as e:
        logger.exception("to_edge_transform_and_lower failed: %s", e)
        raise

    logger.info("Analyzing delegation...")
    delegate_analysis = analyze_delegation(edge)
    logger.info("Delegation analysis complete")

    with open(final_path.parent / "delegate_analysis.json", "w") as f:
        json.dump(delegate_analysis, f, indent=4)
    logger.info(
        "Saved delegation analysis to %s", final_path.parent / "delegate_analysis.json"
    )
    et_program = edge.to_executorch()
    logger.info("ExecuTorch lowering complete")
    pte_path = final_path.with_suffix(".pte")
    with open(pte_path, "wb") as f:
        f.write(et_program.buffer)
    logger.info("Saved ExecuTorch program to %s", pte_path)

    # Build sidecar JSON for export_pipeline validate compatibility
    sidecar = {
        "tag": pte_path.stem,
        "config": {
            "checkpoint_path": config["model"]["checkpoint_path"],
            "model_type": config["model"]["model_type"],
            "se_attention": config["model"]["se_attention"],
            "norm_type": config["model"]["norm_type"],
            "input_h": image_h,
            "input_w": image_w,
            "keep_aspect": False,
            "export_mode": config["model"].get("export_mode", True),
            "backend": config["quantization"]["backend"],
            "quantize": True,
            "precision": "int8",
            "cosine_similarity_threshold": 0.99,
        },
        "checkpoint_hash": compute_checkpoint_hash(config["model"]["checkpoint_path"]),
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pytorch_version": str(torch.__version__),
        "input_shape": [1, 3, image_h, image_w],
        "output_shape": [1, 3, image_h, image_w],
        "delegation_analysis": delegate_analysis,
        "qat": True,
    }
    sidecar_path = pte_path.with_suffix(".json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    logger.info("Saved sidecar to %s", sidecar_path)

    wandb.finish()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    qat_train(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
