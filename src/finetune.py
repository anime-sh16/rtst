# ruff: noqa: F841

"""
Fine-tune a pretrained style transfer model with temporal consistency loss.
Tune all parameters in the model.

Usage:
    uv run python src/finetune.py --config configs/finetune_config.yaml --weights models/fast-nst.pth
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any
import time
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
    load_image,
    normalize,
    denormalize,
    IMAGENET_MEAN_RESHAPED,
    IMAGENET_STD_RESHAPED,
)
from src.utils.loss import (
    compute_content_loss,
    compute_style_loss,
    compute_temporal_loss_output,
    compute_tv_loss,
)
from src.utils.gram import gram_matrix
from tqdm import tqdm

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

    # --- Video DataLoader ---
    dataloader, sampler = build_video_dataloader(
        root=config["data"]["sintel_root"],
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
    with loss_net as extractor:
        for epoch in range(start_epoch, config["training"]["epochs"]):
            if sampler is not None:
                sampler.set_epoch(epoch)

            pbar = tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                desc=f"Epoch {epoch + 1}/{config['training']['epochs']}",
                disable=not is_main,
            )

            for step, video_frame_pair in pbar:
                step_start = time.monotonic()
                optimizer.zero_grad(set_to_none=True)

                frame_t = video_frame_pair.frame_t.to(device)
                frame_t1 = video_frame_pair.frame_t1.to(device)
                flow = video_frame_pair.flow.to(device)
                occlusion_mask = video_frame_pair.occlusion_mask.to(device)

                frame_t_raw = denormalize(frame_t, mean=img_mean, std=img_std)
                frame_t1_raw = denormalize(frame_t1, mean=img_mean, std=img_std)

                # Single forward pass on stacked pair — then split.
                # (frame_t1 path is only used for temporal loss.)
                frames_pair = torch.cat([frame_t, frame_t1], dim=0)
                outputs_pair, features_pair = trans_net(frames_pair)
                output_t, output_t1 = outputs_pair.chunk(2, dim=0)
                feature_t, feature_t1 = features_pair.chunk(2, dim=0)

                output_t_norm = normalize(output_t, mean=img_mean, std=img_std)
                generated_features_vgg = extractor(output_t_norm)

                with torch.no_grad():
                    frame_features_vgg = extractor(frame_t)

                # Content Loss
                content_loss_t = compute_content_loss(
                    generated_features=generated_features_vgg.content,
                    content_features=frame_features_vgg.content,
                )

                # Style Loss
                style_loss_t = compute_style_loss(
                    generated_features=generated_features_vgg.style,
                    style_gram_targets=style_target_features,
                )

                # TV loss
                tv_loss_t = compute_tv_loss(output_t_norm)

                # Temporal loss
                temporal_loss_output, temporal_loss_feature = (
                    compute_temporal_loss_output(
                        frame_t=frame_t_raw,
                        frame_t1=frame_t1_raw,
                        output_t=output_t,
                        output_t1=output_t1,
                        feature_t=feature_t,
                        feature_t1=feature_t1,
                        flow=flow,
                        occlusion_mask=occlusion_mask,
                    )
                )

                total_loss = (
                    config["training"]["content_weight"] * content_loss_t
                    + config["training"]["style_weight"] * style_loss_t
                    + config["training"]["tv_weight"] * tv_loss_t
                    + config["training"]["temporal_output_weight"]
                    * temporal_loss_output
                    + config["training"]["temporal_feature_weight"]
                    * temporal_loss_feature
                )

                total_loss.backward()
                grads = [p.grad for p in trans_net.parameters() if p.grad is not None]
                grad_norm = torch.nn.utils.get_total_norm(grads)

                optimizer.step()
                scheduler.step()
                batch_time = time.monotonic() - step_start

                # Logging
                global_step = epoch * len(dataloader) + step
                w_content = config["training"]["content_weight"] * content_loss_t.item()
                w_style = config["training"]["style_weight"] * style_loss_t.item()
                w_tv = config["training"]["tv_weight"] * tv_loss_t.item()
                w_temp_o = (
                    config["training"]["temporal_output_weight"]
                    * temporal_loss_output.item()
                )
                w_temp_f = (
                    config["training"]["temporal_feature_weight"]
                    * temporal_loss_feature.item()
                )

                pbar.set_postfix(
                    total=f"{total_loss.item():.4f}",
                    temp_o=f"{w_temp_o:.4f}",
                    temp_f=f"{w_temp_f:.4f}",
                    content=f"{w_content:.4f}",
                    style=f"{w_style:.4f}",
                )

                if is_main and step % config["wandb"]["log_every_n_steps"] == 0:
                    wandb.log(
                        {
                            "loss/total": total_loss.item(),
                            "loss/temporal_output": w_temp_o,
                            "loss/temporal_feature": w_temp_f,
                            "loss/content": w_content,
                            "loss/style": w_style,
                            "loss/tv": w_tv,
                            "raw_loss/temporal_output": temporal_loss_output.item(),
                            "raw_loss/temporal_feature": temporal_loss_feature.item(),
                            "raw_loss/content": content_loss_t.item(),
                            "raw_loss/style": style_loss_t.item(),
                            "raw_loss/tv": tv_loss_t.item(),
                            "training/learning_rate": optimizer.param_groups[0]["lr"],
                            "training/global_step": global_step,
                            "training/grad_norm": grad_norm.item(),
                            "training/batch_time": batch_time,
                            "weights/temporal_output_loss": config["training"][
                                "temporal_output_weight"
                            ],
                            "weights/temporal_feature_loss": config["training"][
                                "temporal_feature_weight"
                            ],
                            "weights/content_loss": config["training"][
                                "content_weight"
                            ],
                            "weights/style_loss": config["training"]["style_weight"],
                            "weights/tv_loss": config["training"]["tv_weight"],
                        },
                        step=global_step,
                    )

            if is_main:
                model_to_save = (
                    trans_net.module if isinstance(trans_net, DDP) else trans_net
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
