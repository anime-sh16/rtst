import torch
import torch.nn as nn
from src.utils.gram import gram_matrix
from src.utils.warp import backward_warp
from src.utils.image import get_luminance


def compute_content_loss(
    generated_features: dict[str, torch.Tensor],
    content_features: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    MSE between content feature maps of the generated and content image.

    Args:
        generated_features: Feature maps from the generated image.
        content_features:   Feature maps from the original content image.

    Returns:
        Scalar loss tensor.
    """
    loss = sum(
        nn.functional.mse_loss(generated_features[k], content_features[k])
        for k in generated_features
    )
    return loss


def compute_style_loss(
    generated_features: dict[str, torch.Tensor],
    style_gram_targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    MSE between Gram matrices of the generated image and the precomputed style targets.

    Args:
        generated_features:  Feature maps from the generated image.
        style_gram_targets:  Precomputed Gram matrices for the style image.

    Returns:
        Scalar loss tensor.
    """
    loss = torch.tensor(0.0, device=next(iter(generated_features.values())).device)
    for k in generated_features:
        gen_gram = gram_matrix(generated_features[k])
        loss = loss + nn.functional.mse_loss(
            gen_gram, style_gram_targets[k].expand_as(gen_gram)
        )

    return loss


def compute_tv_loss(generated: torch.Tensor) -> torch.Tensor:
    """
    Total variation loss — penalises high-frequency noise in the output. Uses L1 regularisation.

    Args:
        generated: Batch of stylised images, shape (B, 3, H, W).

    Returns:
        Scalar loss tensor.
    """
    x_diff = torch.abs(generated[:, :, 1:, :] - generated[:, :, :-1, :])
    y_diff = torch.abs(generated[:, :, :, 1:] - generated[:, :, :, :-1])
    loss = torch.mean(x_diff[:, :, :, :-1] + y_diff[:, :, :-1, :])
    return loss


def compute_temporal_loss_output(
    frame_t: torch.Tensor,
    frame_t1: torch.Tensor,
    output_t: torch.Tensor,
    output_t1: torch.Tensor,
    feature_t: torch.Tensor,
    feature_t1: torch.Tensor,
    flow: torch.Tensor,
    occlusion_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ReCoNet-style temporal consistency losses between consecutive frames.

    Three components:
      1.1. Output temporal loss  — warped stylized_t vs stylized_t+1
      1.2. Luminance temporal loss — warped luminance_t vs luminance_t+1
      2. Feature temporal loss — warped encoded_t vs encoded_t+1

    All losses are masked by (1 - occlusion_mask) so that occluded regions
    (where flow is unreliable) don't contribute to the loss.

    Args:
        frame_t:         Input content frame t,            shape (B, 3, H, W).
        frame_t1:        Input content frame t+1,          shape (B, 3, H, W).
        output_t:        Stylized output for frame t,     shape (B, 3, H, W).
        output_t1:       Stylized output for frame t+1,   shape (B, 3, H, W).
        feature_t:       Encoded features for frame t,    shape (B, C, H', W').
        feature_t1:      Encoded features for frame t+1,  shape (B, C, H', W').
        flow:            Optical flow t→t+1,              shape (B, 2, H, W).
        occlusion_mask:  Occlusion mask,                  shape (B, 1, H, W).
                         1 = occluded, 0 = visible.

    Returns:
        Tuple of (output_temporal_loss, feature_temporal_loss).
    """

    # Visibility mask: invert occlusion (we want loss only on VISIBLE pixels)
    visibility_mask = 1.0 - occlusion_mask  # (B, 1, H, W)

    # --- 1. Output temporal loss ---
    B, _, out_h, out_w = output_t.shape
    warped_output_t = backward_warp(output_t, flow)
    output_diff = output_t1 - warped_output_t

    # --- 2. Luminance temporal loss ---
    lum_t = get_luminance(frame_t)
    lum_t1 = get_luminance(frame_t1)
    warped_lum_t = backward_warp(lum_t, flow)
    lum_diff = lum_t1 - warped_lum_t

    output_sq = (output_diff - lum_diff) ** 2  # (B, 3, H, W)
    output_temporal_loss = torch.mean(visibility_mask * output_sq)

    # --- 3. Feature temporal loss ---
    _, feat_c, feat_h, feat_w = feature_t.shape
    flow_resized = nn.functional.interpolate(
        flow, size=(feat_h, feat_w), mode="bilinear", align_corners=True
    )
    flow_resized[:, 0] *= feat_w / flow.shape[3]  # rescale horizontal
    flow_resized[:, 1] *= feat_h / flow.shape[2]  # rescale vertical
    mask_resized = nn.functional.interpolate(
        visibility_mask, size=(feat_h, feat_w), mode="nearest"
    )

    warped_feature_t = backward_warp(feature_t, flow_resized)
    feature_diff = mask_resized * (warped_feature_t - feature_t1)
    feature_temporal_loss = torch.mean(feature_diff**2)

    return output_temporal_loss, feature_temporal_loss
