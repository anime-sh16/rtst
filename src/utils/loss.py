import torch
import torch.nn as nn
from src.utils.gram import gram_matrix


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


def compute_temporal_loss(
    output_t: torch.Tensor,
    output_t1: torch.Tensor,
    feature_t: torch.Tensor,
    feature_t1: torch.Tensor,
    flow: torch.Tensor,
    occlusion_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute ReCoNet-style temporal consistency losses between consecutive frames.

    Three components:
      1. Output temporal loss  — warped stylized_t vs stylized_t+1
      2. Feature temporal loss — warped encoded_t vs encoded_t+1
      3. Luminance temporal loss — warped luminance_t vs luminance_t+1

    All losses are masked by (1 - occlusion_mask) so that occluded regions
    (where flow is unreliable) don't contribute to the loss.

    Args:
        output_t:        Stylized output for frame t,     shape (B, 3, H, W).
        output_t1:       Stylized output for frame t+1,   shape (B, 3, H, W).
        feature_t:       Encoded features for frame t,    shape (B, C, H', W').
        feature_t1:      Encoded features for frame t+1,  shape (B, C, H', W').
        flow:            Optical flow t→t+1,              shape (B, 2, H, W).
        occlusion_mask:  Occlusion mask,                  shape (B, 1, H, W).
                         1 = occluded, 0 = visible.

    Returns:
        Tuple of (output_temporal_loss, feature_temporal_loss, luminance_temporal_loss).
    """

    raise NotImplementedError
