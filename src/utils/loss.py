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
