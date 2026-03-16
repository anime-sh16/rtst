import torch


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    Compute the normalised Gram matrix of a feature map.

    Args:
        features: Tensor of shape (B, C, H, W).

    Returns:
        Gram matrix of shape (B, C, C), normalised by C*H*W.

    Raises:
        ValueError: If `features` is not 4-dimensional.
    """

    if features.ndim != 4:
        raise ValueError(f"Expected 4D feature map, got {features.ndim}D")

    # (B,C,H,W) -> (B,C,H*W)
    features_flattened = features.reshape(features.shape[0], features.shape[1], -1)
    features_flattened_transposed = features_flattened.transpose(-1, -2)

    # (B,C,H*W) * (B, H*W,C) -> (B,C,C)
    gram_matrix = torch.bmm(
        features_flattened, features_flattened_transposed
    )  # bmm because no brodcasting necessary

    return gram_matrix / (features_flattened.shape[-1] * features_flattened.shape[-2])
