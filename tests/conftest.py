"""
Shared pytest fixtures for the NST test suite.
"""

import pytest
import torch
from src.utils.image import get_device


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Return the best available device for this test session."""
    return get_device()


@pytest.fixture()
def image_batch(device: torch.device) -> torch.Tensor:
    """A small batch of random 'images' — shape (2, 3, 64, 64) in [0, 1]."""
    return torch.rand(2, 3, 64, 64, device=device)


@pytest.fixture()
def normalised_batch(image_batch: torch.Tensor) -> torch.Tensor:
    """Same batch normalised with ImageNet mean/std, as expected by the loss network."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_batch.device).view(
        1, 3, 1, 1
    )
    std = torch.tensor([0.229, 0.224, 0.225], device=image_batch.device).view(
        1, 3, 1, 1
    )
    return (image_batch - mean) / std


@pytest.fixture()
def feature_map(device: torch.device) -> torch.Tensor:
    """A random feature map — shape (2, 64, 16, 16) — for gram matrix tests."""
    return torch.rand(2, 64, 16, 16, device=device)


@pytest.fixture()
def feature_dict(device: torch.device) -> dict[str, torch.Tensor]:
    """Two-layer feature dict mimicking LossNetwork output — keys 'relu2_2' and 'relu3_3'."""
    return {
        "relu2_2": torch.rand(2, 128, 16, 16, device=device),
        "relu3_3": torch.rand(2, 256, 8, 8, device=device),
    }


@pytest.fixture()
def style_gram_targets(device: torch.device) -> dict[str, torch.Tensor]:
    """Precomputed style gram matrices matching the layers in feature_dict."""
    from src.utils.gram import gram_matrix

    return {
        "relu2_2": gram_matrix(torch.rand(2, 128, 16, 16, device=device)),
        "relu3_3": gram_matrix(torch.rand(2, 256, 8, 8, device=device)),
    }


@pytest.fixture()
def constant_image(device: torch.device) -> torch.Tensor:
    """A spatially constant image — TV loss should be exactly zero on this."""
    return torch.ones(2, 3, 64, 64, device=device) * 0.5
