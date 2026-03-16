import pytest
import torch

from src.utils.loss import compute_tv_loss


class TestTVLoss:
    def test_returns_scalar(self, image_batch: torch.Tensor) -> None:
        """TV loss on a (B, 3, H, W) image batch must return a scalar."""
        loss = compute_tv_loss(image_batch)
        assert loss.shape == ()

    def test_zero_on_constant_image(self, constant_image: torch.Tensor) -> None:
        """TV loss on a spatially constant image must be exactly zero."""
        loss = compute_tv_loss(constant_image)
        assert loss.item() == pytest.approx(0.0)
