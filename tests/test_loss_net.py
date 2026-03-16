import pytest
import torch
from torchvision import models
from torchvision.models import VGG16_Weights

from src.models.loss_net import CONTENT_LAYERS, STYLE_LAYERS, LossNetwork


@pytest.fixture()
def vgg(device: torch.device) -> torch.nn.Module:
    return models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)


@pytest.fixture()
def loss_net(vgg: torch.nn.Module, device: torch.device) -> LossNetwork:
    return LossNetwork(model=vgg).to(device)


class TestLossNetwork:
    def test_all_parameters_frozen(self, loss_net: LossNetwork) -> None:
        """VGG16 weights must not accumulate gradients during training."""
        for name, param in loss_net.named_parameters():
            assert not param.requires_grad, f"Parameter {name} should be frozen"

    def test_forward_returns_correct_layer_keys(
        self, loss_net: LossNetwork, normalised_batch: torch.Tensor
    ) -> None:
        """LossFeatures dicts must contain exactly the configured layer names."""
        with torch.no_grad(), loss_net as net:
            features = net(normalised_batch)
        assert set(features.content.keys()) == set(CONTENT_LAYERS)
        assert set(features.style.keys()) == set(STYLE_LAYERS)

    def test_hooks_cleaned_up_after_context_exit(self, loss_net: LossNetwork) -> None:
        """After the `with` block, handles and extracted_layers must be cleared."""
        with torch.no_grad(), loss_net:
            pass
        assert len(loss_net.handles) == 0, "Handles not cleared after context exit"
        assert len(loss_net.extracted_layers) == 0, (
            "extracted_layers not cleared after context exit"
        )
