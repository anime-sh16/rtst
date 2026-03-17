from typing import NamedTuple

import torch
import torch.nn as nn
from typing import Self


CONTENT_LAYERS: tuple[str, ...] = ("relu2_2",)
STYLE_LAYERS: tuple[str, ...] = ("relu1_2", "relu2_2", "relu3_3", "relu4_3")


class LossFeatures(NamedTuple):
    content: dict[str, torch.Tensor]
    style: dict[str, torch.Tensor]


class LossNetwork(nn.Module):
    """
    Wraps a pretrained, frozen VGG16 and returns intermediate feature maps
    needed to compute content and style losses.
    """

    def __init__(
        self,
        model,
        content_layers: tuple[str, ...] = CONTENT_LAYERS,
        style_layers: tuple[str, ...] = STYLE_LAYERS,
    ) -> None:
        super().__init__()
        self.content_layers = content_layers
        self.style_layers = style_layers
        self._required_layers = set(content_layers) | set(style_layers)

        # TODO: add early exit after relu4_3
        self.vgg16 = model
        self.vgg16.requires_grad_(False)
        self.vgg16.eval()
        self.layer_map = {
            "relu1_1": "1",
            "relu1_2": "3",
            "relu2_1": "6",
            "relu2_2": "8",
            "relu3_1": "11",
            "relu3_2": "13",
            "relu3_3": "15",
            "relu4_1": "18",
            "relu4_2": "20",
            "relu4_3": "22",
            "relu5_1": "25",
            "relu5_2": "27",
            "relu5_3": "29",
        }
        self.handles = []
        self.extracted_layers = {}

    def __enter__(self) -> Self:
        """Triggered automatically when the 'with' block starts."""
        self._register_hook()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Triggered automatically when the 'with' block ends, even on an error."""
        self._unregister_hook()
        self.extracted_layers.clear()  # Clear the heavy tensors from memory

    def _register_hook(self) -> None:
        for layer_name in self._required_layers:
            layer_index = self.layer_map.get(layer_name, None)
            if layer_index is None:
                raise ValueError(f"Invalid layer name: {layer_name}")
            layer = self.vgg16.features[int(layer_index)]
            handle = layer.register_forward_hook(
                lambda module, input, output, name=layer_name: (
                    self.extracted_layers.update({name: output})
                )
            )
            self.handles.append(handle)

    def _unregister_hook(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def forward(self, x: torch.Tensor) -> LossFeatures:
        """
        Pass `x` through the sliced VGG16 and collect feature maps at the
        content and style layers.

        Args:
            x: Image batch, shape (B, 3, H, W), values normalised to ImageNet mean/std.

        Returns:
            LossFeatures with two dicts: {layer_name → feature_tensor}.
        """
        self.vgg16.features(x)
        content = {}
        style = {}
        res = self.extracted_layers.copy()
        for key, value in res.items():
            if key in self.content_layers:
                content[key] = value
            if key in self.style_layers:
                style[key] = value

        return LossFeatures(content, style)
