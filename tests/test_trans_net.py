import pytest
import torch

from src.models.trans_net import TransformationNetwork


class TestTransformationNetwork:
    def test_output_shape_matches_input(self, image_batch: torch.Tensor) -> None:
        """The network must be spatially lossless — same HxW in and out."""
        model = TransformationNetwork().to(image_batch.device).eval()
        with torch.no_grad():
            output = model(image_batch)
        assert output.shape == image_batch.shape

    def test_output_range(self, image_batch: torch.Tensor) -> None:
        """Sigmoid at the tail of Upsample must keep all values in [0, 1]."""
        model = TransformationNetwork().to(image_batch.device).eval()
        with torch.no_grad():
            output = model(image_batch)
        assert output.min() >= 0.0, "Output has values below 0"
        assert output.max() <= 1.0, "Output has values above 1"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_runs_on_cuda(self) -> None:
        """Forward pass must complete without error on CUDA."""
        device = torch.device("cuda")
        model = TransformationNetwork().to(device).eval()
        x = torch.rand(1, 3, 64, 64, device=device)
        with torch.no_grad():
            model(x)

    @pytest.mark.skipif(not torch.mps.is_available(), reason="MPS not available")
    def test_runs_on_mps(self) -> None:
        """Forward pass must complete without error on MPS."""
        device = torch.device("mps")
        model = TransformationNetwork().to(device).eval()
        x = torch.rand(1, 3, 64, 64, device=device)
        with torch.no_grad():
            model(x)
