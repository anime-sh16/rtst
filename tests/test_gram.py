import pytest
import torch

from src.utils.gram import gram_matrix


class TestGramMatrix:
    def test_output_shape(self, feature_map: torch.Tensor) -> None:
        """gram_matrix must return (B, C, C) from a (B, C, H, W) input."""
        B, C, _, _ = feature_map.shape
        result = gram_matrix(feature_map)
        assert result.shape == (B, C, C)

    def test_symmetry(self, feature_map: torch.Tensor) -> None:
        """Gram matrices are symmetric by construction: G[b] == G[b].T."""
        result = gram_matrix(feature_map)
        assert torch.allclose(result, result.transpose(-1, -2), atol=1e-5)

    def test_raises_on_non_4d_input(self) -> None:
        """Should raise ValueError for tensors that are not 4-dimensional."""
        with pytest.raises(ValueError):
            gram_matrix(torch.rand(3, 16, 16))  # 3D — wrong

    def test_normalisation_scaling(self, feature_map: torch.Tensor) -> None:
        """Scaling input by k should scale the Gram matrix by k²."""
        k = 3.0
        g1 = gram_matrix(feature_map)
        g2 = gram_matrix(feature_map * k)
        assert torch.allclose(g2, g1 * (k**2), atol=1e-4)
