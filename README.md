# Real-Time Style Transfer

A PyTorch reimplementation of [*Perceptual Losses for Real-Time Style Transfer and Super-Resolution*](docs/paper/1603.08155v1.pdf) (Johnson et al., 2016).

The goal is to train a feed-forward transformer network that applies artistic style to arbitrary images in a single forward pass, then deploy the trained model on Android via ExecuTorch.

> **Status:** Training complete — trained model available at `models/fast-nst.pth`.

## Architecture

The system has two networks:

| Network | Role | Trainable |
|---|---|---|
| **Transformer Net** | Feed-forward image transformation network (encoder → residual blocks → decoder) | Yes |
| **Loss Net** (VGG-16) | Extracts feature maps for computing perceptual (content + style) losses | No (frozen) |

During training, the transformer net learns to minimize a weighted combination of content loss, style loss, and total variation loss. At inference time, only the transformer net is needed.

## Project Structure

```
├── configs/
│   └── train_config.yaml        # All hyperparameters
├── src/
│   ├── models/
│   │   ├── trans_net.py          # Feed-forward transformer network
│   │   └── loss_net.py           # VGG-16 feature extractor
│   ├── data/
│   │   └── dataset.py            # MS COCO dataset loader
│   ├── utils/
│   │   ├── gram.py               # Gram matrix computation
│   │   ├── loss.py               # Perceptual loss functions
│   │   └── image.py              # Image I/O and transforms
│   ├── train.py                  # Training loop with DDP & W&B logging
│   ├── inference.py              # Single-image stylization
│   └── export.py                 # Model export (TorchScript / ONNX)
├── tests/                        # Unit tests (pytest)
├── docs/
│   ├── paper/                    # Original paper PDF
│   └── guides/                   # Planning and deployment guides
└── pyproject.toml
```

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)

Key dependencies: PyTorch, TorchVision, Weights & Biases, ExecuTorch.

## Setup

```bash
# Clone the repository
git clone https://github.com/anime-sh16/rtst.git
cd rtst

# Create environment and install dependencies
uv sync --all-extras
```

## Usage

**Training** (requires MS COCO dataset):

Single GPU:
```bash
uv run python src/train.py --config configs/train_config.yaml
```

Multi-GPU with DDP via `torchrun`:
```bash
uv run torchrun --nproc_per_node=<NUM_GPUS> src/train.py --config configs/train_config.yaml
```

Resume from checkpoint:
```bash
uv run torchrun --nproc_per_node=<NUM_GPUS> src/train.py --config configs/train_config.yaml --resume models/checkpoints/checkpoint_0.pth
```

**Inference** (requires trained weights):
```bash
uv run python src/inference.py --image path/to/image.jpg --model path/to/weights.pth
```

**Export for mobile** (requires trained weights):
```bash
uv run python src/export.py --weights path/to/weights.pth --format torchscript
```

## Development

```bash
# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Test
uv run pytest tests/ -v
```

CI runs lint and tests on every push and pull request via GitHub Actions.

## References

- Johnson, J., Alahi, A., & Fei-Fei, L. (2016). *Perceptual Losses for Real-Time Style Transfer and Super-Resolution.* [arXiv:1603.08155](https://arxiv.org/abs/1603.08155)

## License

MIT
