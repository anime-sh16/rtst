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
│   ├── inference.py              # Single/batch image stylization
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

**Inference** (uses `models/fast-nst.pth` by default):

Single image:
```bash
uv run python src/inference.py --image path/to/image.jpg
```

Directory (batched, square resize):
```bash
uv run python src/inference.py --image path/to/dir/ --image-size 512 --batch-size 4
```

Directory (sequential, preserve aspect ratio):
```bash
uv run python src/inference.py --image path/to/dir/ --image-size 512 --keep-aspect
```

Results are saved to `data/results/` by default (override with `--output-dir`).

**Export for mobile** (requires trained weights):
```bash
uv run python src/export.py --weights path/to/weights.pth --format torchscript
```

## Training Configuration

The final model was trained on **MS COCO 2017** (~118k images) with the **mosaic** style image, using the following hyperparameters (after tuning):

| Parameter | Value |
|---|---|
| Image size | 256 × 256 |
| Batch size | 24 (per GPU) |
| GPU | T4 x2 (kaggle) |
| Epochs | 3 |
| Learning rate | 1e-3 (cosine schedule → 1e-6) |
| Content weight | 5.0 |
| Style weight | 2e5 |
| Normalization | InstanceNorm2d |
| Optimizer | Adam |

Style image: `data/styles/mosaic.jpg`

## Results

The trained model was evaluated at multiple resolutions to test generalization beyond the 256 × 256 training size:

- **256 × 256** — Training resolution. Strongest stylization.
- **512 × 512** — Upscaled. Style transfer still clearly visible, with slightly reduced intensity.
- **1024 × 1024** — 4× training size. Stylization becomes subtler but content is well preserved with visible stylization.
- **keep-aspect-512 / keep-aspect-1024** — Resized to target on the longer edge while preserving the original aspect ratio. Results are very similar to their square-resized counterparts, showing that maintaining aspect ratio has minimal impact on output quality.

**Key takeaways:**
1. The model generalizes well to resolutions higher than the training size.
2. Stylization intensity decreases gradually as resolution increases.
3. Aspect ratio preservation vs. square resize produces nearly identical results at the same target size.

### Comparison Table

| Input Image | Original | 256 | 512 | 1024 | keep-aspect-1024 |
|---|---|---|---|---|---|
| flower | ![](data/test_inference/flower.jpg) | ![](data/results/256/flower_style.jpg) | ![](data/results/512/flower_style.jpg) | ![](data/results/1024/flower_style.jpg) | ![](data/results/keep-aspect-1024/flower_style.jpg) |
| me-1 | ![](data/test_inference/me-1.jpg) | ![](data/results/256/me-1_style.jpg) | ![](data/results/512/me-1_style.jpg) | ![](data/results/1024/me-1_style.jpg) | ![](data/results/keep-aspect-1024/me-1_style.jpg) |
| neko | ![](data/test_inference/neko.jpg) | ![](data/results/256/neko_style.jpg) | ![](data/results/512/neko_style.jpg) | ![](data/results/1024/neko_style.jpg) | ![](data/results/keep-aspect-1024/neko_style.jpg) |
| pancake | ![](data/test_inference/pancake.jpg) | ![](data/results/256/pancake_style.jpg) | ![](data/results/512/pancake_style.jpg) | ![](data/results/1024/pancake_style.jpg) | ![](data/results/keep-aspect-1024/pancake_style.jpg) |
| random-scene-1 | ![](data/test_inference/random-scene-1.jpg) | ![](data/results/256/random-scene-1_style.jpg) | ![](data/results/512/random-scene-1_style.jpg) | ![](data/results/1024/random-scene-1_style.jpg) | ![](data/results/keep-aspect-1024/random-scene-1_style.jpg) |
| random-scene-2 | ![](data/test_inference/random-scene-2.jpg) | ![](data/results/256/random-scene-2_style.jpg) | ![](data/results/512/random-scene-2_style.jpg) | ![](data/results/1024/random-scene-2_style.jpg) | ![](data/results/keep-aspect-1024/random-scene-2_style.jpg) |

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
