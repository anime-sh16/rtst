# Real-Time Neural Style Transfer on Android

Shipping a neural style transfer model to Android at ~30 FPS real-time video. The project started as a PyTorch reimplementation of [Johnson et al. (2016)](docs/paper/1603.08155v1.pdf), but the bulk of the work is the deployment path: picking a mobile backend and precision, recovering INT8 quality with quantization-aware training, and swapping the transformer for a MobileNet-style architecture to push latency down further.

<p align="center">
  <video src="data/results/video/mobilenet_vulkan_fp16_320x240.mp4" controls width="60%"></video>
  <br/>
  <em>Final model on OnePlus 11 — MobileNet transformer, Vulkan FP16, 320×240.</em>
</p>

## Results

Three iterations, each solving a specific deployment problem. Chosen configs only — full tables live inside each section.

| Version | Model | Backend / Precision | Resolution | Mean Latency | P95 | Notes |
|---|---|---|---|---|---|---|
| v1 | Johnson BN | Vulkan FP16 | 320×240 | 119 ms | 122 ms | First usable config |
| v2 | Johnson BN | XNNPACK INT8 (QAT) | 320×240 | 75 ms | 143 ms | INT8 without quality loss |
| **v3** | **MobileNet** | **Vulkan FP16** | **320×240** | **25.9 ms** | **27.3 ms** | **Shipped — ~30 FPS** |

## Starting point: PyTorch baseline

Johnson-style feed-forward style transfer: a transformer net (encoder → 5 residual blocks → decoder) trained against a frozen VGG-16 loss net using a weighted sum of content, style, and total-variation losses. At inference only the transformer net runs — a single forward pass stylizes an image.

Trained on **MS COCO 2017** (~118k images) with the mosaic style image on Kaggle (T4 × 2):

| Parameter | Value |
|---|---|
| Input size | 256 × 256 |
| Batch size | 24 per GPU |
| Epochs | 3 |
| Learning rate | 1e-3 → 1e-6 (cosine) |
| Content / style weight | 5.0 / 1.5e5 |
| Optimizer | Adam |

The model generalizes beyond its 256×256 training resolution:

| Original | 256 (train) | 512 | 1024 |
|---|---|---|---|
| ![](data/test_inference/flower.jpg) | ![](data/results/256/flower_style.jpg) | ![](data/results/512/flower_style.jpg) | ![](data/results/1024/flower_style.jpg) |

This baseline is the starting line, not the point of the project. The rest of this README is about getting it onto a phone.

## v1 — Picking a backend and precision

**Problem:** the paper-faithful model uses `InstanceNorm2d`, reflection padding, and nearest-neighbor upsampling — all of which hurt on mobile. On XNNPACK, only 32.5% of ops got delegated (the rest fell back to CPU), since InstanceNorm isn't a native XNNPACK op.

**Fix:** a second variant with `BatchNorm2d` and bilinear upsampling. Padding is reflection at train time (to avoid edge fading) and swapped to zero padding at inference — this swap is transparent to quality but is natively delegated. Bilinear upsampling, on the other hand, had to be used at train time as well: swapping nearest → bilinear only at inference introduced slight blurring, so the model was trained with bilinear from the start. Delegation jumped from 32.5% to 98.57%. Same loss recipe, same training data, negligible quality difference.

Then exported the BN variant across `{XNNPACK, Vulkan} × {FP32, FP16, INT8}` and benchmarked on device (OnePlus 11, Android 16). QNN (Qualcomm NPU) was also exported but couldn't be benchmarked — Android blocks direct Hexagon/NPU access for third-party apps via driver signing and SELinux. A LiteRT migration with the [Qualcomm QNN Accelerator](https://ai.google.dev/edge/litert/android/npu/qualcomm) would be the path forward there.

| Config | Backend | Mean | P95 |
|---|---|---|---|
| BN / FP32 / 640×480 | Vulkan | 603 ms | 608 ms |
| BN / FP16 / 320×240 | Vulkan | 119 ms | 122 ms |
| BN / INT8 (PTQ) / 320×240 | XNNPACK | 56 ms | 69 ms |

INT8 on XNNPACK was the fastest — but post-training quantization visibly trashed the output (washed colors, lost texture). Vulkan INT8 in ExecuTorch is limited for conv-heavy models (coverage is mostly linear ops), so FP16 was the ceiling on Vulkan. XNNPACK at FP16/FP32 was too slow on CPU (~934 ms at 640×480).

**v1 pick: Vulkan FP16** — best speed/quality trade-off at this stage.

## v2 — Recovering INT8 quality with QAT

**Problem:** INT8 on XNNPACK was 2× faster than Vulkan FP16 but the quality was unusable.

**Fix:** quantization-aware training. Two copies of the same architecture — one FP32, one with fake-quantization nodes inserted via `prepare_qat_pt2e`. The FP32 copy is frozen and supervises the fake-quant copy; the fake-quant copy's weights update so the model learns to be robust to the quantization error it will see post-`convert_pt2e`. (Same architecture on both sides — this isn't distillation in the usual cross-architecture sense, just FP32 supervision of a fake-quantized twin.)

- **Loss:** weighted `(1 − cosine_similarity)` + L2 between the two outputs
- **Schedule:** a few epochs on MS COCO, cosine LR
- **Export:** `convert_pt2e` → real INT8 ops → lowered directly to ExecuTorch `.pte`

```bash
uv run python src/qat.py --config configs/qat_config.yaml
```

| Config | Backend | Mean | P95 |
|---|---|---|---|
| v1: BN / FP16 / 640×480 | Vulkan | 454 ms | 457 ms |
| v1: BN / INT8 (PTQ) / 320×240 | XNNPACK | 56 ms | 69 ms |
| **v2: BN / INT8 (QAT) / 320×240** | **XNNPACK** | **75 ms** | **143 ms** |

<p align="center">
  <img src="data/results/int8/johnson%20xnn%20int8.jpg" width="45%" alt="XNNPACK INT8 (PTQ)" />
  <img src="data/results/int8/johnson%20xnn%20int8%20distilled.jpg" width="45%" alt="XNNPACK INT8 (QAT)" />
  <br/>
  <em>Left: post-training INT8 — washed-out colors, lost texture. Right: QAT INT8 — quality restored, latency unchanged.</em>
</p>

Output quality matches FP16 Vulkan; latency is ~1.5× faster. ~12 FPS peak on device.

**v2 pick: XNNPACK INT8 (QAT).**

## v3 — MobileNet transformer

**Problem:** even QAT INT8 capped around 12 FPS. To get to real-time video, the architecture itself had to change.

**Fix:** replace the transformer net's standard conv blocks with a MobileNet-style design — depthwise-separable convolutions, inverted residual blocks with linear bottlenecks (MobileNetV2-style), and the residual stack reduced from 5 → 3 blocks. Squeeze-and-excitation (SE) attention layers from MobileNetV3 were deliberately *not* used — they fragment the graph and hurt mobile delegation. To compensate for the reduced capacity, the **style weight was increased at train time** (the MobileNet variant needs a stronger style signal to match the Johnson BN output). Loss recipe and training data were otherwise unchanged.

| Config | Backend | Mean | P95 |
|---|---|---|---|
| MobileNet / FP16 / 320×240 | Vulkan | 25.9 ms | 27.3 ms |
| MobileNet / FP16 / 640×480 | Vulkan | 90.1 ms | 93.5 ms |
| MobileNet / FP32 / 320×240 | XNNPACK | 52.2 ms | 85.6 ms |
| MobileNet / FP32 / 640×480 | XNNPACK | 196.5 ms | 239.4 ms |
| MobileNet / INT8 (PTQ) / 320×240 | XNNPACK | 23.9 ms | 38.0 ms |

INT8 PTQ was fastest on paper but degraded quality again, and only saved ~2 ms over Vulkan FP16. Vulkan FP16 matched INT8 latency *without* needing QAT, so it became the shipped config.

**v3 pick (shipped): MobileNet / Vulkan FP16 / 320×240** — ~30 FPS, quality on par with Johnson BN.

## Android app

The app (`android/`) loads `.pte` models via the ExecuTorch Android runtime and supports:

- **Photo mode** — pick or capture a photo, apply style transfer, view in gallery
- **Video mode** — real-time stylization on the camera feed (CameraX)
- **Benchmark mode** — on-device latency, memory, and FPS
- **Model switching** — multiple `.pte` models bundled as assets, selectable at runtime

Kotlin, CameraX 1.4.1, ExecuTorch Android SDK 1.1.0 (Vulkan variant).

## Repo layout

```
├── src/
│   ├── models/           # Transformer (Johnson + MobileNet variants), VGG-16 loss net
│   ├── data/             # MS COCO dataset loader
│   ├── utils/            # Gram matrix, loss functions, image I/O
│   ├── train.py          # DDP training, W&B, cosine LR, resume
│   ├── inference.py      # Single/batch stylization
│   ├── export_pipeline.py # ExecuTorch export + host validation + device benchmark
│   └── qat.py            # Quantization-aware training (PT2E)
├── android/              # Kotlin Android app (ExecuTorch runtime)
├── cpp_eval/             # C++ runner for host-side .pte validation
├── configs/              # Training, export, and QAT YAML configs
├── data/                 # Styles, test images, inference results
├── exports/              # .pte files and JSON sidecars
├── tests/                # pytest
└── docs/                 # Guides and original paper
```

## Setup & usage

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/anime-sh16/rtst.git
cd rtst
uv sync --all-extras
```

```bash
# Train
uv run torchrun --nproc_per_node=<N> src/train.py --config configs/train_config.yaml

# Inference
uv run python src/inference.py --image path/to/image.jpg

# Export → validate → benchmark on device
uv run python src/export_pipeline.py full --config configs/export_configs/<config>.yaml --device-id <adb_device>

# QAT
uv run python src/qat.py --config configs/template/qat_config.yaml

# Dev
uv run ruff check src/ tests/ && uv run ruff format src/ tests/
uv run pytest tests/ -v
```

## References

- Johnson, J., Alahi, A., & Fei-Fei, L. (2016). *Perceptual Losses for Real-Time Style Transfer and Super-Resolution.* [arXiv:1603.08155](https://arxiv.org/abs/1603.08155)
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)
- Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q. V., & Adam, H. (2019). *Searching for MobileNetV3.* [arXiv:1905.02244](https://arxiv.org/abs/1905.02244)

## License

MIT
