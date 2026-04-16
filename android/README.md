# Android app

Kotlin app that runs the exported `.pte` style-transfer models on-device via the ExecuTorch Android runtime.

## Modes

- **Photo** — pick from gallery or capture, stylize, save to gallery.
- **Video** — real-time stylization of the camera feed (CameraX).
- **Benchmark** — on-device latency, memory, and FPS for the selected model.

## Models

Bundled `.pte` files live in [app/src/main/assets/](app/src/main/assets/) and are registered in [ModelConfig.kt](app/src/main/kotlin/com/rtst/app/ModelConfig.kt). Each entry declares its asset name, backend (`vulkan` / `xnnpack`), and input H×W. The default shipped config is MobileNet / Vulkan FP16 / 320×240 (~30 FPS on OnePlus 11). See the root [README](../README.md) for the full results table.

To add a new model, drop the `.pte` into `app/src/main/assets/` and append a `ModelConfig` entry — it shows up in the runtime model picker.

## Stack

- Kotlin, `minSdk = 28`, `targetSdk = 35`
- CameraX 1.4.1
- ExecuTorch Android SDK 1.1.0 (Vulkan variant)

## Build

```bash
cd android
./gradlew assembleDebug
./gradlew installDebug   # device must be connected via adb
```

Open in Android Studio for normal dev. `local.properties` needs `sdk.dir` pointing to your Android SDK.

## Layout

```
app/src/main/
├── kotlin/com/rtst/app/
│   ├── MainActivity.kt          # mode picker, model selector
│   ├── CameraActivity.kt        # video mode (CameraX)
│   ├── BenchmarkActivity.kt     # benchmark mode UI
│   ├── BenchmarkRunner.kt       # latency/memory/FPS measurement
│   ├── BenchmarkResult.kt
│   ├── GalleryAdapter.kt        # photo mode gallery
│   ├── StyleTransferRunner.kt   # ExecuTorch module wrapper
│   └── ModelConfig.kt           # registry of available .pte models
├── assets/                      # bundled .pte models + sample image
├── res/                         # layouts, drawables, values
└── AndroidManifest.xml
```
