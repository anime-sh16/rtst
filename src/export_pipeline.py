import argparse
import hashlib
import json
import logging
import subprocess
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import shutil
import os
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a clean, readable logger with color-coded levels."""

    class ColorFormatter(logging.Formatter):
        COLORS = {
            logging.DEBUG: "\033[90m",  # gray
            logging.INFO: "\033[36m",  # cyan
            logging.WARNING: "\033[33m",  # yellow
            logging.ERROR: "\033[31m",  # red
            logging.CRITICAL: "\033[1;31m",  # bold red
        }
        RESET = "\033[0m"
        BOLD = "\033[1m"

        def format(self, record: logging.LogRecord) -> str:
            color = self.COLORS.get(record.levelno, self.RESET)
            ts = time.strftime("%H:%M:%S", time.localtime(record.created))
            ms = f"{record.created % 1:.3f}"[1:]
            level = record.levelname.ljust(8)
            name = record.name.split(".")[-1]
            msg = record.getMessage()

            return (
                f"{self.BOLD}{color}{ts}{ms}{self.RESET}  "
                f"{color}{level}{self.RESET}  "
                f"\033[90m{name:<12}{self.RESET}  "
                f"{msg}"
            )

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())
    logging.root.handlers.clear()
    logging.root.addHandler(handler)
    logging.root.setLevel(level)


# 1. Config Schema


class NormType(str, Enum):
    INSTANCE = "in"
    BATCH = "bn"


class Backend(str, Enum):
    XNNPACK = "xnnpack"
    VULKAN = "vulkan"
    CPU = "cpu"  # no delegation, pure portable ops


@dataclass
class ExportConfig:
    # Model
    checkpoint_path: str  # path to .pth
    norm_type: NormType  # "in" or "bn"
    style_name: str  # e.g. "starry_night", for naming

    # Export
    backend: Backend  # "xnnpack", "vulkan", "cpu"
    quantize: bool = False  # whether to apply ptq via torchao
    # input_size: int | None = 256
    input_h: int = 256
    input_w: int = 256
    keep_aspect: bool = False
    export_mode: bool = True  # Using export mode changes the reflection tp zero padding and nearest to bilinear upsampling

    # Validation
    cosine_similarity_threshold: float = 0.99

    # Paths (auto-populated)
    export_dir: str = "exports"
    results_dir: str = "results"

    @property
    def tag(self) -> str:
        """Unique tag encoding this config. Used for filenames."""
        q = "int8" if self.quantize else "fp32"
        return f"johnson_{self.norm_type.value}_{self.style_name}_{self.backend.value}_{q}_{self.input_h}x{self.input_w}{'_aspect' if self.keep_aspect else ''}{'_export_mode' if self.export_mode else ''}"

    @property
    def pte_path(self) -> Path:
        return Path(self.export_dir) / f"{self.tag}" / f"{self.tag}.pte"

    @property
    def sidecar_path(self) -> Path:
        return Path(self.export_dir) / f"{self.tag}" / f"{self.tag}.json"

    @property
    def debug_path(self) -> Path:
        return Path(self.results_dir) / "debug" / f"{self.tag}"

    @staticmethod
    def from_yaml(path: str) -> "ExportConfig":
        """Load config from YAML file."""
        import yaml

        with open(path) as f:
            d = yaml.safe_load(f)
        d["norm_type"] = NormType(d["norm_type"])
        d["backend"] = Backend(d["backend"])
        return ExportConfig(**d)


# 2. Export
#    config-driven naming + sidecar generation.


def compute_checkpoint_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def analyze_delegation(edge_program) -> dict:
    """
    Walk the graph of a lowered EdgeProgramManager and count ops
    delegated to each backend vs. CPU fallback.

    Returns dict with:
        total_ops, delegated_ops_total, cpu_fallback_ops_total,
        delegation_pct, backend_details (dict of backend->ops->count),
        cpu_op_names (dict of op->count)
    """
    graph_module = edge_program.exported_program().graph_module

    result = {
        "total_ops": 0,
        "delegated_ops_total": 0,
        "cpu_fallback_ops_total": 0,
        "delegation_pct": 0.0,
        "backend_details": {},
        "cpu_op_names": {},
    }

    for node in graph_module.graph.nodes:
        # Skip pure IO/memory nodes in the main graph
        if node.op in ["placeholder", "output", "get_attr"]:
            continue

        # 1. HANDLE DELEGATED OPS
        if (
            node.op == "call_function"
            and getattr(node.target, "__name__", "") == "executorch_call_delegate"
        ):
            lowered_module_node = node.args[0]
            lowered_module = getattr(graph_module, lowered_module_node.target)
            backend_id = lowered_module.backend_id

            if backend_id not in result["backend_details"]:
                result["backend_details"][backend_id] = {}

            subgraph = lowered_module.original_module.graph_module.graph

            for sub_node in subgraph.nodes:
                if sub_node.op in ["placeholder", "output", "get_attr"]:
                    continue

                # Standardize op name
                op_name = (
                    getattr(sub_node.target, "__name__", str(sub_node.target))
                    if sub_node.op == "call_function"
                    else (
                        sub_node.target if sub_node.op == "call_method" else sub_node.op
                    )
                )

                # Tally delegated op
                result["backend_details"][backend_id][op_name] = (
                    result["backend_details"][backend_id].get(op_name, 0) + 1
                )
                result["delegated_ops_total"] += 1

        # 2. HANDLE CPU FALLBACK OPS
        else:
            # Standardize op name
            op_name = (
                getattr(node.target, "__name__", str(node.target))
                if node.op == "call_function"
                else (node.target if node.op == "call_method" else node.op)
            )

            # Tally CPU op
            result["cpu_op_names"][op_name] = result["cpu_op_names"].get(op_name, 0) + 1
            result["cpu_fallback_ops_total"] += 1

    # 3. CALCULATE TOTALS AND PERCENTAGE
    result["total_ops"] = (
        result["delegated_ops_total"] + result["cpu_fallback_ops_total"]
    )
    if result["total_ops"] > 0:
        result["delegation_pct"] = round(
            (result["delegated_ops_total"] / result["total_ops"]) * 100, 2
        )

    return result


CALIB_IMAGES_DIR = Path("data/coco/test1000")


def _load_calib_inputs(
    calib_dir: Path, input_h: int, input_w: int, max_images: int = 500
) -> list[torch.Tensor]:
    """Load images from calib_dir, preprocess them, and return a list of tensors.

    Each tensor should be shape (1, 3, input_h, input_w), normalized to [0, 1].
    """
    from PIL import Image
    from torchvision.transforms import v2

    transform = v2.Compose(
        [
            v2.Resize((input_h, input_w)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    image_paths = sorted(
        p for p in calib_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_paths:
        raise FileNotFoundError(f"No calibration images found in {calib_dir}")

    tensors = []
    for path in image_paths[:max_images]:
        img = Image.open(path).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        tensors.append(tensor)

    logger.info("Loaded %d calibration images from %s", len(tensors), calib_dir)
    return tensors


def _quantize_vulkan_int8(exported_program, calib_inputs=None):
    """TODO: Update the default calib inputs and caliberation method"""

    from executorch.backends.vulkan.quantizer.vulkan_quantizer import (
        get_symmetric_quantization_config,
        VulkanQuantizer,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    quantizer = VulkanQuantizer()
    quantizer.set_global(
        get_symmetric_quantization_config(is_dynamic=False, weight_bits=8)
    )

    quantized_module = prepare_pt2e(exported_program.module(), quantizer)

    for inp in calib_inputs:
        quantized_module(inp)
    quantized_module = convert_pt2e(quantized_module)

    return quantized_module


def _quantize_xnnpack_int8(exported_program, calib_inputs=None):
    """TODO: Update the default calib inputs and caliberation method"""

    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    qparams = get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False)
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(qparams)

    prepared_model = prepare_pt2e(exported_program.module(), quantizer)

    for inp in calib_inputs:
        prepared_model(inp)
    quantized_module = convert_pt2e(prepared_model)

    return quantized_module


def export_model(cfg: ExportConfig):
    """
    TODO: Add the keep aspect flag usage for the dimension range to `export()`
    """
    Path(cfg.pte_path.parent).mkdir(parents=True, exist_ok=True)

    logger.info(f"[export] Loading checkpoint: {cfg.checkpoint_path}")
    logger.info(f"[export] Config tag: {cfg.tag}")

    from src.models.trans_net import TransformationNetwork
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )

    norm_type = (
        nn.InstanceNorm2d if cfg.norm_type == NormType.INSTANCE else nn.BatchNorm2d
    )
    model = TransformationNetwork(
        norm_layer_type=norm_type, export_mode=cfg.export_mode
    )
    model.load_state_dict(torch.load(cfg.checkpoint_path, map_location="cpu"))
    model.eval()

    if cfg.keep_aspect:
        # TODO: implement keep_aspect
        # For now, only support square inputs
        logger.error(
            "[export] Keep aspect not supported yet. Please use square inputs."
        )
        raise NotImplementedError

    example_input = (torch.randn(1, 3, cfg.input_h, cfg.input_w),)
    exported = torch.export.export(model, example_input)

    if cfg.quantize and cfg.backend == Backend.VULKAN:
        calib_inputs = _load_calib_inputs(CALIB_IMAGES_DIR, cfg.input_h, cfg.input_w)
        quantized_module = _quantize_vulkan_int8(exported, calib_inputs)
        exported = torch.export.export(quantized_module, example_input)
    elif cfg.quantize and cfg.backend == Backend.XNNPACK:
        calib_inputs = _load_calib_inputs(CALIB_IMAGES_DIR, cfg.input_h, cfg.input_w)
        quantized_module = _quantize_xnnpack_int8(exported, calib_inputs)
        exported = torch.export.export(quantized_module, example_input)

    partitioner = {
        Backend.XNNPACK: [XnnpackPartitioner()],
        Backend.VULKAN: [VulkanPartitioner(), XnnpackPartitioner()],
        Backend.CPU: None,
    }[cfg.backend]

    if partitioner:
        edge = to_edge_transform_and_lower(
            exported, partitioner=partitioner, generate_etrecord=True
        )
    else:
        edge = to_edge_transform_and_lower(exported, generate_etrecord=True)

    delegate_analysis = analyze_delegation(edge)

    et_program = edge.to_executorch()

    etrecord = et_program.get_etrecord()
    etrecord.update_representative_inputs(example_input)
    etrecord_path = cfg.pte_path.with_suffix(".bin")
    etrecord.save(str(etrecord_path))

    logger.info(f"[export] ETRecord saved: {etrecord_path}")

    with open(cfg.pte_path, "wb") as f:
        f.write(et_program.buffer)

    size_mb = cfg.pte_path.stat().st_size / 1024 / 1024
    logger.info(f"[export] Exported: {cfg.pte_path} ({size_mb:.2f} MB)")

    from importlib.metadata import version

    ckpt_hash = compute_checkpoint_hash(cfg.checkpoint_path)
    sidecar = {
        "tag": cfg.tag,
        "config": asdict(cfg),
        "checkpoint_hash": ckpt_hash,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "executorch_version": str(version("executorch")),
        "pytorch_version": str(torch.__version__),
        "input_shape": [1, 3, cfg.input_h, cfg.input_w],
        "output_shape": [1, 3, cfg.input_h, cfg.input_w],
        "delegation_analysis": delegate_analysis,
    }
    # Enum serialization fix
    sidecar["config"]["norm_type"] = cfg.norm_type.value
    sidecar["config"]["backend"] = cfg.backend.value

    with open(cfg.sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    logger.info(f"[export] Sidecar written: {cfg.sidecar_path}")
    return cfg.pte_path


# 3. Host Validation
#    Run .pte on desktop, compare vs PyTorch.


def load_test_image(
    path: str, h: int, w: int | None, keep_aspect: bool | None
) -> torch.Tensor:
    """
    Load an image from disk into a normalised tensor (1, C, H, W).
    """
    if keep_aspect:
        from src.utils.image import load_image

        img = load_image(path, size=min(h, w), keep_aspect=True)
    else:
        from src.utils.image import load_image_h_w

        img = load_image_h_w(path, h, w)

    img = img.unsqueeze(0)
    return img


def validate_on_host(pte_path: str, ref_image_path: str):
    """
    Compare .pte output vs original PyTorch model output on the same input.
    Uses ExecuTorch Python Runtime API with .contiguous() workaround for
    the channels-last stride bug.
    """
    from executorch.runtime import Runtime, Verification
    from src.models.trans_net import TransformationNetwork
    from src.utils.image import save_image

    sidecar_path = pte_path.replace(".pte", ".json")
    with open(sidecar_path) as f:
        sidecar = json.load(f)

    cfg = sidecar["config"]
    h, w, keep_aspect, export_mode = (
        cfg["input_h"],
        cfg["input_w"],
        cfg["keep_aspect"],
        cfg["export_mode"],
    )

    logger.info(f"[validate] Loading test image: {ref_image_path}")
    test_input = load_test_image(ref_image_path, h, w, keep_aspect)

    # --- PyTorch reference inference ---
    norm_type = (
        nn.InstanceNorm2d
        if NormType(cfg["norm_type"]) == NormType.INSTANCE
        else nn.BatchNorm2d
    )

    model = TransformationNetwork(norm_layer_type=norm_type, export_mode=export_mode)
    model.load_state_dict(torch.load(cfg["checkpoint_path"], map_location="cpu"))
    model.eval()
    with torch.no_grad():
        ref_output = model(test_input)

    # --- ExecuTorch Python Runtime inference ---
    # .contiguous() is required to avoid the channels-last stride bug
    # where the pybinding reads raw data pointers as NCHW regardless of
    # actual tensor strides. See scripts/minimal_repro.py for details.
    runtime = Runtime.get()
    program = runtime.load_program(pte_path, verification=Verification.Minimal)
    method = program.load_method("forward")
    et_outputs = method.execute([test_input.contiguous()])
    et_output = et_outputs[0]

    if not isinstance(et_output, torch.Tensor):
        et_output = torch.tensor(et_output)
    et_output = et_output.reshape(1, 3, h, w)

    logger.info(
        f"[validate] ExecuTorch output shape={et_output.shape}, "
        f"dtype={et_output.dtype}, range=[{et_output.min():.3f}, {et_output.max():.3f}]"
    )

    # --- C++ runner path (kept for reference / future Android host validation) ---
    # CPP_RUNNER = Path("cpp_eval/build/rtst")
    #
    # import tempfile
    # import numpy as np
    #
    # with tempfile.TemporaryDirectory() as tmp_dir:
    #     input_bin = Path(tmp_dir) / "input.bin"
    #     output_bin = Path(tmp_dir) / "output.bin"
    #
    #     test_input.contiguous().numpy().tofile(input_bin)
    #
    #     try:
    #         result = subprocess.run(
    #             [
    #                 str(CPP_RUNNER),
    #                 "validate",
    #                 pte_path,
    #                 str(input_bin),
    #                 str(output_bin),
    #                 str(h),
    #                 str(w),
    #             ],
    #             check=True,
    #             capture_output=True,
    #             text=True,
    #         )
    #         logger.info(f"[validate] C++ runner: {result.stdout.strip()}")
    #     except subprocess.CalledProcessError as e:
    #         logger.error(f"[validate] C++ runner failed:\n{e.stderr.strip()}")
    #         raise
    #
    #     et_output = torch.from_numpy(
    #         np.fromfile(output_bin, dtype=np.float32).reshape(1, 3, h, w)
    #     )

    # --- Compare ---
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_output.flatten().unsqueeze(0), et_output.flatten().unsqueeze(0)
    ).item()

    l2_diff = torch.norm(ref_output - et_output).item()
    linf_diff = torch.max(torch.abs(ref_output - et_output)).item()

    result = {
        "pte": pte_path,
        "tag": sidecar["tag"],
        "ref_image": ref_image_path,
        "cosine_similarity": cos_sim,
        "l2_norm_diff": l2_diff,
        "linf_diff": linf_diff,
        "pass": cos_sim > cfg["cosine_similarity_threshold"],
        "sidecar": sidecar,
    }

    status = "PASS" if result["pass"] else "FAIL"
    logger.info(
        f"[validate] {status}  cosine={cos_sim:.6f}  L2={l2_diff:.4f}  Linf={linf_diff:.4f}"
    )

    # Save validation result
    val_dir = Path("results") / "host_validation" / f"{sidecar['tag']}"
    val_dir.mkdir(parents=True, exist_ok=True)
    out_path = val_dir / f"{sidecar['tag']}_validation.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"[validate] Validation result written: {out_path}")

    save_image(ref_output[0], val_dir / f"{sidecar['tag']}_ref.png")
    save_image(et_output[0], val_dir / f"{sidecar['tag']}_et.png")

    logger.info(f"[validate] Validation images written: {val_dir}")

    return result


# 4. Device Benchmark (adb orchestration)
#    Push .pte to device, run harness, pull results.
DEVICE_BENCH_DIR = "/data/local/tmp/rtst_bench"


def adb(cmd: str, device_id: str | None = None) -> str:
    """Run an adb command, return stdout."""
    # 1. Try to find adb in the system PATH
    adb_path = shutil.which("adb")

    # 2. If Python can't find it in PATH, hardcode the macOS default path
    if not adb_path:
        mac_default = os.path.expanduser("~/Library/Android/sdk/platform-tools/adb")
        if os.path.exists(mac_default):
            adb_path = mac_default
        else:
            raise FileNotFoundError(
                "Could not find 'adb'. Ensure Android Studio is installed and "
                "the platform-tools directory exists."
            )

    prefix = [adb_path]
    if device_id:
        prefix += ["-s", device_id]
    full = prefix + cmd.split()
    r = subprocess.run(full, capture_output=True, text=True, check=True)
    return r.stdout.strip()


def _resolve_adb_path() -> str:
    """Find the adb binary path."""
    adb_path = shutil.which("adb")
    if not adb_path:
        mac_default = os.path.expanduser("~/Library/Android/sdk/platform-tools/adb")
        if os.path.exists(mac_default):
            adb_path = mac_default
        else:
            raise FileNotFoundError(
                "Could not find 'adb'. Ensure Android Studio is installed and "
                "the platform-tools directory exists."
            )
    return adb_path


def adb_pull_from_app(
    app_id: str, remote_path: str, local_path: str, device_id: str | None = None
) -> None:
    """
    Pull a file from an app's private storage using `adb exec-out run-as <app> cat <path>`.

    `exec-out` streams raw bytes (no PTY) so binary files like JPEGs stay intact.
    The file is written locally in binary mode.
    """
    adb_path = _resolve_adb_path()
    prefix = [adb_path]
    if device_id:
        prefix += ["-s", device_id]
    cmd = prefix + ["exec-out", "run-as", app_id, "cat", remote_path]
    with open(local_path, "wb") as f:
        _ = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, check=True)
    logger.info(f"[adb] Pulled {remote_path} -> {local_path}")


def device_benchmark(
    pte_path: str, ref_image: str, device_id: str | None = None, n_iters: int = 50
):
    """
    Push .pte + test image to device, run the benchmark binary/app,
    pull results back.

    Uses ExecuTorch's built-in `executor_runner` binary (build for Android ARM64)

    This function handles the adb orchestration around whichever you use.
    """
    sidecar_path = pte_path.replace(".pte", ".json")
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    tag = sidecar["tag"]
    backend = sidecar["config"]["backend"]

    logger.info(f"[device] Benchmarking {tag} on device {device_id or '(default)'}")

    # Setup device directory
    adb(f"shell mkdir -p {DEVICE_BENCH_DIR}", device_id)

    # Push .pte and test image
    logger.info(f"[device] Pushing {pte_path} ...")
    adb(f"push {pte_path} {DEVICE_BENCH_DIR}/model.pte", device_id)
    adb(f"push {ref_image} {DEVICE_BENCH_DIR}/input.jpg", device_id)

    logger.info("[device] Granting read/write permissions to app sandbox...")
    adb(f"shell chmod -R 777 {DEVICE_BENCH_DIR}", device_id)

    # Clear stale results from previous runs so the poll doesn't find old files
    try:
        adb("shell run-as com.rtst.app rm files/bench.json files/output.jpg", device_id)
    except subprocess.CalledProcessError:
        pass  # files don't exist yet, that's fine

    # Benchmark app via intent to measure actual app lifecycle preformance
    logger.info("[device] Launching Android BenchmarkActivity...")
    adb(
        f"shell am start -n com.rtst.app/.BenchmarkActivity "
        f"--es model_path {DEVICE_BENCH_DIR}/model.pte "
        f"--es tag {tag} "
        f"--es backend {backend} "
        f"--ei warmup 10 "
        f"--ei iters {n_iters}",
        device_id,
    )

    # 4. Wait for the app to finish processing
    # We poll the device to see if 'bench.json' has been written yet.
    logger.info("[device] Waiting for benchmark to complete...")
    max_wait_seconds = (n_iters + 10) * 5
    poll_interval = 5
    elapsed = 0
    benchmark_complete = False

    while elapsed < max_wait_seconds:
        try:
            # 'test -f' checks if the file exists. It returns 0 (success) if it does.
            adb("shell run-as com.rtst.app test -f files/bench.json", device_id)
            benchmark_complete = True
            break
        except subprocess.CalledProcessError:
            # File doesn't exist yet, wait and try again
            time.sleep(poll_interval)
            elapsed += poll_interval

    if not benchmark_complete:
        raise TimeoutError(
            f"Benchmark timed out after {max_wait_seconds} seconds. App may have crashed."
        )

    logger.info("[device] Benchmark finished successfully.")

    # Pull results
    results_dir = Path("results") / "device" / tag
    results_dir.mkdir(parents=True, exist_ok=True)

    adb_pull_from_app(
        "com.rtst.app", "files/bench.json", f"{results_dir}/bench.json", device_id
    )
    adb_pull_from_app(
        "com.rtst.app", "files/output.jpg", f"{results_dir}/output.jpg", device_id
    )

    # Cleanup device
    adb(f"shell rm -rf {DEVICE_BENCH_DIR}", device_id)

    logger.info(f"[device] Results saved: {results_dir}")
    return results_dir


# 5. Result Aggregation
#    Collect all results into one comparison table.


def aggregate_results():
    """Scan results/ directory, build comparison matrix."""
    results_dir = Path("results") / "device"
    if not results_dir.exists():
        logger.warning("[aggregate] No device results found.")
        return

    rows = []
    for tag_dir in sorted(results_dir.iterdir()):
        bench_path = tag_dir / "bench.json"
        if bench_path.exists():
            with open(bench_path) as f:
                bench = json.load(f)
            rows.append(bench)

    if not rows:
        logger.warning("[aggregate] No benchmark JSONs found.")
        return

    def fmt_num(val):
        return f"{val:.1f}" if isinstance(val, (int, float)) else str(val)

    # Print comparison table
    header = f"{'Tag':<30} | {'Backend':<10} | {'Mean (ms)':<10} | {'P95 (ms)':<10} | {'Mem Delta (MB)':<15} | {'Device':<20}"
    separator = "-" * 105
    lines = [header, separator]
    for r in rows:
        tag = str(r.get("tag", "?"))
        # tag_disp = tag if len(tag) <= 29 else tag[:26] + "..."
        tag_disp = tag
        backend = str(r.get("backend", "?"))
        mean = fmt_num(r.get("mean_latency_ms", "?"))
        p95 = fmt_num(r.get("p95_latency_ms", "?"))
        mem = fmt_num(r.get("model_memory_delta_mb", "?"))
        device = str(r.get("device_model", "?"))

        lines.append(
            f"{tag_disp:<30} | {backend:<10} | {mean:<10} | {p95:<10} | {mem:<15} | {device:<20}"
        )

    logger.info(f"\n[aggregate] Comparison table:\n{chr(10).join(lines)}\n")

    # Also dump as CSV for spreadsheet/W&B import
    csv_path = Path("results") / "comparison.csv"
    import csv

    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    # Sort keys for a consistent CSV column order, putting important ones first
    priority_keys = [
        "tag",
        "backend",
        "mean_latency_ms",
        "p95_latency_ms",
        "model_memory_delta_mb",
        "device_model",
    ]
    sorted_keys = priority_keys + sorted(list(all_keys - set(priority_keys)))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted_keys)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"[aggregate] CSV written: {csv_path}")


# 6. Main


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="ExecuTorch Export & Benchmark Pipeline"
    )
    sub = parser.add_subparsers(dest="command")

    # export
    p_export = sub.add_parser("export", help="Export model to .pte via ExecuTorch")
    p_export.add_argument(
        "--config", required=True, help="Path to export YAML config file"
    )

    # validate
    p_val = sub.add_parser(
        "validate", help="Validate exported .pte against PyTorch on host"
    )
    p_val.add_argument("--pte", required=True, help="Path to exported .pte file")
    p_val.add_argument(
        "--ref-input", required=True, help="Path to reference test image"
    )

    # export and validate
    p_e_v = sub.add_parser("export-and-val", help="Export and validate .pte on host")
    p_e_v.add_argument(
        "--config", required=True, help="Path to export YAML config file"
    )
    p_e_v.add_argument(
        "--ref-input", required=True, help="Path to reference test image"
    )

    # device-bench
    p_dev = sub.add_parser(
        "device-bench", help="Benchmark .pte on Android device via adb"
    )
    p_dev.add_argument("--pte", required=True, help="Path to exported .pte file")
    p_dev.add_argument(
        "--ref-input",
        default="data/test_inference/flower.jpg",
        help="Path to reference test image",
    )
    p_dev.add_argument(
        "--device-id",
        default=None,
        help="adb device serial (default: first connected device)",
    )
    p_dev.add_argument(
        "--n-iters", type=int, default=50, help="Number of benchmark iterations"
    )

    # full (export -> validate -> device)
    p_full = sub.add_parser(
        "full", help="Run full pipeline: export -> validate -> device benchmark"
    )
    p_full.add_argument(
        "--config", required=True, help="Path to export YAML config file"
    )
    p_full.add_argument(
        "--ref-input",
        default="data/test_inference/flower.jpg",
        help="Path to reference test image",
    )
    p_full.add_argument(
        "--device-id",
        default=None,
        help="adb device serial (default: first connected device)",
    )
    p_full.add_argument(
        "--n-iters", type=int, default=50, help="Number of benchmark iterations"
    )

    # aggregate
    sub.add_parser("aggregate")

    args = parser.parse_args()

    if args.command == "export":
        cfg = ExportConfig.from_yaml(args.config)
        export_model(cfg)

    elif args.command == "validate":
        validate_on_host(args.pte, args.ref_input)

    elif args.command == "export-and-val":
        cfg = ExportConfig.from_yaml(args.config)
        pte = export_model(cfg)
        validate_on_host(str(pte), args.ref_input)

    elif args.command == "device-bench":
        device_benchmark(args.pte, args.ref_input, args.device_id, args.n_iters)

    elif args.command == "full":
        cfg = ExportConfig.from_yaml(args.config)
        pte = export_model(cfg)
        validate_on_host(str(pte), args.ref_input)
        device_benchmark(str(pte), args.ref_input, args.device_id, args.n_iters)

    elif args.command == "aggregate":
        aggregate_results()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
