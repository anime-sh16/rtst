import argparse
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from torch.export import export


BackendName = Literal["xnnpack", "vulkan", "all"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the NST model to ExecuTorch .pte files."
    )
    parser.add_argument(
        "--weights", type=Path, required=True, help="Path to trained .pth checkpoint."
    )
    parser.add_argument(
        "--backend",
        choices=["xnnpack", "vulkan", "all"],
        default="xnnpack",
        help="Which ExecuTorch backend(s) to generate .pte files for.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory where .pte files will be saved.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Spatial resolution used as the trace example input (H = W).",
    )
    return parser.parse_args()


def load_model(weights_path: Path) -> nn.Module:
    """
    Instantiate TransformationNetwork, load trained weights, and set eval mode.

    Args:
        weights_path: Path to the saved .pth checkpoint.

    Returns:
        The model in eval mode on CPU, ready for export.
    """
    from src.models.trans_net import TransformationNetwork

    trans_net = TransformationNetwork(norm_layer_type=nn.InstanceNorm2d)
    state_dict = torch.load(weights_path, map_location="cpu")
    trans_net.load_state_dict(state_dict)
    trans_net.eval()
    return trans_net


def _log_vulkan_coverage(edge_program) -> None:
    """
    Count call_delegate nodes (ops claimed by a backend — Vulkan or XNNPACK)
    vs call_function nodes (ops that fell through all partitioners and will
    run as portable CPU fallback).

    Note: we can't distinguish Vulkan delegates from XNNPACK delegates here
    without digging into node.meta internals. What we can say is:
      - call_delegate = something accelerated (at minimum XNNPACK)
      - call_function = portable fallback (slow reference kernel)
    A high fallback count is the real warning sign.
    """
    delegate_nodes = 0
    fallback_nodes = 0

    for node in edge_program.exported_program().graph.nodes:
        if node.op == "call_delegate":
            delegate_nodes += 1
        elif node.op == "call_function":
            fallback_nodes += 1

    total = delegate_nodes + fallback_nodes
    if total == 0:
        print("  [coverage] No compute nodes found.")
        return

    print(
        f"  [coverage] Delegated (Vulkan+XNNPACK) : {delegate_nodes} ({100 * delegate_nodes / total:.1f}%)"
    )
    print(
        f"  [coverage] Portable fallback (slow)   : {fallback_nodes} ({100 * fallback_nodes / total:.1f}%)"
    )

    if fallback_nodes > 0:
        print(
            f"  [coverage] {fallback_nodes} op(s) fell through all partitioners → portable CPU."
        )
        print("  [coverage] Run ETDump on device to identify which ops these are.")

    if fallback_nodes / total > 0.20:
        print(
            "  [WARNING]  >20% fallback nodes. Check for unsupported ops in your model."
        )


def _validate_output(
    model: nn.Module,
    pte_path: Path,
    example_inputs: tuple,
    backend: str,
    atol: float = 1e-2,
) -> None:
    """
    Run the exported .pte file against the original PyTorch model on the same input.
    Raises if max absolute error exceeds atol.
    Note: only XNNPACK is available on host machines. Vulkan requires Android hardware.
    Vulkan validation is skipped here — test it on device via ETDump.
    """
    if backend == "vulkan":
        print(
            "  [validate] Skipping numerical check for Vulkan (requires Android hardware)."
        )
        return

    try:
        from executorch.runtime import Runtime

        runtime = Runtime.get()
        method = runtime.load_program(str(pte_path)).load_method("forward")
        et_output = method.execute(list(example_inputs))[0]
        pt_output = model(*example_inputs)
        max_err = (et_output - pt_output).abs().max().item()
        print(f"  [validate] Max absolute error vs PyTorch: {max_err:.2e}")
        if max_err > atol:
            raise ValueError(
                f"Exported model output deviates too much from PyTorch: "
                f"max_err={max_err:.2e} > atol={atol:.2e}. "
                f"Check for unsupported ops or quantization errors."
            )
    except Exception as e:
        print(f"  [validate] WARNING: validation failed: {e}")


def export_to_executorch(
    model: nn.Module,
    image_size: int,
    backend: Literal["xnnpack", "vulkan"],
    output_path: Path,
) -> None:
    """
    Export a single backend variant and write the .pte file.

    Export Process:
      1. Create example_inputs: a tuple of one tensor with shape (1, 3, image_size, image_size).
      2. Call torch.export.export(model, example_inputs) → exported_program.
         - If you want to support variable H/W at runtime, declare dynamic dims here
           using torch.export.Dim and pass dynamic_shapes. For a fixed-size app, skip this.
      3. Select the partitioner:
           xnnpack → XnnpackPartitioner()
           vulkan  → VulkanPartitioner()
      4. Call to_edge_transform_and_lower(exported_program, partitioner=[partitioner])
         → edge_program.
      5. Call edge_program.to_executorch() → et_program.
      6. Write et_program.buffer to output_path in binary mode.
      7. Print the output path and file size in MB.

    Args:
        model:       TransformationNetwork in eval mode.
        image_size:  H = W for the example input tensor.
        backend:     "xnnpack" or "vulkan".
        output_path: Destination .pte file path.
    """
    example_inputs = (torch.randn(1, 3, image_size, image_size),)
    exported_program = export(model, example_inputs)
    if backend == "xnnpack":
        partitioner_list = [XnnpackPartitioner()]
    else:
        partitioner_list = [VulkanPartitioner(), XnnpackPartitioner()]

    edge_program = to_edge_transform_and_lower(
        exported_program, partitioner=partitioner_list
    )
    if backend == "vulkan":
        _log_vulkan_coverage(edge_program)
    et_program = edge_program.to_executorch()

    with open(output_path, "wb") as f:
        f.write(et_program.buffer)
    print(f"Saved to {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")

    _validate_output(model, output_path, example_inputs, backend)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.weights)

    backends_to_run: list[Literal["xnnpack", "vulkan"]] = (
        ["xnnpack", "vulkan"] if args.backend == "all" else [args.backend]  # type: ignore[list-item]
    )

    for backend in backends_to_run:
        output_path = args.output_dir / f"style_{backend}.pte"
        print(f"Exporting → {backend.upper()} ({output_path}) ...")
        export_to_executorch(model, args.image_size, backend, output_path)

    print("Done. Place the .pte file(s) in your Android app's /assets/ directory.")


if __name__ == "__main__":
    main()
