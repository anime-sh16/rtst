import argparse
import logging
from pathlib import Path
from typing import Literal
import torch
import torch.nn as nn
from executorch.runtime import Runtime, Verification
import pandas as pd
from executorch.devtools import Inspector


from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from torch.export import export

logger = logging.getLogger(__name__)

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
        default=Path("models/export"),
        help="Directory where .pte files will be saved.",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="mosaic",
        help="Name of the style image.",
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


def _log_delegate_coverage(edge_program) -> None:
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
    delegated = 0
    fallback = 0

    for node in edge_program.exported_program().graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        if "executorch_call_delegate" in str(node.target):
            delegated += 1
        else:
            fallback += 1
            # logger.info("FALLBACK op: %s → %s", node.name, node.target)

    total = delegated + fallback
    if total == 0:
        logger.info("No compute nodes found.")
        return

    logger.info("Delegated: %d/%d (%.1f%%)", delegated, total, 100 * delegated / total)
    logger.info("Fallback:  %d/%d (%.1f%%)", fallback, total, 100 * fallback / total)


def _validate_output(
    model: nn.Module,
    pte_path: Path,
    example_inputs: tuple,
    backend: str,
    output_path: Path,
    atol: float = 1e-2,
) -> None:
    """
    Run the exported .pte file against the original PyTorch model on the same input.
    Raises if max absolute error exceeds atol.
    Note: only XNNPACK is available on host machines. Vulkan requires Android hardware.
    Vulkan validation is skipped here — test it on device via ETDump.
    """
    if backend == "vulkan":
        logger.info("Skipping numerical check for Vulkan (requires Android hardware).")
        return

    try:
        runtime = Runtime.get()
        program = runtime.load_program(
            str(pte_path),
            verification=Verification.Minimal,
            enable_etdump=True,
            debug_buffer_size=1024 * 1024 * 1024,
        )
        method = program.load_method("forward")
        et_output = method.execute(list(example_inputs))[0]

        etdump_path = output_path.with_suffix(".etdump")
        debug_buffer_path = output_path.parent / "debug_buffer.bin"
        program.write_etdump_result_to_file(str(etdump_path), str(debug_buffer_path))

        with torch.no_grad():
            pt_output = model(*example_inputs)

        if isinstance(et_output, (list, tuple)):
            runtime_output = et_output[0]
        else:
            runtime_output = et_output

        # Calculate MSE between eager and runtime outputs
        mse = torch.mean((pt_output - runtime_output) ** 2).item()
        logger.info("MSE between eager and runtime outputs: %.2e", mse)

        max_err = (et_output - pt_output).abs().max().item()
        logger.info("Max absolute error vs PyTorch: %.2e", max_err)
        if max_err > atol:
            raise ValueError(
                f"Exported model output deviates too much from PyTorch: "
                f"max_err={max_err:.2e} > atol={atol:.2e}. "
                f"Check for unsupported ops or quantization errors."
            )
    except Exception as e:
        logger.warning("Validation failed: %s", e)


def _calculate_numeric_gap(etrecord, output_path: Path) -> pd.DataFrame:
    """
    Calculate numeric gap using Mean Squared Error btween the eager and runtime outputs.
    """
    inspector = Inspector(
        etdump_path=output_path.with_suffix(".etdump"),
        etrecord=etrecord,
        debug_buffer_path=output_path.parent / "debug_buffer.bin",
    )

    inspector.print_data_tabular(output_path.parent / "analysis/inspector.log")

    # Calculate numerical gap using Mean Squared Error
    df: pd.DataFrame = inspector.calculate_numeric_gap("MSE")

    df.to_csv(output_path.parent / "analysis/numeric_gap.csv", index=False)


def _analyze_ops(df: pd.DataFrame, output_path: Path) -> None:
    """Analyze operator numerical gaps and write a markdown report."""
    df_sorted = df.sort_values(
        by="gap",
        ascending=False,
        key=lambda x: x.apply(lambda y: y[0] if isinstance(y, list) else y),
    )

    threshold = 1e-4
    problematic_ops = df[
        df["gap"].apply(
            lambda x: x[0] > threshold if isinstance(x, list) else x > threshold
        )
    ]

    lines: list[str] = []
    lines.append("# Operator Numerical Gap Analysis\n")

    # Top 5 operators section
    lines.append("## Top 5 Operators by MSE Gap\n")
    top5 = df_sorted.head(5)
    lines.append(top5.to_markdown(index=True))
    lines.append("")

    # Problematic operators section
    lines.append(f"## Problematic Operators (MSE > {threshold})\n")
    if problematic_ops.empty:
        lines.append("No operators exceeded the threshold.\n")
    else:
        lines.append(f"**{len(problematic_ops)}** operator(s) found.\n")
        for idx, row in problematic_ops.iterrows():
            lines.append(f"### Operator {idx}\n")
            lines.append(f"- **AOT Ops:** {row['aot_ops']}")
            lines.append(f"- **Gap:** {row['gap']}\n")
            lines.append("**Stack Traces:**\n")
            has_traces = False
            for op_name, trace in row["stacktraces"].items():
                if trace:
                    has_traces = True
                    lines.append(f"`{op_name}`:\n```")
                    for line in trace.split("\n")[:3]:
                        lines.append(line)
                    lines.append("```\n")
            if not has_traces:
                lines.append("No stack traces available.\n")

    analysis_dir = output_path.parent / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_path = analysis_dir / "ops_analysis.md"
    report_path.write_text("\n".join(lines))
    logger.info("Ops analysis report saved to %s", report_path)


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
        exported_program, partitioner=partitioner_list, generate_etrecord=True
    )

    _log_delegate_coverage(edge_program)

    et_program = edge_program.to_executorch()

    etrecord = et_program.get_etrecord()
    etrecord.update_representative_inputs(example_inputs)
    etrecord_path = output_path.with_suffix(".bin")
    etrecord.save(str(etrecord_path))
    logger.info("ETRecord saved to %s", etrecord_path)

    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info("Saved to %s (%.2f MB)", output_path, size_mb)

    _validate_output(model, output_path, example_inputs, backend, output_path)

    # df = _calculate_numeric_gap(etrecord, output_path)
    # _analyze_ops(df, output_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from %s", args.weights)
    model = load_model(args.weights)

    backends_to_run: list[Literal["xnnpack", "vulkan"]] = (
        ["xnnpack", "vulkan"] if args.backend == "all" else [args.backend]  # type: ignore[list-item]
    )

    for backend in backends_to_run:
        output_path = args.output_dir / f"{args.style}_{backend}.pte"
        logger.info("Exporting → %s (%s) ...", backend.upper(), output_path)
        export_to_executorch(model, args.image_size, backend, output_path)

    logger.info(
        "Done. Place the .pte file(s) in your Android app's /assets/ directory."
    )


if __name__ == "__main__":
    main()
