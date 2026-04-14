"""
Diagnostic script: compare Python Runtime API vs C++ runner output
to pinpoint what the Python bindings do wrong.

Usage:
    uv run python scripts/diagnose_runtime.py \
        --pte exports/johnson_in_mosaic_xnnpack_fp32_640x480_export_mode/johnson_in_mosaic_xnnpack_fp32_640x480_export_mode.pte \
        --ref-input data/test_inference/flower.jpg
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

CPP_RUNNER = Path("cpp_eval/build/rtst")


def get_cpp_output(
    pte_path: str, test_input: torch.Tensor, h: int, w: int
) -> torch.Tensor:
    """Run the C++ runner and return its output tensor (known-good baseline)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_bin = Path(tmp_dir) / "input.bin"
        output_bin = Path(tmp_dir) / "output.bin"

        test_input.contiguous().numpy().tofile(input_bin)

        result = subprocess.run(
            [
                str(CPP_RUNNER),
                "validate",
                pte_path,
                str(input_bin),
                str(output_bin),
                str(h),
                str(w),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"[C++] {result.stdout.strip()}")

        return torch.from_numpy(
            np.fromfile(output_bin, dtype=np.float32).reshape(1, 3, h, w)
        )


def get_python_runtime_output(pte_path: str, test_input: torch.Tensor) -> torch.Tensor:
    """Run the ExecuTorch Python Runtime API and return its output tensor."""
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    method = program.load_method("forward")

    # ── Diagnostic: inspect what the Runtime sees ──
    print("\n[Python Runtime] Input tensor:")
    print(f"  shape:      {test_input.shape}")
    print(f"  dtype:      {test_input.dtype}")
    print(f"  contiguous: {test_input.is_contiguous()}")
    print(f"  strides:    {test_input.stride()}")
    print(f"  data_ptr:   {test_input.data_ptr()}")
    print(
        f"  min/max:    {test_input.min().item():.6f} / {test_input.max().item():.6f}"
    )
    print(f"  mean:       {test_input.mean().item():.6f}")
    print(f"  first 10:   {test_input.flatten()[:10].tolist()}")

    # Also try with an explicit contiguous copy to see if that changes anything
    input_contig = test_input.contiguous().clone()
    print("\n[Python Runtime] Contiguous clone check:")
    print(f"  same data?  {torch.equal(test_input, input_contig)}")
    print(f"  strides:    {input_contig.stride()}")

    et_outputs = method.execute([test_input])
    et_output = et_outputs[0]

    print("\n[Python Runtime] Output tensor:")
    print(f"  type:       {type(et_output)}")
    print(f"  shape:      {et_output.shape}")
    print(f"  dtype:      {et_output.dtype}")
    print(f"  strides:    {et_output.stride()}")
    print(f"  contiguous: {et_output.is_contiguous()}")
    print(f"  min/max:    {et_output.min().item():.6f} / {et_output.max().item():.6f}")
    print(f"  mean:       {et_output.mean().item():.6f}")
    print(f"  first 10:   {et_output.flatten()[:10].tolist()}")

    # Also run with the contiguous clone to see if input handling differs
    print("\n[Python Runtime] Re-running with explicit contiguous clone...")
    method2 = program.load_method("forward")
    et_outputs2 = method2.execute([input_contig])
    et_output_contig = et_outputs2[0]
    print(
        f"  clone output same? {torch.allclose(et_output, et_output_contig, atol=1e-6)}"
    )
    if not torch.allclose(et_output, et_output_contig, atol=1e-6):
        diff = torch.abs(et_output - et_output_contig)
        print(f"  clone diff max:    {diff.max().item():.6f}")
        print(f"  clone diff mean:   {diff.mean().item():.6f}")

    return et_output, et_output_contig


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
    ).item()


def pairwise_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Compute cosine similarity, L2, and Linf between two tensors."""
    return {
        "cosine": cosine_sim(a, b),
        "l2": torch.norm(a - b).item(),
        "linf": torch.max(torch.abs(a - b)).item(),
        "mean_abs_diff": torch.mean(torch.abs(a - b)).item(),
    }


def compare_outputs(
    py_output: torch.Tensor,
    cpp_output: torch.Tensor,
    ref_output: torch.Tensor,
) -> None:
    """Deep comparison of all three outputs to isolate the Python Runtime bug."""

    # ── 1. Overall pairwise metrics ──
    print("\n── 1. Overall Pairwise Metrics ──")
    pairs = [
        ("Python RT vs PyTorch ref", py_output, ref_output),
        ("C++ runner vs PyTorch ref", cpp_output, ref_output),
        ("Python RT vs C++ runner", py_output, cpp_output),
    ]
    for label, a, b in pairs:
        m = pairwise_metrics(a, b)
        print(f"\n  {label}:")
        print(f"    cosine_sim:   {m['cosine']:.8f}")
        print(f"    L2:           {m['l2']:.6f}")
        print(f"    Linf:         {m['linf']:.6f}")
        print(f"    mean_abs:     {m['mean_abs_diff']:.6f}")

    # ── 2. Per-channel analysis ──
    print("\n── 2. Per-Channel Analysis (Python RT vs C++ runner) ──")
    for c in range(3):
        ch_py = py_output[:, c, :, :]
        ch_cpp = cpp_output[:, c, :, :]
        ch_ref = ref_output[:, c, :, :]
        cos_py_cpp = cosine_sim(ch_py, ch_cpp)
        cos_py_ref = cosine_sim(ch_py, ch_ref)
        cos_cpp_ref = cosine_sim(ch_cpp, ch_ref)
        mean_diff = torch.mean(torch.abs(ch_py - ch_cpp)).item()
        print(f"\n  Channel {c} (R/G/B):")
        print(f"    py vs cpp cosine: {cos_py_cpp:.8f}  mean_abs_diff: {mean_diff:.6f}")
        print(f"    py vs ref cosine: {cos_py_ref:.8f}")
        print(f"    cpp vs ref cosine: {cos_cpp_ref:.8f}")
        print(
            f"    py  stats: mean={ch_py.mean().item():.6f} std={ch_py.std().item():.6f}"
        )
        print(
            f"    cpp stats: mean={ch_cpp.mean().item():.6f} std={ch_cpp.std().item():.6f}"
        )

    # ── 3. Spatial error analysis ──
    print("\n── 3. Spatial Error Analysis (Python RT vs C++ runner) ──")
    diff = torch.abs(py_output - cpp_output)
    # Check if error is uniform or structured
    # Split image into quadrants and compare error in each
    _, _, h, w = diff.shape
    quadrants = {
        "top-left": diff[:, :, : h // 2, : w // 2],
        "top-right": diff[:, :, : h // 2, w // 2 :],
        "bottom-left": diff[:, :, h // 2 :, : w // 2],
        "bottom-right": diff[:, :, h // 2 :, w // 2 :],
    }
    for name, q in quadrants.items():
        print(
            f"  {name:15s}: mean_err={q.mean().item():.6f}  max_err={q.max().item():.6f}"
        )

    # Check error distribution along rows and columns
    row_err = diff.mean(dim=(0, 1, 3))  # mean error per row
    col_err = diff.mean(dim=(0, 1, 2))  # mean error per col
    print(f"\n  Row error: first5={row_err[:5].tolist()}")
    print(f"             last5={row_err[-5:].tolist()}")
    print(f"             std across rows: {row_err.std().item():.6f}")
    print(f"  Col error: first5={col_err[:5].tolist()}")
    print(f"             last5={col_err[-5:].tolist()}")
    print(f"             std across cols: {col_err.std().item():.6f}")

    # ── 4. Transform hypotheses ──
    print("\n── 4. Transform Hypotheses (does a simple transform fix it?) ──")

    hypotheses = {
        "BGR swap [2,1,0]": py_output[:, [2, 1, 0], :, :],
        "Spatial transpose (H↔W)": py_output.permute(0, 1, 3, 2),
        "Reshape as (1,3,W,H)": py_output.reshape(1, 3, w, h),
        "Channel rotate [1,2,0]": py_output[:, [1, 2, 0], :, :],
        "Channel rotate [2,0,1]": py_output[:, [2, 0, 1], :, :],
        "Flip H": py_output.flip(2),
        "Flip W": py_output.flip(3),
        "Flip H+W": py_output.flip(2).flip(3),
    }

    for name, transformed in hypotheses.items():
        if transformed.shape == cpp_output.shape:
            cos = cosine_sim(transformed, cpp_output)
            marker = " <<<< MATCH!" if cos > 0.999 else ""
            print(f"  {name:30s}: cosine vs C++ = {cos:.8f}{marker}")
        else:
            print(
                f"  {name:30s}: shape mismatch {transformed.shape} vs {cpp_output.shape}"
            )

    # ── 5. Raw data comparison ──
    print("\n── 5. Raw Data Comparison (first 20 values, flat) ──")
    py_flat = py_output.flatten()[:20]
    cpp_flat = cpp_output.flatten()[:20]
    ref_flat = ref_output.flatten()[:20]

    print(
        f"\n  {'idx':>4s}  {'PyTorch ref':>12s}  {'C++ runner':>12s}  {'Python RT':>12s}  {'py-cpp diff':>12s}"
    )
    print(f"  {'─' * 4}  {'─' * 12}  {'─' * 12}  {'─' * 12}  {'─' * 12}")
    for i in range(20):
        r = ref_flat[i].item()
        c = cpp_flat[i].item()
        p = py_flat[i].item()
        d = p - c
        print(f"  {i:4d}  {r:12.6f}  {c:12.6f}  {p:12.6f}  {d:12.6f}")

    # ── 6. Check if Python RT output looks like a different spatial location ──
    print("\n── 6. Cross-correlation check ──")
    # Flatten spatial dims and check if py_output is a shifted version of cpp_output
    # by finding the offset that maximizes correlation for channel 0
    py_ch0 = py_output[0, 0].flatten()
    cpp_ch0 = cpp_output[0, 0].flatten()
    # Normalize
    py_norm = (py_ch0 - py_ch0.mean()) / (py_ch0.std() + 1e-8)
    cpp_norm = (cpp_ch0 - cpp_ch0.mean()) / (cpp_ch0.std() + 1e-8)
    # Simple dot product at a few offsets
    n = len(py_norm)
    offsets_to_check = [0, 1, -1, w, -w, h, -h, 3, -3]
    print("  Checking correlation at spatial offsets (channel 0):")
    for offset in offsets_to_check:
        if offset >= 0:
            corr = torch.dot(py_norm[offset:], cpp_norm[: n - offset]).item() / (
                n - abs(offset)
            )
        else:
            corr = torch.dot(py_norm[: n + offset], cpp_norm[-offset:]).item() / (
                n - abs(offset)
            )
        marker = " <<<< HIGH" if abs(corr) > 0.95 else ""
        print(f"    offset={offset:>6d}: correlation={corr:.6f}{marker}")

    # ── 7. Save debug images ──
    print("\n── 7. Saving debug images ──")
    debug_dir = Path("results") / "debug" / "runtime_diagnosis"
    debug_dir.mkdir(parents=True, exist_ok=True)

    from src.utils.image import save_image

    save_image(ref_output[0], debug_dir / "ref_pytorch.png")
    save_image(cpp_output[0], debug_dir / "cpp_runner.png")
    save_image(py_output[0], debug_dir / "python_runtime.png")

    # Save the diff as a heatmap (amplified for visibility)
    diff_vis = diff[0].mean(dim=0, keepdim=True).repeat(3, 1, 1)  # grayscale diff
    diff_vis = diff_vis / (diff_vis.max() + 1e-8)  # normalize to [0,1]
    save_image(diff_vis, debug_dir / "diff_heatmap.png")

    # Save best transform hypothesis as image
    best_name, best_cos = "", 0.0
    for name, transformed in hypotheses.items():
        if transformed.shape == cpp_output.shape:
            cos = cosine_sim(transformed, cpp_output)
            if cos > best_cos:
                best_cos = cos
                best_name = name
    if best_cos > 0.95:
        print(f"  Best transform: '{best_name}' (cosine={best_cos:.8f})")
        best_transformed = hypotheses[best_name]
        save_image(
            best_transformed[0],
            debug_dir / f"best_transform_{best_name.replace(' ', '_')}.png",
        )

    print(f"\n  Debug images saved to: {debug_dir}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    py_vs_cpp = pairwise_metrics(py_output, cpp_output)
    cpp_vs_ref = pairwise_metrics(cpp_output, ref_output)
    print(
        f"  C++ runner vs PyTorch:    cosine={cpp_vs_ref['cosine']:.8f} (should be ~1.0)"
    )
    print(f"  Python RT vs C++ runner:  cosine={py_vs_cpp['cosine']:.8f} (the bug)")
    print(f"  Best transform fix:       '{best_name}' -> cosine={best_cos:.8f}")
    if best_cos > 0.999:
        print(f"\n  DIAGNOSIS: The Python Runtime API appears to apply a '{best_name}'")
        print(
            "  transformation to the data. This is a bug in the ExecuTorch Python bindings."
        )
    else:
        print("\n  DIAGNOSIS: No simple transform explains the difference.")
        print(
            "  The issue may be in how the Python Runtime handles tensor memory/strides internally."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose Python Runtime API vs C++ runner"
    )
    parser.add_argument("--pte", required=True, help="Path to .pte file")
    parser.add_argument("--ref-input", required=True, help="Path to test image")
    args = parser.parse_args()

    # Load sidecar config
    sidecar_path = args.pte.replace(".pte", ".json")
    with open(sidecar_path) as f:
        sidecar = json.load(f)

    cfg = sidecar["config"]
    h, w = cfg["input_h"], cfg["input_w"]

    # Load test image (same as export_pipeline does)
    from src.export_pipeline import load_test_image

    test_input = load_test_image(args.ref_input, h, w, cfg["keep_aspect"])

    print(f"Test input shape: {test_input.shape}, dtype: {test_input.dtype}")
    print(f"Model: {sidecar['tag']}")
    print(f"ExecuTorch version: {sidecar['executorch_version']}")
    print("=" * 70)

    # ── Step 1: Get PyTorch reference output ──
    import torch.nn as nn
    from src.models.trans_net import TransformationNetwork
    from src.export_pipeline import NormType

    norm_type = (
        nn.InstanceNorm2d
        if NormType(cfg["norm_type"]) == NormType.INSTANCE
        else nn.BatchNorm2d
    )
    model = TransformationNetwork(
        norm_layer_type=norm_type, export_mode=cfg["export_mode"]
    )
    model.load_state_dict(torch.load(cfg["checkpoint_path"], map_location="cpu"))
    model.eval()
    with torch.no_grad():
        ref_output = model(test_input)

    print(f"\n[PyTorch ref] output shape: {ref_output.shape}")
    print(
        f"[PyTorch ref] min/max: {ref_output.min().item():.6f} / {ref_output.max().item():.6f}"
    )

    # ── Step 2: Get C++ runner output (known good) ──
    cpp_output = get_cpp_output(args.pte, test_input, h, w)
    print(f"\n[C++ runner] output shape: {cpp_output.shape}")
    print(
        f"[C++ runner] min/max: {cpp_output.min().item():.6f} / {cpp_output.max().item():.6f}"
    )

    # ── Step 3: Get Python Runtime output (the broken one) ──
    py_output, py_output_contig = get_python_runtime_output(args.pte, test_input)
    print(f"\n[Python RT] output shape: {py_output.shape}")
    print(
        f"[Python RT] min/max: {py_output.min().item():.6f} / {py_output.max().item():.6f}"
    )

    # ── Step 3b: Compare contiguous clone output vs reference ──
    print("\n" + "=" * 70)
    print("CONTIGUOUS CLONE FIX TEST")
    print("=" * 70)
    contig_vs_ref = pairwise_metrics(py_output_contig, ref_output)
    contig_vs_cpp = pairwise_metrics(py_output_contig, cpp_output)
    print("  Python RT (contiguous input) vs PyTorch ref:")
    print(f"    cosine_sim: {contig_vs_ref['cosine']:.8f}")
    print(f"    L2:         {contig_vs_ref['l2']:.6f}")
    print(f"    Linf:       {contig_vs_ref['linf']:.6f}")
    print("  Python RT (contiguous input) vs C++ runner:")
    print(f"    cosine_sim: {contig_vs_cpp['cosine']:.8f}")
    print(f"    L2:         {contig_vs_cpp['l2']:.6f}")
    print(f"    Linf:       {contig_vs_cpp['linf']:.6f}")
    if contig_vs_ref["cosine"] > 0.999:
        print("\n  >>> CONFIRMED: Passing .contiguous() input FIXES the bug!")
        print(
            "  >>> The Python Runtime API ignores tensor strides and reads raw memory as NCHW."
        )
    else:
        print("\n  >>> Contiguous input does NOT fix the issue — the bug is elsewhere.")

    # ── Step 4: Deep comparison (non-contiguous input, the buggy path) ──
    print("\n" + "=" * 70)
    print("COMPARISON (non-contiguous input — the buggy path)")
    print("=" * 70)
    compare_outputs(py_output, cpp_output, ref_output)


if __name__ == "__main__":
    main()
