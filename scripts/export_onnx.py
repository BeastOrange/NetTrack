"""
ONNX export + ONNXRuntime FPS sanity check.

Wraps `yolo export` so the deployment-feasibility section of the thesis has
real numbers to point at, then runs onnxruntime against a few test images
to confirm the export is loadable and produces consistent shapes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml
from loguru import logger


def _load_data_yaml(path: Path) -> Path:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Path(cfg.get("path", path.parent)).expanduser().resolve()


def export_onnx(weights: Path, imgsz: int, opset: int) -> Path:
    from ultralytics import YOLO

    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")
    model = YOLO(str(weights))
    out_path_str = model.export(format="onnx", imgsz=imgsz, opset=opset, dynamic=False)
    out_path = Path(out_path_str)
    logger.info(f"Exported ONNX → {out_path}")
    return out_path


def benchmark_onnx(
    onnx_path: Path,
    sample_dir: Path,
    iters: int,
    imgsz: int,
    providers: list[str],
) -> float:
    import cv2
    import onnxruntime as ort

    samples = sorted(sample_dir.glob("*.jpg"))
    if not samples:
        raise SystemExit(f"No sample images in {sample_dir}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)
    input_name = session.get_inputs()[0].name
    logger.info(f"ONNXRuntime providers active: {session.get_providers()}")

    def preprocess(image_path: Path) -> np.ndarray:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise RuntimeError(f"Failed to read {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (imgsz, imgsz))
        chw = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        return chw[np.newaxis, ...]

    for sample in samples[: min(3, len(samples))]:
        session.run(None, {input_name: preprocess(sample)})

    start = time.perf_counter()
    for i in range(iters):
        sample = samples[i % len(samples)]
        session.run(None, {input_name: preprocess(sample)})
    elapsed = time.perf_counter() - start
    fps = iters / elapsed if elapsed > 0 else 0.0
    logger.info(f"ONNXRuntime FPS over {iters} iters: {fps:.2f}")
    return fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--data", type=Path, default=Path("data/cumt_belt_yolo/data.yaml"))
    parser.add_argument("--bench-iters", type=int, default=200)
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["CPUExecutionProvider"],
        help='ONNXRuntime providers, e.g. "CUDAExecutionProvider CPUExecutionProvider".',
    )
    parser.add_argument("--skip-bench", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_path = export_onnx(args.weights, args.imgsz, args.opset)
    if args.skip_bench:
        return
    root = _load_data_yaml(args.data)
    benchmark_onnx(
        onnx_path=onnx_path,
        sample_dir=root / "images" / "test",
        iters=args.bench_iters,
        imgsz=args.imgsz,
        providers=args.providers,
    )


if __name__ == "__main__":
    main()
