"""속도 및 품질 벤치마크.

NILUT / ImageAdaptive3DLUT 모델의 추론 속도, 모델 크기, 품질 기준을
해상도별·디바이스별로 측정하고 JSON 리포트를 저장한다.

Usage:
    python scripts/benchmark.py --model checkpoints/nilut/best.pt
    python scripts/benchmark.py --model checkpoints/lut3d/best.pt --resolutions 256x256 1920x1080 3840x2160
    python scripts/benchmark.py --model checkpoints/nilut/best.pt --iterations 20
"""

import argparse
import json
import logging
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QUALITY_THRESHOLDS = {
    "delta_e_max": 1.0,
    "nilut_size_mb": 1.0,
    "lut3d_size_mb": 10.0,
    "nilut_latency_ms": 16.0,
    "lut3d_latency_ms": 2.0,
}

RESOLUTIONS = {
    "256x256": (256, 256),
    "1920x1080": (1920, 1080),
    "3840x2160": (3840, 2160),
}


def get_available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def load_model(model_path: Path, device: str):
    from src.inference.export import (
        _detect_model_type,
        _load_nilut_from_checkpoint,
        _load_lut3d_from_checkpoint,
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model_type = _detect_model_type(checkpoint)
    if model_type == "lut3d":
        model = _load_lut3d_from_checkpoint(model_path, device=device)
    else:
        model = _load_nilut_from_checkpoint(model_path, device=device)
    return model, model_type


def run_nilut_inference(model, image_t: torch.Tensor, device: str) -> torch.Tensor:
    b, c, h, w = image_t.shape
    pixels = image_t.permute(0, 2, 3, 1).reshape(-1, 3)
    # 다중 스타일 모드일 경우 style_idx=0 Tensor로 기본 실행
    if getattr(model, "num_styles", None) is not None:
        n = pixels.shape[0]
        idx = torch.zeros(n, dtype=torch.long, device=device)
        out = model(pixels, style_idx=idx)
    else:
        out = model(pixels)
    return out.reshape(b, h, w, c).permute(0, 3, 1, 2)


def run_lut3d_inference(model, image_t: torch.Tensor, device: str) -> torch.Tensor:
    return model(image_t)


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()  # type: ignore[attr-defined]


def benchmark_resolution(
    model,
    model_type: str,
    resolution_name: str,
    h: int,
    w: int,
    device: str,
    iterations: int,
) -> dict[str, Any]:
    image_t = torch.rand(1, 3, h, w, device=device)
    infer_fn = run_lut3d_inference if model_type == "lut3d" else run_nilut_inference

    with torch.no_grad():
        for _ in range(3):
            infer_fn(model, image_t, device)

    latencies_ms: list[float] = []
    with torch.no_grad():
        for _ in range(iterations):
            _sync(device)
            t0 = time.perf_counter()
            infer_fn(model, image_t, device)
            _sync(device)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = statistics.mean(latencies_ms)
    std_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    p95_ms = float(np.percentile(latencies_ms, 95))
    p99_ms = float(np.percentile(latencies_ms, 99))

    return {
        "resolution": resolution_name,
        "height": h,
        "width": w,
        "device": device,
        "iterations": iterations,
        "mean_ms": round(mean_ms, 3),
        "std_ms": round(std_ms, 3),
        "min_ms": round(min(latencies_ms), 3),
        "max_ms": round(max(latencies_ms), 3),
        "p95_ms": round(p95_ms, 3),
        "p99_ms": round(p99_ms, 3),
    }


def measure_model_size(model_path: Path, model) -> dict[str, Any]:
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    file_size_bytes = model_path.stat().st_size
    file_size_mb = file_size_bytes / (1024**2)
    return {
        "param_count": param_count,
        "trainable_param_count": trainable_count,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_mb, 4),
    }


def measure_identity_delta_e(model, model_type: str, device: str) -> dict[str, Any]:
    """항등 입력에 대한 ΔE 측정. 초기화 직후 모델의 색상 보존 수준을 확인한다."""
    from src.utils.color import delta_e_cie76, rgb_to_lab

    np.random.seed(42)
    image_np = np.random.rand(256, 256, 3).astype(np.float32)
    image_t = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)

    infer_fn = run_lut3d_inference if model_type == "lut3d" else run_nilut_inference
    with torch.no_grad():
        out_t = infer_fn(model, image_t, device)

    out_np = np.clip(out_t.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)
    lab_in = rgb_to_lab(image_np.reshape(-1, 3))
    lab_out = rgb_to_lab(out_np.reshape(-1, 3))
    de_map = delta_e_cie76(lab_in, lab_out)
    mean_de = float(de_map.mean())

    return {
        "mean_delta_e": round(mean_de, 4),
        "max_delta_e": round(float(de_map.max()), 4),
        "p95_delta_e": round(float(np.percentile(de_map, 95)), 4),
        "passes_threshold": mean_de < QUALITY_THRESHOLDS["delta_e_max"],
    }


def check_quality_thresholds(
    model_type: str,
    size_info: dict[str, Any],
    latency_results: list[dict[str, Any]],
    delta_e_info: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}

    size_mb = size_info.get("file_size_mb", 0)
    threshold_mb = (
        QUALITY_THRESHOLDS["lut3d_size_mb"]
        if model_type == "lut3d"
        else QUALITY_THRESHOLDS["nilut_size_mb"]
    )
    checks["model_size"] = size_mb < threshold_mb
    checks["delta_e"] = delta_e_info.get("passes_threshold", False)

    gpu_results = [r for r in latency_results if r["device"] in ("cuda", "mps")]
    if gpu_results:
        lat_threshold = (
            QUALITY_THRESHOLDS["lut3d_latency_ms"]
            if model_type == "lut3d"
            else QUALITY_THRESHOLDS["nilut_latency_ms"]
        )
        fhd = [r for r in gpu_results if r["resolution"] == "1920x1080"]
        if fhd:
            checks["latency_1080p_gpu"] = fhd[0]["mean_ms"] < lat_threshold
        k4 = [r for r in gpu_results if r["resolution"] == "3840x2160"]
        if k4:
            checks["latency_4k_gpu"] = k4[0]["mean_ms"] < lat_threshold * 4

    all_pass = all(checks.values())
    return {
        "checks": checks,
        "all_pass": all_pass,
        "summary": "PASS" if all_pass else "FAIL",
    }


def print_table(rows: list[list[str]], headers: list[str]) -> None:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    fmt = "|" + "|".join(f" {{:<{w}}} " for w in col_widths) + "|"
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))
    print(sep)


def print_benchmark_report(report: dict[str, Any]) -> None:
    model_info = report["model_info"]
    size = model_info["size"]
    qc = report["quality_check"]

    print("\n" + "=" * 62)
    print(f"  벤치마크 결과: {Path(model_info['model_path']).name}")
    print(f"  모델 타입: {model_info['model_type']}")
    print(f"  파라미터 수: {size['param_count']:,}")
    print(f"  파일 크기: {size['file_size_mb']:.3f} MB")
    print("=" * 62)

    print("\n[추론 속도]")
    headers = ["해상도", "디바이스", "평균(ms)", "P95(ms)", "P99(ms)", "std(ms)"]
    rows = [
        [
            r["resolution"],
            r["device"],
            f"{r['mean_ms']:.2f}",
            f"{r['p95_ms']:.2f}",
            f"{r['p99_ms']:.2f}",
            f"{r['std_ms']:.2f}",
        ]
        for r in report["latency"]
    ]
    if rows:
        print_table(rows, headers)
    else:
        print("  (측정 결과 없음)")

    de = report["delta_e"]
    print(
        f"\n[항등 변환 ΔE  (배포 기준: 평균 < {QUALITY_THRESHOLDS['delta_e_max']:.1f})]"
    )
    print(f"  평균 ΔE : {de['mean_delta_e']:.4f}")
    print(f"  최대 ΔE : {de['max_delta_e']:.4f}")
    print(f"  P95 ΔE  : {de['p95_delta_e']:.4f}")

    print(f"\n[배포 기준 검증]  ->  {qc['summary']}")
    for check_name, passed in qc["checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}]  {check_name}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NILUT / ImageAdaptive3DLUT 모델 벤치마크"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="학습된 모델 경로 (.pt)"
    )
    parser.add_argument(
        "--resolutions",
        type=str,
        nargs="+",
        default=["256x256", "1920x1080", "3840x2160"],
        help="테스트 해상도 목록",
    )
    parser.add_argument(
        "--devices", type=str, nargs="+", default=None, help="테스트 디바이스"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="반복 횟수 (기본값: 10)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="reports", help="JSON 저장 디렉토리"
    )
    parser.add_argument("--no-save", action="store_true", help="JSON 리포트 저장 생략")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        logger.error(f"모델 파일을 찾을 수 없음: {model_path}")
        sys.exit(1)

    target_resolutions: dict[str, tuple[int, int]] = {}
    for res_str in args.resolutions:
        if res_str in RESOLUTIONS:
            target_resolutions[res_str] = RESOLUTIONS[res_str]
        else:
            try:
                parts = res_str.lower().split("x")
                target_resolutions[res_str] = (int(parts[1]), int(parts[0]))
            except (ValueError, IndexError):
                logger.warning(f"해상도 파싱 실패, 건너뜀: {res_str}")

    if not target_resolutions:
        logger.error("유효한 해상도가 없음")
        sys.exit(1)

    available = get_available_devices()
    if args.devices:
        devices = [d for d in args.devices if d in available] or ["cpu"]
    else:
        devices = available

    logger.info(
        f"벤치마크 시작: model={model_path.name}, devices={devices}, "
        f"resolutions={list(target_resolutions)}, iterations={args.iterations}"
    )

    latency_results: list[dict[str, Any]] = []
    size_info: dict[str, Any] = {}
    delta_e_info: dict[str, Any] = {}
    model_type_global = "unknown"

    for device in devices:
        logger.info(f"디바이스: {device}")
        try:
            model, model_type = load_model(model_path, device)
            model_type_global = model_type
        except Exception as exc:
            logger.error(f"{device} 모델 로드 실패: {exc}")
            continue

        if not size_info:
            size_info = measure_model_size(model_path, model)
            logger.info(
                f"  파라미터: {size_info['param_count']:,}  파일: {size_info['file_size_mb']:.3f} MB"
            )

        if not delta_e_info and device == "cpu":
            logger.info("  항등 변환 ΔE 측정 중...")
            try:
                delta_e_info = measure_identity_delta_e(model, model_type, device)
                logger.info(f"  평균 ΔE: {delta_e_info['mean_delta_e']:.4f}")
            except Exception as exc:
                logger.warning(f"  ΔE 측정 실패: {exc}")
                delta_e_info = {
                    "mean_delta_e": -1,
                    "max_delta_e": -1,
                    "p95_delta_e": -1,
                    "passes_threshold": False,
                }

        for res_name, (h, w) in target_resolutions.items():
            logger.info(f"  {res_name} 측정 중...")
            try:
                result = benchmark_resolution(
                    model, model_type, res_name, h, w, device, args.iterations
                )
                latency_results.append(result)
                logger.info(
                    f"    평균: {result['mean_ms']:.2f}ms  P95: {result['p95_ms']:.2f}ms"
                )
            except Exception as exc:
                logger.error(f"  {res_name} ({device}) 측정 실패: {exc}")

    if not delta_e_info and devices:
        try:
            model, model_type = load_model(model_path, devices[0])
            delta_e_info = measure_identity_delta_e(model, model_type, devices[0])
        except Exception:
            delta_e_info = {
                "mean_delta_e": -1,
                "max_delta_e": -1,
                "p95_delta_e": -1,
                "passes_threshold": False,
            }

    quality_check = check_quality_thresholds(
        model_type_global, size_info, latency_results, delta_e_info
    )

    report: dict[str, Any] = {
        "benchmark_timestamp": datetime.now().isoformat(),
        "model_info": {
            "model_path": str(model_path),
            "model_type": model_type_global,
            "size": size_info,
        },
        "latency": latency_results,
        "delta_e": delta_e_info,
        "quality_check": quality_check,
        "thresholds": QUALITY_THRESHOLDS,
    }

    print_benchmark_report(report)

    if not args.no_save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"benchmark_{ts}.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        logger.info(f"리포트 저장: {report_path}")

    if not quality_check["all_pass"]:
        logger.warning("배포 기준 미충족 항목이 있습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
