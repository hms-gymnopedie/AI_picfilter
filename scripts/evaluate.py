"""평가 실행 스크립트.

Usage:
    python scripts/evaluate.py --model checkpoints/nilut/best.pt \
        --input-dir data/test/input --target-dir data/test/target --output results/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NILUT 모델 평가")
    parser.add_argument("--model", type=str, required=True, help="학습된 모델 경로")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="원본 이미지 디렉토리")
    parser.add_argument("--target-dir", type=str, required=True,
                        help="보정 이미지 디렉토리 (ground truth)")
    parser.add_argument("--output", type=str, default="results/",
                        help="결과 출력 디렉토리")
    parser.add_argument("--style", type=int, default=0, help="스타일 인덱스")
    parser.add_argument("--device", type=str, default="auto", help="디바이스")
    parser.add_argument("--no-visual", action="store_true",
                        help="시각화 생성 생략 (빠른 평가)")
    return parser.parse_args()


def detect_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    """평가 메인 함수."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    device = detect_device(args.device)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 이미지 파일 목록 수집
    input_dir = Path(args.input_dir)
    target_dir = Path(args.target_dir)
    exts = {".jpg", ".jpeg", ".png"}

    input_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in exts])
    if not input_files:
        logger.error(f"입력 이미지가 없음: {input_dir}")
        sys.exit(1)

    logger.info(f"평가 이미지: {len(input_files)}장")

    from src.utils.io import load_image, save_image
    from src.inference.apply_filter import apply_filter
    from src.evaluation.metrics import compute_delta_e, compute_psnr, compute_ssim
    from src.evaluation.visualize import create_comparison_grid, create_delta_e_heatmap

    all_de, all_psnr, all_ssim = [], [], []

    for input_path in input_files:
        target_path = target_dir / input_path.name
        if not target_path.exists():
            logger.warning(f"타겟 이미지 없음, 건너뜀: {target_path}")
            continue

        # 이미지 로드
        inp = load_image(str(input_path), as_float=True)
        tgt = load_image(str(target_path), as_float=True)

        # 필터 적용
        result = apply_filter(inp, model_path=args.model, style_idx=args.style, device=device)

        # 지표 계산
        de = compute_delta_e(result, tgt)
        psnr = compute_psnr(result, tgt)
        ssim = compute_ssim(result, tgt)

        all_de.append(de)
        all_psnr.append(psnr)
        all_ssim.append(ssim)

        logger.info(f"  {input_path.name}: ΔE={de:.3f}, PSNR={psnr:.2f}dB, SSIM={ssim:.4f}")

        # 시각화 저장
        if not args.no_visual:
            stem = input_path.stem
            create_comparison_grid(
                images=[inp, result, tgt],
                labels=["Original", "AI Filter", "Ground Truth"],
                output_path=output_dir / f"{stem}_comparison.png",
            )
            create_delta_e_heatmap(
                image1=result,
                image2=tgt,
                output_path=output_dir / f"{stem}_delta_e.png",
            )

    # 요약
    if not all_de:
        logger.error("평가 가능한 이미지 쌍이 없습니다.")
        sys.exit(1)

    import numpy as np
    summary = {
        "n_images": len(all_de),
        "mean_delta_e": float(np.mean(all_de)),
        "std_delta_e": float(np.std(all_de)),
        "mean_psnr": float(np.mean(all_psnr)),
        "mean_ssim": float(np.mean(all_ssim)),
        "target_delta_e_lt_1": float(np.mean(np.array(all_de) < 1.0)),
    }

    print("\n====== 평가 결과 요약 ======")
    print(f"이미지 수:         {summary['n_images']}")
    print(f"평균 ΔE:          {summary['mean_delta_e']:.4f} (목표: < 1.0)")
    print(f"ΔE < 1.0 비율:   {summary['target_delta_e_lt_1'] * 100:.1f}%")
    print(f"평균 PSNR:        {summary['mean_psnr']:.2f} dB")
    print(f"평균 SSIM:        {summary['mean_ssim']:.4f}")
    print("===========================\n")

    # JSON 결과 저장
    result_json = output_dir / "evaluation_results.json"
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"결과 저장: {result_json}")


if __name__ == "__main__":
    main()
