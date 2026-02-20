"""필터 적용 CLI 스크립트.

Usage:
    python scripts/apply.py --model checkpoints/nilut/best.pt --input photo.jpg --output result.jpg
    python scripts/apply.py --cube filter.cube --input photo.jpg --output result.jpg
    python scripts/apply.py --model checkpoints/nilut/best.pt --input photo.jpg --output result.jpg --style 3
"""

import argparse
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="이미지에 AI 필터 적용")
    parser.add_argument("--model", type=str, default=None,
                        help="학습된 NILUT 모델(.pt) 경로")
    parser.add_argument("--cube", type=str, default=None,
                        help=".cube 파일 경로")
    parser.add_argument("--input", type=str, required=True,
                        help="입력 이미지 경로")
    parser.add_argument("--output", type=str, required=True,
                        help="출력 이미지 경로")
    parser.add_argument("--style", type=int, default=0,
                        help="스타일 인덱스 (다중 스타일 모델용, 기본값: 0)")
    parser.add_argument("--device", type=str, default="auto",
                        help="디바이스 (cpu/cuda/mps/auto)")
    return parser.parse_args()


def detect_device(requested: str) -> str:
    """디바이스 자동 감지."""
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    """필터 적용 메인 함수."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()

    if args.model is None and args.cube is None:
        print("오류: --model 또는 --cube 중 하나를 지정해야 합니다.")
        sys.exit(1)

    device = detect_device(args.device)
    logger.info(f"디바이스: {device}")

    # 입력 이미지 로드
    from src.utils.io import load_image, save_image

    logger.info(f"이미지 로드: {args.input}")
    image = load_image(args.input, as_float=False)  # uint8로 로드
    logger.info(f"이미지 크기: {image.shape}")

    # 필터 적용
    from src.inference.apply_filter import apply_filter

    logger.info("필터 적용 중...")
    result = apply_filter(
        image=image,
        model_path=args.model,
        cube_path=args.cube,
        style_idx=args.style,
        device=device,
    )

    # 결과 저장
    save_image(result, args.output)
    logger.info(f"결과 저장 완료: {args.output}")


if __name__ == "__main__":
    main()
