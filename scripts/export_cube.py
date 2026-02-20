"""학습된 모델을 .cube 파일로 내보내기.

Usage:
    python scripts/export_cube.py --model checkpoints/nilut/best.pt --output filter.cube --size 33
    python scripts/export_cube.py --model checkpoints/nilut/best.pt --output style3.cube --style 3 --size 64
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NILUT 모델을 .cube 파일로 내보내기")
    parser.add_argument("--model", type=str, required=True, help="학습된 모델 경로")
    parser.add_argument("--output", type=str, required=True, help="출력 .cube 파일 경로")
    parser.add_argument("--size", type=int, default=33,
                        help="LUT 그리드 크기 (17/33/64). 클수록 정밀하지만 파일이 커짐")
    parser.add_argument("--style", type=int, default=0, help="스타일 인덱스")
    parser.add_argument("--device", type=str, default="cpu", help="디바이스")
    return parser.parse_args()


def main() -> None:
    """Export 메인 함수."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    from src.inference.export import export_to_cube

    logger.info(f"모델: {args.model}")
    logger.info(f"출력: {args.output} (LUT 크기: {args.size}x{args.size}x{args.size})")
    logger.info(f"스타일 인덱스: {args.style}")

    export_to_cube(
        model_path=args.model,
        output_path=args.output,
        cube_size=args.size,
        style_idx=args.style,
        device=args.device,
    )

    logger.info("Export 완료")


if __name__ == "__main__":
    main()
