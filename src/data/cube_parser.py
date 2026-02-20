""".cube 파일 파서.

3D LUT 파일(.cube)을 읽고 쓰는 유틸리티.
Adobe/DaVinci Resolve 호환 포맷을 지원한다.
"""

import numpy as np
from pathlib import Path


class CubeParser:
    """.cube 파일 파서.

    .cube 파일의 읽기(parse)와 쓰기(export)를 담당한다.
    LUT 크기를 자동 감지하며, 메타데이터(TITLE, DOMAIN_MIN, DOMAIN_MAX)를 처리한다.
    """

    def __init__(self) -> None:
        self.title: str = ""
        self.size: int = 0
        self.domain_min: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.domain_max: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.lut: np.ndarray | None = None

    def read(self, path: str | Path) -> np.ndarray:
        """Read a .cube file and return the 3D LUT as a numpy array.

        .cube 파일 형식:
        - 헤더: TITLE, LUT_3D_SIZE, DOMAIN_MIN, DOMAIN_MAX
        - 데이터: B가 가장 빠르게 변하는 순서 (B-major)로 나열된 RGB 값

        Args:
            path: .cube 파일 경로

        Returns:
            3D LUT 배열 [size, size, size, 3], float32, 범위 [0, 1]

        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
            ValueError: 파일 형식이 올바르지 않을 때
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없음: {path}")

        data_rows: list[list[float]] = []
        size: int = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # 빈 줄 및 주석 건너뜀
                if not line or line.startswith("#"):
                    continue

                if line.upper().startswith("TITLE"):
                    self.title = line[5:].strip().strip('"')

                elif line.upper().startswith("LUT_3D_SIZE"):
                    size = int(line.split()[-1])
                    self.size = size

                elif line.upper().startswith("DOMAIN_MIN"):
                    vals = line.split()[1:]
                    self.domain_min = np.array(
                        [float(v) for v in vals], dtype=np.float32
                    )

                elif line.upper().startswith("DOMAIN_MAX"):
                    vals = line.split()[1:]
                    self.domain_max = np.array(
                        [float(v) for v in vals], dtype=np.float32
                    )

                else:
                    # 데이터 행: "R G B"
                    parts = line.split()
                    if len(parts) == 3:
                        try:
                            data_rows.append([float(p) for p in parts])
                        except ValueError:
                            # 숫자가 아닌 행은 건너뜀 (알 수 없는 키워드 등)
                            pass

        if size == 0:
            raise ValueError(f"LUT_3D_SIZE 헤더를 찾을 수 없음: {path}")

        expected = size**3
        if len(data_rows) != expected:
            raise ValueError(
                f"데이터 행 수 불일치: 예상 {expected}, 실제 {len(data_rows)}"
            )

        # [size^3, 3] -> [size, size, size, 3]
        # .cube 파일은 B-major 순서: B(가장 빠름), G, R(가장 느림)
        lut_flat = np.array(data_rows, dtype=np.float32)
        # reshape: [R, G, B, 3] 순서로 변환
        # 데이터 순서: R 고정, G 고정, B 변화 -> B-major
        self.lut = lut_flat.reshape(size, size, size, 3)
        return self.lut

    def write(
        self,
        lut: np.ndarray,
        path: str | Path,
        title: str = "AI_picfilter Generated LUT",
    ) -> None:
        """Write a 3D LUT to a .cube file.

        Args:
            lut: 3D LUT 배열 [size, size, size, 3], 범위 [0, 1]
            path: 출력 .cube 파일 경로
            title: LUT 제목 (메타데이터)
        """
        lut = np.asarray(lut, dtype=np.float32)
        if (
            lut.ndim != 4
            or lut.shape[3] != 3
            or lut.shape[0] != lut.shape[1] != lut.shape[2]
        ):
            raise ValueError(
                f"LUT shape 오류: {lut.shape} — [size, size, size, 3]이어야 함"
            )

        size = lut.shape[0]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"TITLE {title}\n")
            f.write("\n")
            f.write(f"LUT_3D_SIZE {size}\n")
            f.write("\n")
            f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
            f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
            f.write("\n")

            # B-major 순서로 출력: R 루프 -> G 루프 -> B 루프
            for r in range(size):
                for g in range(size):
                    for b in range(size):
                        rv, gv, bv = lut[r, g, b]
                        f.write(f"{rv:.6f} {gv:.6f} {bv:.6f}\n")

    @staticmethod
    def create_identity_lut(size: int = 33) -> np.ndarray:
        """항등 LUT 생성 (입력 = 출력).

        Args:
            size: LUT 그리드 크기 (일반적으로 17, 33, 64)

        Returns:
            항등 3D LUT [size, size, size, 3], float32
        """
        coords = np.linspace(0.0, 1.0, size, dtype=np.float32)
        # [size, size, size, 3] 그리드 생성: 첫 번째 축이 R
        r, g, b = np.meshgrid(coords, coords, coords, indexing="ij")
        identity = np.stack([r, g, b], axis=-1)
        return identity
