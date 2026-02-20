""".cube 파서 테스트."""

import pytest
import numpy as np
from pathlib import Path


class TestCubeParser:
    """.cube 파서 테스트."""

    def test_identity_lut(self):
        """항등 LUT 생성 검증."""
        # TODO: create_identity_lut(size=17) 생성
        # TODO: shape == [17, 17, 17, 3] 검증
        # TODO: 코너 값 검증 (0,0,0 -> [0,0,0], 16,16,16 -> [1,1,1])
        pass

    def test_write_and_read_roundtrip(self, tmp_path):
        """쓰기 -> 읽기 왕복 검증."""
        # TODO: 항등 LUT 생성
        # TODO: .cube 파일로 쓰기
        # TODO: 다시 읽기
        # TODO: 원본과 일치하는지 검증 (np.allclose)
        pass

    def test_invalid_file(self, tmp_path):
        """잘못된 파일 형식 에러 처리."""
        # TODO: 잘못된 내용의 파일 생성
        # TODO: ValueError 발생 확인
        pass
