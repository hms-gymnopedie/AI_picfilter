# AI_picfilter PyTorch 프로젝트 구조

## 디렉토리 레이아웃

```
AI_picfilter/
├── CLAUDE.md                       # Claude Code 가이드
├── pre_report.md                   # 기술 분석 보고서
├── README.md                       # 프로젝트 소개 및 설치 가이드
├── pyproject.toml                  # 프로젝트 메타데이터 및 의존성
├── requirements.txt                # pip 의존성 (fallback)
│
├── docs/                           # 문서
│   ├── architecture.md             # 기술 아키텍처
│   ├── ai_roadmap.md               # 구현 로드맵
│   └── project_structure.md        # 본 문서
│
├── configs/                        # 실험 설정 파일
│   ├── default.yaml                # 기본 설정
│   ├── nilut.yaml                  # NILUT 학습 설정
│   ├── adaptive_lut.yaml           # Image-Adaptive 3D LUT 설정
│   └── dlut.yaml                   # D-LUT 설정
│
├── src/                            # 메인 소스 코드
│   ├── __init__.py
│   │
│   ├── models/                     # 모델 정의
│   │   ├── __init__.py
│   │   ├── nilut.py                # NILUT (MLP 기반 연속 색상 변환)
│   │   ├── adaptive_lut.py         # Image-Adaptive 3D LUT (CNN + Basis LUT)
│   │   ├── dlut.py                 # Diffusion-based LUT
│   │   └── backbone.py             # 공용 backbone 네트워크
│   │
│   ├── data/                       # 데이터 로딩 및 전처리
│   │   ├── __init__.py
│   │   ├── dataset.py              # PyTorch Dataset 클래스
│   │   ├── cube_parser.py          # .cube 파일 파서 (읽기/쓰기)
│   │   ├── hald.py                 # Hald 이미지 생성 및 처리
│   │   └── transforms.py           # 이미지 변환 (정규화, 색공간, 증강)
│   │
│   ├── training/                   # 학습 관련
│   │   ├── __init__.py
│   │   ├── trainer.py              # 학습 루프 (train/val/test)
│   │   ├── losses.py               # 손실 함수 (L1, perceptual, ΔE, smoothness)
│   │   └── schedulers.py           # 학습률 스케줄러
│   │
│   ├── inference/                  # 추론 및 필터 적용
│   │   ├── __init__.py
│   │   ├── apply_filter.py         # 이미지에 필터 적용
│   │   └── export.py               # 모델 export (.cube, ONNX, TorchScript)
│   │
│   ├── evaluation/                 # 평가 지표
│   │   ├── __init__.py
│   │   ├── metrics.py              # ΔE, PSNR, SSIM 계산
│   │   └── visualize.py            # 결과 시각화 (비교 그리드)
│   │
│   └── utils/                      # 유틸리티
│       ├── __init__.py
│       ├── color.py                # 색공간 변환 (sRGB, Lab, XYZ)
│       ├── lut.py                  # 3D LUT 연산 (trilinear interpolation 등)
│       └── io.py                   # 이미지 I/O 헬퍼
│
├── scripts/                        # 실행 스크립트
│   ├── train.py                    # 학습 실행 진입점
│   ├── evaluate.py                 # 평가 실행
│   ├── apply.py                    # 필터 적용 CLI
│   ├── export_cube.py              # .cube 파일 내보내기
│   └── benchmark.py                # 속도/메모리 벤치마크
│
├── tests/                          # 테스트
│   ├── test_models.py              # 모델 forward pass 테스트
│   ├── test_cube_parser.py         # .cube 파서 테스트
│   ├── test_metrics.py             # 평가 지표 테스트
│   ├── test_transforms.py          # 데이터 변환 테스트
│   └── test_lut.py                 # LUT 연산 테스트
│
├── notebooks/                      # Jupyter 노트북 (실험/시각화)
│   ├── 01_data_exploration.ipynb
│   ├── 02_nilut_training.ipynb
│   └── 03_style_blending.ipynb
│
├── data/                           # 데이터 (gitignore 대상)
│   ├── raw/                        # 원본 이미지
│   ├── processed/                  # 전처리된 데이터
│   ├── cube_files/                 # .cube LUT 파일
│   └── hald/                       # Hald 이미지
│
└── checkpoints/                    # 학습된 모델 (gitignore 대상)
    ├── nilut/
    ├── adaptive_lut/
    └── dlut/
```

---

## 주요 모듈 설명

### `src/models/` — 모델 정의

| 파일 | 역할 | 핵심 클래스 |
|---|---|---|
| `nilut.py` | MLP 기반 연속 색상 변환 모델 | `NILUT(nn.Module)` — 입력 RGB(+조건) -> 출력 RGB |
| `adaptive_lut.py` | CNN으로 basis LUT 가중치를 예측하는 적응형 모델 | `AdaptiveLUT(nn.Module)` — backbone + learnable basis LUTs |
| `dlut.py` | 확산 모델 기반 LUT 학습 | `DiffusionLUT(nn.Module)` — score network + sampling |
| `backbone.py` | 공용 feature extractor | `LightweightCNN` — 이미지 분석용 경량 CNN |

### `src/data/` — 데이터 파이프라인

| 파일 | 역할 |
|---|---|
| `dataset.py` | `ImagePairDataset` (보정 전/후 쌍), `CubeDataset` (.cube 기반 학습) |
| `cube_parser.py` | .cube 파일 읽기(parse) 및 쓰기(export). LUT 크기 자동 감지 |
| `hald.py` | Hald CLUT 이미지 생성. 레퍼런스 이미지에 적용하여 스타일 추출 |
| `transforms.py` | `Normalize`, `RandomCrop`, `ColorJitter`, `sRGBToLinear` 등 |

### `src/training/` — 학습

| 파일 | 역할 |
|---|---|
| `trainer.py` | 학습/검증/테스트 루프. 체크포인트 저장, early stopping, 로깅 통합 |
| `losses.py` | `DeltaELoss`, `PerceptualLoss`, `SmoothnessLoss`, `MonotonicityLoss` |
| `schedulers.py` | CosineAnnealing, WarmupCosine 등 학습률 스케줄러 |

### `src/inference/` — 추론 및 배포

| 파일 | 역할 |
|---|---|
| `apply_filter.py` | 학습된 모델(또는 .cube)을 이미지에 적용. 배치/단건 처리 |
| `export.py` | PyTorch 모델 -> .cube, ONNX, TorchScript 변환 |

### `src/evaluation/` — 평가

| 파일 | 역할 |
|---|---|
| `metrics.py` | `compute_delta_e()`, `compute_psnr()`, `compute_ssim()` |
| `visualize.py` | 원본/스타일/결과 비교 그리드, ΔE 히트맵 생성 |

---

## 데이터 파이프라인 흐름

### 학습 시 데이터 흐름

```
[원본 이미지]          [보정된 이미지 / .cube 파일]
     |                          |
     v                          v
 transforms.py              cube_parser.py
 (resize, normalize,        (.cube -> 3D numpy array)
  color space convert)           |
     |                          v
     v                     hald.py
 dataset.py                (Hald 이미지 생성 ->
 (ImagePairDataset)         보정 적용 -> 쌍 생성)
     |                          |
     +----------+---------------+
                |
                v
         DataLoader (batch, shuffle, num_workers)
                |
                v
         trainer.py
         (forward -> loss -> backward -> optimizer.step)
                |
                v
         checkpoints/ (model.pt 저장)
```

### 추론 시 데이터 흐름

```
[대상 이미지] + [학습된 모델 or .cube 파일]
       |                    |
       v                    v
  transforms.py        모델 로드 / cube_parser.py
  (전처리)              (체크포인트 로드)
       |                    |
       +--------+-----------+
                |
                v
         apply_filter.py
         (모델 추론 or LUT 적용)
                |
                v
         [필터 적용된 출력 이미지]
```

### .cube Export 흐름

```
[학습된 NILUT / Adaptive LUT 모델]
                |
                v
         export.py
         (identity LUT 그리드 생성 ->
          모델로 변환 -> .cube 포맷 저장)
                |
                v
         [output.cube 파일]
         (Photoshop / Premiere Pro 호환)
```

---

## 설정 관리

`configs/` 디렉토리에서 YAML 파일로 실험 설정을 관리한다. Hydra 또는 OmegaConf를 사용하여 CLI에서 오버라이드가 가능하다.

```yaml
# configs/nilut.yaml 예시
model:
  name: nilut
  hidden_dims: [256, 256, 256]
  activation: gelu
  num_styles: 512

data:
  dataset: adobe_fivek
  batch_size: 64
  num_workers: 4
  image_size: 256

training:
  epochs: 100
  lr: 1e-3
  scheduler: cosine
  loss:
    l1_weight: 1.0
    delta_e_weight: 0.5

export:
  cube_size: 64
  format: cube
```

실행 예시:
```bash
# NILUT 학습
python scripts/train.py --config configs/nilut.yaml

# 필터 적용
python scripts/apply.py --model checkpoints/nilut/best.pt --input photo.jpg --output result.jpg

# .cube 내보내기
python scripts/export_cube.py --model checkpoints/nilut/best.pt --size 64 --output filter.cube

# 벤치마크
python scripts/benchmark.py --model checkpoints/nilut/best.pt --resolution 3840x2160
```
