# ML 모델 배포 가이드

AI_picfilter ML 컴포넌트(NILUT, Image-Adaptive 3D LUT)의 배포 환경 설정 및 운영 가이드.

---

## 1. GPU 환경 설정

### 1.1 CUDA (NVIDIA GPU)

권장 CUDA 버전: **11.8 이상** (CUDA 12.x도 지원)

```bash
# CUDA 버전 확인
nvidia-smi
nvcc --version

# PyTorch CUDA 설치 (CUDA 11.8 기준)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 기준
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 설치 확인
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

### 1.2 MPS (Apple Silicon — M1/M2/M3)

macOS 12.3 이상, PyTorch 1.12 이상에서 지원.

```bash
# 표준 pip 설치 — MPS 자동 포함
pip install torch torchvision

# 확인
python -c "import torch; print(torch.backends.mps.is_available())"
```

### 1.3 CPU Fallback

GPU가 없는 환경에서는 자동으로 CPU를 사용한다.
`src/api/inference_api.py`의 `_resolve_device()`가 CUDA → MPS → CPU 순으로 감지한다.

```bash
# CPU-only 설치 (용량 절약)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 2. 학습 데이터 준비

### 2.1 Adobe FiveK 데이터셋 (권장)

5,000장의 RAW 사진과 5명의 전문 리터처가 보정한 이미지 쌍으로 구성.

```bash
# MIT 라이선스. 다운로드 방법:
# https://data.csail.mit.edu/graphics/fivek/

# 권장 디렉토리 구조
data/
  raw/        # 원본(보정 전) 이미지
  processed/  # 타겟(보정 후) 이미지 — 파일명 일치 필수
```

`ImagePairDataset`이 `raw/`와 `processed/` 파일명을 매칭하여 쌍을 구성한다.

### 2.2 .cube 파일 기반 학습

기존 LUT 파일(.cube)에서 픽셀 쌍을 추출하여 NILUT를 학습시킬 수 있다.

```bash
# CubeDataset 사용 예시
python scripts/train.py \
  --config configs/nilut.yaml \
  --data-type cube \
  --cube-dir data/luts/
```

`CubeDataset`은 .cube 파일에서 RGB 그리드를 읽어 (입력, 타겟) 픽셀 쌍을 반환한다.

### 2.3 레퍼런스 이미지 기반 학습 (API 경유)

```python
from src.api.inference_api import run_style_learning

result = run_style_learning(
    reference_images=["path/to/ref1.jpg", "path/to/ref2.jpg"],
    output_path="checkpoints/my_style.pt",
    model_type="adaptive_3dlut",  # 또는 "nilut"
    config={"epochs": 200, "lr": 1e-4, "n_basis": 3},
)
```

---

## 3. 사전 학습 모델 파일 관리

### 3.1 S3 경로 규칙

```
s3://<bucket>/models/
  nilut/
    v{version}/
      best.pt           # 최고 성능 체크포인트
      last.pt           # 마지막 에폭 체크포인트
      config.yaml       # 학습 설정 복사본
  lut3d/
    v{version}/
      best.pt
      last.pt
      config.yaml
  exports/
    {style_name}_{size}.cube   # 배포용 .cube 파일
```

버전 규칙: `v1`, `v2`, ... (시맨틱 버저닝 미사용, 정수 증가)

### 3.2 체크포인트 다운로드

```bash
# AWS CLI
aws s3 cp s3://<bucket>/models/nilut/v1/best.pt checkpoints/nilut/best.pt

# 로컬 디렉토리 구조
checkpoints/
  nilut/
    best.pt
  lut3d/
    best.pt
```

### 3.3 모델 메타데이터 확인

```python
from src.api.inference_api import get_model_info

info = get_model_info("checkpoints/nilut/best.pt")
# {
#   "model_type": "nilut",
#   "model_config": {"hidden_dims": [256, 256, 256], ...},
#   "file_size_bytes": 204800,
#   "param_count": 527619
# }
```

---

## 4. Docker 컨테이너 내 GPU 사용

### 4.1 GPU 지원 Docker 실행

```bash
# NVIDIA Container Toolkit 필요
# 설치: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# GPU 전체 사용
docker run --gpus all \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/data:/app/data \
  ai-picfilter-ml:latest

# 특정 GPU 지정
docker run --gpus '"device=0"' \
  ai-picfilter-ml:latest

# GPU 개수 제한
docker run --gpus 2 \
  ai-picfilter-ml:latest
```

### 4.2 ML 서비스 Dockerfile 예시

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# 모델 파일은 볼륨 마운트 또는 런타임에 S3에서 다운로드
# COPY checkpoints/ ./checkpoints/  # 이미지 크기 증가 주의

# Celery 워커 실행
CMD ["celery", "-A", "backend.celery_app", "worker", "--loglevel=info", "-Q", "ml_tasks"]
```

### 4.3 docker-compose GPU 설정

```yaml
# docker-compose.yml 중 ml-worker 서비스
ml-worker:
  image: ai-picfilter-ml:latest
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1           # GPU 1개 사용
            capabilities: [gpu]
  volumes:
    - ./checkpoints:/app/checkpoints
    - ./data:/app/data
  environment:
    - CUDA_VISIBLE_DEVICES=0
```

---

## 5. 벤치마크 및 배포 기준 검증

학습 완료 후 배포 전 벤치마크를 반드시 실행한다.

```bash
# 전체 해상도 벤치마크 (기본 10회 반복)
python scripts/benchmark.py --model checkpoints/nilut/best.pt

# 특정 해상도만
python scripts/benchmark.py \
  --model checkpoints/lut3d/best.pt \
  --resolutions 1920x1080 3840x2160 \
  --iterations 20

# JSON 리포트 저장 위치: reports/benchmark_YYYYMMDD_HHMMSS.json
```

### 5.1 배포 기준 (자동 검증)

| 항목 | NILUT 기준 | 3DLUT 기준 |
|------|-----------|-----------|
| 모델 크기 | < 1 MB | < 10 MB |
| 항등 변환 평균 ΔE | < 1.0 | < 1.0 |
| 추론 속도 (1080p, GPU) | < 16ms | < 2ms |

벤치마크 스크립트가 기준 미충족 시 exit code 1을 반환하므로 CI/CD 파이프라인에서 활용할 수 있다.

---

## 6. 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `CUDA_VISIBLE_DEVICES` | 사용할 GPU 인덱스 | (전체) |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | MPS 메모리 상한 비율 | `0.0` (제한 없음) |
| `OMP_NUM_THREADS` | CPU 스레드 수 | PyTorch 기본값 |

```bash
# CPU 추론 강제
CUDA_VISIBLE_DEVICES="" python scripts/benchmark.py --model ...

# MPS 메모리 제한 (Apple Silicon)
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 python ...
```
