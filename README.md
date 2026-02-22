# AI_picfilter

[![CI](https://github.com/hms-gymnopedie/AI_picfilter/actions/workflows/ci.yml/badge.svg)](https://github.com/hms-gymnopedie/AI_picfilter/actions/workflows/ci.yml)

AI 기반 이미지 스타일 학습 및 자동 필터 생성 시스템.

레퍼런스 이미지 한 장을 업로드하면 AI가 색상, 조명, 질감, 분위기를 학습하고, 다른 이미지에 그 스타일을 적용하거나 Photoshop/Premiere와 호환되는 `.cube` LUT 파일로 내보낼 수 있습니다.

---

## 주요 기능

- **스타일 학습**: 레퍼런스 이미지로부터 색감/분위기를 자동 학습
- **필터 적용**: 학습된 스타일을 다른 이미지에 실시간 적용
- **다중 스타일 블렌딩**: 두 스타일 사이를 연속적으로 보간
- **`.cube` 내보내기**: Photoshop, Premiere Pro, DaVinci Resolve 호환
- **실시간 WebGL 프리뷰**: 브라우저에서 4K 이미지 즉시 확인

## AI 모델

| 모델 | 특징 | 속도 | 크기 |
|------|------|------|------|
| **NILUT** | 경량 MLP, 다중 스타일 블렌딩, CPU 가능 | <16ms (4K) | <1MB |
| **Image-Adaptive 3D LUT** | CNN 기반 이미지 적응형, 더 자연스러운 결과 | <2ms (4K GPU) | <10MB |

## 기술 스택

**ML**
- PyTorch 2.x, NILUT, Image-Adaptive 3D LUT
- CIE76 ΔE < 1.0 색상 정확도 목표

**Backend**
- FastAPI, Celery + Redis, PostgreSQL, MinIO (S3 호환)

**Frontend**
- Next.js 14 App Router, TypeScript, Tailwind CSS
- WebGL2 실시간 LUT 프리뷰, SSE 학습 진행 스트리밍

**Infra**
- Docker Compose (개발/프로덕션), Nginx 리버스 프록시

---

## 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/hms-gymnopedie/AI_picfilter.git
cd AI_picfilter
```

### 2. 환경변수 설정

```bash
cp .env.example .env
# .env 파일을 열어 필수 값 설정 (JWT_SECRET_KEY, DB 비밀번호 등)
```

### 3. 개발 환경 실행

```bash
docker compose up -d
```

서비스 접속:
- 프론트엔드: http://localhost:3000
- API 문서: http://localhost:8000/docs
- MinIO 콘솔: http://localhost:9001

### 4. DB 초기화

```bash
docker compose exec app alembic upgrade head
```

---

## ML 모델 단독 사용

```bash
pip install -r requirements.txt

# 스타일 학습 (.cube 파일 기반)
python scripts/train.py --data-type cube --data-dir ./luts --output ./checkpoints/my_style

# 이미지에 필터 적용
python scripts/apply.py --model ./checkpoints/my_style/best.pt --input photo.jpg --output result.jpg

# .cube 파일로 내보내기
python scripts/export_cube.py --model ./checkpoints/my_style/best.pt --output my_filter.cube

# 벤치마크 실행
python scripts/benchmark.py --model ./checkpoints/my_style/best.pt
```

---

## 프로젝트 구조

```
AI_picfilter/
├── src/                    # ML 코어
│   ├── models/             # NILUT, Image-Adaptive 3D LUT
│   ├── data/               # .cube 파서, 데이터셋, 전처리
│   ├── training/           # 학습 루프, 손실 함수
│   ├── inference/          # 필터 적용, .cube/.onnx 내보내기
│   ├── evaluation/         # ΔE, PSNR, SSIM 지표
│   └── api/                # 백엔드 연동 인터페이스
├── backend/                # FastAPI 서버
│   ├── api/v1/             # auth, images, styles, filters
│   ├── models/             # ORM (users, styles, jobs ...)
│   └── workers/            # Celery ML 태스크
├── frontend/               # Next.js 14 앱
│   └── src/
│       ├── app/            # 페이지 (Dashboard, Learn, Studio ...)
│       ├── components/     # UI 컴포넌트
│       └── lib/webgl/      # LUT 실시간 프리뷰 셰이더
├── nginx/                  # 리버스 프록시 설정
├── configs/                # 학습 하이퍼파라미터
├── scripts/                # CLI 도구
├── tests/                  # 유닛/통합 테스트 (38개)
└── docs/                   # 설계 문서
```

---

## 테스트

```bash
pip install pytest
python -m pytest tests/ -v
# 38 passed
```

---

## 프로덕션 배포

```bash
docker compose -f docker-compose.prod.yml up -d
```

자세한 내용은 [`docs/deployment_backend.md`](docs/deployment_backend.md)를 참고하세요.

---

## 문서

| 문서 | 설명 |
|------|------|
| [docs/architecture.md](docs/architecture.md) | AI 모델 기술 아키텍처 |
| [docs/ai_roadmap.md](docs/ai_roadmap.md) | Phase 1~3 구현 로드맵 |
| [docs/model_comparison.md](docs/model_comparison.md) | 모델 정량 비교표 |
| [docs/api_spec.md](docs/api_spec.md) | REST API 명세 |
| [docs/db_schema.md](docs/db_schema.md) | 데이터베이스 스키마 |
| [docs/deployment_ml.md](docs/deployment_ml.md) | ML 배포 가이드 |
| [docs/deployment_backend.md](docs/deployment_backend.md) | 백엔드 배포 가이드 |
| [docs/deployment_frontend.md](docs/deployment_frontend.md) | 프론트엔드 배포 가이드 |

---

## 품질 기준

- 색상 정확도: CIE76 ΔE < 1.0
- 추론 속도: NILUT < 16ms / 3D LUT < 2ms (4K GPU)
- 모델 크기: NILUT < 1MB / 3D LUT < 10MB
