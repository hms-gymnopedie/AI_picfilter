# Backend Architecture

## System Overview

```
                           ┌─────────────────────────────────────────────┐
                           │              Client (Web/Mobile)            │
                           └──────────────┬──────────────────────────────┘
                                          │
                                          │ HTTPS
                                          ▼
                           ┌──────────────────────────────┐
                           │         Nginx (Reverse Proxy) │
                           │         + SSL Termination     │
                           └──────────┬───────────┬────────┘
                                      │           │
                          REST API    │           │  Direct Upload
                                      ▼           ▼
                ┌─────────────────────────┐   ┌──────────────────┐
                │    FastAPI Application   │   │  S3 / MinIO      │
                │                         │   │  (Object Storage) │
                │  - Auth (JWT)           │   └──────────────────┘
                │  - Image Management     │           ▲
                │  - Style CRUD           │           │ presigned URL
                │  - Community            │           │
                │  - Job Submission       │───────────┘
                └────┬──────────┬─────────┘
                     │          │
            ┌────────┘          └────────┐
            ▼                            ▼
  ┌──────────────────┐        ┌──────────────────┐
  │   PostgreSQL     │        │   Redis          │
  │                  │        │                  │
  │  - Users         │        │  - Job Queue     │
  │  - Images        │        │  - Session Cache │
  │  - Styles        │        │  - Rate Limiting │
  │  - Jobs          │        │  - Pub/Sub       │
  │  - Comments      │        └────────┬─────────┘
  │  - Ratings       │                 │
  └──────────────────┘                 │
                                       ▼
                            ┌──────────────────────┐
                            │   Celery Workers     │
                            │                      │
                            │  - Style Learning    │
                            │  - Filter Apply      │
                            │  - .cube Export      │
                            │  - Thumbnail Gen     │
                            │                      │
                            │  (GPU instances for  │
                            │   ML workloads)      │
                            └──────────────────────┘
```

## Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Web Framework** | FastAPI (Python 3.11+) | async 지원, 자동 OpenAPI 문서, PyTorch 생태계와 동일 언어 |
| **ML Framework** | PyTorch 2.x | 프로젝트 참조 모델들이 모두 PyTorch 기반 |
| **Task Queue** | Celery 5.x + Redis Broker | ML 학습/추론의 비동기 처리, GPU worker 분리 |
| **Database** | PostgreSQL 16 | JSONB 지원, 풍부한 인덱싱, 안정성 |
| **Cache / Broker** | Redis 7.x | Celery broker, 세션 캐시, rate limiting |
| **Object Storage** | MinIO (dev) / AWS S3 (prod) | 이미지 및 .cube 파일 저장, presigned URL 지원 |
| **ORM** | SQLAlchemy 2.x + Alembic | async 지원, 마이그레이션 관리 |
| **Auth** | python-jose (JWT) + passlib (bcrypt) | 토큰 기반 인증 |
| **Containerization** | Docker + Docker Compose | 로컬 개발 환경 표준화 |

## ML Model Serving Strategy

### 선택: FastAPI + Celery Worker (직접 로드 방식)

TorchServe나 별도 gRPC 서비스 대신, Celery worker 프로세스 내에서 PyTorch 모델을 직접 로드하는 방식을 채택합니다.

**근거:**
1. 프로젝트 초기 단계에서 서비스 간 통신 복잡도를 최소화
2. 모델 크기가 10MB 이하로 매우 가벼움 -- worker 프로세스 메모리에 충분히 적재 가능
3. 모델 종류가 3~4개로 제한적 -- 복잡한 모델 레지스트리 불필요
4. Celery worker를 GPU 인스턴스에 배치하여 GPU 활용 가능

**Worker 구성:**
```
Worker Pool:
  - ml-worker (GPU): 스타일 학습, 필터 적용, .cube 내보내기
  - io-worker (CPU): 썸네일 생성, 이미지 메타데이터 추출, 파일 정리
```

**향후 확장:**
- 트래픽 증가 시 TorchServe나 Triton Inference Server로 ML 서빙 분리
- 모델별 autoscaling이 필요할 때 Kubernetes + KServe 도입 검토

### 모델 로딩 전략

```python
# Worker 시작 시 모델을 한 번 로드하고 재사용
# Celery worker_init signal 활용

@worker_init.connect
def load_models(**kwargs):
    global MODEL_REGISTRY
    MODEL_REGISTRY = {
        "adaptive_3dlut": load_adaptive_3dlut_model(),
        "nilut": load_nilut_model(),
        "dlut": load_dlut_model(),
    }
```

## Async Job Processing

### Job Lifecycle

```
Client Request
    │
    ▼
FastAPI: 유효성 검증 → DB에 Job 레코드 생성 (status=queued) → Celery에 task 전송
    │
    ▼
Redis: 메시지 큐에 task 적재
    │
    ▼
Celery Worker: task 수신 → DB 상태 업데이트 (status=processing)
    │                    → ML 모델 실행
    │                    → 결과를 S3에 저장
    │                    → DB 상태 업데이트 (status=completed)
    │
    ▼
Client: GET /jobs/{id}로 상태 폴링
```

### Polling vs WebSocket

초기 구현은 클라이언트 폴링 방식을 사용합니다.
- 스타일 학습: 2~5분 소요 -- 5초 간격 폴링
- 필터 적용: 수 초 소요 -- 1초 간격 폴링
- .cube 내보내기: 수 초 소요 -- 1초 간격 폴링

향후 실시간 알림이 필요할 경우 Redis Pub/Sub + WebSocket 추가를 검토합니다.

### 실패 처리

- Celery task에 `max_retries=3`, `default_retry_delay=30` 설정
- 3회 재시도 후에도 실패 시 `status=failed`로 기록하고 에러 메시지 저장
- GPU OOM 에러 시 이미지 크기를 줄여 재시도하는 fallback 로직 포함

## Image Storage Design

### Storage Layer

```
MinIO / S3 Bucket Structure:
  picfilter-storage/
    ├── images/
    │   ├── {user_id}/{image_id}/original.{ext}
    │   └── {user_id}/{image_id}/thumbnail.webp
    ├── results/
    │   └── {user_id}/{job_id}/result.{ext}
    ├── styles/
    │   ├── {style_id}/model.pt          # 학습된 모델 가중치
    │   └── {style_id}/preview.webp      # 스타일 미리보기
    └── cubes/
        └── {style_id}/{export_job_id}.cube
```

### Upload Flow (Presigned URL)

```
1. Client → POST /images/upload (filename, content_type, size)
2. Server → S3 presigned PUT URL 생성 (10분 유효)
3. Server → DB에 image 레코드 생성 (status=pending)
4. Client → S3에 직접 PUT 업로드
5. Client → POST /images/{id}/confirm
6. Server → S3 HeadObject로 파일 존재 확인
7. Server → 이미지 메타데이터 추출 (dimension, format)
8. Server → 썸네일 생성 작업 큐에 전송
9. Server → DB 상태 업데이트 (status=confirmed)
```

이 방식의 장점:
- 대용량 이미지가 백엔드 서버를 경유하지 않아 서버 부하 최소화
- S3의 multipart upload 기능을 직접 활용 가능
- 네트워크 효율성 극대화

### CDN Integration

- Production 환경에서는 CloudFront (또는 동등한 CDN)를 S3 앞에 배치
- 썸네일과 결과 이미지에 대해 캐시 적용
- 원본 이미지는 CDN을 통하지 않고 presigned URL로 직접 접근

### Image Processing Pipeline

```
Upload Confirmed
    │
    ├─► Thumbnail Generation (io-worker)
    │     - 400x400 max, WebP format
    │     - quality 80, EXIF orientation 적용
    │
    └─► Metadata Extraction (io-worker)
          - width, height, format, color profile
          - EXIF data (optional, privacy 고려하여 선택적 저장)
```

## Project Directory Structure

```
AI_picfilter/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry
│   ├── config.py               # Settings (env vars)
│   ├── dependencies.py         # Dependency injection
│   ├── api/
│   │   ├── __init__.py
│   │   ├── auth.py             # POST /auth/*
│   │   ├── images.py           # /images/*
│   │   ├── styles.py           # /styles/*
│   │   └── community.py        # comments, ratings
│   ├── models/                 # SQLAlchemy ORM models
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── image.py
│   │   ├── style.py
│   │   ├── job.py
│   │   └── community.py
│   ├── schemas/                # Pydantic request/response schemas
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── image.py
│   │   ├── style.py
│   │   └── community.py
│   ├── services/               # Business logic
│   │   ├── __init__.py
│   │   ├── auth_service.py
│   │   ├── image_service.py
│   │   ├── style_service.py
│   │   └── storage_service.py
│   ├── workers/                # Celery tasks
│   │   ├── __init__.py
│   │   ├── celery_app.py
│   │   ├── ml_tasks.py         # Style learning, filter apply
│   │   └── io_tasks.py         # Thumbnail, metadata
│   └── ml/                     # ML model wrappers
│       ├── __init__.py
│       ├── adaptive_3dlut.py
│       ├── nilut.py
│       ├── dlut.py
│       └── cube_export.py
├── alembic/                    # DB migrations
│   ├── versions/
│   └── env.py
├── tests/
│   ├── test_api/
│   ├── test_services/
│   └── test_workers/
├── docs/
├── docker-compose.yml
├── Dockerfile
├── alembic.ini
├── requirements.txt
└── .env.example
```

## Security Considerations

- JWT access token TTL: 1시간, refresh token TTL: 7일
- 비밀번호: bcrypt 해싱 (cost factor 12)
- S3 presigned URL: 10분 유효, PUT 전용 (GET은 별도 발급)
- 이미지 업로드: content-type 검증 + magic bytes 확인
- 파일 크기 제한: 50MB (nginx + presigned URL 조건 모두에서 적용)
- SQL injection: SQLAlchemy ORM 사용으로 parameterized query 보장
- Rate limiting: Redis 기반 sliding window 방식
- CORS: 허용 origin 명시적 설정

## Environment Configuration

환경별 설정은 환경 변수로 관리합니다:

```env
# .env.example
DATABASE_URL=postgresql+asyncpg://picfilter:password@localhost:5432/picfilter
REDIS_URL=redis://localhost:6379/0
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=picfilter-storage
JWT_SECRET_KEY=change-me-in-production
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
```
