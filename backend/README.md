# AI_picfilter Backend

FastAPI 기반의 AI 스타일 필터 생성 및 적용 백엔드 API

## 기술 스택

- **프레임워크**: FastAPI
- **데이터베이스**: PostgreSQL (async SQLAlchemy)
- **작업 큐**: Celery + Redis
- **저장소**: MinIO (S3 호환)
- **인증**: JWT (python-jose)

## 프로젝트 구조

```
backend/
├── main.py                 # FastAPI 앱 진입점
├── core/
│   ├── config.py          # 환경 변수 설정
│   └── database.py        # 데이터베이스 연결
├── models/                # SQLAlchemy ORM 모델
│   ├── user.py
│   ├── image.py
│   ├── style.py
│   ├── job.py
│   ├── community.py
│   └── token.py
├── schemas/               # Pydantic 요청/응답 스키마
│   ├── auth.py
│   ├── image.py
│   ├── style.py
│   ├── job.py
│   └── community.py
├── api/
│   ├── dependencies.py    # 의존성 주입 (JWT 인증)
│   └── v1/
│       ├── auth.py        # 회원가입, 로그인, 토큰 갱신
│       ├── images.py      # 이미지 업로드/조회/삭제
│       └── styles.py      # 스타일 학습/조회/수정/삭제
├── workers/
│   ├── celery_app.py      # Celery 설정
│   └── ml_tasks.py        # ML 작업 태스크 (스텁)
├── alembic/               # 데이터베이스 마이그레이션
│   ├── env.py
│   ├── versions/
│   └── 001_initial_schema.py
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

## 설치 및 실행

### 1. 환경 변수 설정

```bash
cp .env.example .env
# .env 파일을 환경에 맞게 수정
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 데이터베이스 마이그레이션

```bash
alembic upgrade head
```

### 4. FastAPI 앱 실행

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

API 문서: http://localhost:8000/docs

### 5. Celery Worker 실행 (별도 터미널)

```bash
celery -A backend.workers.ml_tasks worker --loglevel=info
```

## API 엔드포인트

### 인증 (Authentication)

- `POST /v1/auth/register` - 회원가입
- `POST /v1/auth/login` - 로그인
- `POST /v1/auth/refresh` - 토큰 갱신

### 이미지 (Images)

- `POST /v1/images/upload` - 업로드 URL 발급 (Presigned URL)
- `POST /v1/images/{image_id}/confirm` - 업로드 완료 확인
- `GET /v1/images` - 이미지 목록 조회
- `GET /v1/images/{image_id}` - 이미지 상세 조회
- `DELETE /v1/images/{image_id}` - 이미지 삭제

### 스타일 (Styles)

- `POST /v1/styles/learn` - 스타일 학습 작업 생성
- `GET /v1/styles/jobs/{job_id}` - 학습 작업 상태 조회
- `GET /v1/styles` - 스타일 목록 조회
- `GET /v1/styles/{style_id}` - 스타일 상세 조회
- `PATCH /v1/styles/{style_id}` - 스타일 정보 수정
- `DELETE /v1/styles/{style_id}` - 스타일 삭제

## Docker 실행

```bash
docker-compose up -d
```

개발 환경의 전체 스택(FastAPI, PostgreSQL, Redis, MinIO)이 시작됩니다.

## 마이그레이션

### 새 마이그레이션 생성

```bash
alembic revision --autogenerate -m "description"
```

### 마이그레이션 적용

```bash
alembic upgrade head
```

### 마이그레이션 롤백

```bash
alembic downgrade -1
```

## 구현 예정 사항

- [ ] 커뮤니티 엔드포인트 (댓글, 평점)
- [ ] 필터 적용 엔드포인트
- [ ] .cube 파일 내보내기
- [ ] 이미지 처리 작업 (썸네일, 메타데이터)
- [ ] ML 모델 통합
- [ ] Rate limiting 미들웨어
- [ ] API 테스트 작성

## 참고

- [API 명세서](../docs/api_spec.md)
- [백엔드 아키텍처](../docs/backend_architecture.md)
- [데이터베이스 스키마](../docs/db_schema.md)
