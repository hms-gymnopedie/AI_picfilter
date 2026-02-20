# Database Schema

PostgreSQL 16 기반 스키마 설계입니다. 모든 테이블은 `public` 스키마에 생성됩니다.

## Entity Relationship Diagram

```
┌──────────┐       ┌──────────┐       ┌──────────────┐
│  users   │───1:N─│  images  │       │    styles    │
└──────────┘       └──────────┘       └──────┬───────┘
     │                                       │
     │ 1:N                            1:N    │    1:N
     │         ┌─────────────────────────────┼──────────────┐
     │         │                             │              │
     ▼         ▼                             ▼              ▼
┌──────────────────┐              ┌──────────────┐  ┌────────────┐
│       jobs       │              │   comments   │  │  ratings   │
│                  │              └──────────────┘  └────────────┘
│ (learn / apply / │
│  export)         │
└──────────────────┘

users 1:N → images        (사용자가 업로드한 이미지)
users 1:N → styles        (사용자가 생성한 스타일)
users 1:N → jobs          (사용자의 작업 이력)
users 1:N → comments      (사용자가 작성한 댓글)
users 1:N → ratings       (사용자가 남긴 평점)
styles 1:N → comments     (스타일에 달린 댓글)
styles 1:N → ratings      (스타일에 대한 평점)
styles 1:N → jobs         (스타일 관련 작업)
images 1:N → jobs         (이미지 관련 작업)
```

## Tables

### users

사용자 계정 정보

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `UUID` | PK, DEFAULT gen_random_uuid() | 사용자 고유 ID |
| `email` | `VARCHAR(255)` | UNIQUE, NOT NULL | 이메일 (로그인 ID) |
| `username` | `VARCHAR(50)` | UNIQUE, NOT NULL | 표시 이름 |
| `password_hash` | `VARCHAR(255)` | NOT NULL | bcrypt 해시 |
| `is_active` | `BOOLEAN` | DEFAULT TRUE | 계정 활성 상태 |
| `is_admin` | `BOOLEAN` | DEFAULT FALSE | 관리자 여부 |
| `created_at` | `TIMESTAMPTZ` | DEFAULT NOW() | 가입일 |
| `updated_at` | `TIMESTAMPTZ` | DEFAULT NOW() | 수정일 |

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users (email);
```

### images

업로드된 이미지 메타데이터. 실제 파일은 S3/MinIO에 저장됩니다.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `UUID` | PK, DEFAULT gen_random_uuid() | 이미지 고유 ID |
| `user_id` | `UUID` | FK → users.id, NOT NULL | 업로드한 사용자 |
| `filename` | `VARCHAR(255)` | NOT NULL | 원본 파일명 |
| `storage_key` | `VARCHAR(512)` | NOT NULL | S3 object key |
| `thumbnail_key` | `VARCHAR(512)` | | 썸네일 S3 key |
| `content_type` | `VARCHAR(50)` | NOT NULL | MIME type |
| `size_bytes` | `BIGINT` | NOT NULL | 파일 크기 |
| `width` | `INTEGER` | | 이미지 너비 (px) |
| `height` | `INTEGER` | | 이미지 높이 (px) |
| `format` | `VARCHAR(10)` | | 이미지 포맷 (jpeg, png 등) |
| `status` | `VARCHAR(20)` | DEFAULT 'pending' | `pending`, `confirmed`, `deleted` |
| `created_at` | `TIMESTAMPTZ` | DEFAULT NOW() | 업로드 시각 |
| `deleted_at` | `TIMESTAMPTZ` | | soft delete 시각 |

```sql
CREATE TABLE images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    storage_key VARCHAR(512) NOT NULL,
    thumbnail_key VARCHAR(512),
    content_type VARCHAR(50) NOT NULL,
    size_bytes BIGINT NOT NULL,
    width INTEGER,
    height INTEGER,
    format VARCHAR(10),
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);

CREATE INDEX idx_images_user_id ON images (user_id);
CREATE INDEX idx_images_status ON images (status) WHERE status != 'deleted';
```

### styles

학습된 스타일(필터) 정보. 모델 가중치 파일은 S3에 저장됩니다.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `UUID` | PK, DEFAULT gen_random_uuid() | 스타일 고유 ID |
| `user_id` | `UUID` | FK → users.id, NOT NULL | 생성한 사용자 |
| `name` | `VARCHAR(100)` | NOT NULL | 스타일 이름 |
| `description` | `TEXT` | | 스타일 설명 |
| `model_type` | `VARCHAR(30)` | NOT NULL | `adaptive_3dlut`, `nilut`, `dlut` |
| `model_key` | `VARCHAR(512)` | | 학습된 모델 가중치 S3 key |
| `model_size_bytes` | `INTEGER` | | 모델 파일 크기 |
| `preview_key` | `VARCHAR(512)` | | 미리보기 이미지 S3 key |
| `is_public` | `BOOLEAN` | DEFAULT FALSE | 커뮤니티 공개 여부 |
| `rating_sum` | `INTEGER` | DEFAULT 0 | 평점 합계 (평균 계산용) |
| `rating_count` | `INTEGER` | DEFAULT 0 | 평점 수 |
| `download_count` | `INTEGER` | DEFAULT 0 | .cube 다운로드 횟수 |
| `status` | `VARCHAR(20)` | DEFAULT 'active' | `active`, `deleted` |
| `created_at` | `TIMESTAMPTZ` | DEFAULT NOW() | 생성일 |
| `updated_at` | `TIMESTAMPTZ` | DEFAULT NOW() | 수정일 |
| `deleted_at` | `TIMESTAMPTZ` | | soft delete 시각 |

```sql
CREATE TABLE styles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    model_type VARCHAR(30) NOT NULL,
    model_key VARCHAR(512),
    model_size_bytes INTEGER,
    preview_key VARCHAR(512),
    is_public BOOLEAN DEFAULT FALSE,
    rating_sum INTEGER DEFAULT 0,
    rating_count INTEGER DEFAULT 0,
    download_count INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);

CREATE INDEX idx_styles_user_id ON styles (user_id);
CREATE INDEX idx_styles_public ON styles (is_public, created_at DESC) WHERE status = 'active';
CREATE INDEX idx_styles_model_type ON styles (model_type) WHERE status = 'active';
```

### style_reference_images

스타일 학습에 사용된 레퍼런스 이미지 연결 테이블

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `style_id` | `UUID` | FK → styles.id, NOT NULL | 스타일 ID |
| `image_id` | `UUID` | FK → images.id, NOT NULL | 레퍼런스 이미지 ID |
| `created_at` | `TIMESTAMPTZ` | DEFAULT NOW() | 연결 생성일 |

```sql
CREATE TABLE style_reference_images (
    style_id UUID NOT NULL REFERENCES styles(id) ON DELETE CASCADE,
    image_id UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (style_id, image_id)
);
```

### jobs

비동기 작업 이력 (스타일 학습, 필터 적용, .cube 내보내기)

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `UUID` | PK, DEFAULT gen_random_uuid() | 작업 고유 ID |
| `user_id` | `UUID` | FK → users.id, NOT NULL | 요청한 사용자 |
| `job_type` | `VARCHAR(30)` | NOT NULL | `style_learn`, `filter_apply`, `cube_export` |
| `status` | `VARCHAR(20)` | DEFAULT 'queued' | `queued`, `processing`, `completed`, `failed` |
| `progress` | `REAL` | DEFAULT 0.0 | 진행률 (0.0 ~ 1.0) |
| `style_id` | `UUID` | FK → styles.id | 관련 스타일 (학습 완료 후 설정) |
| `input_image_id` | `UUID` | FK → images.id | 입력 이미지 (filter_apply) |
| `result_image_id` | `UUID` | FK → images.id | 결과 이미지 (filter_apply) |
| `result_key` | `VARCHAR(512)` | | 결과 파일 S3 key (.cube 등) |
| `params` | `JSONB` | | 작업 파라미터 (strength, lut_size 등) |
| `error_message` | `TEXT` | | 실패 시 에러 메시지 |
| `processing_time_ms` | `INTEGER` | | 실제 처리 시간 (ms) |
| `celery_task_id` | `VARCHAR(255)` | | Celery task ID (추적용) |
| `created_at` | `TIMESTAMPTZ` | DEFAULT NOW() | 작업 생성 시각 |
| `started_at` | `TIMESTAMPTZ` | | 처리 시작 시각 |
| `completed_at` | `TIMESTAMPTZ` | | 처리 완료 시각 |

```sql
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    job_type VARCHAR(30) NOT NULL,
    status VARCHAR(20) DEFAULT 'queued',
    progress REAL DEFAULT 0.0,
    style_id UUID REFERENCES styles(id) ON DELETE SET NULL,
    input_image_id UUID REFERENCES images(id) ON DELETE SET NULL,
    result_image_id UUID REFERENCES images(id) ON DELETE SET NULL,
    result_key VARCHAR(512),
    params JSONB,
    error_message TEXT,
    processing_time_ms INTEGER,
    celery_task_id VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_jobs_user_id ON jobs (user_id, created_at DESC);
CREATE INDEX idx_jobs_status ON jobs (status) WHERE status IN ('queued', 'processing');
CREATE INDEX idx_jobs_style_id ON jobs (style_id);
CREATE INDEX idx_jobs_celery_task_id ON jobs (celery_task_id);
```

### comments

스타일에 대한 사용자 댓글

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `UUID` | PK, DEFAULT gen_random_uuid() | 댓글 고유 ID |
| `style_id` | `UUID` | FK → styles.id, NOT NULL | 대상 스타일 |
| `user_id` | `UUID` | FK → users.id, NOT NULL | 작성자 |
| `content` | `TEXT` | NOT NULL, CHECK length > 0 | 댓글 내용 |
| `is_deleted` | `BOOLEAN` | DEFAULT FALSE | soft delete |
| `created_at` | `TIMESTAMPTZ` | DEFAULT NOW() | 작성 시각 |

```sql
CREATE TABLE comments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    style_id UUID NOT NULL REFERENCES styles(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL CHECK (length(content) > 0),
    is_deleted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_comments_style_id ON comments (style_id, created_at DESC) WHERE NOT is_deleted;
```

### ratings

스타일에 대한 사용자 평점 (사용자당 스타일당 1개)

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `style_id` | `UUID` | FK → styles.id, NOT NULL | 대상 스타일 |
| `user_id` | `UUID` | FK → users.id, NOT NULL | 평가자 |
| `score` | `SMALLINT` | NOT NULL, CHECK 1~5 | 평점 |
| `created_at` | `TIMESTAMPTZ` | DEFAULT NOW() | 최초 평가 시각 |
| `updated_at` | `TIMESTAMPTZ` | DEFAULT NOW() | 수정 시각 |

```sql
CREATE TABLE ratings (
    style_id UUID NOT NULL REFERENCES styles(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    score SMALLINT NOT NULL CHECK (score >= 1 AND score <= 5),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (style_id, user_id)
);
```

### refresh_tokens

JWT refresh token 관리

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `UUID` | PK, DEFAULT gen_random_uuid() | 토큰 ID |
| `user_id` | `UUID` | FK → users.id, NOT NULL | 사용자 |
| `token_hash` | `VARCHAR(255)` | UNIQUE, NOT NULL | 토큰 SHA256 해시 |
| `expires_at` | `TIMESTAMPTZ` | NOT NULL | 만료 시각 |
| `created_at` | `TIMESTAMPTZ` | DEFAULT NOW() | 발급 시각 |
| `revoked_at` | `TIMESTAMPTZ` | | 폐기 시각 |

```sql
CREATE TABLE refresh_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    revoked_at TIMESTAMPTZ
);

CREATE INDEX idx_refresh_tokens_user_id ON refresh_tokens (user_id);
CREATE INDEX idx_refresh_tokens_expires ON refresh_tokens (expires_at) WHERE revoked_at IS NULL;
```

---

## Rating Aggregation Strategy

`styles` 테이블에 `rating_sum`과 `rating_count`를 비정규화하여 저장합니다. 평균은 `rating_sum / rating_count`로 계산합니다.

평점 등록/수정 시 트리거로 집계를 갱신합니다:

```sql
CREATE OR REPLACE FUNCTION update_style_rating()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE styles
        SET rating_sum = rating_sum + NEW.score,
            rating_count = rating_count + 1
        WHERE id = NEW.style_id;
    ELSIF TG_OP = 'UPDATE' THEN
        UPDATE styles
        SET rating_sum = rating_sum - OLD.score + NEW.score
        WHERE id = NEW.style_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE styles
        SET rating_sum = rating_sum - OLD.score,
            rating_count = rating_count - 1
        WHERE id = OLD.style_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_ratings_update
AFTER INSERT OR UPDATE OR DELETE ON ratings
FOR EACH ROW EXECUTE FUNCTION update_style_rating();
```

## Index Design Notes

| Index | Purpose |
|-------|---------|
| `idx_users_email` | 로그인 시 이메일 조회 |
| `idx_images_user_id` | 사용자별 이미지 목록 조회 |
| `idx_images_status` | 활성 이미지만 필터링 (partial index) |
| `idx_styles_public` | 커뮤니티 탐색: 공개 스타일을 최신순 정렬 |
| `idx_styles_model_type` | 모델 타입별 필터링 |
| `idx_jobs_user_id` | 사용자 작업 이력 조회 |
| `idx_jobs_status` | 대기/처리 중 작업 모니터링 (partial index) |
| `idx_jobs_celery_task_id` | Celery task 결과 콜백 시 조회 |
| `idx_comments_style_id` | 스타일별 댓글 목록 (partial index, 삭제 제외) |
| `idx_refresh_tokens_expires` | 만료 토큰 정리 배치 (partial index) |

**Partial index 활용:** 삭제되거나 비활성 상태의 행을 인덱스에서 제외하여 인덱스 크기와 조회 성능을 최적화합니다.

## Migration Strategy

Alembic을 사용하여 스키마 변경을 관리합니다:

```bash
# 새 마이그레이션 생성
alembic revision --autogenerate -m "description"

# 마이그레이션 적용
alembic upgrade head

# 롤백
alembic downgrade -1
```

모든 스키마 변경은 마이그레이션 파일로 관리되며, 직접 DDL 실행은 금지합니다.
