# AI_picfilter REST API Specification

## Base URL

```
Production: https://api.picfilter.io/v1
Development: http://localhost:8000/v1
```

## Authentication

JWT 기반 Bearer Token 인증을 사용합니다.

| Header | Value |
|--------|-------|
| `Authorization` | `Bearer <access_token>` |

### 인증 엔드포인트

#### POST /auth/register

회원 가입

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword",
  "username": "displayname"
}
```

**Response (201):**
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "username": "displayname",
  "created_at": "2026-01-15T09:00:00Z"
}
```

#### POST /auth/login

로그인 후 JWT 토큰 발급

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

**Response (200):**
```json
{
  "access_token": "eyJhbGciOi...",
  "refresh_token": "eyJhbGciOi...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

#### POST /auth/refresh

Access Token 갱신

**Request Body:**
```json
{
  "refresh_token": "eyJhbGciOi..."
}
```

**Response (200):**
```json
{
  "access_token": "eyJhbGciOi...",
  "expires_in": 3600
}
```

---

## Images (이미지 관리)

#### POST /images/upload

이미지 업로드를 위한 presigned URL 발급. 클라이언트는 반환된 URL로 직접 S3에 업로드합니다.

**Auth:** Required

**Request Body:**
```json
{
  "filename": "photo.jpg",
  "content_type": "image/jpeg",
  "size_bytes": 5242880
}
```

**Response (200):**
```json
{
  "image_id": "uuid",
  "upload_url": "https://storage.picfilter.io/...",
  "expires_in": 600
}
```

**Validation:**
- `content_type`: `image/jpeg`, `image/png`, `image/tiff`, `image/webp`
- `size_bytes`: 최대 50MB

#### POST /images/{image_id}/confirm

클라이언트가 S3 업로드 완료 후 서버에 확인 요청. 서버는 파일 존재 확인 후 메타데이터를 저장합니다.

**Auth:** Required

**Response (200):**
```json
{
  "image_id": "uuid",
  "status": "confirmed",
  "width": 3840,
  "height": 2160,
  "format": "jpeg",
  "size_bytes": 5242880,
  "thumbnail_url": "https://cdn.picfilter.io/thumbs/...",
  "created_at": "2026-01-15T09:00:00Z"
}
```

#### GET /images

사용자의 업로드된 이미지 목록 조회

**Auth:** Required

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `page` | int | 1 | 페이지 번호 |
| `per_page` | int | 20 | 페이지당 항목 수 (max 100) |
| `sort` | string | `created_at` | 정렬 기준 (`created_at`, `name`) |
| `order` | string | `desc` | 정렬 방향 (`asc`, `desc`) |

**Response (200):**
```json
{
  "items": [
    {
      "image_id": "uuid",
      "filename": "photo.jpg",
      "thumbnail_url": "https://cdn.picfilter.io/thumbs/...",
      "width": 3840,
      "height": 2160,
      "size_bytes": 5242880,
      "created_at": "2026-01-15T09:00:00Z"
    }
  ],
  "total": 42,
  "page": 1,
  "per_page": 20
}
```

#### GET /images/{image_id}

이미지 상세 정보 조회

**Auth:** Required

**Response (200):**
```json
{
  "image_id": "uuid",
  "filename": "photo.jpg",
  "url": "https://cdn.picfilter.io/images/...",
  "thumbnail_url": "https://cdn.picfilter.io/thumbs/...",
  "width": 3840,
  "height": 2160,
  "format": "jpeg",
  "size_bytes": 5242880,
  "created_at": "2026-01-15T09:00:00Z"
}
```

#### DELETE /images/{image_id}

이미지 삭제 (soft delete)

**Auth:** Required

**Response (204):** No Content

---

## Styles (스타일 학습)

#### POST /styles/learn

레퍼런스 이미지들로부터 스타일을 학습하는 비동기 작업을 생성합니다.

**Auth:** Required

**Request Body:**
```json
{
  "name": "Warm Sunset",
  "description": "따뜻한 석양 분위기의 필터",
  "reference_image_ids": ["uuid1", "uuid2", "uuid3"],
  "model_type": "adaptive_3dlut",
  "options": {
    "strength": 1.0,
    "preserve_structure": true
  }
}
```

`model_type` 옵션:
- `adaptive_3dlut` -- Image-Adaptive 3D LUT (기본값, 빠른 추론)
- `nilut` -- Neural Implicit LUT (다중 스타일 블렌딩 지원)
- `dlut` -- Diffusion-based LUT (.cube 파일 직접 생성)

**Response (202):**
```json
{
  "job_id": "uuid",
  "status": "queued",
  "estimated_duration_sec": 120,
  "created_at": "2026-01-15T09:00:00Z"
}
```

#### GET /styles/jobs/{job_id}

학습 작업 상태 조회

**Auth:** Required

**Response (200):**
```json
{
  "job_id": "uuid",
  "status": "processing",
  "progress": 0.45,
  "model_type": "adaptive_3dlut",
  "created_at": "2026-01-15T09:00:00Z",
  "started_at": "2026-01-15T09:00:05Z",
  "completed_at": null,
  "error": null
}
```

`status` 값: `queued`, `processing`, `completed`, `failed`

#### GET /styles

사용자의 학습된 스타일(필터) 목록 조회

**Auth:** Required

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `page` | int | 1 | 페이지 번호 |
| `per_page` | int | 20 | 페이지당 항목 수 |
| `model_type` | string | - | 모델 타입 필터 |
| `is_public` | bool | - | 공개 여부 필터 |

**Response (200):**
```json
{
  "items": [
    {
      "style_id": "uuid",
      "name": "Warm Sunset",
      "description": "따뜻한 석양 분위기의 필터",
      "model_type": "adaptive_3dlut",
      "thumbnail_url": "https://cdn.picfilter.io/styles/...",
      "is_public": false,
      "rating_avg": 4.5,
      "download_count": 0,
      "created_at": "2026-01-15T09:02:00Z"
    }
  ],
  "total": 5,
  "page": 1,
  "per_page": 20
}
```

#### GET /styles/{style_id}

스타일 상세 정보 조회

**Auth:** Required (공개 스타일은 인증 불필요)

**Response (200):**
```json
{
  "style_id": "uuid",
  "name": "Warm Sunset",
  "description": "따뜻한 석양 분위기의 필터",
  "model_type": "adaptive_3dlut",
  "thumbnail_url": "https://cdn.picfilter.io/styles/...",
  "preview_images": [
    {
      "before_url": "https://cdn.picfilter.io/...",
      "after_url": "https://cdn.picfilter.io/..."
    }
  ],
  "is_public": false,
  "owner": {
    "user_id": "uuid",
    "username": "displayname"
  },
  "rating_avg": 4.5,
  "rating_count": 12,
  "download_count": 150,
  "created_at": "2026-01-15T09:02:00Z"
}
```

#### PATCH /styles/{style_id}

스타일 정보 수정

**Auth:** Required (소유자만)

**Request Body:**
```json
{
  "name": "Updated Name",
  "description": "Updated description",
  "is_public": true
}
```

**Response (200):** 수정된 스타일 객체 반환

#### DELETE /styles/{style_id}

스타일 삭제 (soft delete)

**Auth:** Required (소유자만)

**Response (204):** No Content

---

## Filter Application (필터 적용)

#### POST /styles/{style_id}/apply

학습된 스타일을 대상 이미지에 적용하는 비동기 작업을 생성합니다.

**Auth:** Required

**Request Body:**
```json
{
  "target_image_id": "uuid",
  "strength": 0.8,
  "output_format": "jpeg",
  "output_quality": 95
}
```

**Response (202):**
```json
{
  "job_id": "uuid",
  "status": "queued",
  "estimated_duration_sec": 5
}
```

#### GET /styles/apply-jobs/{job_id}

필터 적용 작업 결과 조회

**Auth:** Required

**Response (200):**
```json
{
  "job_id": "uuid",
  "status": "completed",
  "result_image_id": "uuid",
  "result_url": "https://cdn.picfilter.io/results/...",
  "processing_time_ms": 15,
  "created_at": "2026-01-15T09:10:00Z",
  "completed_at": "2026-01-15T09:10:01Z"
}
```

---

## .cube File (LUT 파일 다운로드)

#### POST /styles/{style_id}/export-cube

스타일을 .cube LUT 파일로 내보내는 작업을 생성합니다. D-LUT 모델은 직접 .cube 생성이 가능하며, 다른 모델은 변환 과정이 필요합니다.

**Auth:** Required

**Request Body:**
```json
{
  "lut_size": 33,
  "title": "Warm Sunset LUT"
}
```

`lut_size`: 17, 33, 65 중 선택 (기본값: 33)

**Response (202):**
```json
{
  "job_id": "uuid",
  "status": "queued"
}
```

#### GET /styles/export-jobs/{job_id}

.cube 내보내기 작업 상태 및 다운로드 URL 조회

**Auth:** Required

**Response (200):**
```json
{
  "job_id": "uuid",
  "status": "completed",
  "download_url": "https://storage.picfilter.io/cubes/...",
  "download_expires_in": 3600,
  "file_size_bytes": 1048576,
  "lut_size": 33
}
```

---

## Community (커뮤니티)

#### GET /community/styles

공개된 스타일 탐색

**Auth:** Optional

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `page` | int | 1 | 페이지 번호 |
| `per_page` | int | 20 | 페이지당 항목 수 |
| `sort` | string | `popular` | 정렬: `popular`, `recent`, `rating` |
| `model_type` | string | - | 모델 타입 필터 |
| `q` | string | - | 검색어 (이름, 설명) |

**Response (200):** 스타일 목록 (GET /styles 응답과 동일 형식)

#### POST /styles/{style_id}/comments

스타일에 댓글 작성

**Auth:** Required

**Request Body:**
```json
{
  "content": "이 필터 정말 좋아요!"
}
```

**Response (201):**
```json
{
  "comment_id": "uuid",
  "style_id": "uuid",
  "user": {
    "user_id": "uuid",
    "username": "displayname"
  },
  "content": "이 필터 정말 좋아요!",
  "created_at": "2026-01-15T10:00:00Z"
}
```

#### GET /styles/{style_id}/comments

스타일의 댓글 목록 조회

**Auth:** Optional

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `page` | int | 1 | 페이지 번호 |
| `per_page` | int | 20 | 페이지당 항목 수 |
| `sort` | string | `recent` | 정렬: `recent`, `oldest` |

**Response (200):**
```json
{
  "items": [
    {
      "comment_id": "uuid",
      "user": {
        "user_id": "uuid",
        "username": "displayname"
      },
      "content": "이 필터 정말 좋아요!",
      "created_at": "2026-01-15T10:00:00Z"
    }
  ],
  "total": 3,
  "page": 1,
  "per_page": 20
}
```

#### DELETE /styles/{style_id}/comments/{comment_id}

댓글 삭제

**Auth:** Required (작성자 또는 관리자)

**Response (204):** No Content

#### POST /styles/{style_id}/rate

스타일 평점 등록/수정 (사용자당 1회)

**Auth:** Required

**Request Body:**
```json
{
  "score": 5
}
```

`score`: 1~5 정수

**Response (200):**
```json
{
  "style_id": "uuid",
  "user_score": 5,
  "rating_avg": 4.5,
  "rating_count": 13
}
```

---

## Error Responses

모든 에러는 아래 형식으로 반환됩니다:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Human-readable error description",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format"
      }
    ]
  }
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | 성공 |
| 201 | 리소스 생성됨 |
| 202 | 비동기 작업 수락됨 |
| 204 | 성공 (응답 본문 없음) |
| 400 | 잘못된 요청 (유효성 검증 실패) |
| 401 | 인증 실패 (토큰 없음/만료) |
| 403 | 권한 없음 |
| 404 | 리소스 없음 |
| 409 | 충돌 (중복 리소스) |
| 413 | 파일 크기 초과 |
| 415 | 지원하지 않는 미디어 타입 |
| 422 | 처리 불가능한 엔티티 |
| 429 | 요청 횟수 제한 초과 |
| 500 | 서버 내부 오류 |

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | 입력 유효성 검증 실패 |
| `AUTH_INVALID_CREDENTIALS` | 401 | 잘못된 이메일 또는 비밀번호 |
| `AUTH_TOKEN_EXPIRED` | 401 | 토큰 만료 |
| `FORBIDDEN` | 403 | 리소스 접근 권한 없음 |
| `NOT_FOUND` | 404 | 리소스를 찾을 수 없음 |
| `DUPLICATE_RESOURCE` | 409 | 이미 존재하는 리소스 |
| `FILE_TOO_LARGE` | 413 | 업로드 파일 크기 초과 (50MB) |
| `UNSUPPORTED_FORMAT` | 415 | 지원하지 않는 이미지 형식 |
| `RATE_LIMIT_EXCEEDED` | 429 | API 호출 제한 초과 |
| `JOB_FAILED` | 500 | ML 작업 처리 실패 |
| `INTERNAL_ERROR` | 500 | 서버 내부 오류 |

### Rate Limiting

| Tier | Limit |
|------|-------|
| 일반 API | 60 req/min |
| 이미지 업로드 | 10 req/min |
| 스타일 학습 | 5 req/hour |
| 필터 적용 | 30 req/min |

Rate limit 정보는 응답 헤더에 포함됩니다:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1705312800
```
