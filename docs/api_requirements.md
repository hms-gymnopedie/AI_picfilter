# AI_picfilter Backend API Requirements

This document defines the API endpoints, data formats, real-time communication strategy, error handling, and file transfer flows required by the frontend application.

**Base URL**: `/api/v1`

---

## 1. API Endpoints

### 1.1 Style Learning

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/styles` | Create a new style learning job |
| `GET` | `/styles` | List all user styles (paginated) |
| `GET` | `/styles/{styleId}` | Get style details and status |
| `DELETE` | `/styles/{styleId}` | Delete a style and its associated models |
| `GET` | `/styles/{styleId}/progress` | **SSE stream** -- real-time learning progress |
| `POST` | `/styles/{styleId}/cancel` | Cancel an in-progress learning job |

### 1.2 Image Upload

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/uploads/presign` | Get presigned URL(s) for direct-to-storage upload |
| `POST` | `/uploads/confirm` | Confirm upload completion, trigger validation |

### 1.3 Filter Application

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/styles/{styleId}/lut` | Download LUT data for client-side WebGL preview |
| `POST` | `/styles/{styleId}/apply` | Server-side filter application (WebGL fallback) |

### 1.4 Export

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/styles/{styleId}/export` | Generate export file (.cube, .png hald, .json weights) |
| `GET` | `/exports/{exportId}` | Check export generation status |
| `GET` | `/exports/{exportId}/download` | Download the generated export file |

### 1.5 Filter Library

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/filters` | List saved filters (paginated, searchable) |
| `POST` | `/filters` | Save a filter preset (style + blend config) |
| `GET` | `/filters/{filterId}` | Get filter details |
| `PATCH` | `/filters/{filterId}` | Update filter name or blend configuration |
| `DELETE` | `/filters/{filterId}` | Delete a saved filter |

---

## 2. Request / Response Data Formats

### 2.1 Create Style Learning Job

**`POST /styles`**

Request:
```json
{
  "name": "Warm Sunset",
  "method": "nilut",
  "imageKeys": [
    "uploads/abc123-ref1.jpg",
    "uploads/abc123-ref2.jpg",
    "uploads/abc123-ref3.jpg"
  ]
}
```

- `method`: `"adaptive_3dlut"` | `"nilut"` | `"dlut"`
- `imageKeys`: Storage keys returned from the presigned upload flow

Response (`201 Created`):
```json
{
  "id": "style_a1b2c3d4",
  "name": "Warm Sunset",
  "method": "nilut",
  "status": "queued",
  "imageCount": 3,
  "createdAt": "2026-02-18T10:30:00Z"
}
```

### 2.2 List Styles

**`GET /styles?page=1&limit=20&sort=createdAt:desc`**

Response:
```json
{
  "items": [
    {
      "id": "style_a1b2c3d4",
      "name": "Warm Sunset",
      "method": "nilut",
      "status": "completed",
      "thumbnailUrl": "https://cdn.example.com/thumbs/style_a1b2c3d4.jpg",
      "styleCount": 3,
      "createdAt": "2026-02-18T10:30:00Z",
      "completedAt": "2026-02-18T10:32:15Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 42,
    "totalPages": 3
  }
}
```

### 2.3 Get Style Details

**`GET /styles/{styleId}`**

Response:
```json
{
  "id": "style_a1b2c3d4",
  "name": "Warm Sunset",
  "method": "nilut",
  "status": "completed",
  "imageCount": 3,
  "referenceImages": [
    {
      "key": "uploads/abc123-ref1.jpg",
      "thumbnailUrl": "https://cdn.example.com/thumbs/abc123-ref1.jpg",
      "width": 4000,
      "height": 2667
    }
  ],
  "metrics": {
    "deltaE": 0.83,
    "epochs": 200,
    "trainingTimeSeconds": 135
  },
  "styleFeatures": {
    "warmTones": "high",
    "contrast": "medium",
    "saturation": "high",
    "shadowDetail": "low"
  },
  "createdAt": "2026-02-18T10:30:00Z",
  "completedAt": "2026-02-18T10:32:15Z"
}
```

### 2.4 Presigned Upload

**`POST /uploads/presign`**

Request:
```json
{
  "files": [
    { "filename": "sunset_ref1.jpg", "contentType": "image/jpeg", "sizeBytes": 4521890 },
    { "filename": "sunset_ref2.png", "contentType": "image/png", "sizeBytes": 8932100 }
  ]
}
```

Response:
```json
{
  "uploads": [
    {
      "key": "uploads/a1b2c3-sunset_ref1.jpg",
      "uploadUrl": "https://storage.example.com/uploads/a1b2c3-sunset_ref1.jpg?X-Amz-Signature=...",
      "expiresAt": "2026-02-18T11:00:00Z"
    },
    {
      "key": "uploads/d4e5f6-sunset_ref2.png",
      "uploadUrl": "https://storage.example.com/uploads/d4e5f6-sunset_ref2.png?X-Amz-Signature=...",
      "expiresAt": "2026-02-18T11:00:00Z"
    }
  ]
}
```

**Upload Flow**:
1. Frontend calls `POST /uploads/presign` with file metadata
2. Frontend uploads files directly to storage via presigned URLs (PUT)
3. Frontend calls `POST /uploads/confirm` with the keys
4. Backend validates images (format, dimensions, corruption check)

**`POST /uploads/confirm`**

Request:
```json
{
  "keys": [
    "uploads/a1b2c3-sunset_ref1.jpg",
    "uploads/d4e5f6-sunset_ref2.png"
  ]
}
```

Response:
```json
{
  "confirmed": [
    {
      "key": "uploads/a1b2c3-sunset_ref1.jpg",
      "valid": true,
      "width": 4000,
      "height": 2667,
      "thumbnailUrl": "https://cdn.example.com/thumbs/a1b2c3-sunset_ref1.jpg"
    },
    {
      "key": "uploads/d4e5f6-sunset_ref2.png",
      "valid": true,
      "width": 3840,
      "height": 2160,
      "thumbnailUrl": "https://cdn.example.com/thumbs/d4e5f6-sunset_ref2.png"
    }
  ]
}
```

### 2.5 Download LUT Data (for WebGL preview)

**`GET /styles/{styleId}/lut?format=texture`**

Query parameters:
- `format`: `"texture"` (binary 3D texture for WebGL) | `"json"` (flattened array)
- `size`: `17` | `33` | `65` (LUT grid dimension, default `33`)

Response for `format=texture`:
- `Content-Type: application/octet-stream`
- Binary data: flattened RGB float32 array, dimensions `size x size x size x 3`
- Header `X-LUT-Size: 33` indicating grid dimension

Response for `format=json`:
```json
{
  "size": 33,
  "data": [0.0, 0.0, 0.0, 0.003, 0.0, 0.0, ...],
  "domain": { "min": [0, 0, 0], "max": [1, 1, 1] }
}
```

For NILUT multi-style, additional parameter:
- `styleWeights`: `"0.55,0.30,0.15"` (comma-separated blend weights)

### 2.6 Server-side Filter Application (Fallback)

**`POST /styles/{styleId}/apply`**

Request (`multipart/form-data`):
- `image`: Target image file
- `intensity`: `0.0` - `1.0` (default `1.0`)
- `styleWeights`: JSON string of blend weights (NILUT only), e.g. `"[0.55, 0.30, 0.15]"`

Response:
```json
{
  "resultUrl": "https://cdn.example.com/results/r1e2s3u4l5t.jpg",
  "expiresAt": "2026-02-18T12:00:00Z"
}
```

### 2.7 Export Filter

**`POST /styles/{styleId}/export`**

Request:
```json
{
  "format": "cube",
  "lutSize": 33,
  "styleWeights": [0.55, 0.30, 0.15]
}
```

- `format`: `"cube"` | `"hald_png"` | `"nilut_json"`
- `lutSize`: `17` | `33` | `65` (for cube and hald formats)

Response (`202 Accepted`):
```json
{
  "exportId": "export_x1y2z3",
  "status": "generating",
  "estimatedSeconds": 5
}
```

**`GET /exports/{exportId}`**

Response:
```json
{
  "exportId": "export_x1y2z3",
  "status": "ready",
  "filename": "Warm_Sunset_33x33x33.cube",
  "sizeBytes": 1134592,
  "downloadUrl": "/api/v1/exports/export_x1y2z3/download",
  "expiresAt": "2026-02-19T10:30:00Z"
}
```

- `status`: `"generating"` | `"ready"` | `"failed"`

**`GET /exports/{exportId}/download`**

Response:
- `Content-Type: application/octet-stream`
- `Content-Disposition: attachment; filename="Warm_Sunset_33x33x33.cube"`
- Binary file data

### 2.8 Save Filter Preset

**`POST /filters`**

Request:
```json
{
  "name": "My Warm Blend",
  "styleId": "style_a1b2c3d4",
  "intensity": 0.7,
  "styleWeights": [0.55, 0.30, 0.15]
}
```

Response (`201 Created`):
```json
{
  "id": "filter_m1n2o3",
  "name": "My Warm Blend",
  "styleId": "style_a1b2c3d4",
  "intensity": 0.7,
  "styleWeights": [0.55, 0.30, 0.15],
  "thumbnailUrl": "https://cdn.example.com/thumbs/filter_m1n2o3.jpg",
  "createdAt": "2026-02-18T11:00:00Z"
}
```

---

## 3. Real-time Processing Strategy

### 3.1 Chosen Approach: Server-Sent Events (SSE)

**Endpoint**: `GET /styles/{styleId}/progress`

**Rationale**:
- Style learning progress is unidirectional (server to client)
- SSE has native browser support via `EventSource` API
- Simpler infrastructure than WebSocket (no upgrade handshake, works through standard HTTP proxies/CDNs)
- Automatic reconnection built into the browser `EventSource` API
- Sufficient for the progress update frequency needed (1-2 updates per second)

**Event Stream Format**:

```
event: progress
data: {"phase":"preprocessing","progress":0.10,"message":"Preprocessing 5 reference images...","elapsed":3}

event: progress
data: {"phase":"training","progress":0.45,"epoch":90,"totalEpochs":200,"deltaE":1.42,"message":"Training NILUT model (epoch 90/200)...","elapsed":58}

event: progress
data: {"phase":"training","progress":0.72,"epoch":144,"totalEpochs":200,"deltaE":0.91,"message":"Training NILUT model (epoch 144/200)...","elapsed":95}

event: metrics
data: {"colorDistribution":{"r":[12,45,78,...],"g":[8,32,65,...],"b":[5,22,48,...]},"dominantColors":["#E8A040","#D4783C","#2B1810"]}

event: features
data: {"warmTones":"high","contrast":"medium","saturation":"high","shadowDetail":"low"}

event: complete
data: {"styleId":"style_a1b2c3d4","deltaE":0.83,"trainingTimeSeconds":135}

event: error
data: {"code":"TRAINING_DIVERGED","message":"Model training failed to converge. Try with more reference images."}
```

**Event Types**:

| Event | Frequency | Purpose |
|-------|-----------|---------|
| `progress` | Every 1-2 seconds | Overall progress %, phase, epoch, deltaE |
| `metrics` | Once (after preprocessing) | Color distribution histogram data |
| `features` | Once (after analysis) | Detected style feature summary |
| `complete` | Once | Final results, signals stream end |
| `error` | On failure | Error details, signals stream end |

**Reconnection**:
- Server sets `retry: 3000` (3 second reconnect delay)
- Each event includes an `id` field for `Last-Event-ID` resumption
- Frontend shows "Reconnecting..." indicator during gaps
- If reconnection fails 5 times, frontend falls back to polling `GET /styles/{styleId}` every 5 seconds

### 3.2 Why Not WebSocket or Polling?

| Approach | Pros | Cons |
|----------|------|------|
| **SSE (chosen)** | Native browser API, auto-reconnect, simple server impl | Unidirectional only (sufficient for this use case) |
| WebSocket | Bidirectional | Over-engineered for progress updates; complex reconnect handling; proxy/firewall issues |
| Polling | Simplest implementation | Wasteful network traffic; delayed updates; poor UX for progress bars |

---

## 4. Error Handling

### 4.1 Standard Error Response Format

All error responses follow a consistent shape:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Human-readable description of what went wrong",
    "details": [
      { "field": "imageKeys", "issue": "At least 1 reference image is required" }
    ]
  }
}
```

### 4.2 HTTP Status Codes

| Status | Usage |
|--------|-------|
| `400` | Validation error (bad input) |
| `401` | Authentication required |
| `403` | Insufficient permissions |
| `404` | Resource not found |
| `409` | Conflict (e.g., style already being trained) |
| `413` | File too large |
| `415` | Unsupported file type |
| `422` | Semantically invalid (e.g., cannot blend with adaptive_3dlut method) |
| `429` | Rate limited |
| `500` | Internal server error |
| `503` | Service temporarily unavailable (GPU workers busy) |

### 4.3 Error Cases by Domain

**Image Upload**:

| Code | Status | Cause | Frontend Action |
|------|--------|-------|-----------------|
| `FILE_TOO_LARGE` | 413 | Image exceeds 50MB limit | Show size limit in toast, keep other files |
| `UNSUPPORTED_FORMAT` | 415 | Not JPG/PNG/TIFF | Show supported formats, reject file |
| `IMAGE_CORRUPTED` | 422 | Cannot decode image data | Toast with "File may be corrupted" |
| `UPLOAD_EXPIRED` | 410 | Presigned URL expired | Re-request presigned URL automatically |
| `TOO_MANY_FILES` | 400 | Exceeded 20 image limit | Show count limit |

**Style Learning**:

| Code | Status | Cause | Frontend Action |
|------|--------|-------|-----------------|
| `TRAINING_DIVERGED` | 500 | Model failed to converge | Suggest more/different reference images |
| `INSUFFICIENT_IMAGES` | 400 | Fewer than required images for method | Show minimum count |
| `GPU_UNAVAILABLE` | 503 | No GPU workers available | Show queue position, auto-retry |
| `TRAINING_TIMEOUT` | 504 | Exceeded max training time | Suggest simpler method or fewer images |
| `STYLE_NOT_FOUND` | 404 | Invalid style ID | Redirect to dashboard |
| `ALREADY_TRAINING` | 409 | Style is already being trained | Show current progress |

**Filter Application**:

| Code | Status | Cause | Frontend Action |
|------|--------|-------|-----------------|
| `STYLE_NOT_READY` | 422 | Style learning not yet complete | Show learning status |
| `BLEND_NOT_SUPPORTED` | 422 | Method doesn't support blending | Disable blend UI, show method info |
| `INVALID_WEIGHTS` | 400 | Blend weights don't sum to ~1.0 | Normalize weights client-side |
| `LUT_GENERATION_FAILED` | 500 | Failed to generate LUT from model | Retry or suggest re-training |

**Export**:

| Code | Status | Cause | Frontend Action |
|------|--------|-------|-----------------|
| `EXPORT_FAILED` | 500 | Export generation error | Retry button |
| `EXPORT_EXPIRED` | 410 | Download link expired | Re-trigger export |
| `UNSUPPORTED_EXPORT` | 422 | Method doesn't support format | Disable unavailable format options |

### 4.4 Rate Limiting

Response header `X-RateLimit-Remaining` included on all responses. When `429` is returned:

```json
{
  "error": {
    "code": "RATE_LIMITED",
    "message": "Too many requests",
    "retryAfter": 30
  }
}
```

Frontend uses `retryAfter` value to show countdown before retry.

---

## 5. File Upload / Download Flows

### 5.1 Image Upload Flow (Presigned URL Pattern)

```
Frontend                          Backend                        Storage (S3/GCS)
   |                                |                                |
   |-- POST /uploads/presign ------>|                                |
   |   {files: [{name, type, size}]}|                                |
   |                                |-- Generate presigned URLs ---->|
   |<-- {uploads: [{key, url}]} ----|                                |
   |                                                                 |
   |-- PUT (direct upload) ---------------------------------------->|
   |   Content-Type: image/jpeg                                      |
   |   Body: <raw file bytes>                                        |
   |<-- 200 OK ---------------------------------------------------- |
   |                                                                 |
   |-- POST /uploads/confirm ------>|                                |
   |   {keys: [...]}                |-- Validate images ----------->|
   |                                |<-- Image metadata ------------|
   |<-- {confirmed: [{valid, w, h}]}|                                |
```

**Why presigned URLs?**
- Large files (up to 50MB) bypass the backend server entirely
- Reduces backend memory/CPU load
- Enables parallel multi-file upload from the browser
- Progress tracking via `XMLHttpRequest` or `fetch` with `ReadableStream`

**Frontend upload implementation notes**:
- Use `XMLHttpRequest` (not `fetch`) for upload progress events
- Show per-file progress bars during upload
- Generate client-side thumbnails via `<canvas>` immediately (before upload completes)
- Retry failed uploads up to 3 times with exponential backoff

### 5.2 LUT Data Download (for WebGL Preview)

```
Frontend                          Backend
   |                                |
   |-- GET /styles/{id}/lut ------->|
   |   ?format=texture&size=33      |
   |                                |-- Generate/cache LUT data
   |<-- Binary (float32 array) -----|
   |   Content-Type: application/octet-stream
   |   X-LUT-Size: 33
   |   Cache-Control: public, max-age=86400
   |
   |-- Load as WebGL 3D texture
   |-- Render preview
```

**Caching strategy**:
- LUT data is immutable once generated -- cache aggressively
- Backend returns `Cache-Control: public, max-age=86400` and `ETag`
- Frontend caches in `IndexedDB` keyed by `styleId + size + weights`
- On subsequent visits, check `ETag` via conditional `GET` (If-None-Match)

### 5.3 Export File Download

```
Frontend                          Backend
   |                                |
   |-- POST /styles/{id}/export --->|
   |   {format, lutSize, weights}   |
   |                                |-- Queue export job
   |<-- 202 {exportId, status} -----|
   |                                |
   |-- GET /exports/{id} ---------->|  (poll every 2s)
   |<-- {status: "generating"} -----|
   |                                |
   |-- GET /exports/{id} ---------->|
   |<-- {status: "ready", url} -----|
   |                                |
   |-- GET /exports/{id}/download ->|
   |<-- Binary file ----------------|
   |   Content-Disposition: attachment; filename="..."
   |
   |-- Trigger browser download via <a download>
```

**Export file size estimates**:

| Format | LUT Size | Approximate File Size |
|--------|----------|----------------------|
| `.cube` | 17x17x17 | ~150 KB |
| `.cube` | 33x33x33 | ~1.1 MB |
| `.cube` | 65x65x65 | ~8.3 MB |
| `.png` (Hald) | 33 equivalent | ~3.5 MB |
| `.json` (NILUT weights) | N/A | <250 KB |

---

## 6. Authentication & Headers

### 6.1 Required Headers

| Header | Value | Notes |
|--------|-------|-------|
| `Authorization` | `Bearer <token>` | JWT token from auth provider |
| `Content-Type` | `application/json` | For JSON request bodies |
| `Accept-Language` | `ko` or `en` | For localized error messages |

### 6.2 CORS

Backend must allow:
- Origins: frontend deployment domain(s)
- Methods: `GET, POST, PATCH, DELETE, OPTIONS`
- Headers: `Authorization, Content-Type, Accept-Language, X-Request-ID`
- Exposed headers: `X-RateLimit-Remaining, X-LUT-Size, ETag`

---

## 7. API Versioning

- Version is included in the URL path: `/api/v1/...`
- Breaking changes require a new version (`v2`)
- Non-breaking additions (new optional fields) are added to existing version
- Deprecated endpoints return `Sunset` header with removal date
