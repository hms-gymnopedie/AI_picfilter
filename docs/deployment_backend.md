# Backend 배포 가이드

AI_picfilter 백엔드를 프로덕션 환경에 배포하는 방법을 설명합니다.

## 목차

- [빠른 시작](#빠른-시작)
- [환경 설정](#환경-설정)
- [데이터베이스 초기화](#데이터베이스-초기화)
- [로그 확인](#로그-확인)
- [헬스체크](#헬스체크)
- [모니터링](#모니터링)
- [트러블슈팅](#트러블슈팅)

## 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/yourusername/AI_picfilter.git
cd AI_picfilter
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# 편집기로 열어서 필수 값 수정
# - JWT_SECRET_KEY
# - POSTGRES_PASSWORD
# - S3_SECRET_KEY
nano .env
```

### 3. Docker Compose로 실행

```bash
# 프로덕션 환경
docker-compose -f docker-compose.prod.yml up -d

# 개발 환경
docker-compose up -d
```

## 환경 설정

### 필수 환경변수

| 변수 | 설명 | 예시 |
|------|------|------|
| `ENVIRONMENT` | 환경 타입 | `production` |
| `DATABASE_URL` | PostgreSQL 연결 | `postgresql+asyncpg://...` |
| `REDIS_URL` | Redis 연결 | `redis://redis:6379/0` |
| `JWT_SECRET_KEY` | JWT 서명 키 | (보안상 공개 금지) |
| `S3_ENDPOINT` | S3/MinIO 엔드포인트 | `https://s3.example.com` |
| `S3_ACCESS_KEY` | S3 액세스 키 | `minioadmin` |
| `S3_SECRET_KEY` | S3 비밀 키 | (보안상 공개 금지) |
| `S3_BUCKET` | S3 버킷 이름 | `picfilter-storage` |

### 선택적 환경변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `DEBUG` | 디버그 모드 | `false` |
| `CORS_ORIGINS` | CORS 허용 오리진 | `http://localhost:3000` |
| `MAX_UPLOAD_SIZE_MB` | 최대 업로드 크기 | `50` |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | 액세스 토큰 유효기간 | `60` |
| `JWT_REFRESH_TOKEN_EXPIRE_DAYS` | 리프레시 토큰 유효기간 | `7` |

### JWT 비밀 키 생성

보안상 강력한 무작위 키를 사용해야 합니다:

```bash
python3 << 'EOF'
import secrets
print(secrets.token_urlsafe(32))
EOF
```

생성된 값을 `.env` 파일의 `JWT_SECRET_KEY`에 복사합니다.

## 데이터베이스 초기화

### 1. 마이그레이션 실행

앱 서비스가 시작될 때 자동으로 실행됩니다:

```bash
# 수동으로 실행하려면:
docker-compose -f docker-compose.prod.yml exec app \
  alembic upgrade head
```

### 2. 마이그레이션 상태 확인

```bash
docker-compose -f docker-compose.prod.yml exec app \
  alembic current
```

### 3. 마이그레이션 이력 보기

```bash
docker-compose -f docker-compose.prod.yml exec app \
  alembic history
```

### 4. 마이그레이션 롤백 (위험!)

```bash
# 이전 버전으로 롤백
docker-compose -f docker-compose.prod.yml exec app \
  alembic downgrade -1
```

## 로그 확인

### 실시간 로그 보기

```bash
# 모든 서비스 로그
docker-compose -f docker-compose.prod.yml logs -f

# 특정 서비스만
docker-compose -f docker-compose.prod.yml logs -f app
docker-compose -f docker-compose.prod.yml logs -f ml-worker
docker-compose -f docker-compose.prod.yml logs -f postgres
```

### 로그 필터링

```bash
# FastAPI 오류만
docker-compose -f docker-compose.prod.yml logs app | grep -i error

# Celery 작업 진행 상황
docker-compose -f docker-compose.prod.yml logs ml-worker | grep -i "task"

# 마지막 100줄만
docker-compose -f docker-compose.prod.yml logs --tail=100
```

## 헬스체크

### 기본 헬스체크

```bash
curl http://localhost/v1/health
```

응답:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "timestamp": "2026-02-20T10:30:00.000000"
}
```

### 준비 상태 확인 (Readiness Check)

```bash
curl http://localhost/v1/health/ready
```

응답:
```json
{
  "status": "ready",
  "checks": {
    "db": "ok",
    "redis": "ok",
    "storage": "ok"
  },
  "timestamp": "2026-02-20T10:30:00.000000"
}
```

만약 어떤 서비스가 다운되면:
```json
{
  "status": "not_ready",
  "checks": {
    "db": "ok",
    "redis": "error: Connection refused",
    "storage": "ok"
  }
}
```

## 모니터링

### Docker 리소스 사용량 모니터링

```bash
docker stats

# 또는 지속적 모니터링
watch docker stats
```

### 데이터베이스 연결 상태

```bash
docker-compose -f docker-compose.prod.yml exec postgres \
  psql -U picfilter -d picfilter -c "SELECT datname, usename FROM pg_stat_activity;"
```

### Redis 정보

```bash
docker-compose -f docker-compose.prod.yml exec redis \
  redis-cli INFO
```

### Celery 작업 큐 상태

```bash
docker-compose -f docker-compose.prod.yml exec ml-worker \
  celery -A backend.workers.ml_tasks inspect active
```

### S3 버킷 확인

```bash
docker-compose -f docker-compose.prod.yml exec minio-init \
  mc ls local/
```

## 주요 포트

| 서비스 | 포트 | 용도 |
|--------|------|------|
| Nginx (리버스 프록시) | 80, 443 | HTTP/HTTPS |
| FastAPI | 8000 | API 서버 (내부) |
| Next.js | 3000 | 프론트엔드 (내부) |
| PostgreSQL | 5432 | 데이터베이스 (내부) |
| Redis | 6379 | 캐시/브로커 (내부) |
| MinIO | 9000 | S3 객체 저장소 (내부) |
| MinIO Console | 9001 | MinIO 웹 UI (내부) |

## 백업 및 복구

### 데이터베이스 백업

```bash
# PostgreSQL 덤프
docker-compose -f docker-compose.prod.yml exec postgres \
  pg_dump -U picfilter picfilter > backup_$(date +%Y%m%d_%H%M%S).sql
```

### 데이터베이스 복구

```bash
# 백업 파일에서 복구
docker-compose -f docker-compose.prod.yml exec -T postgres \
  psql -U picfilter picfilter < backup_20260220_103000.sql
```

### S3 데이터 백업

```bash
# 로컬로 동기화
docker-compose -f docker-compose.prod.yml exec minio \
  mc mirror local/picfilter-storage /backup/
```

## SSL/TLS 설정 (Let's Encrypt)

### 1. Certbot 설치

```bash
sudo apt-get install certbot python3-certbot-nginx
```

### 2. 인증서 발급

```bash
sudo certbot certonly --standalone -d yourdomain.com
```

### 3. Nginx 설정 업데이트

`nginx/nginx.conf`의 주석 처리된 SSL 섹션 활성화:

```nginx
server {
    listen 443 ssl http2;
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ...
}
```

### 4. 자동 갱신 설정

```bash
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer
```

## 프로덕션 체크리스트

- [ ] `.env` 파일 생성 및 모든 필수 값 설정
- [ ] JWT_SECRET_KEY를 강력한 무작위 키로 변경
- [ ] 데이터베이스 비밀번호 변경
- [ ] S3 비밀 키 변경
- [ ] 데이터베이스 자동 백업 설정
- [ ] SSL/TLS 인증서 설정
- [ ] 모니터링 및 알람 설정
- [ ] 로그 로테이션 설정
- [ ] 방화벽 설정 (필요한 포트만 열기)
- [ ] 헬스체크 엔드포인트 모니터링 설정

## 트러블슈팅

### 데이터베이스 연결 실패

```bash
# 데이터베이스 상태 확인
docker-compose -f docker-compose.prod.yml ps postgres

# 데이터베이스 로그 확인
docker-compose -f docker-compose.prod.yml logs postgres

# 재시작
docker-compose -f docker-compose.prod.yml restart postgres
```

### Redis 연결 실패

```bash
# Redis 상태 확인
docker-compose -f docker-compose.prod.yml ps redis

# Redis ping 테스트
docker-compose -f docker-compose.prod.yml exec redis redis-cli ping

# 재시작
docker-compose -f docker-compose.prod.yml restart redis
```

### 메모리 부족

```bash
# 메모리 사용량 확인
docker stats

# 컨테이너 메모리 한계 확인
docker-compose -f docker-compose.prod.yml config | grep -A 5 "mem_limit"

# docker-compose.prod.yml의 메모리 제한값 조정
```

### Celery 워커 동작 안 함

```bash
# 워커 상태 확인
docker-compose -f docker-compose.prod.yml exec ml-worker \
  celery -A backend.workers.ml_tasks inspect active

# 워커 로그 확인
docker-compose -f docker-compose.prod.yml logs ml-worker

# 워커 재시작
docker-compose -f docker-compose.prod.yml restart ml-worker
```

### API 응답 느림

```bash
# Nginx 로그 확인
docker-compose -f docker-compose.prod.yml logs nginx

# 백엔드 로그 확인
docker-compose -f docker-compose.prod.yml logs app

# 데이터베이스 쿼리 성능 확인
docker-compose -f docker-compose.prod.yml exec postgres \
  psql -U picfilter -d picfilter -c "SELECT * FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;"
```

## 추가 리소스

- [FastAPI 배포 문서](https://fastapi.tiangolo.com/deployment/)
- [Docker Compose 문서](https://docs.docker.com/compose/)
- [PostgreSQL 문서](https://www.postgresql.org/docs/)
- [Redis 문서](https://redis.io/documentation)
- [Celery 문서](https://docs.celeryproject.io/)
- [Nginx 문서](https://nginx.org/en/docs/)
