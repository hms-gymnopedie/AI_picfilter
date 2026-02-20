# Frontend Deployment Guide

This document covers local development, production builds, environment variable configuration, and Docker-based deployment for the AI_picfilter Next.js frontend.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Variables](#2-environment-variables)
3. [Local Development (without Docker)](#3-local-development-without-docker)
4. [Local Development (with Docker Compose)](#4-local-development-with-docker-compose)
5. [Production Build (without Docker)](#5-production-build-without-docker)
6. [Production Build (with Docker)](#6-production-build-with-docker)
7. [Full Stack with Docker Compose (production)](#7-full-stack-with-docker-compose-production)
8. [Dockerfile Stages Explained](#8-dockerfile-stages-explained)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites

| Tool | Minimum version |
|------|----------------|
| Node.js | 18.x |
| npm | 9.x |
| Docker | 24.x |
| Docker Compose | v2.20 |

---

## 2. Environment Variables

The frontend uses two types of environment variables:

| Prefix | Availability | Notes |
|--------|-------------|-------|
| `NEXT_PUBLIC_*` | Browser + server | **Baked into the JS bundle at build time.** Must be set before `npm run build` or `docker build`. |
| *(no prefix)* | Server only | Not currently used by this project. |

### Variable reference

| Variable | Description | Example |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API base URL (no trailing slash) | `https://api.example.com` |
| `NEXT_PUBLIC_APP_URL` | Public URL of this frontend | `https://example.com` |

### Setup

```bash
# Copy the template and fill in values
cp frontend/.env.example frontend/.env.local
```

The `.env.local` file is **gitignored** and loaded automatically by Next.js for local development. Do not commit it.

---

## 3. Local Development (without Docker)

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server with hot reload
npm run dev
```

The app will be available at `http://localhost:3000`.

The backend API must be running separately (default: `http://localhost:8000`). Set `NEXT_PUBLIC_API_URL` in `.env.local` if using a different address.

---

## 4. Local Development (with Docker Compose)

The `docker-compose.yml` file runs the frontend in development mode with live code reloading via volume mount.

```bash
# From the project root
docker compose up frontend

# Or start the full stack (backend + workers + DB + frontend)
docker compose up
```

The `frontend` service:
- Uses the `deps` build stage (installs node_modules only)
- Mounts `./frontend` into the container for hot reload
- Uses anonymous volumes for `node_modules` and `.next` to avoid conflicts with the host filesystem
- Runs `npm run dev` on port 3000

> **Note:** The first startup may be slow while npm installs dependencies. Subsequent starts reuse the anonymous volume cache.

---

## 5. Production Build (without Docker)

```bash
cd frontend

# Set public env vars before building
export NEXT_PUBLIC_API_URL=https://api.example.com
export NEXT_PUBLIC_APP_URL=https://example.com

# Build the optimised application
npm run build

# Start the production server
npm start
```

The build output uses Next.js [standalone mode](https://nextjs.org/docs/pages/api-reference/next-config-js/output), which produces a self-contained `./next/standalone` directory that does not require the full `node_modules` tree to run.

---

## 6. Production Build (with Docker)

### Build the image

```bash
# From the project root
docker build \
  --target runner \
  --build-arg NEXT_PUBLIC_API_URL=https://api.example.com \
  --build-arg NEXT_PUBLIC_APP_URL=https://example.com \
  -t picfilter-frontend:latest \
  ./frontend
```

### Run the container

```bash
docker run -d \
  --name picfilter-frontend \
  -p 3000:3000 \
  -e PORT=3000 \
  -e HOSTNAME=0.0.0.0 \
  picfilter-frontend:latest
```

The container runs as a non-root user (`nextjs`, UID 1001) and listens on port 3000.

---

## 7. Full Stack with Docker Compose (production)

```bash
# From the project root

# 1. Copy and fill in .env for the backend
cp .env.example .env   # if it exists, otherwise create it

# 2. Export public frontend vars so docker compose passes them as build args
export NEXT_PUBLIC_API_URL=https://api.example.com
export NEXT_PUBLIC_APP_URL=https://example.com

# 3. Build all production images
docker compose \
  -f docker-compose.yml \
  -f docker-compose.prod.yml \
  build

# 4. Start all services in the background
docker compose \
  -f docker-compose.yml \
  -f docker-compose.prod.yml \
  up -d

# 5. Check health
docker compose \
  -f docker-compose.yml \
  -f docker-compose.prod.yml \
  ps
```

The production compose file (`docker-compose.prod.yml`):
- Sets `target: runner` so the full multi-stage Dockerfile runs
- Passes `NEXT_PUBLIC_*` as Docker build args (baked into JS bundles)
- Removes development volume mounts
- Adds resource limits for all services
- Configures `healthcheck` using `wget` (available in the Alpine image)

---

## 8. Dockerfile Stages Explained

```
Stage 1 — deps
  node:18-alpine
  Installs node_modules (npm ci) for deterministic, cached installs.
  Used directly in the dev Docker Compose target.

Stage 2 — builder
  node:18-alpine
  Copies node_modules from deps, copies source, runs `npm run build`.
  NEXT_PUBLIC_* vars are injected as build ARGs here.

Stage 3 — runner
  node:18-alpine
  Copies only the standalone output:
    .next/standalone/   → self-contained server
    .next/static/       → compiled client assets
    public/             → static public files
  Runs as non-root user (nextjs:1001).
  Final image size is typically 150–250 MB.
```

### Image size optimisation

- `output: 'standalone'` in `next.config.ts` eliminates the need to ship all of `node_modules`
- `.dockerignore` excludes `.git`, `node_modules`, `.next`, and documentation from the build context
- Alpine base image keeps the OS footprint minimal

---

## 9. Troubleshooting

### `NEXT_PUBLIC_API_URL` is undefined at runtime

`NEXT_PUBLIC_*` variables are baked into the JS bundle **at build time**. If they are undefined when you run `docker build`, they will be empty in the browser. Always pass them as `--build-arg` flags or set them in the shell environment before calling `docker compose build`.

### Hot reload not working in Docker dev

Check that:
1. `./frontend` is correctly volume-mounted into `/app`
2. The anonymous `node_modules` volume exists (it is created on first `docker compose up`)
3. `CHOKIDAR_USEPOLLING=1` may be needed on some host filesystems (WSL2, VirtualBox):
   ```yaml
   environment:
     - CHOKIDAR_USEPOLLING=1
   ```

### Port 3000 already in use

```bash
# Find and kill the process using port 3000
lsof -ti:3000 | xargs kill -9
```

Or change the host port mapping in `docker-compose.yml`:
```yaml
ports:
  - "3001:3000"
```

### `standalone` directory missing after build

Ensure `output: 'standalone'` is set in `frontend/next.config.ts`. Without it, the `runner` stage will fail to copy `.next/standalone`.

### WebGL not available in server-rendered context

WebGL runs entirely in the browser. If you see server-side rendering errors related to `WebGL2RenderingContext`, ensure the `LUTPreview` component is only used inside `'use client'` files (it already is) and that it is not imported in server components.
