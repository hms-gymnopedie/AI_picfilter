from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from uuid import UUID, uuid4
from datetime import datetime

from backend.core.database import get_db
from backend.models import User, Style, Job, Image
from backend.schemas import (
    StyleLearnRequest,
    StyleResponse,
    StyleListResponse,
    StyleDetailResponse,
    JobResponse,
)
from backend.api.dependencies import get_current_user
from backend.workers.ml_tasks import learn_style as learn_style_task
from backend.core.storage import storage

router = APIRouter()


@router.post("/learn", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_style_learn_job(
    request: StyleLearnRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Verify reference images belong to user
    for image_id in request.reference_image_ids:
        result = await db.execute(
            select(Image).where(
                (Image.id == image_id) & (Image.user_id == current_user.id)
            )
        )
        if not result.scalars().first():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image {image_id} not found",
            )

    # Create style
    style = Style(
        user_id=current_user.id,
        name=request.name,
        description=request.description,
        model_type=request.model_type,
        status="active",
    )
    db.add(style)
    await db.flush()

    # Create job
    job = Job(
        user_id=current_user.id,
        job_type="style_learn",
        status="queued",
        style_id=style.id,
        params={
            "reference_image_ids": [str(id) for id in request.reference_image_ids],
            "model_type": request.model_type,
            "options": request.options or {},
        },
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Get S3 keys for reference images
    result = await db.execute(
        select(Image).where(Image.id.in_(request.reference_image_ids))
    )
    reference_images = result.scalars().all()
    reference_image_s3_keys = [img.storage_key for img in reference_images]

    # Submit to Celery
    task = learn_style_task.delay(
        job_id=str(job.id),
        style_id=str(style.id),
        reference_image_s3_keys=reference_image_s3_keys,
        config={
            "model_type": request.model_type,
            "strength": request.options.get("strength", 1.0)
            if request.options
            else 1.0,
            "preserve_structure": request.options.get("preserve_structure", True)
            if request.options
            else True,
        },
    )
    job.celery_task_id = task.id
    await db.commit()

    return job


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Job).where((Job.id == job_id) & (Job.user_id == current_user.id))
    )
    job = result.scalars().first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Job not found"
        )

    return job


@router.get("", response_model=StyleListResponse)
async def list_styles(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    model_type: str | None = None,
    is_public: bool | None = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Build query
    query = select(Style).where(
        (Style.user_id == current_user.id) & (Style.status != "deleted")
    )

    if model_type:
        query = query.where(Style.model_type == model_type)
    if is_public is not None:
        query = query.where(Style.is_public == is_public)

    # Count total
    count_result = await db.execute(
        select(func.count(Style.id)).where(query.whereclause)
    )
    total = count_result.scalar() or 0

    # Apply pagination
    offset = (page - 1) * per_page
    query = query.order_by(Style.created_at.desc()).offset(offset).limit(per_page)

    result = await db.execute(query)
    styles = result.scalars().all()

    return StyleListResponse(items=styles, total=total, page=page, per_page=per_page)


@router.get("/{style_id}", response_model=StyleDetailResponse)
async def get_style(
    style_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Style).where((Style.id == style_id) & (Style.user_id == current_user.id))
    )
    style = result.scalars().first()

    if not style:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Style not found"
        )

    rating_avg = (
        style.rating_sum / style.rating_count if style.rating_count > 0 else 0.0
    )

    return StyleDetailResponse(
        id=style.id,
        name=style.name,
        description=style.description,
        model_type=style.model_type,
        is_public=style.is_public,
        owner={"user_id": current_user.id, "username": current_user.username},
        rating_avg=rating_avg,
        rating_count=style.rating_count,
        download_count=style.download_count,
        created_at=style.created_at,
    )


@router.patch("/{style_id}", response_model=StyleDetailResponse)
async def update_style(
    style_id: UUID,
    request: dict,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Style).where((Style.id == style_id) & (Style.user_id == current_user.id))
    )
    style = result.scalars().first()

    if not style:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Style not found"
        )

    if "name" in request:
        style.name = request["name"]
    if "description" in request:
        style.description = request["description"]
    if "is_public" in request:
        style.is_public = request["is_public"]

    style.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(style)

    rating_avg = (
        style.rating_sum / style.rating_count if style.rating_count > 0 else 0.0
    )

    return StyleDetailResponse(
        id=style.id,
        name=style.name,
        description=style.description,
        model_type=style.model_type,
        is_public=style.is_public,
        owner={"user_id": current_user.id, "username": current_user.username},
        rating_avg=rating_avg,
        rating_count=style.rating_count,
        download_count=style.download_count,
        created_at=style.created_at,
    )


@router.delete("/{style_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_style(
    style_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Style).where((Style.id == style_id) & (Style.user_id == current_user.id))
    )
    style = result.scalars().first()

    if not style:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Style not found"
        )

    style.status = "deleted"
    style.deleted_at = datetime.utcnow()
    await db.commit()


@router.get("/jobs/{job_id}/progress")
async def get_job_progress(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get job progress as Server-Sent Events (SSE)."""
    import redis
    from fastapi.responses import StreamingResponse

    result = await db.execute(
        select(Job).where((Job.id == job_id) & (Job.user_id == current_user.id))
    )
    job = result.scalars().first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Job not found"
        )

    async def event_generator():
        from backend.core.config import settings

        redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        last_status = job.status
        last_progress = 0.0

        while True:
            # Get current progress from Redis
            progress_key = f"job_progress:{job_id}"
            progress_str = redis_client.get(progress_key)
            current_progress = float(progress_str) if progress_str else last_progress

            # Refresh job status from DB
            result = await db.execute(select(Job).where(Job.id == job_id))
            job_updated = result.scalars().first()
            current_status = job_updated.status if job_updated else last_status

            # Send progress update
            if current_progress != last_progress or current_status != last_status:
                yield f"data: {{'progress': {current_progress}, 'status': '{current_status}'}}\n\n"
                last_progress = current_progress
                last_status = current_status

            # Exit if job is complete or failed
            if current_status in ["completed", "failed"]:
                break

            # Wait before next check
            import asyncio

            await asyncio.sleep(1)

        redis_client.close()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/model-types")
async def get_model_types():
    """Get available model types for style learning."""
    return {
        "nilut": {
            "name": "NILUT",
            "description": "경량 MLP, 다중 스타일 블렌딩, CPU 가능",
            "model_size": "<1MB",
            "inference_speed": "<16ms (4K)",
            "advantages": [
                "매우 가볍고 빠른 추론",
                "여러 스타일을 한 모델에서 지원",
                "CPU 환경에서 실행 가능",
                "모바일 배포에 최적화",
            ],
            "training_time_estimate_sec": 120,
        },
        "lut3d": {
            "name": "Image-Adaptive 3D LUT",
            "description": "CNN 기반 이미지 적응형, 더 자연스러운 결과",
            "model_size": "<10MB",
            "inference_speed": "<2ms (4K GPU)",
            "advantages": [
                "더 정밀한 스타일 전이",
                "이미지에 적응하는 동적 처리",
                "GPU에서 매우 빠른 속도",
                "Photoshop과 호환되는 LUT 내보내기",
            ],
            "training_time_estimate_sec": 300,
        },
    }
