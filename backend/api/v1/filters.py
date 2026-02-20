from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID
from datetime import datetime

from backend.core.database import get_db
from backend.models import User, Job, Image, Style
from backend.schemas import JobResponse
from backend.api.dependencies import get_current_user
from backend.workers.ml_tasks import (
    apply_filter as apply_filter_task,
    export_cube as export_cube_task,
)

router = APIRouter()


@router.post("/apply", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def apply_filter(
    style_id: UUID,
    target_image_id: UUID,
    strength: float = Query(0.8, ge=0.0, le=1.0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Apply learned style filter to image."""
    # Verify style exists and user owns it
    result = await db.execute(
        select(Style).where((Style.id == style_id) & (Style.user_id == current_user.id))
    )
    style = result.scalars().first()
    if not style:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Style not found"
        )

    # Verify image exists and user owns it
    result = await db.execute(
        select(Image).where(
            (Image.id == target_image_id) & (Image.user_id == current_user.id)
        )
    )
    image = result.scalars().first()
    if not image or image.status == "deleted":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Image not found"
        )

    # Create job
    job = Job(
        user_id=current_user.id,
        job_type="filter_apply",
        status="queued",
        style_id=style_id,
        input_image_id=target_image_id,
        params={"strength": strength},
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Submit to Celery
    task = apply_filter_task.delay(
        job_id=str(job.id),
        input_image_s3_key=image.storage_key,
        style_id=str(style_id),
        intensity=strength,
    )
    job.celery_task_id = task.id
    await db.commit()

    return job


@router.post(
    "/{style_id}/export",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def export_cube(
    style_id: UUID,
    lut_size: int = Query(33, regex="^(17|33|65)$"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Export learned style as .cube LUT file."""
    # Verify style exists and user owns it
    result = await db.execute(
        select(Style).where((Style.id == style_id) & (Style.user_id == current_user.id))
    )
    style = result.scalars().first()
    if not style:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Style not found"
        )

    # Create job
    job = Job(
        user_id=current_user.id,
        job_type="cube_export",
        status="queued",
        style_id=style_id,
        params={"lut_size": lut_size},
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Submit to Celery
    task = export_cube_task.delay(
        job_id=str(job.id),
        style_id=str(style_id),
        lut_size=lut_size,
    )
    job.celery_task_id = task.id
    await db.commit()

    return job


@router.get("/jobs/{job_id}/result", status_code=status.HTTP_200_OK)
async def get_job_result(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get filter application or .cube export result."""
    result = await db.execute(
        select(Job).where((Job.id == job_id) & (Job.user_id == current_user.id))
    )
    job = result.scalars().first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Job not found"
        )

    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job status is {job.status}, not completed",
        )

    # Return appropriate response based on job type
    if job.job_type == "filter_apply":
        return {
            "job_id": job.id,
            "status": job.status,
            "result_image_id": job.result_image_id,
            "result_url": f"/images/{job.result_image_id}",
            "processing_time_ms": job.processing_time_ms,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
        }
    elif job.job_type == "cube_export":
        from backend.core.storage import storage

        download_url = storage.generate_presigned_url(job.result_key, expires_in=3600)
        return {
            "job_id": job.id,
            "status": job.status,
            "download_url": download_url,
            "download_expires_in": 3600,
            "file_size_bytes": job.params.get("file_size"),
            "lut_size": job.params.get("lut_size"),
            "created_at": job.created_at,
            "completed_at": job.completed_at,
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown job type"
        )
