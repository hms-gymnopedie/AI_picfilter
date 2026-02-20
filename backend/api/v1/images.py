from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from uuid import UUID, uuid4
import boto3

from backend.core.database import get_db
from backend.core.config import settings
from backend.models import User, Image
from backend.schemas import (
    ImageUploadRequest,
    ImageUploadResponse,
    ImageResponse,
    ImageDetailResponse,
    ImageListResponse,
)
from backend.api.dependencies import get_current_user

router = APIRouter()


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
        region_name=settings.S3_REGION,
    )


@router.post("/upload", response_model=ImageUploadResponse)
async def create_upload_url(
    request: ImageUploadRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Validate file size
    if request.size_bytes > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large")

    # Validate content type
    if request.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Unsupported image format")

    # Create image record
    image = Image(
        user_id=current_user.id,
        filename=request.filename,
        storage_key=f"images/{current_user.id}/{uuid4()}/original",
        content_type=request.content_type,
        size_bytes=request.size_bytes,
        status="pending",
    )
    db.add(image)
    await db.commit()
    await db.refresh(image)

    # Generate presigned URL
    s3_client = get_s3_client()
    upload_url = s3_client.generate_presigned_url(
        "put_object",
        Params={"Bucket": settings.S3_BUCKET, "Key": image.storage_key, "ContentType": request.content_type},
        ExpiresIn=600,
    )

    return ImageUploadResponse(image_id=image.id, upload_url=upload_url, expires_in=600)


@router.post("/{image_id}/confirm", response_model=ImageDetailResponse)
async def confirm_upload(
    image_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Fetch image
    result = await db.execute(select(Image).where((Image.id == image_id) & (Image.user_id == current_user.id)))
    image = result.scalars().first()

    if not image:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    # Verify file exists in S3
    s3_client = get_s3_client()
    try:
        s3_client.head_object(Bucket=settings.S3_BUCKET, Key=image.storage_key)
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File not found in storage")

    # Update image status
    image.status = "confirmed"
    await db.commit()
    await db.refresh(image)

    return image


@router.get("", response_model=ImageListResponse)
async def list_images(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    sort: str = Query("created_at", regex="^(created_at|name)$"),
    order: str = Query("desc", regex="^(asc|desc)$"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Build query
    query = select(Image).where((Image.user_id == current_user.id) & (Image.status != "deleted"))

    # Count total
    count_result = await db.execute(select(func.count(Image.id)).where(query.whereclause))
    total = count_result.scalar() or 0

    # Apply sorting
    if sort == "created_at":
        query = query.order_by(Image.created_at.desc() if order == "desc" else Image.created_at.asc())
    else:
        query = query.order_by(Image.filename.desc() if order == "desc" else Image.filename.asc())

    # Apply pagination
    offset = (page - 1) * per_page
    query = query.offset(offset).limit(per_page)

    result = await db.execute(query)
    images = result.scalars().all()

    return ImageListResponse(items=images, total=total, page=page, per_page=per_page)


@router.get("/{image_id}", response_model=ImageDetailResponse)
async def get_image(
    image_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Image).where((Image.id == image_id) & (Image.user_id == current_user.id)))
    image = result.scalars().first()

    if not image:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    return image


@router.delete("/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_image(
    image_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Image).where((Image.id == image_id) & (Image.user_id == current_user.id)))
    image = result.scalars().first()

    if not image:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    from datetime import datetime

    image.status = "deleted"
    image.deleted_at = datetime.utcnow()
    await db.commit()
