from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID


class ImageUploadRequest(BaseModel):
    filename: str
    content_type: str
    size_bytes: int

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "photo.jpg",
                "content_type": "image/jpeg",
                "size_bytes": 5242880,
            }
        }


class ImageUploadResponse(BaseModel):
    image_id: UUID
    upload_url: str
    expires_in: int = 600


class ImageResponse(BaseModel):
    image_id: UUID = Field(alias="id")
    filename: str
    thumbnail_url: str | None = None
    width: int | None = None
    height: int | None = None
    size_bytes: int
    created_at: datetime

    class Config:
        from_attributes = True
        populate_by_name = True


class ImageDetailResponse(BaseModel):
    image_id: UUID = Field(alias="id")
    filename: str
    url: str | None = None
    thumbnail_url: str | None = None
    width: int | None = None
    height: int | None = None
    format: str | None = None
    size_bytes: int
    created_at: datetime

    class Config:
        from_attributes = True
        populate_by_name = True


class ImageListResponse(BaseModel):
    items: list[ImageResponse]
    total: int
    page: int
    per_page: int
