from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID


class StyleLearnRequest(BaseModel):
    name: str
    description: str | None = None
    reference_image_ids: list[UUID]
    model_type: str = "nilut"
    options: dict | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Warm Sunset",
                "description": "따뜻한 석양 분위기의 필터",
                "reference_image_ids": ["uuid1", "uuid2", "uuid3"],
                "model_type": "nilut",
                "options": {"strength": 1.0, "preserve_structure": True},
            }
        }


class StyleResponse(BaseModel):
    style_id: UUID = Field(alias="id")
    name: str
    description: str | None = None
    model_type: str
    thumbnail_url: str | None = None
    is_public: bool
    rating_avg: float = 0.0
    download_count: int
    created_at: datetime

    class Config:
        from_attributes = True
        populate_by_name = True


class StyleDetailResponse(BaseModel):
    style_id: UUID = Field(alias="id")
    name: str
    description: str | None = None
    model_type: str
    thumbnail_url: str | None = None
    preview_images: list[dict] = []
    is_public: bool
    owner: dict
    rating_avg: float = 0.0
    rating_count: int
    download_count: int
    created_at: datetime

    class Config:
        from_attributes = True
        populate_by_name = True


class StyleListResponse(BaseModel):
    items: list[StyleResponse]
    total: int
    page: int
    per_page: int
