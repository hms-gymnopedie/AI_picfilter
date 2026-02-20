from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID


class CommentRequest(BaseModel):
    content: str

    class Config:
        json_schema_extra = {
            "example": {
                "content": "이 필터 정말 좋아요!",
            }
        }


class UserInfo(BaseModel):
    user_id: UUID = Field(alias="id")
    username: str

    class Config:
        from_attributes = True
        populate_by_name = True


class CommentResponse(BaseModel):
    comment_id: UUID = Field(alias="id")
    style_id: UUID
    user: UserInfo
    content: str
    created_at: datetime

    class Config:
        from_attributes = True
        populate_by_name = True


class CommentListResponse(BaseModel):
    items: list[CommentResponse]
    total: int
    page: int
    per_page: int


class RatingRequest(BaseModel):
    score: int

    class Config:
        json_schema_extra = {
            "example": {
                "score": 5,
            }
        }


class RatingResponse(BaseModel):
    style_id: UUID
    user_score: int
    rating_avg: float
    rating_count: int

    class Config:
        from_attributes = True
