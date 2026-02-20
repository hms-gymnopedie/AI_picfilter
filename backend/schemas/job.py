from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID


class JobResponse(BaseModel):
    job_id: UUID = Field(alias="id")
    status: str
    progress: float | None = None
    model_type: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None

    class Config:
        from_attributes = True
        populate_by_name = True
