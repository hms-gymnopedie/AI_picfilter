from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.types import Float
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from .user import Base


class Job(Base):
    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    job_type = Column(String(30), nullable=False)
    status = Column(String(20), default="queued", index=True)
    progress = Column(Float, default=0.0)
    style_id = Column(
        UUID(as_uuid=True), ForeignKey("styles.id", ondelete="SET NULL"), index=True
    )
    input_image_id = Column(
        UUID(as_uuid=True), ForeignKey("images.id", ondelete="SET NULL")
    )
    result_image_id = Column(
        UUID(as_uuid=True), ForeignKey("images.id", ondelete="SET NULL")
    )
    result_key = Column(String(512))
    params = Column(JSONB, default={})
    error_message = Column(Text)
    processing_time_ms = Column(Integer)
    celery_task_id = Column(String(255), index=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))

    # Relationships
    user = relationship("User", back_populates="jobs")
    style = relationship("Style", back_populates="jobs")

    def __repr__(self):
        return f"<Job {self.job_type}:{self.status}>"
