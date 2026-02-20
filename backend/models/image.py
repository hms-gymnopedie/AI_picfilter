from sqlalchemy import Column, String, Integer, BigInteger, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from .user import Base


class Image(Base):
    __tablename__ = "images"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename = Column(String(255), nullable=False)
    storage_key = Column(String(512), nullable=False)
    thumbnail_key = Column(String(512))
    content_type = Column(String(50), nullable=False)
    size_bytes = Column(BigInteger, nullable=False)
    width = Column(Integer)
    height = Column(Integer)
    format = Column(String(10))
    status = Column(String(20), default="pending", index=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    deleted_at = Column(DateTime(timezone=True))

    # Relationships
    user = relationship("User", back_populates="images")

    def __repr__(self):
        return f"<Image {self.filename}>"
