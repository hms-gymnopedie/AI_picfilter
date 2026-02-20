from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, ForeignKey, Table, PrimaryKeyConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from .user import Base


class Style(Base):
    __tablename__ = "styles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    model_type = Column(String(30), nullable=False, default="nilut")
    model_key = Column(String(512))
    model_size_bytes = Column(Integer)
    preview_key = Column(String(512))
    is_public = Column(Boolean, default=False)
    rating_sum = Column(Integer, default=0)
    rating_count = Column(Integer, default=0)
    download_count = Column(Integer, default=0)
    status = Column(String(20), default="active")
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime(timezone=True))

    # Relationships
    user = relationship("User", back_populates="styles")
    reference_images = relationship(
        "StyleReferenceImage",
        back_populates="style",
        cascade="all, delete-orphan",
    )
    jobs = relationship("Job", back_populates="style")
    comments = relationship("Comment", back_populates="style", cascade="all, delete-orphan")
    ratings = relationship("Rating", back_populates="style", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Style {self.name}>"


class StyleReferenceImage(Base):
    __tablename__ = "style_reference_images"

    style_id = Column(UUID(as_uuid=True), ForeignKey("styles.id", ondelete="CASCADE"), nullable=False)
    image_id = Column(UUID(as_uuid=True), ForeignKey("images.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (PrimaryKeyConstraint("style_id", "image_id"),)

    # Relationships
    style = relationship("Style", back_populates="reference_images")

    def __repr__(self):
        return f"<StyleReferenceImage style={self.style_id} image={self.image_id}>"
