from sqlalchemy import Column, String, Text, Boolean, DateTime, ForeignKey, SmallInteger, PrimaryKeyConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from .user import Base


class Comment(Base):
    __tablename__ = "comments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    style_id = Column(UUID(as_uuid=True), ForeignKey("styles.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    is_deleted = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="comments")
    style = relationship("Style", back_populates="comments")

    def __repr__(self):
        return f"<Comment {self.id}>"


class Rating(Base):
    __tablename__ = "ratings"

    style_id = Column(UUID(as_uuid=True), ForeignKey("styles.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    score = Column(SmallInteger, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (PrimaryKeyConstraint("style_id", "user_id"),)

    # Relationships
    user = relationship("User", back_populates="ratings")
    style = relationship("Style", back_populates="ratings")

    def __repr__(self):
        return f"<Rating {self.style_id}:{self.user_id}={self.score}>"
