from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from .user import Base


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    token_hash = Column(String(255), unique=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    revoked_at = Column(DateTime(timezone=True))

    # Relationships
    user = relationship("User", back_populates="refresh_tokens")

    def __repr__(self):
        return f"<RefreshToken user={self.user_id}>"
