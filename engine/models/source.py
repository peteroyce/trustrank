import uuid
from datetime import datetime
from sqlalchemy import Float, Integer, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from engine.db.session import Base

class SourceCredibility(Base):
    __tablename__ = "source_credibility"
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("entities.id"), primary_key=True)
    credibility_score: Mapped[float] = mapped_column(Float, default=0.5)
    review_count: Mapped[int] = mapped_column(Integer, default=0)
    review_diversity: Mapped[float] = mapped_column(Float, default=0.0)
    accuracy_score: Mapped[float] = mapped_column(Float, default=0.5)
    account_age_days: Mapped[int] = mapped_column(Integer, default=0)
    flagged_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
