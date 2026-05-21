import uuid
import enum
from datetime import datetime
from sqlalchemy import String, Float, Boolean, Text, Enum, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import Mapped, mapped_column
from engine.db.session import Base

class Dimension(str, enum.Enum):
    quality = "quality"
    reliability = "reliability"
    responsiveness = "responsiveness"
    trust = "trust"

class SignalType(str, enum.Enum):
    review = "review"
    transaction = "transaction"
    complaint = "complaint"
    verification = "verification"
    dispute = "dispute"

class Signal(Base):
    __tablename__ = "signals"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("entities.id"), nullable=False, index=True)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("entities.id"), nullable=False, index=True)
    dimension: Mapped[Dimension] = mapped_column(Enum(Dimension), nullable=False)
    type: Mapped[SignalType] = mapped_column(Enum(SignalType), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    sentiment: Mapped[float] = mapped_column(Float, default=0.0)
    fake_probability: Mapped[float] = mapped_column(Float, default=0.0)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    raw_weight: Mapped[float] = mapped_column(Float, default=1.0)
    dampened: Mapped[bool] = mapped_column(Boolean, default=False)
    dampening_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
