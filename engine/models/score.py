import uuid
import enum
from datetime import datetime
from sqlalchemy import String, Float, Integer, Enum, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column
from engine.db.session import Base
from engine.models.signal import Dimension

class Tier(str, enum.Enum):
    platinum = "platinum"
    gold = "gold"
    silver = "silver"
    bronze = "bronze"
    untrusted = "untrusted"

class Trend(str, enum.Enum):
    improving = "improving"
    stable = "stable"
    declining = "declining"

class DimensionScore(Base):
    __tablename__ = "dimension_scores"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("entities.id"), nullable=False, index=True)
    dimension: Mapped[Dimension] = mapped_column(Enum(Dimension), nullable=False)
    wilson_lower: Mapped[float] = mapped_column(Float, default=0.0)
    bayesian_score: Mapped[float] = mapped_column(Float, default=3.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    signal_count: Mapped[int] = mapped_column(Integer, default=0)
    alpha: Mapped[float] = mapped_column(Float, default=2.0)
    beta_param: Mapped[float] = mapped_column(Float, default=2.0)
    dirichlet: Mapped[list[float] | None] = mapped_column(ARRAY(Float), default=[1.0, 1.0, 1.0, 1.0, 1.0])
    trend: Mapped[Trend] = mapped_column(Enum(Trend), default=Trend.stable)
    trend_slope: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class OverallScore(Base):
    __tablename__ = "overall_scores"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("entities.id"), nullable=False, unique=True, index=True)
    score: Mapped[float] = mapped_column(Float, default=3.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    tier: Mapped[Tier] = mapped_column(Enum(Tier), default=Tier.bronze)
    total_signals: Mapped[int] = mapped_column(Integer, default=0)
    trust_bonus: Mapped[float] = mapped_column(Float, default=0.0)
    manipulation_penalty: Mapped[float] = mapped_column(Float, default=0.0)
    alerts: Mapped[dict] = mapped_column(JSONB, default=list)
    breakdown: Mapped[dict] = mapped_column(JSONB, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class ScoreHistory(Base):
    __tablename__ = "score_history"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("entities.id"), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    dimension_scores: Mapped[dict] = mapped_column(JSONB, default=dict)
    trigger_signal_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("signals.id"), nullable=True)
    breakdown: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
