import uuid
import enum
from datetime import datetime
from sqlalchemy import String, Float, Enum, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from engine.db.session import Base

class EvidenceType(str, enum.Enum):
    transaction = "transaction"
    vouching = "vouching"
    verification = "verification"

class TrustEdge(Base):
    __tablename__ = "trust_edges"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("entities.id"), nullable=False, index=True)
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("entities.id"), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(100), default="general")
    weight: Mapped[float] = mapped_column(Float, default=0.5)
    evidence_type: Mapped[EvidenceType] = mapped_column(Enum(EvidenceType), default=EvidenceType.transaction)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
