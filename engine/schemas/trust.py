from uuid import UUID
from pydantic import BaseModel, Field
from engine.models.trust import EvidenceType


class TrustEdgeCreate(BaseModel):
    target_id: UUID
    category: str = "general"
    weight: float = Field(ge=0.0, le=1.0, default=0.5)
    evidence_type: EvidenceType = EvidenceType.transaction
