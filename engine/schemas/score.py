from uuid import UUID
from pydantic import BaseModel


class DimensionScoreResponse(BaseModel):
    score: float
    wilson_lower: float
    trend: str
    signals: int
    confidence: float


class ScoreResponse(BaseModel):
    entity_id: UUID
    overall: float
    confidence: float
    tier: str
    dimensions: dict[str, DimensionScoreResponse]
    breakdown: dict
    alerts: list[dict]
    counterfactual: dict


class ScoreHistoryItem(BaseModel):
    score: float
    breakdown: dict
    created_at: str
