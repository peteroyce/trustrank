from uuid import UUID
from pydantic import BaseModel, Field
from engine.models.signal import Dimension, SignalType


class SignalCreate(BaseModel):
    source_id: UUID
    dimension: Dimension
    type: SignalType
    value: float = Field(ge=1.0, le=5.0)
    text: str | None = None


class SignalResponse(BaseModel):
    id: UUID
    entity_id: UUID
    source_id: UUID
    dimension: str
    type: str
    value: float
    tags: list[str] | None
    sentiment: float
    fake_probability: float
    dampened: bool
    dampening_reason: str | None
    weight: float
    created_at: str

    model_config = {"from_attributes": True}
