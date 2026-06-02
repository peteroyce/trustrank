from uuid import UUID
from datetime import datetime
from pydantic import BaseModel
from engine.models.entity import EntityType


class EntityCreate(BaseModel):
    type: EntityType
    name: str
    metadata: dict = {}


class EntityResponse(BaseModel):
    id: UUID
    type: EntityType
    name: str
    metadata: dict
    created_at: datetime
    score: float | None = None
    tier: str | None = None

    model_config = {"from_attributes": True}
