from pydantic import BaseModel


class OverviewResponse(BaseModel):
    total_entities: int
    total_signals: int
    tier_distribution: dict[str, int]
    signals_today: int
    dampened_count: int
