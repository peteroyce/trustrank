from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone
from engine.deps import get_db
from engine.models.entity import Entity
from engine.models.signal import Signal
from engine.models.score import OverallScore

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("/overview")
async def overview(db: AsyncSession = Depends(get_db)):
    entities = (
        await db.execute(
            select(func.count(Entity.id)).where(Entity.deleted == False)
        )
    ).scalar()
    signals = (await db.execute(select(func.count(Signal.id)))).scalar()
    today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    signals_today = (
        await db.execute(
            select(func.count(Signal.id)).where(Signal.created_at >= today)
        )
    ).scalar()
    dampened = (
        await db.execute(
            select(func.count(Signal.id)).where(Signal.dampened == True)
        )
    ).scalar()
    tier_result = await db.execute(
        select(OverallScore.tier, func.count(OverallScore.id)).group_by(
            OverallScore.tier
        )
    )
    tier_dist = {row[0].value: row[1] for row in tier_result.all()}
    return {
        "total_entities": entities,
        "total_signals": signals,
        "tier_distribution": tier_dist,
        "signals_today": signals_today,
        "dampened_count": dampened,
    }


@router.get("/leaderboard")
async def leaderboard(
    limit: int = 20, offset: int = 0, db: AsyncSession = Depends(get_db)
):
    q = (
        select(OverallScore, Entity)
        .join(Entity, OverallScore.entity_id == Entity.id)
        .where(Entity.deleted == False)
    )
    q = q.order_by(OverallScore.score.desc()).offset(offset).limit(limit)
    result = await db.execute(q)
    return [
        {
            "entity_id": str(o.entity_id),
            "name": e.name,
            "type": e.type.value,
            "score": o.score,
            "tier": o.tier.value,
            "signals": o.total_signals,
            "confidence": o.confidence,
        }
        for o, e in result.all()
    ]
