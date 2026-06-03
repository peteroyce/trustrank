from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from engine.deps import get_db, get_analyzer_dep
from engine.models.signal import Signal
from engine.models.entity import Entity
from engine.schemas.signal import SignalCreate, SignalResponse

router = APIRouter(
    prefix="/api/v1/entities/{entity_id}/signals", tags=["signals"]
)


@router.post("", response_model=SignalResponse, status_code=201)
async def submit_signal(
    entity_id: UUID,
    body: SignalCreate,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Entity).where(Entity.id == entity_id, Entity.deleted == False)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(404, "Entity not found")
    analyzer = get_analyzer_dep()
    analysis = analyzer.analyze(body.text or "", {})
    signal = Signal(
        entity_id=entity_id,
        source_id=body.source_id,
        dimension=body.dimension,
        type=body.type,
        value=body.value,
        text=body.text,
        tags=analysis.tags,
        sentiment=analysis.sentiment,
        fake_probability=analysis.fake_probability,
        weight=1.0,
        raw_weight=1.0,
    )
    db.add(signal)
    await db.commit()
    await db.refresh(signal)
    return SignalResponse(
        id=signal.id,
        entity_id=signal.entity_id,
        source_id=signal.source_id,
        dimension=signal.dimension.value,
        type=signal.type.value,
        value=signal.value,
        tags=signal.tags,
        sentiment=signal.sentiment,
        fake_probability=signal.fake_probability,
        dampened=signal.dampened,
        dampening_reason=signal.dampening_reason,
        weight=signal.weight,
        created_at=signal.created_at.isoformat(),
    )


@router.get("", response_model=list[SignalResponse])
async def list_signals(
    entity_id: UUID,
    dimension: str | None = None,
    dampened: bool | None = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    q = select(Signal).where(Signal.entity_id == entity_id)
    if dimension:
        q = q.where(Signal.dimension == dimension)
    if dampened is not None:
        q = q.where(Signal.dampened == dampened)
    q = q.order_by(Signal.created_at.desc()).limit(limit)
    result = await db.execute(q)
    return [
        SignalResponse(
            id=s.id,
            entity_id=s.entity_id,
            source_id=s.source_id,
            dimension=s.dimension.value,
            type=s.type.value,
            value=s.value,
            tags=s.tags,
            sentiment=s.sentiment,
            fake_probability=s.fake_probability,
            dampened=s.dampened,
            dampening_reason=s.dampening_reason,
            weight=s.weight,
            created_at=s.created_at.isoformat(),
        )
        for s in result.scalars().all()
    ]
