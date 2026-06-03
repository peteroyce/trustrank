from uuid import UUID
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from engine.deps import get_db, get_analyzer_dep
from engine.models.signal import Signal
from engine.models.score import ScoreHistory
from engine.models.source import SourceCredibility
from engine.scoring.pipeline import ScoringPipeline
from engine.schemas.score import ScoreResponse, ScoreHistoryItem

router = APIRouter(
    prefix="/api/v1/entities/{entity_id}/score", tags=["scores"]
)


@router.get("", response_model=ScoreResponse)
async def get_score(entity_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Signal)
        .where(Signal.entity_id == entity_id)
        .order_by(Signal.created_at)
    )
    signals = result.scalars().all()
    pipeline = ScoringPipeline(analyzer=get_analyzer_dep())
    signal_dicts = []
    for s in signals:
        cred_result = await db.execute(
            select(SourceCredibility).where(
                SourceCredibility.source_id == s.source_id
            )
        )
        cred = cred_result.scalar_one_or_none()
        source_stats = {
            "review_count": cred.review_count if cred else 1,
            "review_diversity": cred.review_diversity if cred else 0.5,
            "accuracy": cred.accuracy_score if cred else 0.5,
            "account_age_days": cred.account_age_days if cred else 30,
            "flagged_ratio": (
                cred.flagged_count / max(cred.review_count, 1) if cred else 0.0
            ),
        }
        signal_dicts.append(
            {
                "value": s.value,
                "text": s.text or "",
                "dimension": s.dimension.value,
                "source_stats": source_stats,
                "created_at": s.created_at,
            }
        )
    scored = pipeline.score(signal_dicts)
    return ScoreResponse(entity_id=entity_id, **scored)


@router.get("/history", response_model=list[ScoreHistoryItem])
async def get_score_history(
    entity_id: UUID, limit: int = 100, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(ScoreHistory)
        .where(ScoreHistory.entity_id == entity_id)
        .order_by(ScoreHistory.created_at.desc())
        .limit(limit)
    )
    return [
        ScoreHistoryItem(
            score=h.score,
            breakdown=h.breakdown,
            created_at=h.created_at.isoformat(),
        )
        for h in result.scalars().all()
    ]
