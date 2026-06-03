from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from engine.deps import get_db
from engine.models.source import SourceCredibility

router = APIRouter(prefix="/api/v1/sources", tags=["sources"])


@router.get("/{source_id}/credibility")
async def get_credibility(
    source_id: UUID, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(SourceCredibility).where(
            SourceCredibility.source_id == source_id
        )
    )
    cred = result.scalar_one_or_none()
    if not cred:
        raise HTTPException(404, "Source credibility not found")
    return {
        "source_id": str(cred.source_id),
        "credibility_score": cred.credibility_score,
        "review_count": cred.review_count,
        "review_diversity": cred.review_diversity,
        "accuracy_score": cred.accuracy_score,
        "account_age_days": cred.account_age_days,
        "flagged_count": cred.flagged_count,
    }
