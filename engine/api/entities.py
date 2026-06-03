from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from engine.deps import get_db
from engine.models.entity import Entity
from engine.models.score import OverallScore
from engine.schemas.entity import EntityCreate, EntityResponse

router = APIRouter(prefix="/api/v1/entities", tags=["entities"])


@router.post("", response_model=EntityResponse, status_code=201)
async def create_entity(body: EntityCreate, db: AsyncSession = Depends(get_db)):
    entity = Entity(type=body.type, name=body.name, metadata_=body.metadata)
    db.add(entity)
    await db.commit()
    await db.refresh(entity)
    return EntityResponse(
        id=entity.id,
        type=entity.type,
        name=entity.name,
        metadata=entity.metadata_,
        created_at=entity.created_at,
    )


@router.get("/{entity_id}", response_model=EntityResponse)
async def get_entity(entity_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Entity).where(Entity.id == entity_id, Entity.deleted == False)
    )
    entity = result.scalar_one_or_none()
    if not entity:
        raise HTTPException(404, "Entity not found")
    score_result = await db.execute(
        select(OverallScore).where(OverallScore.entity_id == entity_id)
    )
    overall = score_result.scalar_one_or_none()
    return EntityResponse(
        id=entity.id,
        type=entity.type,
        name=entity.name,
        metadata=entity.metadata_,
        created_at=entity.created_at,
        score=overall.score if overall else None,
        tier=overall.tier.value if overall else None,
    )


@router.get("", response_model=list[EntityResponse])
async def list_entities(
    type: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    q = select(Entity).where(Entity.deleted == False)
    if type:
        q = q.where(Entity.type == type)
    q = q.offset(offset).limit(limit)
    result = await db.execute(q)
    entities = result.scalars().all()
    responses = []
    for e in entities:
        sr = await db.execute(
            select(OverallScore).where(OverallScore.entity_id == e.id)
        )
        o = sr.scalar_one_or_none()
        responses.append(
            EntityResponse(
                id=e.id,
                type=e.type,
                name=e.name,
                metadata=e.metadata_,
                created_at=e.created_at,
                score=o.score if o else None,
                tier=o.tier.value if o else None,
            )
        )
    return responses


@router.delete("/{entity_id}", status_code=204)
async def delete_entity(entity_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Entity).where(Entity.id == entity_id))
    entity = result.scalar_one_or_none()
    if not entity:
        raise HTTPException(404, "Entity not found")
    entity.deleted = True
    await db.commit()
