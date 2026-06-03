from uuid import UUID
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from engine.deps import get_db
from engine.models.trust import TrustEdge
from engine.schemas.trust import TrustEdgeCreate
from engine.graph.trust import TrustGraph
from engine.graph.influence import katz_centrality
import networkx as nx

router = APIRouter(
    prefix="/api/v1/entities/{entity_id}/trust", tags=["trust"]
)


@router.post("", status_code=201)
async def create_trust_edge(
    entity_id: UUID,
    body: TrustEdgeCreate,
    db: AsyncSession = Depends(get_db),
):
    edge = TrustEdge(
        source_id=entity_id,
        target_id=body.target_id,
        category=body.category,
        weight=body.weight,
        evidence_type=body.evidence_type,
    )
    db.add(edge)
    await db.commit()
    return {"status": "created"}


@router.get("/graph")
async def get_trust_graph(
    entity_id: UUID, hops: int = 2, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(TrustEdge))
    edges = result.scalars().all()
    tg = TrustGraph(max_hops=hops)
    for e in edges:
        tg.add_edge(
            str(e.source_id),
            str(e.target_id),
            weight=e.weight,
            category=e.category,
        )
    return tg.get_subgraph(str(entity_id), hops=hops)


@router.get("/influence")
async def get_influence(
    entity_id: UUID, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(TrustEdge))
    edges = result.scalars().all()
    G = nx.DiGraph()
    for e in edges:
        G.add_edge(str(e.source_id), str(e.target_id), weight=e.weight)
    scores = katz_centrality(G)
    return {
        "entity_id": str(entity_id),
        "influence": scores.get(str(entity_id), 0.0),
    }
