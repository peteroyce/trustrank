from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from engine.api import entities, signals, scores, trust, analytics, sources
from engine.deps import get_db

app = FastAPI(
    title="trustrank",
    version="0.1.0",
    description="Entity Reputation & Trust Scoring Engine",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(entities.router)
app.include_router(signals.router)
app.include_router(scores.router)
app.include_router(trust.router)
app.include_router(analytics.router)
app.include_router(sources.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/v1/admin/seed")
async def run_seed(db: AsyncSession = Depends(get_db)):
    from engine.seed.seeder import seed_database

    result = await seed_database(db)
    return {"status": "seeded", **result}
