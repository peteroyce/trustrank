# trustrank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade entity reputation engine with multi-dimensional scoring, manipulation resistance, trust graph analysis, and full explainability.

**Architecture:** FastAPI backend with PostgreSQL + Redis, pure-Python scoring/detection/graph algorithms, pluggable AI analyzer (mock default, Claude opt-in), React+Tailwind dashboard. All math is the product — Wilson intervals, Beta-Binomial, Dirichlet-Multinomial, CUSUM, BOCPD, Louvain, Katz centrality.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy, Alembic, PostgreSQL 16, Redis, NetworkX, NumPy, SciPy, scikit-learn, React 19, Tailwind v4, Recharts, D3.js, Docker Compose

**Spec:** `docs/specs/2026-07-13-trustrank-design.md`

---

## File Map

```
trustrank/
├── engine/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app + router registration
│   ├── config.py                  # Pydantic Settings (env-based)
│   ├── deps.py                    # Dependency injection (db session, analyzer)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── entities.py            # Entity CRUD
│   │   ├── signals.py             # Signal submission
│   │   ├── scores.py              # Score breakdown + history + counterfactual
│   │   ├── trust.py               # Trust edge + graph + influence
│   │   ├── analytics.py           # Overview + leaderboard + alerts + detection
│   │   └── sources.py             # Source credibility
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── wilson.py              # Wilson score interval
│   │   ├── bayesian.py            # Beta-Binomial + Dirichlet-Multinomial
│   │   ├── decay.py               # Temporal decay + velocity
│   │   ├── aggregator.py          # Dimension → overall + tier
│   │   └── pipeline.py            # Full scoring pipeline orchestrator
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── burst.py               # CUSUM
│   │   ├── coordination.py        # TF-IDF clustering + Poisson timing
│   │   ├── credibility.py         # Source credibility scoring
│   │   ├── reciprocal.py          # Collusion detection
│   │   └── manager.py             # Runs all detectors
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── trust.py               # Transitive propagation
│   │   ├── community.py           # Louvain
│   │   └── influence.py           # Katz centrality
│   ├── temporal/
│   │   ├── __init__.py
│   │   ├── trend.py               # Rolling linear regression
│   │   └── changepoint.py         # BOCPD
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── base.py                # Analyzer ABC + AnalysisResult
│   │   ├── mock.py                # VADER + TF-IDF + heuristics
│   │   └── claude_analyzer.py     # Claude API
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── tracer.py              # Score audit trail
│   │   └── counterfactual.py      # What-if scenarios
│   ├── models/
│   │   ├── __init__.py            # Re-exports all models
│   │   ├── entity.py
│   │   ├── signal.py
│   │   ├── score.py               # DimensionScore + OverallScore + ScoreHistory
│   │   ├── trust.py               # TrustEdge
│   │   └── source.py              # SourceCredibility
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── entity.py
│   │   ├── signal.py
│   │   ├── score.py
│   │   ├── trust.py
│   │   └── analytics.py
│   ├── db/
│   │   ├── __init__.py
│   │   └── session.py             # Engine + SessionLocal
│   └── seed/
│       ├── __init__.py
│       ├── seeder.py              # Data generation
│       └── scenarios.py           # Attack scenarios
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # DB fixtures, test session
│   ├── test_wilson.py
│   ├── test_bayesian.py
│   ├── test_decay.py
│   ├── test_aggregator.py
│   ├── test_burst.py
│   ├── test_coordination.py
│   ├── test_credibility.py
│   ├── test_reciprocal.py
│   ├── test_trust_graph.py
│   ├── test_community.py
│   ├── test_influence.py
│   ├── test_trend.py
│   ├── test_changepoint.py
│   ├── test_mock_analyzer.py
│   ├── test_tracer.py
│   ├── test_counterfactual.py
│   ├── test_pipeline.py
│   └── test_api.py
├── dashboard/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── lib/api.ts
│       ├── components/
│       │   ├── Layout.tsx
│       │   ├── TierBadge.tsx
│       │   ├── ScoreSparkline.tsx
│       │   └── ForceGraph.tsx
│       └── pages/
│           ├── Leaderboard.tsx
│           ├── EntityDetail.tsx
│           ├── TrustExplorer.tsx
│           ├── Detection.tsx
│           └── Analytics.tsx
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── alembic.ini
├── .env.example
├── .gitignore
└── README.md
```

---

## Phase 1: Foundation

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `engine/__init__.py`, `engine/config.py`, `engine/db/__init__.py`, `engine/db/session.py`, `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "trustrank"
version = "0.1.0"
description = "Entity reputation & trust scoring engine"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "sqlalchemy[asyncio]>=2.0.30",
    "asyncpg>=0.30.0",
    "alembic>=1.13.0",
    "redis>=5.0.0",
    "pydantic>=2.7.0",
    "pydantic-settings>=2.3.0",
    "networkx>=3.3",
    "numpy>=2.0.0",
    "scipy>=1.14.0",
    "scikit-learn>=1.5.0",
    "nltk>=3.9.0",
    "anthropic>=0.30.0",
    "python-community-louvain>=0.16",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2.0",
    "pytest-asyncio>=0.23.0",
    "httpx>=0.27.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Create .gitignore and .env.example**

`.gitignore`:
```
__pycache__/
*.pyc
.env
*.egg-info/
dist/
.venv/
node_modules/
CLAUDE.md
.pytest_cache/
```

`.env.example`:
```
DATABASE_URL=postgresql+asyncpg://trustrank:trustrank@localhost:5432/trustrank
REDIS_URL=redis://localhost:6379/0
ANTHROPIC_API_KEY=
ANALYZER_BACKEND=mock
```

- [ ] **Step 3: Create engine/config.py**

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://trustrank:trustrank@localhost:5432/trustrank"
    redis_url: str = "redis://localhost:6379/0"
    anthropic_api_key: str = ""
    analyzer_backend: str = "mock"

    # Scoring defaults
    decay_half_life_days: float = 90.0
    velocity_window_days: int = 7
    velocity_boost: float = 1.1
    trend_window_days: int = 30
    trend_threshold: float = 0.02

    # Detection defaults
    cusum_allowance_factor: float = 0.5
    cusum_threshold_factor: float = 5.0
    coordination_similarity_threshold: float = 0.85
    coordination_cv_threshold: float = 0.5
    reciprocal_jaccard_threshold: float = 0.3
    reciprocal_sentiment_threshold: float = 0.7

    # Trust graph
    trust_damping: float = 0.7
    trust_max_hops: int = 2
    trust_score_weight: float = 0.15

    # Tier thresholds
    tier_platinum_score: float = 4.5
    tier_platinum_signals: int = 100
    tier_platinum_confidence: float = 0.85
    tier_gold_score: float = 4.0
    tier_gold_signals: int = 50
    tier_gold_confidence: float = 0.75
    tier_silver_score: float = 3.5
    tier_silver_signals: int = 20
    tier_silver_confidence: float = 0.60
    tier_bronze_score: float = 2.5
    tier_bronze_signals: int = 5
    tier_bronze_confidence: float = 0.40

    # Dimension weights (per entity type)
    default_dimension_weights: dict = {
        "quality": 0.35,
        "reliability": 0.25,
        "responsiveness": 0.20,
        "trust": 0.20,
    }

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()
```

- [ ] **Step 4: Create engine/db/session.py**

```python
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from engine.config import settings

engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

async def get_session():
    async with async_session() as session:
        yield session
```

- [ ] **Step 5: Create tests/conftest.py**

```python
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from engine.db.session import Base

TEST_DB_URL = "sqlite+aiosqlite:///test.db"

@pytest.fixture
async def db_session():
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()
```

- [ ] **Step 6: Create __init__.py files and commit**

Create empty `engine/__init__.py`, `engine/db/__init__.py`, `tests/__init__.py`.

```bash
git add -A
git commit -m "chore: project scaffolding — pyproject, config, db session, test fixtures"
```

---

### Task 2: SQLAlchemy Models

**Files:**
- Create: `engine/models/entity.py`, `engine/models/signal.py`, `engine/models/score.py`, `engine/models/trust.py`, `engine/models/source.py`, `engine/models/__init__.py`

- [ ] **Step 1: Create engine/models/entity.py**

```python
import uuid
import enum
from datetime import datetime
from sqlalchemy import String, Enum, DateTime, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column
from engine.db.session import Base

class EntityType(str, enum.Enum):
    merchant = "merchant"
    user = "user"
    service = "service"
    product = "product"

class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type: Mapped[EntityType] = mapped_column(Enum(EntityType), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    deleted: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
```

- [ ] **Step 2: Create engine/models/signal.py**

```python
import uuid
import enum
from datetime import datetime
from sqlalchemy import String, Float, Boolean, Text, Enum, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import Mapped, mapped_column
from engine.db.session import Base

class Dimension(str, enum.Enum):
    quality = "quality"
    reliability = "reliability"
    responsiveness = "responsiveness"
    trust = "trust"

class SignalType(str, enum.Enum):
    review = "review"
    transaction = "transaction"
    complaint = "complaint"
    verification = "verification"
    dispute = "dispute"

class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("entities.id"), nullable=False, index=True)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("entities.id"), nullable=False, index=True)
    dimension: Mapped[Dimension] = mapped_column(Enum(Dimension), nullable=False)
    type: Mapped[SignalType] = mapped_column(Enum(SignalType), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    sentiment: Mapped[float] = mapped_column(Float, default=0.0)
    fake_probability: Mapped[float] = mapped_column(Float, default=0.0)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    raw_weight: Mapped[float] = mapped_column(Float, default=1.0)
    dampened: Mapped[bool] = mapped_column(Boolean, default=False)
    dampening_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
```

- [ ] **Step 3: Create engine/models/score.py**

```python
import uuid
import enum
from datetime import datetime
from sqlalchemy import String, Float, Integer, Enum, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column
from engine.db.session import Base
from engine.models.signal import Dimension

class Tier(str, enum.Enum):
    platinum = "platinum"
    gold = "gold"
    silver = "silver"
    bronze = "bronze"
    untrusted = "untrusted"

class Trend(str, enum.Enum):
    improving = "improving"
    stable = "stable"
    declining = "declining"

class DimensionScore(Base):
    __tablename__ = "dimension_scores"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("entities.id"), nullable=False, index=True)
    dimension: Mapped[Dimension] = mapped_column(Enum(Dimension), nullable=False)
    wilson_lower: Mapped[float] = mapped_column(Float, default=0.0)
    bayesian_score: Mapped[float] = mapped_column(Float, default=3.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    signal_count: Mapped[int] = mapped_column(Integer, default=0)
    alpha: Mapped[float] = mapped_column(Float, default=2.0)
    beta_param: Mapped[float] = mapped_column(Float, default=2.0)
    dirichlet: Mapped[list[float] | None] = mapped_column(ARRAY(Float), default=[1.0, 1.0, 1.0, 1.0, 1.0])
    trend: Mapped[Trend] = mapped_column(Enum(Trend), default=Trend.stable)
    trend_slope: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class OverallScore(Base):
    __tablename__ = "overall_scores"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("entities.id"), nullable=False, unique=True, index=True)
    score: Mapped[float] = mapped_column(Float, default=3.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    tier: Mapped[Tier] = mapped_column(Enum(Tier), default=Tier.bronze)
    total_signals: Mapped[int] = mapped_column(Integer, default=0)
    trust_bonus: Mapped[float] = mapped_column(Float, default=0.0)
    manipulation_penalty: Mapped[float] = mapped_column(Float, default=0.0)
    alerts: Mapped[dict] = mapped_column(JSONB, default=list)
    breakdown: Mapped[dict] = mapped_column(JSONB, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class ScoreHistory(Base):
    __tablename__ = "score_history"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("entities.id"), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    dimension_scores: Mapped[dict] = mapped_column(JSONB, default=dict)
    trigger_signal_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("signals.id"), nullable=True)
    breakdown: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
```

- [ ] **Step 4: Create engine/models/trust.py and engine/models/source.py**

`engine/models/trust.py`:
```python
import uuid
import enum
from datetime import datetime
from sqlalchemy import String, Float, Enum, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from engine.db.session import Base

class EvidenceType(str, enum.Enum):
    transaction = "transaction"
    vouching = "vouching"
    verification = "verification"

class TrustEdge(Base):
    __tablename__ = "trust_edges"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("entities.id"), nullable=False, index=True)
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("entities.id"), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(100), default="general")
    weight: Mapped[float] = mapped_column(Float, default=0.5)
    evidence_type: Mapped[EvidenceType] = mapped_column(Enum(EvidenceType), default=EvidenceType.transaction)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
```

`engine/models/source.py`:
```python
import uuid
from datetime import datetime
from sqlalchemy import Float, Integer, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from engine.db.session import Base

class SourceCredibility(Base):
    __tablename__ = "source_credibility"

    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("entities.id"), primary_key=True)
    credibility_score: Mapped[float] = mapped_column(Float, default=0.5)
    review_count: Mapped[int] = mapped_column(Integer, default=0)
    review_diversity: Mapped[float] = mapped_column(Float, default=0.0)
    accuracy_score: Mapped[float] = mapped_column(Float, default=0.5)
    account_age_days: Mapped[int] = mapped_column(Integer, default=0)
    flagged_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
```

- [ ] **Step 5: Create engine/models/__init__.py and commit**

```python
from engine.models.entity import Entity, EntityType
from engine.models.signal import Signal, Dimension, SignalType
from engine.models.score import DimensionScore, OverallScore, ScoreHistory, Tier, Trend
from engine.models.trust import TrustEdge, EvidenceType
from engine.models.source import SourceCredibility

__all__ = [
    "Entity", "EntityType",
    "Signal", "Dimension", "SignalType",
    "DimensionScore", "OverallScore", "ScoreHistory", "Tier", "Trend",
    "TrustEdge", "EvidenceType",
    "SourceCredibility",
]
```

```bash
git add engine/models/
git commit -m "feat: add all SQLAlchemy models — entity, signal, score, trust, source"
```

---

### Task 3: Docker Compose + Alembic

**Files:**
- Create: `docker-compose.yml`, `alembic.ini`, `engine/db/migrations/env.py`

- [ ] **Step 1: Create docker-compose.yml**

```yaml
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: trustrank
      POSTGRES_PASSWORD: trustrank
      POSTGRES_DB: trustrank
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  pgdata:
```

- [ ] **Step 2: Initialize Alembic**

```bash
pip install -e ".[dev]"
alembic init engine/db/migrations
```

Edit `alembic.ini` — set `sqlalchemy.url` to empty (we override in env.py).

Edit `engine/db/migrations/env.py` to import Base and all models:

```python
from engine.db.session import Base
from engine.models import *  # noqa: F403 — registers all models with Base

target_metadata = Base.metadata
# Set sqlalchemy.url from engine.config
from engine.config import settings
config.set_main_option("sqlalchemy.url", settings.database_url.replace("+asyncpg", ""))
```

- [ ] **Step 3: Start DB, generate and run migration**

```bash
docker compose up -d db redis
alembic revision --autogenerate -m "initial schema"
alembic upgrade head
```

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml alembic.ini engine/db/
git commit -m "infra: docker compose (postgres+redis) + alembic initial migration"
```

---

## Phase 2: Scoring Engine

### Task 4: Wilson Score Interval

**Files:**
- Create: `engine/scoring/__init__.py`, `engine/scoring/wilson.py`, `tests/test_wilson.py`

- [ ] **Step 1: Write failing tests**

`tests/test_wilson.py`:
```python
from engine.scoring.wilson import wilson_score_interval

def test_perfect_score_few_reviews():
    """5/5 from 3 reviews should score lower than 4.7/5 from 200."""
    few = wilson_score_interval(positive=3, total=3)
    many = wilson_score_interval(positive=188, total=200)  # 94% positive = ~4.7/5
    assert many > few

def test_no_reviews_returns_zero():
    assert wilson_score_interval(positive=0, total=0) == 0.0

def test_all_negative():
    result = wilson_score_interval(positive=0, total=100)
    assert result < 0.05

def test_50_50_split():
    result = wilson_score_interval(positive=50, total=100)
    assert 0.4 < result < 0.5

def test_from_star_rating():
    """Convert 1-5 star ratings to Wilson interval."""
    from engine.scoring.wilson import wilson_from_stars
    # 4.5 avg from 200 ratings → high score
    high = wilson_from_stars(mean_rating=4.5, count=200)
    # 4.5 avg from 5 ratings → lower score (less evidence)
    low = wilson_from_stars(mean_rating=4.5, count=5)
    assert high > low
    assert 0.0 <= high <= 1.0
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_wilson.py -v
```

- [ ] **Step 3: Implement wilson.py**

```python
import math

def wilson_score_interval(positive: int, total: int, z: float = 1.96) -> float:
    """Wilson score interval lower bound. Returns 0-1."""
    if total == 0:
        return 0.0
    p = positive / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return max(0.0, (centre - spread) / denominator)

def wilson_from_stars(mean_rating: float, count: int, z: float = 1.96) -> float:
    """Convert 1-5 star mean to Wilson lower bound (0-1 normalized)."""
    if count == 0:
        return 0.0
    p = (mean_rating - 1.0) / 4.0  # normalize to 0-1
    p = max(0.0, min(1.0, p))
    return wilson_score_interval(positive=round(p * count), total=count, z=z)
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_wilson.py -v
```

- [ ] **Step 5: Commit**

```bash
git add engine/scoring/ tests/test_wilson.py
git commit -m "feat(scoring): Wilson score interval with star rating normalization"
```

---

### Task 5: Beta-Binomial Model

**Files:**
- Create: `engine/scoring/bayesian.py`, `tests/test_bayesian.py`

- [ ] **Step 1: Write failing tests**

`tests/test_bayesian.py`:
```python
from engine.scoring.bayesian import BetaBinomial, DirichletMultinomial

class TestBetaBinomial:
    def test_prior_only(self):
        bb = BetaBinomial(alpha=2.0, beta=2.0)
        assert bb.mean() == 0.5

    def test_update_with_successes(self):
        bb = BetaBinomial(alpha=2.0, beta=2.0)
        bb.update(successes=8, failures=2)
        assert bb.mean() > 0.5
        assert bb.alpha == 10.0
        assert bb.beta == 4.0

    def test_confidence_increases_with_data(self):
        few = BetaBinomial(alpha=2.0, beta=2.0)
        few.update(successes=3, failures=1)
        many = BetaBinomial(alpha=2.0, beta=2.0)
        many.update(successes=80, failures=20)
        assert many.confidence() > few.confidence()

    def test_variance_decreases_with_data(self):
        few = BetaBinomial(alpha=2.0, beta=2.0)
        few.update(successes=3, failures=1)
        many = BetaBinomial(alpha=2.0, beta=2.0)
        many.update(successes=80, failures=20)
        assert many.variance() < few.variance()

class TestDirichletMultinomial:
    def test_uniform_prior(self):
        dm = DirichletMultinomial(alphas=[1.0, 1.0, 1.0, 1.0, 1.0])
        means = dm.mean_distribution()
        assert all(abs(m - 0.2) < 0.001 for m in means)

    def test_update_with_counts(self):
        dm = DirichletMultinomial(alphas=[1.0, 1.0, 1.0, 1.0, 1.0])
        dm.update(counts=[0, 0, 5, 10, 85])  # mostly 4-5 stars
        means = dm.mean_distribution()
        assert means[4] > means[0]  # 5-star > 1-star

    def test_weighted_mean_as_score(self):
        dm = DirichletMultinomial(alphas=[1.0, 1.0, 1.0, 1.0, 1.0])
        dm.update(counts=[0, 0, 0, 0, 100])  # all 5-star
        score = dm.weighted_mean()
        assert score > 4.5

    def test_bimodal_has_high_variance(self):
        bimodal = DirichletMultinomial(alphas=[1.0, 1.0, 1.0, 1.0, 1.0])
        bimodal.update(counts=[50, 0, 0, 0, 50])  # half 1-star, half 5-star
        uniform = DirichletMultinomial(alphas=[1.0, 1.0, 1.0, 1.0, 1.0])
        uniform.update(counts=[0, 0, 100, 0, 0])  # all 3-star
        assert bimodal.variance() > uniform.variance()
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_bayesian.py -v
```

- [ ] **Step 3: Implement bayesian.py**

```python
import math

class BetaBinomial:
    """Beta-Binomial model for binary outcomes."""

    def __init__(self, alpha: float = 2.0, beta: float = 2.0):
        self.alpha = alpha
        self.beta = beta

    def update(self, successes: int, failures: int) -> None:
        self.alpha += successes
        self.beta += failures

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def confidence(self) -> float:
        """0-1 confidence based on total evidence. Saturates toward 1."""
        total = self.alpha + self.beta - 4.0  # subtract prior
        return 1.0 - 1.0 / (1.0 + total / 20.0)  # sigmoid-like

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta, "mean": self.mean(), "confidence": self.confidence()}


class DirichletMultinomial:
    """Dirichlet-Multinomial model for 1-5 star rating distributions."""

    def __init__(self, alphas: list[float] | None = None):
        self.alphas = list(alphas) if alphas else [1.0, 1.0, 1.0, 1.0, 1.0]

    def update(self, counts: list[int]) -> None:
        for i in range(5):
            self.alphas[i] += counts[i]

    def mean_distribution(self) -> list[float]:
        total = sum(self.alphas)
        return [a / total for a in self.alphas]

    def weighted_mean(self) -> float:
        """Expected star rating (1-5 scale)."""
        dist = self.mean_distribution()
        return sum((i + 1) * p for i, p in enumerate(dist))

    def variance(self) -> float:
        """Variance of the weighted mean."""
        dist = self.mean_distribution()
        mean = self.weighted_mean()
        return sum(p * (i + 1 - mean) ** 2 for i, p in enumerate(dist))

    def confidence(self) -> float:
        total = sum(self.alphas) - 5.0  # subtract prior
        return 1.0 - 1.0 / (1.0 + total / 30.0)

    def to_dict(self) -> dict:
        return {"alphas": self.alphas, "mean": self.weighted_mean(), "variance": self.variance(), "confidence": self.confidence()}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_bayesian.py -v
```

- [ ] **Step 5: Commit**

```bash
git add engine/scoring/bayesian.py tests/test_bayesian.py
git commit -m "feat(scoring): Beta-Binomial + Dirichlet-Multinomial models"
```

---

### Task 6: Temporal Decay + Velocity

**Files:**
- Create: `engine/scoring/decay.py`, `tests/test_decay.py`

- [ ] **Step 1: Write failing tests**

`tests/test_decay.py`:
```python
from datetime import datetime, timedelta, timezone
from engine.scoring.decay import compute_decay, compute_velocity

def test_recent_signal_full_weight():
    now = datetime.now(timezone.utc)
    weight = compute_decay(signal_time=now, current_time=now, half_life_days=90)
    assert abs(weight - 1.0) < 0.01

def test_half_life_halves_weight():
    now = datetime.now(timezone.utc)
    past = now - timedelta(days=90)
    weight = compute_decay(signal_time=past, current_time=now, half_life_days=90)
    assert abs(weight - 0.5) < 0.01

def test_very_old_signal_near_zero():
    now = datetime.now(timezone.utc)
    ancient = now - timedelta(days=900)
    weight = compute_decay(signal_time=ancient, current_time=now, half_life_days=90)
    assert weight < 0.01

def test_velocity_positive_slope():
    now = datetime.now(timezone.utc)
    # Scores improving over 7 days: 2.0, 3.0, 4.0, 5.0
    scores = [(now - timedelta(days=6), 2.0), (now - timedelta(days=4), 3.0),
              (now - timedelta(days=2), 4.0), (now, 5.0)]
    slope = compute_velocity(scores, window_days=7)
    assert slope > 0

def test_velocity_stable():
    now = datetime.now(timezone.utc)
    scores = [(now - timedelta(days=i), 3.0) for i in range(7)]
    slope = compute_velocity(scores, window_days=7)
    assert abs(slope) < 0.01

def test_velocity_empty_returns_zero():
    assert compute_velocity([], window_days=7) == 0.0
```

- [ ] **Step 2: Run tests — expect FAIL**

- [ ] **Step 3: Implement decay.py**

```python
import math
from datetime import datetime, timezone

def compute_decay(signal_time: datetime, current_time: datetime, half_life_days: float) -> float:
    """Exponential decay weight based on signal age."""
    age_days = (current_time - signal_time).total_seconds() / 86400.0
    if age_days < 0:
        return 1.0
    return math.exp(-math.log(2) * age_days / half_life_days)

def compute_velocity(scores: list[tuple[datetime, float]], window_days: int = 7) -> float:
    """Linear regression slope of scores within window. Returns slope per day."""
    if len(scores) < 2:
        return 0.0
    now = max(t for t, _ in scores)
    cutoff = now - __import__("datetime").timedelta(days=window_days)
    recent = [(t, v) for t, v in scores if t >= cutoff]
    if len(recent) < 2:
        return 0.0
    # Convert to days-from-start
    t0 = min(t for t, _ in recent)
    xs = [(t - t0).total_seconds() / 86400.0 for t, _ in recent]
    ys = [v for _, v in recent]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add engine/scoring/decay.py tests/test_decay.py
git commit -m "feat(scoring): temporal decay with half-life + velocity regression"
```

---

### Task 7: Dimension Aggregator + Tier Classification

**Files:**
- Create: `engine/scoring/aggregator.py`, `tests/test_aggregator.py`

- [ ] **Step 1: Write failing tests**

`tests/test_aggregator.py`:
```python
from engine.scoring.aggregator import aggregate_dimensions, classify_tier

def test_equal_weights():
    dims = {"quality": 4.0, "reliability": 4.0, "responsiveness": 4.0, "trust": 4.0}
    weights = {"quality": 0.25, "reliability": 0.25, "responsiveness": 0.25, "trust": 0.25}
    assert aggregate_dimensions(dims, weights) == 4.0

def test_weighted_aggregation():
    dims = {"quality": 5.0, "reliability": 3.0, "responsiveness": 4.0, "trust": 4.0}
    weights = {"quality": 0.35, "reliability": 0.25, "responsiveness": 0.20, "trust": 0.20}
    result = aggregate_dimensions(dims, weights)
    expected = (5.0*0.35 + 3.0*0.25 + 4.0*0.20 + 4.0*0.20) / 1.0
    assert abs(result - expected) < 0.01

def test_tier_platinum():
    assert classify_tier(score=4.7, signal_count=150, confidence=0.9) == "platinum"

def test_tier_gold():
    assert classify_tier(score=4.2, signal_count=60, confidence=0.8) == "gold"

def test_tier_insufficient_signals():
    """High score but too few signals stays bronze."""
    assert classify_tier(score=4.8, signal_count=3, confidence=0.9) == "bronze"

def test_tier_insufficient_confidence():
    """High score + enough signals but low confidence stays silver."""
    assert classify_tier(score=4.8, signal_count=200, confidence=0.5) == "silver"

def test_tier_untrusted():
    assert classify_tier(score=2.0, signal_count=50, confidence=0.8) == "untrusted"
```

- [ ] **Step 2: Run tests — expect FAIL**

- [ ] **Step 3: Implement aggregator.py**

```python
from engine.config import settings

def aggregate_dimensions(dimension_scores: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """Weighted combination of dimension scores."""
    w = weights or settings.default_dimension_weights
    total_weight = sum(w.get(d, 0) for d in dimension_scores)
    if total_weight == 0:
        return 3.0
    return sum(dimension_scores[d] * w.get(d, 0) for d in dimension_scores) / total_weight

def classify_tier(score: float, signal_count: int, confidence: float) -> str:
    """Classify into tier based on score + signal count + confidence thresholds."""
    tiers = [
        ("platinum", settings.tier_platinum_score, settings.tier_platinum_signals, settings.tier_platinum_confidence),
        ("gold", settings.tier_gold_score, settings.tier_gold_signals, settings.tier_gold_confidence),
        ("silver", settings.tier_silver_score, settings.tier_silver_signals, settings.tier_silver_confidence),
        ("bronze", settings.tier_bronze_score, settings.tier_bronze_signals, settings.tier_bronze_confidence),
    ]
    for name, min_score, min_signals, min_conf in tiers:
        if score >= min_score and signal_count >= min_signals and confidence >= min_conf:
            return name
    if score < settings.tier_bronze_score:
        return "untrusted"
    return "bronze"
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add engine/scoring/aggregator.py tests/test_aggregator.py
git commit -m "feat(scoring): dimension aggregation + tier classification"
```

---

## Phase 3: Manipulation Detection

### Task 8: CUSUM Burst Detection

**Files:**
- Create: `engine/detection/__init__.py`, `engine/detection/burst.py`, `tests/test_burst.py`

- [ ] **Step 1: Write failing tests**

`tests/test_burst.py`:
```python
from datetime import datetime, timedelta, timezone
from engine.detection.burst import CUSUMDetector

def test_no_burst_normal_rate():
    """Steady stream of signals should not trigger."""
    now = datetime.now(timezone.utc)
    times = [now - timedelta(hours=i * 12) for i in range(20)]  # 1 per 12h for 10 days
    detector = CUSUMDetector(allowance_factor=0.5, threshold_factor=5.0)
    result = detector.detect(signal_times=times, current_time=now)
    assert not result.burst_detected

def test_burst_detected():
    """Sudden spike of 15 signals in 2 hours after baseline of 1/day."""
    now = datetime.now(timezone.utc)
    baseline = [now - timedelta(days=i) for i in range(1, 31)]  # 1/day for 30 days
    burst = [now - timedelta(minutes=i * 8) for i in range(15)]  # 15 in 2h
    times = sorted(baseline + burst)
    detector = CUSUMDetector(allowance_factor=0.5, threshold_factor=5.0)
    result = detector.detect(signal_times=times, current_time=now)
    assert result.burst_detected
    assert result.burst_window_start is not None

def test_burst_returns_dampened_indices():
    """Burst result should identify which signals to dampen."""
    now = datetime.now(timezone.utc)
    baseline = [now - timedelta(days=i) for i in range(1, 31)]
    burst = [now - timedelta(minutes=i * 5) for i in range(20)]
    times = sorted(baseline + burst)
    detector = CUSUMDetector(allowance_factor=0.5, threshold_factor=5.0)
    result = detector.detect(signal_times=times, current_time=now)
    assert result.burst_detected
    assert len(result.dampened_indices) > 0

def test_too_few_signals():
    """Less than 5 signals — not enough history for detection."""
    now = datetime.now(timezone.utc)
    times = [now - timedelta(hours=i) for i in range(3)]
    detector = CUSUMDetector()
    result = detector.detect(signal_times=times, current_time=now)
    assert not result.burst_detected
```

- [ ] **Step 2: Run tests — expect FAIL**

- [ ] **Step 3: Implement burst.py**

```python
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

@dataclass
class BurstResult:
    burst_detected: bool = False
    burst_window_start: datetime | None = None
    burst_window_end: datetime | None = None
    dampened_indices: list[int] = field(default_factory=list)
    baseline_rate: float = 0.0
    observed_rate: float = 0.0

class CUSUMDetector:
    def __init__(self, allowance_factor: float = 0.5, threshold_factor: float = 5.0):
        self.allowance_factor = allowance_factor
        self.threshold_factor = threshold_factor

    def detect(self, signal_times: list[datetime], current_time: datetime) -> BurstResult:
        if len(signal_times) < 5:
            return BurstResult()

        sorted_times = sorted(signal_times)
        # Compute inter-arrival times in hours
        intervals = []
        for i in range(1, len(sorted_times)):
            dt = (sorted_times[i] - sorted_times[i - 1]).total_seconds() / 3600.0
            intervals.append(max(dt, 0.001))  # avoid zero

        if len(intervals) < 4:
            return BurstResult()

        # Baseline: mean and std of inter-arrival times (excluding last 20%)
        split = max(3, int(len(intervals) * 0.8))
        baseline_intervals = intervals[:split]
        mean_interval = sum(baseline_intervals) / len(baseline_intervals)
        variance = sum((x - mean_interval) ** 2 for x in baseline_intervals) / len(baseline_intervals)
        std_interval = math.sqrt(variance) if variance > 0 else mean_interval * 0.5

        # CUSUM on arrival rate (inverse of interval)
        # We detect unusually SHORT intervals (high arrival rate)
        k = self.allowance_factor * std_interval
        h = self.threshold_factor * std_interval

        cusum = 0.0
        burst_start_idx = None
        for i, interval in enumerate(intervals):
            # Deviation: expected interval minus actual (positive when interval is shorter than expected)
            deviation = mean_interval - interval - k
            cusum = max(0.0, cusum + deviation)
            if cusum > h and burst_start_idx is None:
                burst_start_idx = i

        if burst_start_idx is None:
            return BurstResult(baseline_rate=1.0 / mean_interval if mean_interval > 0 else 0)

        # Identify dampened signals: all signals from burst_start onward
        burst_window_start = sorted_times[burst_start_idx]
        dampened = [i for i, t in enumerate(sorted_times) if t >= burst_window_start]

        recent_intervals = intervals[burst_start_idx:]
        observed_rate = len(recent_intervals) / (sum(recent_intervals) if sum(recent_intervals) > 0 else 1)

        return BurstResult(
            burst_detected=True,
            burst_window_start=burst_window_start,
            burst_window_end=sorted_times[-1],
            dampened_indices=dampened,
            baseline_rate=1.0 / mean_interval if mean_interval > 0 else 0,
            observed_rate=observed_rate,
        )
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add engine/detection/ tests/test_burst.py
git commit -m "feat(detection): CUSUM burst detection on signal arrival rate"
```

---

### Task 9: Coordination Detection

**Files:**
- Create: `engine/detection/coordination.py`, `tests/test_coordination.py`

- [ ] **Step 1: Write failing tests**

`tests/test_coordination.py`:
```python
from datetime import datetime, timedelta, timezone
from engine.detection.coordination import CoordinationDetector, CoordinationResult

def test_no_coordination_diverse_text():
    now = datetime.now(timezone.utc)
    signals = [
        {"text": "Great food, loved the pasta and wine selection", "time": now - timedelta(hours=i * 24)}
        for i in range(5)
    ]
    # Make texts actually diverse
    signals[1]["text"] = "Terrible service, waited 45 minutes for appetizers"
    signals[2]["text"] = "Average experience, nothing special but not bad either"
    signals[3]["text"] = "The ambiance was wonderful, perfect for a date night"
    signals[4]["text"] = "Overpriced for the quality, would not recommend to friends"
    detector = CoordinationDetector()
    result = detector.detect(signals)
    assert not result.coordinated

def test_coordination_similar_text():
    now = datetime.now(timezone.utc)
    # Same text with minor variations — clearly coordinated
    base = "Amazing product absolutely love it highly recommend to everyone"
    signals = [
        {"text": base, "time": now - timedelta(minutes=i * 10)}
        for i in range(8)
    ]
    signals[1]["text"] = "Amazing product absolutely love this highly recommend to all"
    signals[2]["text"] = "Amazing item absolutely love it highly recommend to everyone"
    signals[3]["text"] = "Amazing product totally love it highly recommend to everyone"
    detector = CoordinationDetector(similarity_threshold=0.85)
    result = detector.detect(signals)
    assert result.coordinated
    assert result.cluster_size >= 3

def test_regular_timing_detected():
    now = datetime.now(timezone.utc)
    # Suspiciously regular: exactly every 30 minutes
    signals = [
        {"text": f"Review number {i} for this place", "time": now - timedelta(minutes=i * 30)}
        for i in range(10)
    ]
    detector = CoordinationDetector(cv_threshold=0.5)
    result = detector.detect(signals)
    assert result.timing_suspicious

def test_empty_signals():
    detector = CoordinationDetector()
    result = detector.detect([])
    assert not result.coordinated
```

- [ ] **Step 2: Run tests — expect FAIL**

- [ ] **Step 3: Implement coordination.py**

```python
import math
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@dataclass
class CoordinationResult:
    coordinated: bool = False
    timing_suspicious: bool = False
    cluster_size: int = 0
    max_similarity: float = 0.0
    cv_score: float = 1.0

class CoordinationDetector:
    def __init__(self, similarity_threshold: float = 0.85, cv_threshold: float = 0.5, min_cluster: int = 3):
        self.similarity_threshold = similarity_threshold
        self.cv_threshold = cv_threshold
        self.min_cluster = min_cluster

    def detect(self, signals: list[dict]) -> CoordinationResult:
        if len(signals) < self.min_cluster:
            return CoordinationResult()

        text_result = self._check_text_similarity(signals)
        timing_result = self._check_timing(signals)

        return CoordinationResult(
            coordinated=text_result["coordinated"] or (text_result["max_similarity"] > 0.7 and timing_result["suspicious"]),
            timing_suspicious=timing_result["suspicious"],
            cluster_size=text_result["cluster_size"],
            max_similarity=text_result["max_similarity"],
            cv_score=timing_result["cv"],
        )

    def _check_text_similarity(self, signals: list[dict]) -> dict:
        texts = [s.get("text", "") for s in signals]
        texts = [t for t in texts if t and len(t) > 10]
        if len(texts) < self.min_cluster:
            return {"coordinated": False, "cluster_size": 0, "max_similarity": 0.0}

        vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        tfidf = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf)

        # Find largest cluster of similar texts (greedy)
        n = len(texts)
        visited = set()
        largest_cluster = 0
        max_sim = 0.0

        for i in range(n):
            if i in visited:
                continue
            cluster = {i}
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= self.similarity_threshold:
                    cluster.add(j)
                    max_sim = max(max_sim, sim_matrix[i, j])
            if len(cluster) >= self.min_cluster:
                largest_cluster = max(largest_cluster, len(cluster))
                visited.update(cluster)

        # Also track overall max similarity
        np.fill_diagonal(sim_matrix, 0)
        overall_max = float(sim_matrix.max()) if sim_matrix.size > 0 else 0.0
        max_sim = max(max_sim, overall_max)

        return {
            "coordinated": largest_cluster >= self.min_cluster,
            "cluster_size": largest_cluster,
            "max_similarity": max_sim,
        }

    def _check_timing(self, signals: list[dict]) -> dict:
        times = sorted([s["time"] for s in signals if "time" in s])
        if len(times) < 3:
            return {"suspicious": False, "cv": 1.0}

        intervals = [(times[i] - times[i - 1]).total_seconds() for i in range(1, len(times))]
        intervals = [max(x, 0.001) for x in intervals]

        mean = sum(intervals) / len(intervals)
        if mean == 0:
            return {"suspicious": True, "cv": 0.0}
        variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
        std = math.sqrt(variance)
        cv = std / mean  # coefficient of variation

        return {"suspicious": cv < self.cv_threshold, "cv": cv}
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add engine/detection/coordination.py tests/test_coordination.py
git commit -m "feat(detection): coordination detection — TF-IDF clustering + Poisson timing"
```

---

### Task 10: Source Credibility Scoring

**Files:**
- Create: `engine/detection/credibility.py`, `tests/test_credibility.py`

- [ ] **Step 1: Write failing tests**

`tests/test_credibility.py`:
```python
from engine.detection.credibility import compute_credibility

def test_new_account_low_credibility():
    score = compute_credibility(review_count=1, review_diversity=0.0, accuracy=0.5, account_age_days=1, flagged_ratio=0.0)
    assert score < 0.3

def test_established_reviewer_high_credibility():
    score = compute_credibility(review_count=100, review_diversity=0.9, accuracy=0.85, account_age_days=365, flagged_ratio=0.0)
    assert score > 0.8

def test_flagged_reviews_reduce_credibility():
    clean = compute_credibility(review_count=50, review_diversity=0.7, accuracy=0.8, account_age_days=180, flagged_ratio=0.0)
    flagged = compute_credibility(review_count=50, review_diversity=0.7, accuracy=0.8, account_age_days=180, flagged_ratio=0.5)
    assert flagged < clean

def test_single_entity_reviewer_low_diversity():
    """Reviewer who only reviews one entity — suspicious."""
    score = compute_credibility(review_count=50, review_diversity=0.0, accuracy=0.8, account_age_days=180, flagged_ratio=0.0)
    assert score < 0.6

def test_score_bounds():
    """Credibility should always be between 0 and 1."""
    score = compute_credibility(review_count=0, review_diversity=0.0, accuracy=0.0, account_age_days=0, flagged_ratio=1.0)
    assert 0.0 <= score <= 1.0
    score = compute_credibility(review_count=1000, review_diversity=1.0, accuracy=1.0, account_age_days=3650, flagged_ratio=0.0)
    assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run tests — expect FAIL**

- [ ] **Step 3: Implement credibility.py**

```python
def _normalize(value: float, target: float) -> float:
    """Sigmoid-like normalization: value/target saturating toward 1."""
    if target <= 0:
        return 0.0
    return min(1.0, value / target)

def compute_credibility(
    review_count: int,
    review_diversity: float,
    accuracy: float,
    account_age_days: int,
    flagged_ratio: float,
) -> float:
    """
    Compute source credibility score (0-1).

    Weights: review_count=0.30, diversity=0.25, accuracy=0.25, age=0.10, flags=0.10
    """
    score = (
        0.30 * _normalize(review_count, 50) +
        0.25 * max(0.0, min(1.0, review_diversity)) +
        0.25 * max(0.0, min(1.0, accuracy)) +
        0.10 * _normalize(account_age_days, 180) +
        0.10 * (1.0 - max(0.0, min(1.0, flagged_ratio)))
    )
    return max(0.0, min(1.0, score))
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add engine/detection/credibility.py tests/test_credibility.py
git commit -m "feat(detection): source credibility scoring with 5-factor model"
```

---

### Task 11: Reciprocal Network Detection + Detection Manager

**Files:**
- Create: `engine/detection/reciprocal.py`, `engine/detection/manager.py`, `tests/test_reciprocal.py`

- [ ] **Step 1: Write failing tests**

`tests/test_reciprocal.py`:
```python
from engine.detection.reciprocal import ReciprocalDetector

def test_no_collusion_independent_reviewers():
    # Entity A's reviewers: {1,2,3}, Entity B's reviewers: {4,5,6}
    entity_reviews = {
        "A": [{"source": "1", "value": 4.0}, {"source": "2", "value": 5.0}, {"source": "3", "value": 4.0}],
        "B": [{"source": "4", "value": 3.0}, {"source": "5", "value": 4.0}, {"source": "6", "value": 5.0}],
    }
    detector = ReciprocalDetector()
    result = detector.detect(entity_reviews)
    assert len(result.flagged_pairs) == 0

def test_collusion_detected():
    # Same reviewers giving both entities high scores
    entity_reviews = {
        "A": [{"source": "1", "value": 5.0}, {"source": "2", "value": 5.0}, {"source": "3", "value": 5.0},
              {"source": "4", "value": 4.0}, {"source": "5", "value": 5.0}],
        "B": [{"source": "1", "value": 5.0}, {"source": "2", "value": 4.0}, {"source": "3", "value": 5.0},
              {"source": "4", "value": 5.0}, {"source": "6", "value": 3.0}],
    }
    detector = ReciprocalDetector(jaccard_threshold=0.3, sentiment_threshold=0.7)
    result = detector.detect(entity_reviews)
    assert len(result.flagged_pairs) > 0
    assert ("A", "B") in result.flagged_pairs or ("B", "A") in result.flagged_pairs

def test_single_entity_no_pairs():
    entity_reviews = {"A": [{"source": "1", "value": 5.0}]}
    detector = ReciprocalDetector()
    result = detector.detect(entity_reviews)
    assert len(result.flagged_pairs) == 0
```

- [ ] **Step 2: Run tests — expect FAIL**

- [ ] **Step 3: Implement reciprocal.py**

```python
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np

@dataclass
class ReciprocalResult:
    flagged_pairs: list[tuple[str, str]] = field(default_factory=list)

class ReciprocalDetector:
    def __init__(self, jaccard_threshold: float = 0.3, sentiment_threshold: float = 0.7):
        self.jaccard_threshold = jaccard_threshold
        self.sentiment_threshold = sentiment_threshold

    def detect(self, entity_reviews: dict[str, list[dict]]) -> ReciprocalResult:
        entities = list(entity_reviews.keys())
        if len(entities) < 2:
            return ReciprocalResult()

        flagged = []
        for a, b in combinations(entities, 2):
            reviews_a = entity_reviews[a]
            reviews_b = entity_reviews[b]
            sources_a = {r["source"] for r in reviews_a}
            sources_b = {r["source"] for r in reviews_b}

            # Jaccard similarity of reviewer sets
            intersection = sources_a & sources_b
            union = sources_a | sources_b
            if not union:
                continue
            jaccard = len(intersection) / len(union)

            if jaccard < self.jaccard_threshold:
                continue

            # Sentiment correlation among shared reviewers
            if len(intersection) < 2:
                continue
            shared = sorted(intersection)
            vals_a = {r["source"]: r["value"] for r in reviews_a}
            vals_b = {r["source"]: r["value"] for r in reviews_b}
            scores_a = [vals_a[s] for s in shared if s in vals_a]
            scores_b = [vals_b[s] for s in shared if s in vals_b]

            if len(scores_a) < 2:
                continue
            corr = np.corrcoef(scores_a, scores_b)[0, 1]
            if np.isnan(corr):
                continue

            if corr >= self.sentiment_threshold:
                flagged.append((a, b))

        return ReciprocalResult(flagged_pairs=flagged)
```

- [ ] **Step 4: Implement detection manager**

`engine/detection/manager.py`:
```python
from dataclasses import dataclass, field
from datetime import datetime
from engine.detection.burst import CUSUMDetector, BurstResult
from engine.detection.coordination import CoordinationDetector, CoordinationResult
from engine.detection.credibility import compute_credibility
from engine.config import settings

@dataclass
class DetectionResult:
    burst: BurstResult = field(default_factory=BurstResult)
    coordination: CoordinationResult = field(default_factory=CoordinationResult)
    source_credibility: float = 0.5
    dampening_factor: float = 1.0
    reasons: list[str] = field(default_factory=list)

class DetectionManager:
    def __init__(self):
        self.burst_detector = CUSUMDetector(
            allowance_factor=settings.cusum_allowance_factor,
            threshold_factor=settings.cusum_threshold_factor,
        )
        self.coordination_detector = CoordinationDetector(
            similarity_threshold=settings.coordination_similarity_threshold,
            cv_threshold=settings.coordination_cv_threshold,
        )

    def run_all(
        self,
        signal_times: list[datetime],
        signal_texts: list[dict],
        source_stats: dict,
        current_time: datetime,
    ) -> DetectionResult:
        result = DetectionResult()

        # Burst detection
        result.burst = self.burst_detector.detect(signal_times, current_time)
        if result.burst.burst_detected:
            result.dampening_factor *= 0.3
            result.reasons.append(f"burst: {len(result.burst.dampened_indices)} signals in burst window")

        # Coordination detection
        result.coordination = self.coordination_detector.detect(signal_texts)
        if result.coordination.coordinated:
            result.dampening_factor *= 0.4
            result.reasons.append(f"coordination: cluster of {result.coordination.cluster_size}, sim={result.coordination.max_similarity:.2f}")

        # Source credibility
        result.source_credibility = compute_credibility(**source_stats)
        result.dampening_factor *= max(0.3, result.source_credibility)

        return result
```

- [ ] **Step 5: Run tests + commit**

```bash
pytest tests/test_reciprocal.py -v
git add engine/detection/ tests/test_reciprocal.py
git commit -m "feat(detection): reciprocal network detection + detection manager"
```

---

## Phase 4: Trust Graph

### Task 12: Transitive Trust Propagation

**Files:**
- Create: `engine/graph/__init__.py`, `engine/graph/trust.py`, `tests/test_trust_graph.py`

- [ ] **Step 1: Write failing tests**

`tests/test_trust_graph.py`:
```python
import networkx as nx
from engine.graph.trust import TrustGraph

def test_direct_trust():
    g = TrustGraph(damping=0.7, max_hops=2)
    g.add_edge("A", "B", weight=0.9, category="general")
    assert g.direct_trust("A", "B") == 0.9

def test_transitive_trust_one_hop():
    g = TrustGraph(damping=0.7, max_hops=2)
    g.add_edge("A", "B", weight=0.9, category="general")
    g.add_edge("B", "C", weight=0.8, category="general")
    trust = g.indirect_trust("A", "C", category="general")
    assert abs(trust - 0.9 * 0.8 * 0.7) < 0.01

def test_transitive_trust_max_hops():
    g = TrustGraph(damping=0.7, max_hops=2)
    g.add_edge("A", "B", weight=0.9, category="general")
    g.add_edge("B", "C", weight=0.8, category="general")
    g.add_edge("C", "D", weight=0.7, category="general")
    trust = g.indirect_trust("A", "D", category="general")
    assert trust == 0.0  # 3 hops > max_hops=2

def test_multiple_paths_takes_max():
    g = TrustGraph(damping=0.7, max_hops=2)
    g.add_edge("A", "B", weight=0.5, category="general")
    g.add_edge("A", "C", weight=0.9, category="general")
    g.add_edge("B", "D", weight=0.8, category="general")
    g.add_edge("C", "D", weight=0.8, category="general")
    trust = g.indirect_trust("A", "D", category="general")
    path_ab = 0.5 * 0.8 * 0.7
    path_ac = 0.9 * 0.8 * 0.7
    assert abs(trust - max(path_ab, path_ac)) < 0.01

def test_category_filtering():
    g = TrustGraph(damping=0.7, max_hops=2)
    g.add_edge("A", "B", weight=0.9, category="food")
    g.add_edge("B", "C", weight=0.8, category="tech")
    trust = g.indirect_trust("A", "C", category="food")
    assert trust == 0.0  # category mismatch blocks propagation

def test_trust_bonus():
    g = TrustGraph(damping=0.7, max_hops=2)
    g.add_edge("X", "A", weight=0.9, category="general")
    g.add_edge("Y", "A", weight=0.8, category="general")
    bonus = g.compute_trust_bonus("A", weight=0.15)
    assert bonus > 0
```

- [ ] **Step 2: Run tests — expect FAIL**

- [ ] **Step 3: Implement trust.py**

```python
import networkx as nx
from collections import defaultdict

class TrustGraph:
    def __init__(self, damping: float = 0.7, max_hops: int = 2):
        self.graph = nx.DiGraph()
        self.damping = damping
        self.max_hops = max_hops

    def add_edge(self, source: str, target: str, weight: float, category: str = "general") -> None:
        self.graph.add_edge(source, target, weight=weight, category=category)

    def direct_trust(self, source: str, target: str) -> float:
        if self.graph.has_edge(source, target):
            return self.graph[source][target]["weight"]
        return 0.0

    def indirect_trust(self, source: str, target: str, category: str = "general") -> float:
        if source == target:
            return 1.0
        if source not in self.graph or target not in self.graph:
            return 0.0

        max_trust = 0.0
        # BFS with depth limit
        queue = [(source, 1.0, 0)]  # (node, accumulated_trust, hops)
        visited = {source}

        while queue:
            current, acc_trust, hops = queue.pop(0)
            if hops >= self.max_hops:
                continue
            for neighbor in self.graph.successors(current):
                edge = self.graph[current][neighbor]
                edge_cat = edge.get("category", "general")
                if edge_cat != "general" and edge_cat != category:
                    continue
                new_trust = acc_trust * edge["weight"] * self.damping
                if neighbor == target:
                    max_trust = max(max_trust, new_trust)
                elif neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, new_trust, hops + 1))

        return max_trust

    def compute_trust_bonus(self, entity_id: str, weight: float = 0.15) -> float:
        if entity_id not in self.graph:
            return 0.0
        incoming = []
        for pred in self.graph.predecessors(entity_id):
            edge = self.graph[pred][entity_id]
            incoming.append(edge["weight"])
        if not incoming:
            return 0.0
        return (sum(incoming) / len(incoming)) * weight

    def get_subgraph(self, entity_id: str, hops: int = 2) -> dict:
        if entity_id not in self.graph:
            return {"nodes": [], "edges": []}
        nodes = {entity_id}
        edges = []
        frontier = {entity_id}
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                for succ in self.graph.successors(node):
                    nodes.add(succ)
                    edges.append({"source": node, "target": succ, **self.graph[node][succ]})
                    next_frontier.add(succ)
                for pred in self.graph.predecessors(node):
                    nodes.add(pred)
                    edges.append({"source": pred, "target": node, **self.graph[pred][node]})
                    next_frontier.add(pred)
            frontier = next_frontier - nodes | next_frontier
        return {"nodes": list(nodes), "edges": edges}
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add engine/graph/ tests/test_trust_graph.py
git commit -m "feat(graph): transitive trust propagation with category filtering"
```

---

### Task 13: Community Detection + Influence Scoring

**Files:**
- Create: `engine/graph/community.py`, `engine/graph/influence.py`, `tests/test_community.py`, `tests/test_influence.py`

- [ ] **Step 1: Write failing tests**

`tests/test_community.py`:
```python
from engine.graph.community import detect_communities
import networkx as nx

def test_two_clear_communities():
    G = nx.Graph()
    # Community 1: tightly connected
    for i in range(5):
        for j in range(i + 1, 5):
            G.add_edge(f"a{i}", f"a{j}", weight=0.9)
    # Community 2: tightly connected
    for i in range(5):
        for j in range(i + 1, 5):
            G.add_edge(f"b{i}", f"b{j}", weight=0.9)
    # Weak bridge
    G.add_edge("a0", "b0", weight=0.1)
    communities = detect_communities(G)
    # Should have at least 2 communities
    assert len(set(communities.values())) >= 2

def test_single_node():
    G = nx.Graph()
    G.add_node("alone")
    communities = detect_communities(G)
    assert "alone" in communities

def test_same_community_flag():
    from engine.graph.community import are_same_community
    communities = {"A": 0, "B": 0, "C": 1}
    assert are_same_community("A", "B", communities)
    assert not are_same_community("A", "C", communities)
```

`tests/test_influence.py`:
```python
import networkx as nx
from engine.graph.influence import katz_centrality

def test_hub_has_highest_influence():
    G = nx.DiGraph()
    for i in range(10):
        G.add_edge("hub", f"node{i}", weight=0.8)
    scores = katz_centrality(G, alpha=0.1)
    assert scores["hub"] == max(scores.values())

def test_isolated_node_low_influence():
    G = nx.DiGraph()
    G.add_node("isolated")
    G.add_edge("A", "B", weight=0.5)
    scores = katz_centrality(G, alpha=0.1)
    assert scores.get("isolated", 0) <= scores.get("A", 0)

def test_empty_graph():
    G = nx.DiGraph()
    scores = katz_centrality(G, alpha=0.1)
    assert scores == {}
```

- [ ] **Step 2: Run tests — expect FAIL**

- [ ] **Step 3: Implement community.py and influence.py**

`engine/graph/community.py`:
```python
import networkx as nx
import community as community_louvain  # python-community-louvain

def detect_communities(G: nx.Graph) -> dict[str, int]:
    if len(G.nodes) == 0:
        return {}
    if len(G.nodes) == 1:
        return {list(G.nodes)[0]: 0}
    if len(G.edges) == 0:
        return {n: i for i, n in enumerate(G.nodes)}
    return community_louvain.best_partition(G, weight="weight")

def are_same_community(a: str, b: str, communities: dict[str, int]) -> bool:
    return communities.get(a) == communities.get(b) and a in communities
```

`engine/graph/influence.py`:
```python
import networkx as nx

def katz_centrality(G: nx.DiGraph, alpha: float = 0.1, max_iter: int = 100, tol: float = 1e-6) -> dict[str, float]:
    if len(G.nodes) == 0:
        return {}
    try:
        return nx.katz_centrality(G, alpha=alpha, max_iter=max_iter, tol=tol, weight="weight")
    except nx.NetworkXError:
        # If alpha is too large for convergence, use a smaller value
        return nx.katz_centrality(G, alpha=alpha * 0.5, max_iter=max_iter, tol=tol, weight="weight")
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add engine/graph/ tests/test_community.py tests/test_influence.py
git commit -m "feat(graph): Louvain community detection + Katz centrality influence"
```

---

## Phase 5: Temporal Analysis

### Task 14: Trend Detection + Regime Change (BOCPD)

**Files:**
- Create: `engine/temporal/__init__.py`, `engine/temporal/trend.py`, `engine/temporal/changepoint.py`, `tests/test_trend.py`, `tests/test_changepoint.py`

- [ ] **Step 1: Write failing tests**

`tests/test_trend.py`:
```python
from datetime import datetime, timedelta, timezone
from engine.temporal.trend import detect_trend

def test_improving_trend():
    now = datetime.now(timezone.utc)
    scores = [(now - timedelta(days=29 - i), 3.0 + i * 0.1) for i in range(30)]
    trend, slope = detect_trend(scores, window_days=30, threshold=0.02)
    assert trend == "improving"
    assert slope > 0.02

def test_declining_trend():
    now = datetime.now(timezone.utc)
    scores = [(now - timedelta(days=29 - i), 5.0 - i * 0.1) for i in range(30)]
    trend, slope = detect_trend(scores, window_days=30, threshold=0.02)
    assert trend == "declining"
    assert slope < -0.02

def test_stable_trend():
    now = datetime.now(timezone.utc)
    scores = [(now - timedelta(days=29 - i), 4.0) for i in range(30)]
    trend, slope = detect_trend(scores, window_days=30, threshold=0.02)
    assert trend == "stable"

def test_empty_scores():
    trend, slope = detect_trend([], window_days=30, threshold=0.02)
    assert trend == "stable"
    assert slope == 0.0
```

`tests/test_changepoint.py`:
```python
from engine.temporal.changepoint import detect_regime_change

def test_no_change_stable_data():
    values = [4.0 + (i % 3) * 0.1 for i in range(50)]  # slight noise around 4.0
    result = detect_regime_change(values, threshold=0.8)
    assert not result.change_detected

def test_clear_regime_change():
    # 30 values around 4.0, then 20 values around 2.0
    values = [4.0 + (i % 3) * 0.1 for i in range(30)] + [2.0 + (i % 3) * 0.1 for i in range(20)]
    result = detect_regime_change(values, threshold=0.8)
    assert result.change_detected
    assert 25 <= result.change_index <= 35  # near the transition

def test_too_few_values():
    result = detect_regime_change([4.0, 3.0], threshold=0.8)
    assert not result.change_detected
```

- [ ] **Step 2: Run tests — expect FAIL**

- [ ] **Step 3: Implement trend.py**

```python
from datetime import datetime, timedelta

def detect_trend(
    scores: list[tuple[datetime, float]],
    window_days: int = 30,
    threshold: float = 0.02,
) -> tuple[str, float]:
    if len(scores) < 2:
        return "stable", 0.0

    now = max(t for t, _ in scores)
    cutoff = now - timedelta(days=window_days)
    recent = [(t, v) for t, v in scores if t >= cutoff]
    if len(recent) < 2:
        return "stable", 0.0

    t0 = min(t for t, _ in recent)
    xs = [(t - t0).total_seconds() / 86400.0 for t, _ in recent]
    ys = [v for _, v in recent]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return "stable", 0.0
    slope = num / den

    if slope > threshold:
        return "improving", slope
    elif slope < -threshold:
        return "declining", slope
    return "stable", slope
```

- [ ] **Step 4: Implement changepoint.py**

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class ChangePointResult:
    change_detected: bool = False
    change_index: int | None = None
    confidence: float = 0.0

def detect_regime_change(values: list[float], threshold: float = 0.8) -> ChangePointResult:
    """Simplified Bayesian change-point detection using max log-likelihood ratio."""
    if len(values) < 10:
        return ChangePointResult()

    arr = np.array(values)
    n = len(arr)
    total_var = np.var(arr)
    if total_var < 1e-10:
        return ChangePointResult()

    best_ratio = 0.0
    best_idx = 0

    for t in range(5, n - 5):
        left = arr[:t]
        right = arr[t:]
        var_left = np.var(left) if len(left) > 1 else total_var
        var_right = np.var(right) if len(right) > 1 else total_var

        # Log-likelihood ratio: reduction in total variance
        weighted_var = (len(left) * var_left + len(right) * var_right) / n
        if weighted_var < 1e-10:
            continue
        ratio = 1.0 - weighted_var / total_var

        # Also check mean shift
        mean_diff = abs(np.mean(left) - np.mean(right))
        combined = ratio * 0.5 + min(1.0, mean_diff / (np.std(arr) + 1e-10)) * 0.5

        if combined > best_ratio:
            best_ratio = combined
            best_idx = t

    if best_ratio >= threshold:
        return ChangePointResult(change_detected=True, change_index=best_idx, confidence=best_ratio)
    return ChangePointResult(confidence=best_ratio)
```

- [ ] **Step 5: Run tests — expect PASS, then commit**

```bash
pytest tests/test_trend.py tests/test_changepoint.py -v
git add engine/temporal/ tests/test_trend.py tests/test_changepoint.py
git commit -m "feat(temporal): trend detection + BOCPD regime change detection"
```

---

## Phase 6: AI Analyzer

### Task 15: Mock Analyzer + Claude Analyzer

**Files:**
- Create: `engine/analysis/__init__.py`, `engine/analysis/base.py`, `engine/analysis/mock.py`, `engine/analysis/claude_analyzer.py`, `tests/test_mock_analyzer.py`

- [ ] **Step 1: Write failing tests**

`tests/test_mock_analyzer.py`:
```python
import pytest
from engine.analysis.mock import MockAnalyzer

@pytest.fixture
def analyzer():
    return MockAnalyzer()

def test_positive_review(analyzer):
    result = analyzer.analyze("Absolutely wonderful experience, loved everything about it! Amazing quality and service.")
    assert result.sentiment > 0.0
    assert len(result.tags) > 0
    assert 0.0 <= result.fake_probability <= 1.0

def test_negative_review(analyzer):
    result = analyzer.analyze("Terrible product, broke after one day. Complete waste of money, very disappointing.")
    assert result.sentiment < 0.0

def test_fake_review_high_exclamation(analyzer):
    result = analyzer.analyze("AMAZING!!! BEST EVER!!! BUY NOW!!! INCREDIBLE!!! WOW!!!")
    assert result.fake_probability > 0.3

def test_empty_text(analyzer):
    result = analyzer.analyze("")
    assert result.sentiment == 0.0
    assert result.tags == []

def test_tags_extracted(analyzer):
    result = analyzer.analyze("The delivery was fast and the pizza was delicious with great cheese and fresh toppings")
    assert len(result.tags) >= 1
```

- [ ] **Step 2: Run tests — expect FAIL**

- [ ] **Step 3: Implement base.py, mock.py, claude_analyzer.py**

`engine/analysis/base.py`:
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class AnalysisResult:
    sentiment: float = 0.0          # -1.0 to 1.0
    fake_probability: float = 0.0   # 0.0 to 1.0
    tags: list[str] = field(default_factory=list)
    summary: str = ""

class BaseAnalyzer(ABC):
    @abstractmethod
    def analyze(self, text: str, context: dict | None = None) -> AnalysisResult:
        pass
```

`engine/analysis/mock.py`:
```python
import re
import math
from collections import Counter
from engine.analysis.base import BaseAnalyzer, AnalysisResult

# Simple sentiment lexicon (VADER-inspired subset)
_POSITIVE = {"good", "great", "excellent", "amazing", "wonderful", "love", "loved", "best",
             "fantastic", "awesome", "perfect", "delicious", "fast", "fresh", "quality",
             "beautiful", "outstanding", "incredible", "superb", "happy", "recommend"}
_NEGATIVE = {"bad", "terrible", "horrible", "worst", "hate", "awful", "poor", "slow",
             "broken", "waste", "disappointing", "disgusting", "rude", "expensive",
             "overpriced", "dirty", "cold", "damaged", "useless", "never"}

class MockAnalyzer(BaseAnalyzer):
    def analyze(self, text: str, context: dict | None = None) -> AnalysisResult:
        if not text or not text.strip():
            return AnalysisResult()

        lower = text.lower()
        words = re.findall(r'\b[a-z]+\b', lower)

        # Sentiment: count positive vs negative words
        pos = sum(1 for w in words if w in _POSITIVE)
        neg = sum(1 for w in words if w in _NEGATIVE)
        total = pos + neg
        sentiment = (pos - neg) / max(total, 1)
        sentiment = max(-1.0, min(1.0, sentiment))

        # Fake probability: heuristics
        exclamation_ratio = text.count("!") / max(len(text), 1)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        short_text = len(words) < 5
        generic_phrases = sum(1 for p in ["buy now", "best ever", "highly recommend", "must buy"] if p in lower)
        fake_prob = min(1.0, exclamation_ratio * 5 + caps_ratio * 2 + generic_phrases * 0.15 + (0.2 if short_text else 0))

        # Tags: top TF keywords (simple term frequency, skip stopwords)
        stopwords = {"the", "a", "an", "is", "it", "to", "and", "of", "in", "for", "was", "with", "on", "at", "this", "that", "i", "my", "we", "but", "not", "very", "so", "just"}
        filtered = [w for w in words if w not in stopwords and len(w) > 2]
        counts = Counter(filtered)
        tags = [word for word, _ in counts.most_common(5)]

        summary = text[:100] + "..." if len(text) > 100 else text

        return AnalysisResult(sentiment=sentiment, fake_probability=fake_prob, tags=tags, summary=summary)
```

`engine/analysis/claude_analyzer.py`:
```python
import json
from engine.analysis.base import BaseAnalyzer, AnalysisResult
from engine.config import settings

class ClaudeAnalyzer(BaseAnalyzer):
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def analyze(self, text: str, context: dict | None = None) -> AnalysisResult:
        if not text or not text.strip():
            return AnalysisResult()

        prompt = f"""Analyze this review and return JSON with exactly these fields:
- sentiment: float from -1.0 (very negative) to 1.0 (very positive)
- fake_probability: float from 0.0 to 1.0 (likelihood this is fake/spam)
- tags: list of 3-5 specific quality tags extracted from the text
- summary: one sentence summary

Review: "{text}"

Return ONLY valid JSON, no markdown."""

        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            data = json.loads(response.content[0].text)
            return AnalysisResult(
                sentiment=float(data.get("sentiment", 0)),
                fake_probability=float(data.get("fake_probability", 0)),
                tags=data.get("tags", []),
                summary=data.get("summary", ""),
            )
        except (json.JSONDecodeError, KeyError, IndexError):
            return AnalysisResult()

def get_analyzer() -> BaseAnalyzer:
    from engine.analysis.mock import MockAnalyzer
    if settings.analyzer_backend == "claude" and settings.anthropic_api_key:
        return ClaudeAnalyzer()
    return MockAnalyzer()
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add engine/analysis/ tests/test_mock_analyzer.py
git commit -m "feat(analysis): pluggable analyzer — mock (VADER+TF-IDF) + Claude"
```

---

## Phase 7: Explainability + Pipeline

### Task 16: Score Tracer + Counterfactual Analysis

**Files:**
- Create: `engine/explainability/__init__.py`, `engine/explainability/tracer.py`, `engine/explainability/counterfactual.py`, `tests/test_tracer.py`, `tests/test_counterfactual.py`

- [ ] **Step 1: Write failing tests**

`tests/test_tracer.py`:
```python
from engine.explainability.tracer import ScoreTracer

def test_tracer_records_factors():
    tracer = ScoreTracer()
    tracer.add("base_prior", 3.0)
    tracer.add("bayesian_update", 1.2)
    tracer.add("trust_propagation", 0.15)
    tracer.add("temporal_decay", -0.1)
    tracer.add("manipulation_dampening", -0.05)
    breakdown = tracer.get_breakdown()
    assert breakdown["base_prior"] == 3.0
    assert breakdown["bayesian_update"] == 1.2
    assert "final" not in breakdown  # not set yet

def test_tracer_finalize():
    tracer = ScoreTracer()
    tracer.add("base_prior", 3.0)
    tracer.add("bayesian_update", 1.0)
    tracer.finalize(4.0)
    breakdown = tracer.get_breakdown()
    assert breakdown["final"] == 4.0

def test_tracer_alerts():
    tracer = ScoreTracer()
    tracer.add_alert("burst_detected", "7 signals in 2h", "medium")
    alerts = tracer.get_alerts()
    assert len(alerts) == 1
    assert alerts[0]["type"] == "burst_detected"
```

`tests/test_counterfactual.py`:
```python
from engine.explainability.counterfactual import compute_counterfactuals

def test_without_dampened():
    signals = [
        {"value": 5.0, "weight": 1.0, "dampened": False},
        {"value": 1.0, "weight": 0.3, "dampened": True},
        {"value": 5.0, "weight": 1.0, "dampened": False},
    ]
    result = compute_counterfactuals(signals, current_score=4.0, trust_bonus=0.1)
    assert "without_dampened" in result
    assert result["without_dampened"] >= 4.0  # removing low dampened signal raises score

def test_without_trust_bonus():
    signals = [{"value": 4.0, "weight": 1.0, "dampened": False}]
    result = compute_counterfactuals(signals, current_score=4.1, trust_bonus=0.1)
    assert abs(result["without_trust_bonus"] - 4.0) < 0.01
```

- [ ] **Step 2: Run tests — expect FAIL**

- [ ] **Step 3: Implement tracer.py and counterfactual.py**

`engine/explainability/tracer.py`:
```python
from dataclasses import dataclass, field

class ScoreTracer:
    def __init__(self):
        self._factors: dict[str, float] = {}
        self._alerts: list[dict] = []

    def add(self, name: str, value: float) -> None:
        self._factors[name] = value

    def add_alert(self, alert_type: str, detail: str, severity: str = "medium") -> None:
        self._alerts.append({"type": alert_type, "detail": detail, "severity": severity})

    def finalize(self, final_score: float) -> None:
        self._factors["final"] = final_score

    def get_breakdown(self) -> dict:
        return dict(self._factors)

    def get_alerts(self) -> list[dict]:
        return list(self._alerts)
```

`engine/explainability/counterfactual.py`:
```python
def compute_counterfactuals(
    signals: list[dict],
    current_score: float,
    trust_bonus: float = 0.0,
) -> dict:
    result = {}

    # Without dampened signals
    undampened = [s for s in signals if not s.get("dampened", False)]
    if undampened:
        total_w = sum(s["weight"] for s in undampened)
        if total_w > 0:
            result["without_dampened"] = sum(s["value"] * s["weight"] for s in undampened) / total_w
        else:
            result["without_dampened"] = current_score
    else:
        result["without_dampened"] = current_score

    # Without trust bonus
    result["without_trust_bonus"] = current_score - trust_bonus

    # Without last 30 days (placeholder — needs timestamps in real usage)
    result["without_last_30d"] = current_score  # simplified

    return result
```

- [ ] **Step 4: Run tests — expect PASS, then commit**

```bash
pytest tests/test_tracer.py tests/test_counterfactual.py -v
git add engine/explainability/ tests/test_tracer.py tests/test_counterfactual.py
git commit -m "feat(explainability): score tracer + counterfactual what-if analysis"
```

---

### Task 17: Full Scoring Pipeline

**Files:**
- Create: `engine/scoring/pipeline.py`, `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test**

`tests/test_pipeline.py`:
```python
from datetime import datetime, timedelta, timezone
from engine.scoring.pipeline import ScoringPipeline
from engine.analysis.mock import MockAnalyzer

def test_pipeline_basic_scoring():
    pipeline = ScoringPipeline(analyzer=MockAnalyzer())
    signals = [
        {"value": 4.5, "text": "Great product", "dimension": "quality",
         "source_stats": {"review_count": 10, "review_diversity": 0.5, "accuracy": 0.7, "account_age_days": 90, "flagged_ratio": 0.0},
         "created_at": datetime.now(timezone.utc) - timedelta(days=i)}
        for i in range(10)
    ]
    result = pipeline.score(signals)
    assert "overall" in result
    assert "dimensions" in result
    assert "breakdown" in result
    assert "tier" in result
    assert result["overall"] > 3.0

def test_pipeline_empty_signals():
    pipeline = ScoringPipeline(analyzer=MockAnalyzer())
    result = pipeline.score([])
    assert result["overall"] == 3.0
    assert result["tier"] == "bronze"
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement pipeline.py**

```python
from datetime import datetime, timezone
from engine.analysis.base import BaseAnalyzer
from engine.scoring.wilson import wilson_from_stars
from engine.scoring.bayesian import BetaBinomial, DirichletMultinomial
from engine.scoring.decay import compute_decay, compute_velocity
from engine.scoring.aggregator import aggregate_dimensions, classify_tier
from engine.detection.burst import CUSUMDetector
from engine.detection.credibility import compute_credibility
from engine.explainability.tracer import ScoreTracer
from engine.explainability.counterfactual import compute_counterfactuals
from engine.config import settings

class ScoringPipeline:
    def __init__(self, analyzer: BaseAnalyzer):
        self.analyzer = analyzer
        self.burst_detector = CUSUMDetector()

    def score(self, signals: list[dict], trust_bonus: float = 0.0) -> dict:
        tracer = ScoreTracer()
        now = datetime.now(timezone.utc)

        if not signals:
            tracer.add("base_prior", 3.0)
            tracer.finalize(3.0)
            return {"overall": 3.0, "confidence": 0.0, "tier": "bronze", "dimensions": {},
                    "breakdown": tracer.get_breakdown(), "alerts": [], "counterfactual": {}}

        # Group by dimension
        by_dim: dict[str, list] = {}
        for s in signals:
            dim = s.get("dimension", "quality")
            by_dim.setdefault(dim, []).append(s)

        # Burst detection
        times = [s["created_at"] for s in signals]
        burst = self.burst_detector.detect(times, now)
        if burst.burst_detected:
            tracer.add_alert("burst_detected", f"{len(burst.dampened_indices)} signals dampened", "medium")
            for idx in burst.dampened_indices:
                if idx < len(signals):
                    signals[idx]["_dampened"] = True

        # Score each dimension
        dim_scores = {}
        dim_details = {}
        total_signals = 0

        for dim_name, dim_signals in by_dim.items():
            # Apply decay + credibility weighting
            weighted_values = []
            for s in dim_signals:
                decay = compute_decay(s["created_at"], now, settings.decay_half_life_days)
                cred = compute_credibility(**s.get("source_stats", {
                    "review_count": 1, "review_diversity": 0.5, "accuracy": 0.5,
                    "account_age_days": 30, "flagged_ratio": 0.0}))
                dampen = 0.3 if s.get("_dampened") else 1.0
                weight = decay * cred * dampen
                weighted_values.append((s["value"], weight))

            # Dirichlet update
            dm = DirichletMultinomial()
            counts = [0, 0, 0, 0, 0]
            for v, w in weighted_values:
                star = max(0, min(4, round(v) - 1))
                counts[star] += max(1, round(w))
            dm.update(counts)
            score = dm.weighted_mean()
            wilson = wilson_from_stars(score, len(dim_signals))
            conf = dm.confidence()

            # Trend
            score_history = [(s["created_at"], s["value"]) for s in dim_signals]
            vel = compute_velocity(score_history, settings.velocity_window_days)
            if vel > settings.trend_threshold:
                trend = "improving"
            elif vel < -settings.trend_threshold:
                trend = "declining"
            else:
                trend = "stable"

            dim_scores[dim_name] = score
            dim_details[dim_name] = {
                "score": round(score, 2), "wilson_lower": round(wilson, 3),
                "trend": trend, "signals": len(dim_signals), "confidence": round(conf, 2),
            }
            total_signals += len(dim_signals)

        # Aggregate
        overall = aggregate_dimensions(dim_scores)
        overall += trust_bonus
        min_conf = min((d["confidence"] for d in dim_details.values()), default=0.0)
        tier = classify_tier(overall, total_signals, min_conf)

        tracer.add("base_prior", 3.0)
        tracer.add("bayesian_update", round(overall - 3.0 - trust_bonus, 2))
        tracer.add("trust_propagation", round(trust_bonus, 2))
        manipulation_penalty = sum(1 for s in signals if s.get("_dampened")) * 0.01
        tracer.add("manipulation_dampening", round(-manipulation_penalty, 3))
        tracer.finalize(round(overall, 2))

        cf = compute_counterfactuals(
            [{"value": s["value"], "weight": 1.0, "dampened": s.get("_dampened", False)} for s in signals],
            current_score=overall, trust_bonus=trust_bonus,
        )

        return {
            "overall": round(overall, 2),
            "confidence": round(min_conf, 2),
            "tier": tier,
            "dimensions": dim_details,
            "breakdown": tracer.get_breakdown(),
            "alerts": tracer.get_alerts(),
            "counterfactual": cf,
        }
```

- [ ] **Step 4: Run tests — expect PASS, then commit**

```bash
pytest tests/test_pipeline.py -v
git add engine/scoring/pipeline.py tests/test_pipeline.py
git commit -m "feat(scoring): full scoring pipeline — decay, Bayesian, detection, explainability"
```

---

## Phase 8: API + Schemas + Main App

### Task 18: Pydantic Schemas

**Files:**
- Create: `engine/schemas/__init__.py`, `engine/schemas/entity.py`, `engine/schemas/signal.py`, `engine/schemas/score.py`, `engine/schemas/trust.py`, `engine/schemas/analytics.py`

- [ ] **Step 1: Create all schema files**

`engine/schemas/entity.py`:
```python
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
```

`engine/schemas/signal.py`:
```python
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
```

`engine/schemas/score.py`:
```python
from uuid import UUID
from pydantic import BaseModel

class DimensionScoreResponse(BaseModel):
    score: float
    wilson_lower: float
    trend: str
    signals: int
    confidence: float

class ScoreResponse(BaseModel):
    entity_id: UUID
    overall: float
    confidence: float
    tier: str
    dimensions: dict[str, DimensionScoreResponse]
    breakdown: dict
    alerts: list[dict]
    counterfactual: dict

class ScoreHistoryItem(BaseModel):
    score: float
    breakdown: dict
    created_at: str
```

`engine/schemas/trust.py`:
```python
from uuid import UUID
from pydantic import BaseModel, Field
from engine.models.trust import EvidenceType

class TrustEdgeCreate(BaseModel):
    target_id: UUID
    category: str = "general"
    weight: float = Field(ge=0.0, le=1.0, default=0.5)
    evidence_type: EvidenceType = EvidenceType.transaction
```

`engine/schemas/analytics.py`:
```python
from pydantic import BaseModel

class OverviewResponse(BaseModel):
    total_entities: int
    total_signals: int
    tier_distribution: dict[str, int]
    signals_today: int
    dampened_count: int
```

- [ ] **Step 2: Commit**

```bash
git add engine/schemas/
git commit -m "feat(api): Pydantic request/response schemas"
```

---

### Task 19: FastAPI App + Entity & Signal Endpoints

**Files:**
- Create: `engine/main.py`, `engine/deps.py`, `engine/api/__init__.py`, `engine/api/entities.py`, `engine/api/signals.py`

- [ ] **Step 1: Create engine/deps.py**

```python
from engine.db.session import async_session
from engine.analysis.claude_analyzer import get_analyzer

async def get_db():
    async with async_session() as session:
        yield session

def get_analyzer_dep():
    return get_analyzer()
```

- [ ] **Step 2: Create engine/api/entities.py**

```python
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
    return EntityResponse(id=entity.id, type=entity.type, name=entity.name,
                          metadata=entity.metadata_, created_at=entity.created_at)

@router.get("/{entity_id}", response_model=EntityResponse)
async def get_entity(entity_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Entity).where(Entity.id == entity_id, Entity.deleted == False))
    entity = result.scalar_one_or_none()
    if not entity:
        raise HTTPException(404, "Entity not found")
    # Attach score if exists
    score_result = await db.execute(select(OverallScore).where(OverallScore.entity_id == entity_id))
    overall = score_result.scalar_one_or_none()
    return EntityResponse(
        id=entity.id, type=entity.type, name=entity.name, metadata=entity.metadata_,
        created_at=entity.created_at,
        score=overall.score if overall else None,
        tier=overall.tier.value if overall else None,
    )

@router.get("", response_model=list[EntityResponse])
async def list_entities(type: str | None = None, tier: str | None = None,
                        limit: int = 50, offset: int = 0,
                        db: AsyncSession = Depends(get_db)):
    q = select(Entity).where(Entity.deleted == False)
    if type:
        q = q.where(Entity.type == type)
    q = q.offset(offset).limit(limit)
    result = await db.execute(q)
    entities = result.scalars().all()
    responses = []
    for e in entities:
        sr = await db.execute(select(OverallScore).where(OverallScore.entity_id == e.id))
        o = sr.scalar_one_or_none()
        responses.append(EntityResponse(
            id=e.id, type=e.type, name=e.name, metadata=e.metadata_,
            created_at=e.created_at,
            score=o.score if o else None, tier=o.tier.value if o else None,
        ))
    return responses

@router.delete("/{entity_id}", status_code=204)
async def delete_entity(entity_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Entity).where(Entity.id == entity_id))
    entity = result.scalar_one_or_none()
    if not entity:
        raise HTTPException(404, "Entity not found")
    entity.deleted = True
    await db.commit()
```

- [ ] **Step 3: Create engine/api/signals.py**

```python
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from engine.deps import get_db, get_analyzer_dep
from engine.models.signal import Signal
from engine.models.entity import Entity
from engine.schemas.signal import SignalCreate, SignalResponse
from engine.analysis.base import BaseAnalyzer

router = APIRouter(prefix="/api/v1/entities/{entity_id}/signals", tags=["signals"])

@router.post("", response_model=SignalResponse, status_code=201)
async def submit_signal(entity_id: UUID, body: SignalCreate, db: AsyncSession = Depends(get_db)):
    # Verify entity exists
    result = await db.execute(select(Entity).where(Entity.id == entity_id, Entity.deleted == False))
    if not result.scalar_one_or_none():
        raise HTTPException(404, "Entity not found")

    # Run analyzer on text
    analyzer = get_analyzer_dep()
    analysis = analyzer.analyze(body.text or "", {})

    signal = Signal(
        entity_id=entity_id, source_id=body.source_id, dimension=body.dimension,
        type=body.type, value=body.value, text=body.text,
        tags=analysis.tags, sentiment=analysis.sentiment,
        fake_probability=analysis.fake_probability, weight=1.0, raw_weight=1.0,
    )
    db.add(signal)
    await db.commit()
    await db.refresh(signal)

    return SignalResponse(
        id=signal.id, entity_id=signal.entity_id, source_id=signal.source_id,
        dimension=signal.dimension.value, type=signal.type.value, value=signal.value,
        tags=signal.tags, sentiment=signal.sentiment, fake_probability=signal.fake_probability,
        dampened=signal.dampened, dampening_reason=signal.dampening_reason,
        weight=signal.weight, created_at=signal.created_at.isoformat(),
    )

@router.get("", response_model=list[SignalResponse])
async def list_signals(entity_id: UUID, dimension: str | None = None,
                       dampened: bool | None = None, limit: int = 50,
                       db: AsyncSession = Depends(get_db)):
    q = select(Signal).where(Signal.entity_id == entity_id)
    if dimension:
        q = q.where(Signal.dimension == dimension)
    if dampened is not None:
        q = q.where(Signal.dampened == dampened)
    q = q.order_by(Signal.created_at.desc()).limit(limit)
    result = await db.execute(q)
    return [SignalResponse(
        id=s.id, entity_id=s.entity_id, source_id=s.source_id,
        dimension=s.dimension.value, type=s.type.value, value=s.value,
        tags=s.tags, sentiment=s.sentiment, fake_probability=s.fake_probability,
        dampened=s.dampened, dampening_reason=s.dampening_reason,
        weight=s.weight, created_at=s.created_at.isoformat(),
    ) for s in result.scalars().all()]
```

- [ ] **Step 4: Create engine/main.py**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from engine.api import entities, signals

app = FastAPI(title="trustrank", version="0.1.0", description="Entity Reputation & Trust Scoring Engine")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(entities.router)
app.include_router(signals.router)

@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 5: Commit**

```bash
git add engine/main.py engine/deps.py engine/api/
git commit -m "feat(api): FastAPI app + entity CRUD + signal submission endpoints"
```

---

### Task 20: Score, Trust, Analytics, Source Endpoints

**Files:**
- Create: `engine/api/scores.py`, `engine/api/trust.py`, `engine/api/analytics.py`, `engine/api/sources.py`

- [ ] **Step 1: Create engine/api/scores.py**

```python
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from engine.deps import get_db, get_analyzer_dep
from engine.models.signal import Signal
from engine.models.score import OverallScore, ScoreHistory
from engine.models.source import SourceCredibility
from engine.scoring.pipeline import ScoringPipeline
from engine.schemas.score import ScoreResponse, ScoreHistoryItem

router = APIRouter(prefix="/api/v1/entities/{entity_id}/score", tags=["scores"])

@router.get("", response_model=ScoreResponse)
async def get_score(entity_id: UUID, db: AsyncSession = Depends(get_db)):
    # Fetch all signals for this entity
    result = await db.execute(select(Signal).where(Signal.entity_id == entity_id).order_by(Signal.created_at))
    signals = result.scalars().all()

    pipeline = ScoringPipeline(analyzer=get_analyzer_dep())
    signal_dicts = []
    for s in signals:
        # Get source credibility
        cred_result = await db.execute(select(SourceCredibility).where(SourceCredibility.source_id == s.source_id))
        cred = cred_result.scalar_one_or_none()
        source_stats = {
            "review_count": cred.review_count if cred else 1,
            "review_diversity": cred.review_diversity if cred else 0.5,
            "accuracy": cred.accuracy_score if cred else 0.5,
            "account_age_days": cred.account_age_days if cred else 30,
            "flagged_ratio": (cred.flagged_count / max(cred.review_count, 1)) if cred else 0.0,
        }
        signal_dicts.append({
            "value": s.value, "text": s.text or "", "dimension": s.dimension.value,
            "source_stats": source_stats, "created_at": s.created_at,
        })

    scored = pipeline.score(signal_dicts)
    return ScoreResponse(entity_id=entity_id, **scored)

@router.get("/history", response_model=list[ScoreHistoryItem])
async def get_score_history(entity_id: UUID, limit: int = 100, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ScoreHistory).where(ScoreHistory.entity_id == entity_id)
        .order_by(ScoreHistory.created_at.desc()).limit(limit)
    )
    return [ScoreHistoryItem(score=h.score, breakdown=h.breakdown, created_at=h.created_at.isoformat())
            for h in result.scalars().all()]
```

- [ ] **Step 2: Create engine/api/trust.py**

```python
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from engine.deps import get_db
from engine.models.trust import TrustEdge
from engine.schemas.trust import TrustEdgeCreate
from engine.graph.trust import TrustGraph
from engine.graph.influence import katz_centrality
import networkx as nx

router = APIRouter(prefix="/api/v1/entities/{entity_id}/trust", tags=["trust"])

@router.post("", status_code=201)
async def create_trust_edge(entity_id: UUID, body: TrustEdgeCreate, db: AsyncSession = Depends(get_db)):
    edge = TrustEdge(source_id=entity_id, target_id=body.target_id,
                     category=body.category, weight=body.weight, evidence_type=body.evidence_type)
    db.add(edge)
    await db.commit()
    return {"status": "created"}

@router.get("/graph")
async def get_trust_graph(entity_id: UUID, hops: int = 2, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(TrustEdge))
    edges = result.scalars().all()
    tg = TrustGraph(max_hops=hops)
    for e in edges:
        tg.add_edge(str(e.source_id), str(e.target_id), weight=e.weight, category=e.category)
    return tg.get_subgraph(str(entity_id), hops=hops)

@router.get("/influence")
async def get_influence(entity_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(TrustEdge))
    edges = result.scalars().all()
    G = nx.DiGraph()
    for e in edges:
        G.add_edge(str(e.source_id), str(e.target_id), weight=e.weight)
    scores = katz_centrality(G)
    return {"entity_id": str(entity_id), "influence": scores.get(str(entity_id), 0.0)}
```

- [ ] **Step 3: Create engine/api/analytics.py and engine/api/sources.py**

`engine/api/analytics.py`:
```python
from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone, timedelta
from engine.deps import get_db
from engine.models.entity import Entity
from engine.models.signal import Signal
from engine.models.score import OverallScore

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])

@router.get("/overview")
async def overview(db: AsyncSession = Depends(get_db)):
    entities = (await db.execute(select(func.count(Entity.id)).where(Entity.deleted == False))).scalar()
    signals = (await db.execute(select(func.count(Signal.id)))).scalar()
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
    signals_today = (await db.execute(select(func.count(Signal.id)).where(Signal.created_at >= today))).scalar()
    dampened = (await db.execute(select(func.count(Signal.id)).where(Signal.dampened == True))).scalar()

    # Tier distribution
    tier_result = await db.execute(select(OverallScore.tier, func.count(OverallScore.id)).group_by(OverallScore.tier))
    tier_dist = {row[0].value: row[1] for row in tier_result.all()}

    return {"total_entities": entities, "total_signals": signals, "tier_distribution": tier_dist,
            "signals_today": signals_today, "dampened_count": dampened}

@router.get("/leaderboard")
async def leaderboard(limit: int = 20, offset: int = 0, tier: str | None = None,
                      db: AsyncSession = Depends(get_db)):
    q = select(OverallScore, Entity).join(Entity, OverallScore.entity_id == Entity.id).where(Entity.deleted == False)
    if tier:
        q = q.where(OverallScore.tier == tier)
    q = q.order_by(OverallScore.score.desc()).offset(offset).limit(limit)
    result = await db.execute(q)
    rows = result.all()
    return [{"entity_id": str(o.entity_id), "name": e.name, "type": e.type.value,
             "score": o.score, "tier": o.tier.value, "signals": o.total_signals,
             "confidence": o.confidence} for o, e in rows]
```

`engine/api/sources.py`:
```python
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from engine.deps import get_db
from engine.models.source import SourceCredibility

router = APIRouter(prefix="/api/v1/sources", tags=["sources"])

@router.get("/{source_id}/credibility")
async def get_credibility(source_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(SourceCredibility).where(SourceCredibility.source_id == source_id))
    cred = result.scalar_one_or_none()
    if not cred:
        raise HTTPException(404, "Source credibility not found")
    return {"source_id": str(cred.source_id), "credibility_score": cred.credibility_score,
            "review_count": cred.review_count, "review_diversity": cred.review_diversity,
            "accuracy_score": cred.accuracy_score, "account_age_days": cred.account_age_days,
            "flagged_count": cred.flagged_count}
```

- [ ] **Step 4: Register all routers in main.py**

Update `engine/main.py` imports:
```python
from engine.api import entities, signals, scores, trust, analytics, sources

app.include_router(entities.router)
app.include_router(signals.router)
app.include_router(scores.router)
app.include_router(trust.router)
app.include_router(analytics.router)
app.include_router(sources.router)
```

- [ ] **Step 5: Commit**

```bash
git add engine/api/ engine/main.py
git commit -m "feat(api): score, trust, analytics, source endpoints + router registration"
```

---

## Phase 9: Seeding + Docker + README

### Task 21: Data Seeder with Attack Scenarios

**Files:**
- Create: `engine/seed/__init__.py`, `engine/seed/seeder.py`, `engine/seed/scenarios.py`

- [ ] **Step 1: Create engine/seed/scenarios.py**

```python
"""Pre-built attack scenarios for demo data."""
import random

def review_bombing_signals(entity_idx: int, source_start: int, count: int = 15) -> list[dict]:
    """Burst of negative reviews in short window."""
    return [{"entity_idx": entity_idx, "source_idx": source_start + i,
             "dimension": "quality", "type": "review", "value": 1.0,
             "text": random.choice([
                 "Terrible experience, complete waste of money",
                 "Awful product, do not buy this garbage",
                 "Worst purchase I have ever made in my life",
                 "Absolutely horrible, I want a refund immediately",
             ]),
             "hours_ago": random.uniform(0, 2)} for i in range(count)]

def coordinated_positive(entity_idx: int, source_start: int, count: int = 8) -> list[dict]:
    """Similar positive reviews from different accounts."""
    base = "Amazing product, absolutely love it, highly recommend to everyone"
    variants = [
        "Amazing product, absolutely love this, highly recommend to all",
        "Amazing item, absolutely love it, highly recommend to everyone",
        "Amazing product, totally love it, highly recommend to everyone",
        "Amazing product, absolutely love it, strongly recommend to all",
        "Amazing product, absolutely adore it, highly recommend to everyone",
        "Amazing goods, absolutely love it, highly recommend to everyone",
        "Amazing product, absolutely love it, highly recommend to anybody",
        "Amazing product, absolutely love it, really recommend to everyone",
    ]
    return [{"entity_idx": entity_idx, "source_idx": source_start + i,
             "dimension": "quality", "type": "review", "value": 5.0,
             "text": variants[i % len(variants)], "hours_ago": i * 0.5} for i in range(count)]

def reciprocal_ring(entities: list[int], source_pool: list[int]) -> list[dict]:
    """Entities reviewing each other positively."""
    signals = []
    for i, ent in enumerate(entities):
        for j, other in enumerate(entities):
            if i != j:
                signals.append({"entity_idx": other, "source_idx": ent,
                                "dimension": "quality", "type": "review", "value": 5.0,
                                "text": f"Great service from entity {other}, very professional",
                                "hours_ago": random.uniform(24, 720)})
    return signals
```

- [ ] **Step 2: Create engine/seed/seeder.py**

```python
import random
import uuid
from datetime import datetime, timedelta, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from engine.models.entity import Entity, EntityType
from engine.models.signal import Signal, Dimension, SignalType
from engine.models.trust import TrustEdge, EvidenceType
from engine.seed.scenarios import review_bombing_signals, coordinated_positive, reciprocal_ring

random.seed(42)

ENTITY_NAMES = {
    "merchant": ["Alpine Kitchen", "Sakura Sushi", "Metro Mart", "Golden Spice", "Blue Harbor",
                  "Sunset Cafe", "Prime Goods", "Fresh Fields", "Ocean Breeze", "Peak Performance"],
    "user": [f"user_{i:03d}" for i in range(100)],
    "service": ["QuickFix Repairs", "SwiftDeliver", "CleanPro", "TechAssist", "PetCare Plus",
                "GreenThumb Gardens", "HomeBright", "AutoCare", "FitCoach", "TravelEase"],
    "product": ["ErgoChair Pro", "SkyBuds Wireless", "AquaFilter X", "SolarPack 5000", "ChefMaster Blender",
                "ZenMat Yoga", "PureAir Purifier", "SmartLock V2", "NightOwl Lamp", "FrostKeep Cooler"],
}

async def seed_database(db: AsyncSession) -> dict:
    entities = []
    entity_map = {}

    # Create entities
    for etype in ["merchant", "service", "product"]:
        for name in ENTITY_NAMES[etype][:10]:
            e = Entity(type=EntityType(etype), name=name, metadata_={"category": etype})
            db.add(e)
            entities.append(e)
    for name in ENTITY_NAMES["user"][:100]:
        e = Entity(type=EntityType.user, name=name, metadata_={})
        db.add(e)
        entities.append(e)
    await db.flush()

    non_users = [e for e in entities if e.type != EntityType.user]
    users = [e for e in entities if e.type == EntityType.user]

    now = datetime.now(timezone.utc)
    signal_count = 0

    # Organic signals: ~5000 spread across entities
    for entity in non_users:
        n_signals = random.randint(30, 200)
        for _ in range(n_signals):
            source = random.choice(users)
            dim = random.choice(list(Dimension))
            # Skew positive (realistic distribution)
            value = min(5.0, max(1.0, random.gauss(3.8, 0.9)))
            value = round(value * 2) / 2  # round to nearest 0.5
            texts = ["Good quality overall", "Met my expectations", "Could be better",
                     "Excellent service", "Average experience", "Very satisfied",
                     "Not worth the price", "Will come back again", "Highly recommended",
                     "Disappointing quality", "Great value for money", ""]
            s = Signal(
                entity_id=entity.id, source_id=source.id, dimension=dim,
                type=SignalType.review, value=value, text=random.choice(texts),
                created_at=now - timedelta(days=random.uniform(1, 365)),
            )
            db.add(s)
            signal_count += 1

    # Attack scenarios
    # 1. Review bombing on entity 0
    for attack in review_bombing_signals(0, 80, 15):
        s = Signal(
            entity_id=non_users[attack["entity_idx"]].id,
            source_id=users[attack["source_idx"]].id,
            dimension=Dimension(attack["dimension"]), type=SignalType(attack["type"]),
            value=attack["value"], text=attack["text"],
            created_at=now - timedelta(hours=attack["hours_ago"]),
        )
        db.add(s)
        signal_count += 1

    # 2. Coordinated positive on entity 5
    for attack in coordinated_positive(5, 60, 8):
        s = Signal(
            entity_id=non_users[attack["entity_idx"]].id,
            source_id=users[attack["source_idx"]].id,
            dimension=Dimension(attack["dimension"]), type=SignalType(attack["type"]),
            value=attack["value"], text=attack["text"],
            created_at=now - timedelta(hours=attack["hours_ago"]),
        )
        db.add(s)
        signal_count += 1

    # Trust edges (3-4 communities)
    trust_count = 0
    community_size = len(non_users) // 3
    for c in range(3):
        members = non_users[c * community_size:(c + 1) * community_size]
        for i in range(len(members)):
            for j in range(i + 1, min(i + 3, len(members))):
                edge = TrustEdge(source_id=members[i].id, target_id=members[j].id,
                                 weight=random.uniform(0.5, 0.95), category="general",
                                 evidence_type=EvidenceType.transaction)
                db.add(edge)
                trust_count += 1

    await db.commit()
    return {"entities": len(entities), "signals": signal_count, "trust_edges": trust_count}
```

- [ ] **Step 3: Add seed CLI command to main.py**

Add to `engine/main.py`:
```python
@app.post("/api/v1/admin/seed")
async def run_seed(db: AsyncSession = Depends(get_db)):
    from engine.seed.seeder import seed_database
    result = await seed_database(db)
    return {"status": "seeded", **result}
```

- [ ] **Step 4: Commit**

```bash
git add engine/seed/ engine/main.py
git commit -m "feat: data seeder with 200 entities, 5000+ signals, attack scenarios"
```

---

### Task 22: Dockerfile + README

**Files:**
- Create: `Dockerfile`, update `docker-compose.yml`, create `README.md`

- [ ] **Step 1: Create Dockerfile**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .
COPY engine/ engine/
COPY alembic.ini .
EXPOSE 8000
CMD ["uvicorn", "engine.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Update docker-compose.yml with API service**

```yaml
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: trustrank
      POSTGRES_PASSWORD: trustrank
      POSTGRES_DB: trustrank
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql+asyncpg://trustrank:trustrank@db:5432/trustrank
      REDIS_URL: redis://redis:6379/0
      ANALYZER_BACKEND: mock
    depends_on:
      - db
      - redis

volumes:
  pgdata:
```

- [ ] **Step 3: Create README.md**

Write a comprehensive README covering: overview, features list (multi-dimensional scoring, manipulation resistance with CUSUM/coordination/credibility/reciprocal, trust graph with Louvain/Katz, temporal analysis with BOCPD, explainability), quickstart with Docker, API docs link, architecture diagram (ASCII), tech stack, and screenshots placeholder.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile docker-compose.yml README.md
git commit -m "feat: Docker setup + README with full documentation"
```

---

## Phase 10: Dashboard

### Task 23: Dashboard Scaffolding

**Files:**
- Create: `dashboard/package.json`, `dashboard/vite.config.ts`, `dashboard/tailwind.config.ts`, `dashboard/index.html`, `dashboard/src/main.tsx`, `dashboard/src/App.tsx`, `dashboard/src/lib/api.ts`, `dashboard/src/components/Layout.tsx`, `dashboard/src/components/TierBadge.tsx`

- [ ] **Step 1: Initialize React + Tailwind + Vite project**

```bash
cd dashboard
npm create vite@latest . -- --template react-ts
npm install tailwindcss @tailwindcss/vite react-router-dom recharts d3 @types/d3
```

- [ ] **Step 2: Create src/lib/api.ts**

```typescript
const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1";

export async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}
```

- [ ] **Step 3: Create Layout.tsx, TierBadge.tsx, App.tsx with routing**

`src/components/TierBadge.tsx`:
```tsx
const COLORS: Record<string, string> = {
  platinum: "bg-violet-100 text-violet-800",
  gold: "bg-amber-100 text-amber-800",
  silver: "bg-gray-200 text-gray-700",
  bronze: "bg-orange-100 text-orange-700",
  untrusted: "bg-red-100 text-red-700",
};

export function TierBadge({ tier }: { tier: string }) {
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${COLORS[tier] || "bg-gray-100"}`}>
      {tier}
    </span>
  );
}
```

`src/App.tsx` — Routes: `/` (Leaderboard), `/entity/:id` (Detail), `/trust` (Explorer), `/detection` (Detection), `/analytics` (Analytics).

- [ ] **Step 4: Commit**

```bash
git add dashboard/
git commit -m "feat(dashboard): React + Tailwind scaffolding with routing"
```

---

### Task 24: Leaderboard + Entity Detail Pages

**Files:**
- Create: `dashboard/src/pages/Leaderboard.tsx`, `dashboard/src/pages/EntityDetail.tsx`, `dashboard/src/components/ScoreSparkline.tsx`

- [ ] **Step 1: Build Leaderboard page**

Sortable table with columns: Rank, Name, Type, Score, Tier (badge), Signals, Confidence, Trend. Fetches from `/analytics/leaderboard`. Click row → navigates to `/entity/:id`.

- [ ] **Step 2: Build EntityDetail page**

Sections:
1. Header: entity name, type, tier badge, overall score
2. Dimension radar chart (Recharts RadarChart)
3. Score timeline (Recharts LineChart from `/score/history`)
4. Signal history table (from `/signals`)
5. Score breakdown JSON viewer
6. Trust graph mini-view (link to full explorer)

- [ ] **Step 3: Commit**

```bash
git add dashboard/src/pages/ dashboard/src/components/
git commit -m "feat(dashboard): leaderboard + entity detail with charts"
```

---

### Task 25: Trust Explorer + Detection + Analytics Pages

**Files:**
- Create: `dashboard/src/pages/TrustExplorer.tsx`, `dashboard/src/pages/Detection.tsx`, `dashboard/src/pages/Analytics.tsx`, `dashboard/src/components/ForceGraph.tsx`

- [ ] **Step 1: Build ForceGraph component with D3**

D3 force-directed graph: nodes colored by community, sized by influence, edges weighted by trust strength. Click node → shows entity detail tooltip.

- [ ] **Step 2: Build TrustExplorer page**

Entity search input + ForceGraph rendering the trust subgraph. Sidebar with node details on click.

- [ ] **Step 3: Build Detection page**

Cards showing: active burst alerts, coordination clusters, flagged signals queue with approve/reject actions, source credibility distribution histogram.

- [ ] **Step 4: Build Analytics page**

Charts: tier distribution donut (Recharts PieChart), signals per day (LineChart), detection rate trends, top tags word cloud.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/
git commit -m "feat(dashboard): trust explorer (D3 force graph) + detection + analytics"
```

---

### Task 26: Final Integration + Docker Dashboard

**Files:**
- Update: `docker-compose.yml`, `dashboard/Dockerfile`

- [ ] **Step 1: Add dashboard to Docker Compose**

```yaml
  dashboard:
    build: ./dashboard
    ports:
      - "3000:3000"
    environment:
      VITE_API_URL: http://localhost:8000/api/v1
    depends_on:
      - api
```

- [ ] **Step 2: Run full stack and verify**

```bash
docker compose up --build
# POST /api/v1/admin/seed to populate data
# Open http://localhost:3000 — verify leaderboard, entity detail, trust graph, detection, analytics
```

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: full-stack trustrank — scoring engine, detection, trust graph, dashboard"
```
