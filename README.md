# trustrank

A multi-signal entity reputation and trust scoring engine with manipulation detection, Bayesian updating, and graph-based trust propagation.

## Features

- **Bayesian Scoring** — Dirichlet-Multinomial model with Wilson lower-bound confidence intervals; scores update incrementally as new signals arrive.
- **Temporal Decay & Velocity** — Exponential half-life decay on signal weights; trend detection (improving / stable / declining) via linear regression over a configurable window.
- **Manipulation Detection** — CUSUM burst detector flags sudden signal influxes; coordination detector catches clusters of semantically similar signals; credibility scorer weights sources by review diversity, accuracy, and account age.
- **Graph Trust Propagation** — Directed trust graph with multi-hop dampening and Katz centrality influence scoring; trust bonus from peer endorsements adds up to 15% to the final score.
- **Explainability** — Per-request `ScoreTracer` records every factor (prior, Bayesian update, trust propagation, manipulation penalty); counterfactual engine answers "what if dampened signals were removed / trust bonus was zero?"
- **Tier Classification** — Five tiers (platinum / gold / silver / bronze / untrusted) gated on score, signal count, and confidence thresholds.

## Quickstart

```bash
# Clone and start all services
git clone https://github.com/your-org/trustrank.git
cd trustrank
docker compose up --build

# The API is now available at http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

Run database migrations (after db container is healthy):

```bash
docker compose exec api alembic upgrade head
```

Seed with 200 entities and 5000+ signals including attack scenarios:

```bash
curl -X POST http://localhost:8000/api/v1/admin/seed
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/entities` | Create entity |
| GET | `/api/v1/entities` | List entities (filterable by type) |
| GET | `/api/v1/entities/{id}` | Get entity with current score |
| DELETE | `/api/v1/entities/{id}` | Soft-delete entity |
| POST | `/api/v1/entities/{id}/signals` | Submit a trust signal |
| GET | `/api/v1/entities/{id}/signals` | List signals (filter by dimension/dampened) |
| GET | `/api/v1/entities/{id}/score` | Compute full score with breakdown |
| GET | `/api/v1/entities/{id}/score/history` | Score history |
| POST | `/api/v1/entities/{id}/trust` | Create trust edge |
| GET | `/api/v1/entities/{id}/trust/graph` | N-hop trust subgraph |
| GET | `/api/v1/entities/{id}/trust/influence` | Katz centrality influence score |
| GET | `/api/v1/analytics/overview` | Platform-wide stats |
| GET | `/api/v1/analytics/leaderboard` | Top entities by score |
| GET | `/api/v1/sources/{id}/credibility` | Source credibility record |
| POST | `/api/v1/admin/seed` | Seed demo data |

## Architecture Overview

```
POST /signals
     │
     ▼
 MockAnalyzer / ClaudeAnalyzer
 (sentiment, fake_probability, tags)
     │
     ▼
 ScoringPipeline.score()
 ├── CUSUMDetector          — burst dampening
 ├── compute_credibility    — per-source weight
 ├── compute_decay          — temporal half-life
 ├── DirichletMultinomial   — Bayesian star update
 ├── wilson_from_stars      — lower confidence bound
 ├── aggregate_dimensions   — weighted dimension merge
 ├── classify_tier          — platinum/gold/silver/bronze/untrusted
 ├── ScoreTracer            — audit trail
 └── compute_counterfactuals
     │
     ▼
 ScoreResponse (JSON)
```

Trust propagation runs independently via `TrustGraph.compute_trust_bonus()` and is injected as `trust_bonus` into the pipeline.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI 0.115 + Uvicorn |
| ORM | SQLAlchemy 2 (async) + asyncpg |
| Database | PostgreSQL 16 |
| Cache | Redis 7 |
| Graph | NetworkX 3 |
| AI Analysis | Anthropic Claude Haiku (optional; falls back to MockAnalyzer) |
| Validation | Pydantic v2 |
| Migrations | Alembic |
| Containerisation | Docker + Docker Compose |

## Configuration

All settings are read from environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://trustrank:trustrank@localhost:5432/trustrank` | Async Postgres DSN |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis DSN |
| `ANALYZER_BACKEND` | `mock` | `mock` or `claude` |
| `ANTHROPIC_API_KEY` | `` | Required when backend is `claude` |
| `DECAY_HALF_LIFE_DAYS` | `90` | Signal decay half-life |
| `VELOCITY_WINDOW_DAYS` | `7` | Trend detection window |
| `TREND_THRESHOLD` | `0.02` | Min slope to call a trend |

See `engine/config.py` for the full list.

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
