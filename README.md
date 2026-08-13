# trustrank

A multi-signal entity reputation engine. It turns a stream of ratings, reviews and peer-trust
edges into a per-entity score that resists review bombing and coordinated praise, and that can
explain every point of the number it returns.

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)

## Features

- **Bayesian scoring per dimension.** Quality, reliability, responsiveness and trust are each
  modelled as a Dirichlet-Multinomial over the five star buckets. Signal weights become integer
  pseudo-counts before the update, so a heavily discounted signal still counts once rather than
  vanishing entirely.
- **Wilson lower bound.** Every dimension reports the lower bound of the 95% Wilson interval
  next to the posterior mean — what you want when ranking an entity with five ratings against
  one with five hundred.
- **Temporal decay and trend.** Influence decays on a configurable exponential half-life (90
  days by default); a least-squares slope over a recent window labels each dimension improving,
  stable or declining.
- **Burst dampening.** A CUSUM detector over inter-arrival times finds where the signal rate
  breaks from its own baseline and discounts that window to 30% weight, rather than using a
  fixed rate limit that a patient attacker can wait out.
- **Source credibility weighting.** Sources are scored on review count, diversity, historical
  accuracy, account age and flagged ratio; the result multiplies the signal weight. Unknown
  sources fall back to neutral defaults.
- **Trust graph.** A directed NetworkX graph gives multi-hop trust with per-hop damping, n-hop
  subgraph extraction, Katz centrality on the reversed graph (so influence flows from
  endorsers) and Louvain community detection.
- **Explainability.** `ScoreTracer` records the prior, Bayesian delta, trust term and dampening
  for each request, and a counterfactual pass answers "what would this be without the dampened
  signals, or without the trust bonus?".
- **Pluggable analysis.** Review text yields sentiment, a fake probability and tags. The
  default analyser is a dependency-free lexicon, so the service runs with no API key;
  `ANALYZER_BACKEND=claude` swaps in Claude Haiku.

## Architecture

```
POST /api/v1/entities/{id}/signals
      ├── analyzer.analyze(text) ─► sentiment, fake_probability, tags
      │      MockAnalyzer (default) │ ClaudeAnalyzer
      └── persist Signal row

GET  /api/v1/entities/{id}/score
      ├── load Signal rows + SourceCredibility for each source
      ▼
  ScoringPipeline.score(signals, trust_bonus=0.0)
      ├── CUSUMDetector.detect ....... burst window → weight x0.3, alert emitted
      ├── compute_decay .............. exponential half-life on signal age
      ├── compute_credibility ........ per-source multiplier
      ├── DirichletMultinomial ....... per-dimension posterior over 1-5 stars
      ├── wilson_from_stars .......... lower confidence bound per dimension
      ├── compute_velocity ........... slope → improving / stable / declining
      ├── aggregate_dimensions ....... weighted merge of the four dimensions
      ├── classify_tier .............. platinum / gold / silver / bronze / untrusted
      ├── ScoreTracer ................ factor-by-factor breakdown + alerts
      └── compute_counterfactuals .... score without dampened signals / trust bonus
      ▼
  ScoreResponse (JSON)

GET  /api/v1/entities/{id}/trust/{graph,influence}
      └── TrustGraph (NetworkX DiGraph) → n-hop subgraph, Katz centrality
```

`ScoringPipeline.score()` accepts `trust_bonus` and `TrustGraph.compute_trust_bonus()` produces
it (mean incoming edge weight scaled by 0.15); the HTTP score route currently scores without it.
`CoordinationDetector`, `ReciprocalDetector`, `detect_regime_change` and `detect_communities`
are unit-tested library components composed by `DetectionManager`, not yet wired into the
request path.

| Directory | Contents |
|---|---|
| `engine/api/`, `engine/schemas/` | FastAPI routers and Pydantic request/response models |
| `engine/scoring/` | Bayesian update, Wilson bound, decay, aggregation, tiers, pipeline |
| `engine/detection/` | CUSUM burst, coordination (TF-IDF), reciprocal, credibility |
| `engine/graph/`, `engine/temporal/` | Trust propagation, Katz, Louvain; trend and change point |
| `engine/explainability/` | Score tracer and counterfactuals |
| `engine/seed/` | Demo data including review-bombing and coordinated-praise scenarios |
| `dashboard/` | React 19 + Vite read-only UI over the API |

## Quickstart

The scoring engine is pure Python and needs no database:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Full stack (PostgreSQL 16, Redis 7, API), plus the dashboard, which proxies `/api` to port 8000:

```bash
cp .env.example .env
docker compose up --build        # API on :8000, docs at /docs, health at /health
cd dashboard && npm install && npm run dev   # http://localhost:3000
```

`alembic.ini` is present but the repository ships no migration scripts, so the tables have to be
created from the SQLAlchemy metadata in `engine/models/` before the persistence-backed endpoints
work. Once they exist, `POST /api/v1/admin/seed` loads 130 entities (10 merchants, 10 services,
10 products, 100 reviewer accounts), a few thousand signals spread over the past year, trust
edges across three communities, and two attack scenarios: a 15-signal review-bombing burst and
8 near-duplicate five-star reviews.

## Usage

Score a set of signals directly, without the HTTP layer:

```python
from datetime import datetime, timezone, timedelta
from engine.analysis.mock import MockAnalyzer
from engine.scoring.pipeline import ScoringPipeline

now = datetime.now(timezone.utc)
signals = [
    {
        "value": 5.0,
        "text": "Fast delivery and the packaging was excellent",
        "dimension": "quality",
        "source_stats": {"review_count": 40, "review_diversity": 0.8, "accuracy": 0.9,
                         "account_age_days": 400, "flagged_ratio": 0.0},
        "created_at": now - timedelta(days=i * 3),
    }
    for i in range(12)
]

result = ScoringPipeline(analyzer=MockAnalyzer()).score(signals, trust_bonus=0.05)
print(result["overall"], result["tier"])
print(result["dimensions"]["quality"])  # score, wilson_lower, trend, signals, confidence
print(result["breakdown"], result["alerts"], result["counterfactual"])
```

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/api/v1/entities` | Create entity (merchant, user, service, product) |
| GET | `/api/v1/entities` | List entities, filterable by type |
| GET | `/api/v1/entities/{id}` | Entity with its stored score and tier |
| DELETE | `/api/v1/entities/{id}` | Soft delete |
| POST | `/api/v1/entities/{id}/signals` | Submit a signal (value 1.0-5.0, optional text) |
| GET | `/api/v1/entities/{id}/signals` | List signals, filter by dimension or dampened |
| GET | `/api/v1/entities/{id}/score` | Recompute the full score with breakdown |
| GET | `/api/v1/entities/{id}/score/history` | Stored score history |
| POST | `/api/v1/entities/{id}/trust` | Create a trust edge |
| GET | `/api/v1/entities/{id}/trust/graph` | N-hop trust subgraph |
| GET | `/api/v1/entities/{id}/trust/influence` | Katz centrality influence |
| GET | `/api/v1/analytics/overview` | Counts, tier distribution, dampened total |
| GET | `/api/v1/analytics/leaderboard` | Top entities by stored score |
| GET | `/api/v1/sources/{id}/credibility` | Source credibility record |
| POST | `/api/v1/admin/seed` | Load demo data |

## Configuration

Read from environment variables or `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://trustrank:trustrank@localhost:5432/trustrank` | Async Postgres DSN |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis DSN |
| `ANALYZER_BACKEND` | `mock` | `mock` or `claude` |
| `ANTHROPIC_API_KEY` | empty | Required when the backend is `claude` |
| `DECAY_HALF_LIFE_DAYS` | `90` | Signal decay half-life |
| `VELOCITY_WINDOW_DAYS` | `7` | Window for the trend slope |
| `TREND_THRESHOLD` | `0.02` | Minimum slope to call a trend |

Detector sensitivities, trust damping, tier thresholds and the dimension weights (quality 0.35,
reliability 0.25, responsiveness 0.20, trust 0.20) are settings too — `engine/config.py` has the
full list.

## Tech stack

Python 3.12, FastAPI, SQLAlchemy 2 (async) with asyncpg, PostgreSQL 16, Redis 7, Pydantic v2,
NetworkX, NumPy, SciPy, scikit-learn, Anthropic SDK, Docker Compose. Dashboard: React 19,
Vite 6, Tailwind CSS 4, Recharts, D3.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Covers the scoring maths, each detector, the trust graph and influence scoring, the temporal
functions, tracer and counterfactuals, and the end-to-end pipeline. No database or network needed.

## License

MIT
