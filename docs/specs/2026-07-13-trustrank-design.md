# trustrank — Entity Reputation & Trust Scoring Engine

**Date:** 2026-07-13
**Status:** Approved
**Author:** Peter Royce Saldanha

---

## Overview

A production-grade reputation scoring engine with multi-dimensional profiling, manipulation resistance, trust graph analysis, and full explainability. Not a CRUD wrapper — the scoring math, detection algorithms, and graph analysis ARE the product.

Ships with a mock AI analyzer by default. Set `ANTHROPIC_API_KEY` to enable real Claude-powered review analysis.

## Tech Stack

- **Backend:** Python 3.12, FastAPI, SQLAlchemy, Alembic
- **Database:** PostgreSQL 16
- **Cache:** Redis (score caching, rate limiting)
- **Graph:** NetworkX (trust graph, community detection, centrality)
- **ML/Stats:** NumPy, SciPy, scikit-learn (for embedding clustering in coordination detection)
- **AI:** Anthropic Claude API (pluggable, mock by default)
- **Frontend:** React 19, Tailwind CSS v4, Recharts, D3.js (force graph)
- **Infra:** Docker Compose (PostgreSQL + Redis + API + Dashboard)

## Data Model

### Entity
Anything being scored — merchant, driver, restaurant, user, service provider.

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| type | enum | merchant, user, service, product |
| name | string | Display name |
| metadata | JSONB | Arbitrary key-value (category, location, etc.) |
| created_at | timestamp | Registration time |

### Signal
A scored event submitted against an entity.

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| entity_id | UUID | FK → Entity |
| source_id | UUID | FK → Entity (the reviewer/submitter) |
| dimension | enum | quality, reliability, responsiveness, trust |
| type | enum | review, transaction, complaint, verification, dispute |
| value | float | Raw score (1.0 - 5.0 for reviews, 0.0 or 1.0 for binary) |
| text | text | Optional review text |
| tags | text[] | AI-extracted or user-supplied tags |
| sentiment | float | -1.0 to 1.0 (from analyzer) |
| fake_probability | float | 0.0 to 1.0 (from analyzer) |
| weight | float | Computed effective weight after all adjustments |
| raw_weight | float | Base weight before adjustments |
| dampened | boolean | True if manipulation detection dampened this signal |
| dampening_reason | string | Why it was dampened |
| created_at | timestamp | When signal was submitted |

### DimensionScore
Per-dimension score snapshot for an entity.

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| entity_id | UUID | FK → Entity |
| dimension | enum | quality, reliability, responsiveness, trust |
| wilson_lower | float | Wilson score interval lower bound |
| bayesian_score | float | Beta-Binomial / Dirichlet posterior mean |
| confidence | float | 0.0 - 1.0, derived from sample size and variance |
| signal_count | int | Total signals for this dimension |
| alpha | float | Beta distribution alpha parameter |
| beta_param | float | Beta distribution beta parameter |
| dirichlet | float[5] | Dirichlet parameters for 1-5 star distribution |
| trend | enum | improving, stable, declining |
| trend_slope | float | Linear regression slope (30-day rolling) |
| updated_at | timestamp | Last recomputation |

### OverallScore
Aggregated entity score across all dimensions.

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| entity_id | UUID | FK → Entity (unique) |
| score | float | Weighted combination of dimension scores |
| confidence | float | Min confidence across dimensions |
| tier | enum | platinum, gold, silver, bronze, untrusted |
| total_signals | int | Sum across all dimensions |
| trust_bonus | float | Score contribution from trust graph propagation |
| manipulation_penalty | float | Total dampening applied |
| alerts | JSONB | Active alerts (burst detected, regime change, etc.) |
| breakdown | JSONB | Full score decomposition for explainability |
| updated_at | timestamp | Last recomputation |

### TrustEdge
Directional trust relationship between entities.

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| source_id | UUID | FK → Entity (the truster) |
| target_id | UUID | FK → Entity (the trusted) |
| category | string | Context-specific (food_quality, delivery, general) |
| weight | float | 0.0 - 1.0 trust strength |
| evidence_type | enum | transaction, vouching, verification |
| created_at | timestamp | When edge was created |

### ScoreHistory
Audit trail of score changes over time.

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| entity_id | UUID | FK → Entity |
| score | float | Score at this point |
| dimension_scores | JSONB | All dimension scores at this point |
| trigger_signal_id | UUID | FK → Signal that caused this snapshot |
| breakdown | JSONB | Full decomposition at this point |
| created_at | timestamp | Snapshot time |

### SourceCredibility
Rolling credibility score for signal sources.

| Field | Type | Description |
|-------|------|-------------|
| source_id | UUID | FK → Entity (unique) |
| credibility_score | float | 0.0 - 1.0 |
| review_count | int | Total reviews submitted |
| review_diversity | float | Entropy of reviewed entities |
| accuracy_score | float | How often this source's reviews align with consensus |
| account_age_days | int | Days since account creation |
| flagged_count | int | How many of their signals were dampened |
| updated_at | timestamp | Last recomputation |

## Scoring Engine

### 2.1 Wilson Score Interval

For confidence-adjusted ratings. An entity with 5/5 from 3 reviews scores LOWER than 4.7 from 200 reviews.

```
wilson_lower = (p̂ + z²/2n - z√(p̂(1-p̂)/n + z²/4n²)) / (1 + z²/n)
```

Where p̂ = positive fraction, z = 1.96 (95% CI), n = sample size. For 1-5 star ratings, normalize to 0-1 range: p̂ = (mean_rating - 1) / 4. Wilson lower is the sort-worthy score for leaderboards (pessimistic ranking). The bayesian_score (from Beta-Binomial/Dirichlet) is used for tier classification and the explainability breakdown.

### 2.2 Beta-Binomial Model

For binary outcomes (transaction success/failure, dispute yes/no).

- Prior: α₀, β₀ (from category-level hierarchical prior)
- Update: α' = α₀ + successes, β' = β₀ + failures
- Posterior mean: α' / (α' + β')
- Posterior variance: α'β' / ((α'+β')²(α'+β'+1))

Category-level hierarchical priors: new restaurants start with the average restaurant's α/β, not a global average. Recomputed on each scoring pipeline run by averaging α/β across all entities of the same type. Cached in Redis with a 1-hour TTL to avoid recomputing on every signal.

### 2.3 Dirichlet-Multinomial Model

For star rating distributions (1-5). Models the FULL distribution, not just the mean.

- Prior: α = [α₁, α₂, α₃, α₄, α₅] (from category hierarchical prior)
- Update: α'ᵢ = αᵢ + count(rating = i)
- Posterior mean for rating k: α'ₖ / Σα'

An entity with all 3-star reviews vs one with half 1-star and half 5-star have the same mean but very different Dirichlet parameters. The variance tells the story.

### 2.4 Temporal Decay

Exponential decay with configurable half-life:

```
weight_decay = exp(-ln(2) * age_days / half_life)
```

Default half-life: 90 days. Recent signals matter more than old ones.

**Velocity awareness:** Track the rate of score change over rolling 7-day windows. If the entity is rapidly improving (positive slope > threshold), apply a boost multiplier (1.1x) to recent positive signals. If rapidly declining, no dampening — let the score fall naturally (the decline itself is information).

### 2.5 Dimension Aggregation

Overall score = weighted combination of dimension scores:

```
overall = Σ(dimension_score × dimension_weight) / Σ(dimension_weight)
```

Default weights: quality=0.35, reliability=0.25, responsiveness=0.20, trust=0.20.

Weights are configurable per entity type. A delivery service might weight reliability at 0.40.

### 2.6 Tier Classification

| Tier | Score Range | Min Signals | Min Confidence |
|------|------------|-------------|----------------|
| Platinum | ≥ 4.5 | 100 | 0.85 |
| Gold | ≥ 4.0 | 50 | 0.75 |
| Silver | ≥ 3.5 | 20 | 0.60 |
| Bronze | ≥ 2.5 | 5 | 0.40 |
| Untrusted | < 2.5 or flagged | — | — |

Tier requires BOTH score threshold AND minimum signal count AND minimum confidence. A 5.0 score from 2 reviews stays Bronze until it accumulates enough evidence.

## Manipulation Resistance Engine

### 3.1 Burst Detection (CUSUM)

Cumulative Sum control chart on signal arrival rate.

- Compute entity's baseline arrival rate λ from historical signals (exponential moving average)
- For each new signal, compute deviation: sₙ = max(0, sₙ₋₁ + (xₙ - λ - k))
- If sₙ > threshold h, trigger burst alert
- Parameters: k = 0.5σ (allowance), h = 5σ (decision boundary)

When burst is detected:
1. All signals in the burst window get `dampened = true`
2. Their weight is reduced by 70%
3. Alert is attached to the entity's score response
4. The signals still contribute, just with reduced influence
5. Manual review can un-dampen signals

### 3.2 Coordination Detection

Two-pronged detection for coordinated fake reviews:

**Textual clustering:**
- Compute sentence embeddings for review text (using sentence-transformers, `all-MiniLM-L6-v2`)
- Cluster recent reviews using DBSCAN (eps=0.15, min_samples=3)
- If a cluster contains >3 reviews with cosine similarity >0.85, flag as coordinated
- In mock mode: uses TF-IDF vectors + cosine similarity instead of transformer embeddings

**Temporal pattern analysis:**
- Fit a Poisson process to signal arrival times
- Compute the inter-arrival time coefficient of variation (CV)
- Natural reviews: CV ≈ 1.0 (exponential inter-arrivals)
- Coordinated reviews: CV < 0.5 (suspiciously regular timing)
- If CV < 0.5 AND cluster detected, flag with high confidence

### 3.3 Source Credibility Scoring

Every signal source (reviewer/submitter) gets a credibility score that affects how much their signals weigh.

```
credibility = (
    0.30 × normalize(review_count, target=50) +
    0.25 × review_diversity_entropy +
    0.25 × accuracy_score +
    0.10 × normalize(account_age_days, target=180) +
    0.10 × (1 - flagged_ratio)
)
```

- **review_diversity**: Shannon entropy of the entity distribution they've reviewed. Reviewing only one entity = 0 entropy = low credibility.
- **accuracy_score**: How often their rating is within 1 star of the entity's consensus score. Consistently contrarian = lower credibility.
- **flagged_ratio**: Fraction of their signals that have been dampened by other detection systems.

Credibility score multiplies the signal's base weight:
```
effective_weight = raw_weight × source_credibility × decay_factor
```

### 3.4 Reciprocal Network Detection

Graph analysis to find "you scratch my back, I scratch yours" collusion.

For each entity pair (A, B):
1. Compute reviewer overlap: Jaccard similarity of their reviewer sets
2. Compute sentiment correlation: Pearson correlation of review scores from shared reviewers
3. If Jaccard > 0.3 AND sentiment correlation > 0.7, flag the relationship
4. All signals between flagged pairs get a 50% weight reduction

Additionally: detect review rings (cycles of length 3-5 in the reviewer→entity bipartite graph where all edges are positive). Uses DFS cycle detection on the filtered graph.

## Trust Graph Analysis

### 4.1 Transitive Trust Propagation

Trust flows directionally through the graph with decay:

```
indirect_trust(A→C) = trust(A→B) × trust(B→C) × damping_factor
```

- damping_factor: 0.7 per hop (configurable)
- Max depth: 2 hops
- If multiple paths exist, take the max (optimistic trust)
- Trust contribution to overall score: `trust_bonus = mean(incoming_indirect_trust) × 0.15`

### 4.2 Context-Specific Edges

Trust edges carry a `category` label. Propagation only flows within matching categories.

Entity A trusts Entity B for "food_quality" — this trust does NOT propagate to B's "delivery" dimension. Category matching uses exact match with a wildcard "general" that propagates everywhere.

### 4.3 Community Detection

Louvain algorithm on the trust graph (undirected projection, weighted by trust strength).

- Communities are computed nightly
- Signals from within the same community get a 1.1x weight boost (community endorsement)
- Signals from outside get no adjustment (neutral, not penalized)

### 4.4 Influence Scoring (Katz Centrality)

```
influence(v) = α × Σ(A^k × 1)  for k=1 to ∞
```

Approximated iteratively. α = 0.1 (below spectral radius).

High-influence entities (top 5%) get their outgoing signals audited more aggressively — their signals pass through ALL detection subsystems regardless of volume. This prevents well-connected entities from gaming the system through their network position.

## Temporal Analysis Engine

### 5.1 Trend Detection

Rolling 30-day linear regression on signal scores:
- slope > 0.02/day → "improving"
- slope < -0.02/day → "declining"
- else → "stable"

Exposed in every score response.

### 5.2 Regime Change Detection (BOCPD)

Bayesian Online Change-Point Detection:

- Maintains a run-length distribution over time
- When a new signal arrives, compute the probability that it came from the same distribution as recent signals vs. a new distribution
- If P(change-point) > 0.8 for any time step, flag a regime change
- The system splits the score history at the change point
- Post-change signals get 2x weight, pre-change signals get accelerated decay

Use case: a restaurant changes management. Old reviews shouldn't drag down (or prop up) the new reality.

### 5.3 Seasonality (Optional)

STL decomposition (Seasonal-Trend-Loess) on weekly signal aggregates:
- Extracts: trend + seasonal + residual
- Score normalization: subtract seasonal component before scoring
- Only activates for entities with 6+ months of signal history

Off by default. Enable per entity type in config.

## AI Analyzer (Pluggable)

### Interface

```python
class BaseAnalyzer(ABC):
    @abstractmethod
    async def analyze(self, text: str, context: dict) -> AnalysisResult:
        """Returns sentiment, fake_probability, tags, summary."""
        pass
```

### MockAnalyzer (default)
- Sentiment: VADER-based (lexicon approach, no API needed)
- Fake probability: heuristic (text length, exclamation density, generic praise patterns)
- Tags: TF-IDF keyword extraction from text
- Ships working out of the box, no external dependencies

### ClaudeAnalyzer (opt-in via ANTHROPIC_API_KEY)
- Sentiment: Claude analysis with nuanced scoring (-1.0 to 1.0)
- Fake probability: Claude assessment with reasoning
- Tags: Claude extraction of domain-specific quality tags
- Summary: One-line review summary
- Rate-limited: 10 req/s with exponential backoff

Selection in config:
```python
ANALYZER_BACKEND = os.getenv("ANALYZER_BACKEND", "mock")  # "mock" | "claude"
```

## Explainability Engine

Every score computation produces an audit trail:

### Score Breakdown

```json
{
  "entity_id": "merchant_429",
  "overall": 4.12,
  "confidence": 0.87,
  "tier": "Gold",
  "dimensions": {
    "quality": {"score": 4.3, "wilson_lower": 4.1, "trend": "stable", "signals": 142, "confidence": 0.91},
    "reliability": {"score": 3.8, "wilson_lower": 3.5, "trend": "improving", "signals": 89, "confidence": 0.84},
    "responsiveness": {"score": 4.4, "wilson_lower": 4.1, "trend": "stable", "signals": 67, "confidence": 0.79},
    "trust": {"score": 4.0, "wilson_lower": 3.7, "trend": "stable", "signals": 31, "confidence": 0.72}
  },
  "factors": {
    "base_prior": 3.2,
    "bayesian_update": "+0.94",
    "trust_propagation": "+0.12",
    "temporal_decay": "-0.08",
    "manipulation_dampening": "-0.06",
    "final": 4.12
  },
  "alerts": [
    {"type": "burst_detected", "detail": "7 signals in 2h on Jun 14, 5 dampened", "severity": "medium"}
  ],
  "counterfactual": {
    "without_dampened": 4.18,
    "without_trust_bonus": 4.00,
    "without_last_30d": 4.05
  },
  "regime": {
    "change_detected": false,
    "last_change": null,
    "current_regime_start": "2026-01-15"
  }
}
```

### Counterfactual Analysis

For any entity, compute "what if" scenarios:
- "Without the 3 most recent dampened signals, score would be X"
- "Without trust graph bonus, score would be X"
- "Without signals older than 30 days, score would be X"

This is computed on-demand per API request, not stored.

## API Endpoints

### Entities
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/entities` | Register entity (type, name, metadata) |
| GET | `/api/v1/entities/{id}` | Get entity with current score |
| GET | `/api/v1/entities` | List entities (filter by type, tier, score range) |
| DELETE | `/api/v1/entities/{id}` | Soft-delete entity |

### Signals
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/entities/{id}/signals` | Submit signal (triggers full scoring pipeline) |
| GET | `/api/v1/entities/{id}/signals` | List signals (filter by dimension, type, dampened) |

### Scores
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/entities/{id}/score` | Full score breakdown with explainability |
| GET | `/api/v1/entities/{id}/score/history` | Score timeline (for charts) |
| GET | `/api/v1/entities/{id}/score/counterfactual` | What-if scenarios |

### Trust Graph
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/entities/{id}/trust` | Create trust edge to another entity |
| GET | `/api/v1/entities/{id}/trust/graph` | Get trust subgraph (1-2 hops) |
| GET | `/api/v1/entities/{id}/trust/influence` | Katz centrality score |

### Analytics
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/analytics/overview` | System-wide stats (total entities, tier distribution, signal volume) |
| GET | `/api/v1/analytics/leaderboard` | Top entities by tier/score with pagination |
| GET | `/api/v1/analytics/alerts` | Active manipulation alerts across all entities |
| GET | `/api/v1/analytics/detection` | Detection system stats (dampened %, false positive rate) |

### Source Credibility
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/sources/{id}/credibility` | Source credibility breakdown |

## Dashboard

### Pages

1. **Leaderboard** — sortable/filterable entity table with tier badges, sparkline score trends, signal counts, alert indicators
2. **Entity Detail** — score timeline (Recharts line chart), dimension radar chart, signal history table with dampening indicators, tag cloud, trust graph visualization (D3 force-directed), regime change markers on timeline
3. **Trust Graph Explorer** — full network D3 visualization with community coloring, click-to-inspect, influence size scaling
4. **Detection Dashboard** — burst timeline, coordination clusters, flagged signals queue, source credibility distribution
5. **Analytics** — tier distribution donut, signals/day chart, detection rate trends, top tags word cloud

## Seeding

Ships with a data seeder that generates:
- 200 entities across 4 types (merchant, user, service, product)
- 5,000 signals with realistic distributions (most positive, some negative, a few coordinated fake clusters)
- 50 trust edges forming 3-4 communities
- 10 "attack" scenarios baked in (review bombing, reciprocal rings, burst patterns) so the detection systems have something to catch
- Deterministic seed (random state=42) for reproducible demos

## Project Structure

```
trustrank/
├── engine/
│   ├── main.py               # FastAPI app entry
│   ├── config.py              # Settings (env-based)
│   ├── api/
│   │   ├── entities.py
│   │   ├── signals.py
│   │   ├── scores.py
│   │   ├── trust.py
│   │   ├── analytics.py
│   │   └── sources.py
│   ├── scoring/
│   │   ├── wilson.py          # Wilson score interval
│   │   ├── bayesian.py        # Beta-Binomial + Dirichlet-Multinomial
│   │   ├── dimensions.py      # Multi-dimensional profile manager
│   │   ├── decay.py           # Temporal decay + velocity awareness
│   │   ├── aggregator.py      # Dimension → overall score + tier
│   │   └── pipeline.py        # Full scoring pipeline orchestrator
│   ├── detection/
│   │   ├── burst.py           # CUSUM change-point detection
│   │   ├── coordination.py    # Embedding clustering + Poisson timing
│   │   ├── credibility.py     # Source reputation scoring
│   │   ├── reciprocal.py      # Graph-based collusion detection
│   │   └── manager.py         # Runs all detectors, aggregates results
│   ├── graph/
│   │   ├── trust.py           # Transitive trust propagation
│   │   ├── community.py       # Louvain clustering
│   │   └── influence.py       # Katz centrality
│   ├── temporal/
│   │   ├── trend.py           # Rolling linear regression
│   │   ├── changepoint.py     # BOCPD regime detection
│   │   └── seasonal.py        # STL decomposition (optional)
│   ├── analysis/
│   │   ├── base.py            # Analyzer ABC
│   │   ├── mock.py            # VADER + TF-IDF + heuristics
│   │   └── claude.py          # Claude API integration
│   ├── explainability/
│   │   ├── tracer.py          # Score audit trail builder
│   │   └── counterfactual.py  # What-if scenario computation
│   ├── models/
│   │   ├── entity.py
│   │   ├── signal.py
│   │   ├── score.py
│   │   ├── trust.py
│   │   └── source.py
│   ├── db/
│   │   ├── session.py         # SQLAlchemy engine + session
│   │   └── migrations/        # Alembic
│   └── seed/
│       ├── seeder.py          # Data generation
│       └── scenarios.py       # Attack scenarios for demo
├── dashboard/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Leaderboard.tsx
│   │   │   ├── EntityDetail.tsx
│   │   │   ├── TrustGraph.tsx
│   │   │   ├── Detection.tsx
│   │   │   └── Analytics.tsx
│   │   ├── components/
│   │   └── lib/
│   └── package.json
├── docker-compose.yml         # PostgreSQL + Redis + API + Dashboard
├── Dockerfile
├── pyproject.toml
├── README.md
└── docs/
    └── specs/
        └── 2026-07-13-trustrank-design.md
```

## Non-Goals

- Real-time streaming (WebSocket for score updates) — not in v1
- Multi-tenancy / API keys — single-tenant for portfolio
- Horizontal scaling — single-instance is fine
- Production deployment — Docker Compose local only
- Email/webhook notifications — dashboard alerts only
