from datetime import datetime, timezone
from engine.analysis.base import BaseAnalyzer
from engine.scoring.wilson import wilson_from_stars
from engine.scoring.bayesian import DirichletMultinomial
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
            return {
                "overall": 3.0,
                "confidence": 0.0,
                "tier": "bronze",
                "dimensions": {},
                "breakdown": tracer.get_breakdown(),
                "alerts": [],
                "counterfactual": {},
            }

        by_dim: dict[str, list] = {}
        for s in signals:
            dim = s.get("dimension", "quality")
            by_dim.setdefault(dim, []).append(s)

        times = [s["created_at"] for s in signals]
        burst = self.burst_detector.detect(times, now)
        if burst.burst_detected:
            tracer.add_alert(
                "burst_detected",
                f"{len(burst.dampened_indices)} signals dampened",
                "medium",
            )
            for idx in burst.dampened_indices:
                if idx < len(signals):
                    signals[idx]["_dampened"] = True

        dim_scores = {}
        dim_details = {}
        total_signals = 0

        for dim_name, dim_signals in by_dim.items():
            weighted_values = []
            for s in dim_signals:
                decay = compute_decay(s["created_at"], now, settings.decay_half_life_days)
                cred = compute_credibility(
                    **s.get(
                        "source_stats",
                        {
                            "review_count": 1,
                            "review_diversity": 0.5,
                            "accuracy": 0.5,
                            "account_age_days": 30,
                            "flagged_ratio": 0.0,
                        },
                    )
                )
                dampen = 0.3 if s.get("_dampened") else 1.0
                weight = decay * cred * dampen
                weighted_values.append((s["value"], weight))

            dm = DirichletMultinomial()
            counts = [0, 0, 0, 0, 0]
            for v, w in weighted_values:
                star = max(0, min(4, round(v) - 1))
                counts[star] += max(1, round(w))
            dm.update(counts)
            score = dm.weighted_mean()
            wilson = wilson_from_stars(score, len(dim_signals))
            conf = dm.confidence()

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
                "score": round(score, 2),
                "wilson_lower": round(wilson, 3),
                "trend": trend,
                "signals": len(dim_signals),
                "confidence": round(conf, 2),
            }
            total_signals += len(dim_signals)

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
            [
                {
                    "value": s["value"],
                    "weight": 1.0,
                    "dampened": s.get("_dampened", False),
                }
                for s in signals
            ],
            current_score=overall,
            trust_bonus=trust_bonus,
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
