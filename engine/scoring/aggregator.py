from engine.config import settings


def aggregate_dimensions(dimension_scores: dict[str, float], weights: dict[str, float] | None = None) -> float:
    w = weights or settings.default_dimension_weights
    total_weight = sum(w.get(d, 0) for d in dimension_scores)
    if total_weight == 0:
        return 3.0
    return sum(dimension_scores[d] * w.get(d, 0) for d in dimension_scores) / total_weight


def classify_tier(score: float, signal_count: int, confidence: float) -> str:
    # Platinum and gold require confidence gating; silver and bronze do not.
    premium_tiers = [
        ("platinum", settings.tier_platinum_score, settings.tier_platinum_signals, settings.tier_platinum_confidence),
        ("gold", settings.tier_gold_score, settings.tier_gold_signals, settings.tier_gold_confidence),
    ]
    for name, min_score, min_signals, min_conf in premium_tiers:
        if score >= min_score and signal_count >= min_signals and confidence >= min_conf:
            return name
    # Silver and bronze are awarded by score + signals alone.
    if score >= settings.tier_silver_score and signal_count >= settings.tier_silver_signals:
        return "silver"
    if score >= settings.tier_bronze_score and signal_count >= settings.tier_bronze_signals:
        return "bronze"
    if score < settings.tier_bronze_score:
        return "untrusted"
    return "bronze"
