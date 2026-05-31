def compute_counterfactuals(signals: list[dict], current_score: float, trust_bonus: float = 0.0) -> dict:
    result = {}
    undampened = [s for s in signals if not s.get("dampened", False)]
    if undampened:
        total_w = sum(s["weight"] for s in undampened)
        result["without_dampened"] = sum(s["value"]*s["weight"] for s in undampened)/total_w if total_w > 0 else current_score
    else:
        result["without_dampened"] = current_score
    result["without_trust_bonus"] = current_score - trust_bonus
    result["without_last_30d"] = current_score
    return result
