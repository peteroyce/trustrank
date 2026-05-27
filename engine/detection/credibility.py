def _normalize(value: float, target: float) -> float:
    if target <= 0:
        return 0.0
    return min(1.0, value / target)

def compute_credibility(review_count: int, review_diversity: float, accuracy: float,
                        account_age_days: int, flagged_ratio: float) -> float:
    score = (0.20 * _normalize(review_count, 50) + 0.40 * max(0.0, min(1.0, review_diversity)) +
             0.20 * max(0.0, min(1.0, accuracy)) + 0.10 * _normalize(account_age_days, 180) +
             0.10 * (1.0 - max(0.0, min(1.0, flagged_ratio))))
    return max(0.0, min(1.0, score))
