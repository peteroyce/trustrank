import math
from datetime import datetime, timedelta


def compute_decay(signal_time: datetime, current_time: datetime, half_life_days: float) -> float:
    age_days = (current_time - signal_time).total_seconds() / 86400.0
    if age_days < 0:
        return 1.0
    return math.exp(-math.log(2) * age_days / half_life_days)


def compute_velocity(scores: list[tuple[datetime, float]], window_days: int = 7) -> float:
    if len(scores) < 2:
        return 0.0
    now = max(t for t, _ in scores)
    cutoff = now - timedelta(days=window_days)
    recent = [(t, v) for t, v in scores if t >= cutoff]
    if len(recent) < 2:
        return 0.0
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
