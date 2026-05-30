from datetime import datetime, timedelta

def detect_trend(scores: list[tuple[datetime, float]], window_days: int = 30, threshold: float = 0.02) -> tuple[str, float]:
    if len(scores) < 2: return "stable", 0.0
    now = max(t for t, _ in scores)
    cutoff = now - timedelta(days=window_days)
    recent = [(t, v) for t, v in scores if t >= cutoff]
    if len(recent) < 2: return "stable", 0.0
    t0 = min(t for t, _ in recent)
    xs = [(t - t0).total_seconds() / 86400.0 for t, _ in recent]
    ys = [v for _, v in recent]
    n = len(xs)
    mean_x, mean_y = sum(xs)/n, sum(ys)/n
    num = sum((x-mean_x)*(y-mean_y) for x,y in zip(xs,ys))
    den = sum((x-mean_x)**2 for x in xs)
    if den == 0: return "stable", 0.0
    slope = num / den
    if slope > threshold: return "improving", slope
    elif slope < -threshold: return "declining", slope
    return "stable", slope
