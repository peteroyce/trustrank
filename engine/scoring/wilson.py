import math


def wilson_score_interval(positive: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    p = positive / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return max(0.0, (centre - spread) / denominator)


def wilson_from_stars(mean_rating: float, count: int, z: float = 1.96) -> float:
    if count == 0:
        return 0.0
    p = (mean_rating - 1.0) / 4.0
    p = max(0.0, min(1.0, p))
    return wilson_score_interval(positive=round(p * count), total=count, z=z)
