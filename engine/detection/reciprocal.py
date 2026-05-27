from dataclasses import dataclass, field
from itertools import combinations
import numpy as np

@dataclass
class ReciprocalResult:
    flagged_pairs: list[tuple[str, str]] = field(default_factory=list)

class ReciprocalDetector:
    def __init__(self, jaccard_threshold: float = 0.3, sentiment_threshold: float = 0.7):
        self.jaccard_threshold = jaccard_threshold
        self.sentiment_threshold = sentiment_threshold

    def detect(self, entity_reviews: dict[str, list[dict]]) -> ReciprocalResult:
        entities = list(entity_reviews.keys())
        if len(entities) < 2:
            return ReciprocalResult()
        flagged = []
        for a, b in combinations(entities, 2):
            sources_a = {r["source"] for r in entity_reviews[a]}
            sources_b = {r["source"] for r in entity_reviews[b]}
            intersection = sources_a & sources_b
            union = sources_a | sources_b
            if not union:
                continue
            jaccard = len(intersection) / len(union)
            if jaccard < self.jaccard_threshold:
                continue
            if len(intersection) < 2:
                continue
            shared = sorted(intersection)
            vals_a = {r["source"]: r["value"] for r in entity_reviews[a]}
            vals_b = {r["source"]: r["value"] for r in entity_reviews[b]}
            scores_a = [vals_a[s] for s in shared if s in vals_a]
            scores_b = [vals_b[s] for s in shared if s in vals_b]
            if len(scores_a) < 2:
                continue
            # Compute Pearson correlation; if not enough variance, fall back to mean-score check
            corr = np.corrcoef(scores_a, scores_b)[0, 1]
            mean_a = sum(scores_a) / len(scores_a)
            mean_b = sum(scores_b) / len(scores_b)
            max_val = max(max(scores_a), max(scores_b), 1.0)
            norm_mean_a = mean_a / max_val
            norm_mean_b = mean_b / max_val
            # Flag if both entities received uniformly high scores from shared reviewers
            # OR if correlation is high (coordinated positive/negative sentiment)
            both_high = norm_mean_a >= self.sentiment_threshold and norm_mean_b >= self.sentiment_threshold
            corr_high = not np.isnan(corr) and abs(corr) >= self.sentiment_threshold
            if both_high or corr_high:
                flagged.append((a, b))
        return ReciprocalResult(flagged_pairs=flagged)
