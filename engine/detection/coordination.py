import math
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@dataclass
class CoordinationResult:
    coordinated: bool = False
    timing_suspicious: bool = False
    cluster_size: int = 0
    max_similarity: float = 0.0
    cv_score: float = 1.0

class CoordinationDetector:
    def __init__(self, similarity_threshold: float = 0.85, cv_threshold: float = 0.5, min_cluster: int = 3):
        self.similarity_threshold = similarity_threshold
        self.cv_threshold = cv_threshold
        self.min_cluster = min_cluster

    def detect(self, signals: list[dict]) -> CoordinationResult:
        if len(signals) < self.min_cluster:
            return CoordinationResult()
        text_result = self._check_text_similarity(signals)
        timing_result = self._check_timing(signals)
        coordinated = text_result["coordinated"] or (text_result["max_similarity"] > 0.7 and timing_result["suspicious"])
        cluster_size = text_result["cluster_size"]
        # When coordination is detected via combined path but strict clustering missed it,
        # recount at a broader threshold to surface the actual community size
        if coordinated and cluster_size < self.min_cluster:
            cluster_size = text_result.get("broad_cluster_size", cluster_size)
        return CoordinationResult(
            coordinated=coordinated,
            timing_suspicious=timing_result["suspicious"],
            cluster_size=cluster_size,
            max_similarity=text_result["max_similarity"],
            cv_score=timing_result["cv"])

    def _check_text_similarity(self, signals: list[dict]) -> dict:
        texts = [s.get("text", "") for s in signals]
        texts = [t for t in texts if t and len(t) > 10]
        if len(texts) < self.min_cluster:
            return {"coordinated": False, "cluster_size": 0, "max_similarity": 0.0}
        vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        tfidf = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf)
        n = len(texts)
        visited = set()
        largest_cluster = 0
        max_sim = 0.0
        for i in range(n):
            if i in visited:
                continue
            cluster = {i}
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= self.similarity_threshold:
                    cluster.add(j)
                    max_sim = max(max_sim, sim_matrix[i, j])
            if len(cluster) >= self.min_cluster:
                largest_cluster = max(largest_cluster, len(cluster))
                visited.update(cluster)
        np.fill_diagonal(sim_matrix, 0)
        overall_max = float(sim_matrix.max()) if sim_matrix.size > 0 else 0.0
        max_sim = max(max_sim, overall_max)
        # Broad cluster: single-linkage at half the threshold to find connected communities
        broad_threshold = max(0.3, self.similarity_threshold * 0.6)
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= broad_threshold:
                    union(i, j)
        from collections import Counter
        roots = [find(i) for i in range(n)]
        broad_largest = max(Counter(roots).values()) if roots else 0
        return {"coordinated": largest_cluster >= self.min_cluster, "cluster_size": largest_cluster,
                "broad_cluster_size": broad_largest, "max_similarity": max_sim}

    def _check_timing(self, signals: list[dict]) -> dict:
        times = sorted([s["time"] for s in signals if "time" in s])
        if len(times) < 3:
            return {"suspicious": False, "cv": 1.0}
        intervals = [(times[i] - times[i - 1]).total_seconds() for i in range(1, len(times))]
        intervals = [max(x, 0.001) for x in intervals]
        mean = sum(intervals) / len(intervals)
        if mean == 0:
            return {"suspicious": True, "cv": 0.0}
        variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
        std = math.sqrt(variance)
        cv = std / mean
        return {"suspicious": cv < self.cv_threshold, "cv": cv}
