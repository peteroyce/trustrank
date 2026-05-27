from dataclasses import dataclass, field
from datetime import datetime
from engine.detection.burst import CUSUMDetector, BurstResult
from engine.detection.coordination import CoordinationDetector, CoordinationResult
from engine.detection.credibility import compute_credibility
from engine.config import settings

@dataclass
class DetectionResult:
    burst: BurstResult = field(default_factory=BurstResult)
    coordination: CoordinationResult = field(default_factory=CoordinationResult)
    source_credibility: float = 0.5
    dampening_factor: float = 1.0
    reasons: list[str] = field(default_factory=list)

class DetectionManager:
    def __init__(self):
        self.burst_detector = CUSUMDetector(allowance_factor=settings.cusum_allowance_factor,
                                            threshold_factor=settings.cusum_threshold_factor)
        self.coordination_detector = CoordinationDetector(
            similarity_threshold=settings.coordination_similarity_threshold,
            cv_threshold=settings.coordination_cv_threshold)

    def run_all(self, signal_times: list[datetime], signal_texts: list[dict],
                source_stats: dict, current_time: datetime) -> DetectionResult:
        result = DetectionResult()
        result.burst = self.burst_detector.detect(signal_times, current_time)
        if result.burst.burst_detected:
            result.dampening_factor *= 0.3
            result.reasons.append(f"burst: {len(result.burst.dampened_indices)} signals in burst window")
        result.coordination = self.coordination_detector.detect(signal_texts)
        if result.coordination.coordinated:
            result.dampening_factor *= 0.4
            result.reasons.append(f"coordination: cluster of {result.coordination.cluster_size}")
        result.source_credibility = compute_credibility(**source_stats)
        result.dampening_factor *= max(0.3, result.source_credibility)
        return result
