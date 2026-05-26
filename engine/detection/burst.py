import math
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class BurstResult:
    burst_detected: bool = False
    burst_window_start: datetime | None = None
    burst_window_end: datetime | None = None
    dampened_indices: list[int] = field(default_factory=list)
    baseline_rate: float = 0.0
    observed_rate: float = 0.0

class CUSUMDetector:
    def __init__(self, allowance_factor: float = 0.5, threshold_factor: float = 5.0):
        self.allowance_factor = allowance_factor
        self.threshold_factor = threshold_factor

    def detect(self, signal_times: list[datetime], current_time: datetime) -> BurstResult:
        if len(signal_times) < 5:
            return BurstResult()
        sorted_times = sorted(signal_times)
        intervals = []
        for i in range(1, len(sorted_times)):
            dt = (sorted_times[i] - sorted_times[i - 1]).total_seconds() / 3600.0
            intervals.append(max(dt, 0.001))
        if len(intervals) < 4:
            return BurstResult()
        split = max(3, int(len(intervals) * 0.8))
        baseline_intervals = intervals[:split]
        mean_interval = sum(baseline_intervals) / len(baseline_intervals)
        variance = sum((x - mean_interval) ** 2 for x in baseline_intervals) / len(baseline_intervals)
        std_interval = math.sqrt(variance) if variance > 0 else mean_interval * 0.5
        k = self.allowance_factor * std_interval
        h = self.threshold_factor * std_interval
        cusum = 0.0
        burst_start_idx = None
        for i, interval in enumerate(intervals):
            deviation = mean_interval - interval - k
            cusum = max(0.0, cusum + deviation)
            if cusum > h and burst_start_idx is None:
                burst_start_idx = i
        if burst_start_idx is None:
            return BurstResult(baseline_rate=1.0 / mean_interval if mean_interval > 0 else 0)
        burst_window_start = sorted_times[burst_start_idx]
        dampened = [i for i, t in enumerate(sorted_times) if t >= burst_window_start]
        recent_intervals = intervals[burst_start_idx:]
        observed_rate = len(recent_intervals) / (sum(recent_intervals) if sum(recent_intervals) > 0 else 1)
        return BurstResult(burst_detected=True, burst_window_start=burst_window_start,
                          burst_window_end=sorted_times[-1], dampened_indices=dampened,
                          baseline_rate=1.0 / mean_interval if mean_interval > 0 else 0, observed_rate=observed_rate)
