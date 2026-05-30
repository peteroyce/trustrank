from dataclasses import dataclass
import numpy as np

@dataclass
class ChangePointResult:
    change_detected: bool = False
    change_index: int | None = None
    confidence: float = 0.0

def detect_regime_change(values: list[float], threshold: float = 0.8) -> ChangePointResult:
    if len(values) < 10: return ChangePointResult()
    arr = np.array(values)
    n = len(arr)
    total_var = np.var(arr)
    if total_var < 1e-10: return ChangePointResult()
    best_ratio, best_idx = 0.0, 0
    for t in range(5, n - 5):
        left, right = arr[:t], arr[t:]
        var_left = np.var(left) if len(left) > 1 else total_var
        var_right = np.var(right) if len(right) > 1 else total_var
        weighted_var = (len(left)*var_left + len(right)*var_right) / n
        if weighted_var < 1e-10: continue
        ratio = 1.0 - weighted_var / total_var
        mean_diff = abs(np.mean(left) - np.mean(right))
        combined = ratio * 0.5 + min(1.0, mean_diff / (np.std(arr) + 1e-10)) * 0.5
        if combined > best_ratio:
            best_ratio, best_idx = combined, t
    if best_ratio >= threshold:
        return ChangePointResult(change_detected=True, change_index=best_idx, confidence=best_ratio)
    return ChangePointResult(confidence=best_ratio)
