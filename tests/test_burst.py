from datetime import datetime, timedelta, timezone
from engine.detection.burst import CUSUMDetector

def test_no_burst_normal_rate():
    now = datetime.now(timezone.utc)
    times = [now - timedelta(hours=i * 12) for i in range(20)]
    detector = CUSUMDetector(allowance_factor=0.5, threshold_factor=5.0)
    result = detector.detect(signal_times=times, current_time=now)
    assert not result.burst_detected

def test_burst_detected():
    now = datetime.now(timezone.utc)
    baseline = [now - timedelta(days=i) for i in range(1, 31)]
    burst = [now - timedelta(minutes=i * 8) for i in range(15)]
    times = sorted(baseline + burst)
    detector = CUSUMDetector(allowance_factor=0.5, threshold_factor=5.0)
    result = detector.detect(signal_times=times, current_time=now)
    assert result.burst_detected
    assert result.burst_window_start is not None

def test_burst_returns_dampened_indices():
    now = datetime.now(timezone.utc)
    baseline = [now - timedelta(days=i) for i in range(1, 31)]
    burst = [now - timedelta(minutes=i * 5) for i in range(20)]
    times = sorted(baseline + burst)
    detector = CUSUMDetector(allowance_factor=0.5, threshold_factor=5.0)
    result = detector.detect(signal_times=times, current_time=now)
    assert result.burst_detected
    assert len(result.dampened_indices) > 0

def test_too_few_signals():
    now = datetime.now(timezone.utc)
    times = [now - timedelta(hours=i) for i in range(3)]
    detector = CUSUMDetector()
    result = detector.detect(signal_times=times, current_time=now)
    assert not result.burst_detected
