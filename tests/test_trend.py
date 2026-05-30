from datetime import datetime, timedelta, timezone
from engine.temporal.trend import detect_trend

def test_improving_trend():
    now = datetime.now(timezone.utc)
    scores = [(now - timedelta(days=29-i), 3.0 + i*0.1) for i in range(30)]
    trend, slope = detect_trend(scores, window_days=30, threshold=0.02)
    assert trend == "improving"

def test_declining_trend():
    now = datetime.now(timezone.utc)
    scores = [(now - timedelta(days=29-i), 5.0 - i*0.1) for i in range(30)]
    trend, slope = detect_trend(scores, window_days=30, threshold=0.02)
    assert trend == "declining"

def test_stable_trend():
    now = datetime.now(timezone.utc)
    scores = [(now - timedelta(days=29-i), 4.0) for i in range(30)]
    trend, slope = detect_trend(scores, window_days=30, threshold=0.02)
    assert trend == "stable"

def test_empty_scores():
    trend, slope = detect_trend([], window_days=30, threshold=0.02)
    assert trend == "stable" and slope == 0.0
