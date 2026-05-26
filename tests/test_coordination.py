from datetime import datetime, timedelta, timezone
from engine.detection.coordination import CoordinationDetector

def test_no_coordination_diverse_text():
    now = datetime.now(timezone.utc)
    signals = [
        {"text": "Great food, loved the pasta and wine selection", "time": now - timedelta(hours=24)},
        {"text": "Terrible service, waited 45 minutes for appetizers", "time": now - timedelta(hours=48)},
        {"text": "Average experience, nothing special but not bad either", "time": now - timedelta(hours=72)},
        {"text": "The ambiance was wonderful, perfect for a date night", "time": now - timedelta(hours=96)},
        {"text": "Overpriced for the quality, would not recommend to friends", "time": now - timedelta(hours=120)},
    ]
    detector = CoordinationDetector()
    result = detector.detect(signals)
    assert not result.coordinated

def test_coordination_similar_text():
    now = datetime.now(timezone.utc)
    signals = [
        {"text": "Amazing product absolutely love it highly recommend to everyone", "time": now - timedelta(minutes=10)},
        {"text": "Amazing product absolutely love this highly recommend to all", "time": now - timedelta(minutes=20)},
        {"text": "Amazing item absolutely love it highly recommend to everyone", "time": now - timedelta(minutes=30)},
        {"text": "Amazing product totally love it highly recommend to everyone", "time": now - timedelta(minutes=40)},
        {"text": "Amazing product absolutely love it strongly recommend to all", "time": now - timedelta(minutes=50)},
        {"text": "Amazing goods absolutely love it highly recommend to everyone", "time": now - timedelta(minutes=60)},
        {"text": "Amazing product absolutely love it highly recommend to anybody", "time": now - timedelta(minutes=70)},
        {"text": "Amazing product absolutely love it really recommend to everyone", "time": now - timedelta(minutes=80)},
    ]
    detector = CoordinationDetector(similarity_threshold=0.85)
    result = detector.detect(signals)
    assert result.coordinated
    assert result.cluster_size >= 3

def test_regular_timing_detected():
    now = datetime.now(timezone.utc)
    signals = [{"text": f"Review number {i} for this place", "time": now - timedelta(minutes=i * 30)} for i in range(10)]
    detector = CoordinationDetector(cv_threshold=0.5)
    result = detector.detect(signals)
    assert result.timing_suspicious

def test_empty_signals():
    detector = CoordinationDetector()
    result = detector.detect([])
    assert not result.coordinated
