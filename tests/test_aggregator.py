from engine.scoring.aggregator import aggregate_dimensions, classify_tier


def test_equal_weights():
    dims = {"quality": 4.0, "reliability": 4.0, "responsiveness": 4.0, "trust": 4.0}
    weights = {"quality": 0.25, "reliability": 0.25, "responsiveness": 0.25, "trust": 0.25}
    assert aggregate_dimensions(dims, weights) == 4.0


def test_weighted_aggregation():
    dims = {"quality": 5.0, "reliability": 3.0, "responsiveness": 4.0, "trust": 4.0}
    weights = {"quality": 0.35, "reliability": 0.25, "responsiveness": 0.20, "trust": 0.20}
    result = aggregate_dimensions(dims, weights)
    expected = (5.0 * 0.35 + 3.0 * 0.25 + 4.0 * 0.20 + 4.0 * 0.20) / 1.0
    assert abs(result - expected) < 0.01


def test_tier_platinum():
    assert classify_tier(score=4.7, signal_count=150, confidence=0.9) == "platinum"


def test_tier_gold():
    assert classify_tier(score=4.2, signal_count=60, confidence=0.8) == "gold"


def test_tier_insufficient_signals():
    assert classify_tier(score=4.8, signal_count=3, confidence=0.9) == "bronze"


def test_tier_insufficient_confidence():
    assert classify_tier(score=4.8, signal_count=200, confidence=0.5) == "silver"


def test_tier_untrusted():
    assert classify_tier(score=2.0, signal_count=50, confidence=0.8) == "untrusted"
