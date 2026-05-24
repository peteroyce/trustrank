from datetime import datetime, timedelta, timezone
from engine.scoring.decay import compute_decay, compute_velocity


def test_recent_signal_full_weight():
    now = datetime.now(timezone.utc)
    weight = compute_decay(signal_time=now, current_time=now, half_life_days=90)
    assert abs(weight - 1.0) < 0.01


def test_half_life_halves_weight():
    now = datetime.now(timezone.utc)
    past = now - timedelta(days=90)
    weight = compute_decay(signal_time=past, current_time=now, half_life_days=90)
    assert abs(weight - 0.5) < 0.01


def test_very_old_signal_near_zero():
    now = datetime.now(timezone.utc)
    ancient = now - timedelta(days=900)
    weight = compute_decay(signal_time=ancient, current_time=now, half_life_days=90)
    assert weight < 0.01


def test_velocity_positive_slope():
    now = datetime.now(timezone.utc)
    scores = [(now - timedelta(days=6), 2.0), (now - timedelta(days=4), 3.0),
              (now - timedelta(days=2), 4.0), (now, 5.0)]
    slope = compute_velocity(scores, window_days=7)
    assert slope > 0


def test_velocity_stable():
    now = datetime.now(timezone.utc)
    scores = [(now - timedelta(days=i), 3.0) for i in range(7)]
    slope = compute_velocity(scores, window_days=7)
    assert abs(slope) < 0.01


def test_velocity_empty_returns_zero():
    assert compute_velocity([], window_days=7) == 0.0
