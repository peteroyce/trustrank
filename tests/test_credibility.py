from engine.detection.credibility import compute_credibility

def test_new_account_low_credibility():
    score = compute_credibility(review_count=1, review_diversity=0.0, accuracy=0.5, account_age_days=1, flagged_ratio=0.0)
    assert score < 0.3

def test_established_reviewer_high_credibility():
    score = compute_credibility(review_count=100, review_diversity=0.9, accuracy=0.85, account_age_days=365, flagged_ratio=0.0)
    assert score > 0.8

def test_flagged_reviews_reduce_credibility():
    clean = compute_credibility(review_count=50, review_diversity=0.7, accuracy=0.8, account_age_days=180, flagged_ratio=0.0)
    flagged = compute_credibility(review_count=50, review_diversity=0.7, accuracy=0.8, account_age_days=180, flagged_ratio=0.5)
    assert flagged < clean

def test_single_entity_reviewer_low_diversity():
    score = compute_credibility(review_count=50, review_diversity=0.0, accuracy=0.8, account_age_days=180, flagged_ratio=0.0)
    assert score < 0.6

def test_score_bounds():
    score = compute_credibility(review_count=0, review_diversity=0.0, accuracy=0.0, account_age_days=0, flagged_ratio=1.0)
    assert 0.0 <= score <= 1.0
    score = compute_credibility(review_count=1000, review_diversity=1.0, accuracy=1.0, account_age_days=3650, flagged_ratio=0.0)
    assert 0.0 <= score <= 1.0
