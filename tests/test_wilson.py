from engine.scoring.wilson import wilson_score_interval, wilson_from_stars


def test_perfect_score_few_reviews():
    few = wilson_score_interval(positive=3, total=3)
    many = wilson_score_interval(positive=188, total=200)
    assert many > few


def test_no_reviews_returns_zero():
    assert wilson_score_interval(positive=0, total=0) == 0.0


def test_all_negative():
    result = wilson_score_interval(positive=0, total=100)
    assert result < 0.05


def test_50_50_split():
    result = wilson_score_interval(positive=50, total=100)
    assert 0.4 < result < 0.5


def test_from_star_rating():
    high = wilson_from_stars(mean_rating=4.5, count=200)
    low = wilson_from_stars(mean_rating=4.5, count=5)
    assert high > low
    assert 0.0 <= high <= 1.0
