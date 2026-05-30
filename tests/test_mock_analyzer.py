from engine.analysis.mock import MockAnalyzer

def test_positive_review():
    a = MockAnalyzer()
    result = a.analyze("Absolutely wonderful experience, loved everything about it! Amazing quality and service.")
    assert result.sentiment > 0.0
    assert len(result.tags) > 0

def test_negative_review():
    a = MockAnalyzer()
    result = a.analyze("Terrible product, broke after one day. Complete waste of money, very disappointing.")
    assert result.sentiment < 0.0

def test_fake_review_high_exclamation():
    a = MockAnalyzer()
    result = a.analyze("AMAZING!!! BEST EVER!!! BUY NOW!!! INCREDIBLE!!! WOW!!!")
    assert result.fake_probability > 0.3

def test_empty_text():
    a = MockAnalyzer()
    result = a.analyze("")
    assert result.sentiment == 0.0 and result.tags == []

def test_tags_extracted():
    a = MockAnalyzer()
    result = a.analyze("The delivery was fast and the pizza was delicious with great cheese and fresh toppings")
    assert len(result.tags) >= 1
