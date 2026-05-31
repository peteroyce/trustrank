from engine.explainability.counterfactual import compute_counterfactuals

def test_without_dampened():
    signals = [{"value":5.0,"weight":1.0,"dampened":False},{"value":1.0,"weight":0.3,"dampened":True},{"value":5.0,"weight":1.0,"dampened":False}]
    result = compute_counterfactuals(signals, current_score=4.0, trust_bonus=0.1)
    assert result["without_dampened"] >= 4.0

def test_without_trust_bonus():
    signals = [{"value":4.0,"weight":1.0,"dampened":False}]
    result = compute_counterfactuals(signals, current_score=4.1, trust_bonus=0.1)
    assert abs(result["without_trust_bonus"] - 4.0) < 0.01
