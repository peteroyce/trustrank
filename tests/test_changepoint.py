from engine.temporal.changepoint import detect_regime_change

def test_no_change_stable_data():
    values = [4.0 + (i%3)*0.1 for i in range(50)]
    result = detect_regime_change(values, threshold=0.8)
    assert not result.change_detected

def test_clear_regime_change():
    values = [4.0 + (i%3)*0.1 for i in range(30)] + [2.0 + (i%3)*0.1 for i in range(20)]
    result = detect_regime_change(values, threshold=0.8)
    assert result.change_detected
    assert 25 <= result.change_index <= 35

def test_too_few_values():
    result = detect_regime_change([4.0, 3.0], threshold=0.8)
    assert not result.change_detected
