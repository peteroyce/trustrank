from engine.detection.reciprocal import ReciprocalDetector

def test_no_collusion_independent_reviewers():
    entity_reviews = {
        "A": [{"source": "1", "value": 4.0}, {"source": "2", "value": 5.0}, {"source": "3", "value": 4.0}],
        "B": [{"source": "4", "value": 3.0}, {"source": "5", "value": 4.0}, {"source": "6", "value": 5.0}],
    }
    detector = ReciprocalDetector()
    result = detector.detect(entity_reviews)
    assert len(result.flagged_pairs) == 0

def test_collusion_detected():
    entity_reviews = {
        "A": [{"source": "1", "value": 5.0}, {"source": "2", "value": 5.0}, {"source": "3", "value": 5.0},
              {"source": "4", "value": 4.0}, {"source": "5", "value": 5.0}],
        "B": [{"source": "1", "value": 5.0}, {"source": "2", "value": 4.0}, {"source": "3", "value": 5.0},
              {"source": "4", "value": 5.0}, {"source": "6", "value": 3.0}],
    }
    detector = ReciprocalDetector(jaccard_threshold=0.3, sentiment_threshold=0.7)
    result = detector.detect(entity_reviews)
    assert len(result.flagged_pairs) > 0

def test_single_entity_no_pairs():
    entity_reviews = {"A": [{"source": "1", "value": 5.0}]}
    detector = ReciprocalDetector()
    result = detector.detect(entity_reviews)
    assert len(result.flagged_pairs) == 0
