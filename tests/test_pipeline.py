from datetime import datetime, timedelta, timezone
from engine.scoring.pipeline import ScoringPipeline
from engine.analysis.mock import MockAnalyzer


def test_pipeline_basic_scoring():
    pipeline = ScoringPipeline(analyzer=MockAnalyzer())
    signals = [
        {
            "value": 4.5,
            "text": "Great product",
            "dimension": "quality",
            "source_stats": {
                "review_count": 10,
                "review_diversity": 0.5,
                "accuracy": 0.7,
                "account_age_days": 90,
                "flagged_ratio": 0.0,
            },
            "created_at": datetime.now(timezone.utc) - timedelta(days=i),
        }
        for i in range(10)
    ]
    result = pipeline.score(signals)
    assert "overall" in result
    assert "dimensions" in result
    assert "breakdown" in result
    assert "tier" in result
    assert result["overall"] > 3.0


def test_pipeline_empty_signals():
    pipeline = ScoringPipeline(analyzer=MockAnalyzer())
    result = pipeline.score([])
    assert result["overall"] == 3.0
    assert result["tier"] == "bronze"


def test_pipeline_returns_all_keys():
    pipeline = ScoringPipeline(analyzer=MockAnalyzer())
    signals = [
        {
            "value": 3.0,
            "text": "Average",
            "dimension": "quality",
            "source_stats": {
                "review_count": 5,
                "review_diversity": 0.5,
                "accuracy": 0.5,
                "account_age_days": 30,
                "flagged_ratio": 0.0,
            },
            "created_at": datetime.now(timezone.utc) - timedelta(days=1),
        }
    ]
    result = pipeline.score(signals)
    for key in ("overall", "confidence", "tier", "dimensions", "breakdown", "alerts", "counterfactual"):
        assert key in result


def test_pipeline_multi_dimension():
    pipeline = ScoringPipeline(analyzer=MockAnalyzer())
    now = datetime.now(timezone.utc)
    signals = [
        {
            "value": 4.0,
            "text": "Good quality",
            "dimension": "quality",
            "source_stats": {
                "review_count": 20,
                "review_diversity": 0.6,
                "accuracy": 0.8,
                "account_age_days": 180,
                "flagged_ratio": 0.0,
            },
            "created_at": now - timedelta(days=i),
        }
        for i in range(5)
    ] + [
        {
            "value": 4.5,
            "text": "Very reliable",
            "dimension": "reliability",
            "source_stats": {
                "review_count": 20,
                "review_diversity": 0.6,
                "accuracy": 0.8,
                "account_age_days": 180,
                "flagged_ratio": 0.0,
            },
            "created_at": now - timedelta(days=i),
        }
        for i in range(5)
    ]
    result = pipeline.score(signals)
    assert "quality" in result["dimensions"]
    assert "reliability" in result["dimensions"]
    assert result["overall"] > 3.0


def test_pipeline_with_trust_bonus():
    pipeline = ScoringPipeline(analyzer=MockAnalyzer())
    signals = [
        {
            "value": 3.5,
            "text": "Good",
            "dimension": "quality",
            "source_stats": {
                "review_count": 5,
                "review_diversity": 0.5,
                "accuracy": 0.5,
                "account_age_days": 30,
                "flagged_ratio": 0.0,
            },
            "created_at": datetime.now(timezone.utc) - timedelta(days=i),
        }
        for i in range(5)
    ]
    result_no_bonus = pipeline.score(signals, trust_bonus=0.0)
    result_with_bonus = pipeline.score(signals, trust_bonus=0.2)
    assert result_with_bonus["overall"] > result_no_bonus["overall"]


def test_pipeline_tier_classification():
    pipeline = ScoringPipeline(analyzer=MockAnalyzer())
    signals = [
        {
            "value": 1.0,
            "text": "Terrible",
            "dimension": "quality",
            "source_stats": {
                "review_count": 5,
                "review_diversity": 0.3,
                "accuracy": 0.3,
                "account_age_days": 10,
                "flagged_ratio": 0.5,
            },
            "created_at": datetime.now(timezone.utc) - timedelta(days=i),
        }
        for i in range(10)
    ]
    result = pipeline.score(signals)
    assert result["tier"] in ("untrusted", "bronze", "silver", "gold", "platinum")


def test_pipeline_breakdown_contains_expected_keys():
    pipeline = ScoringPipeline(analyzer=MockAnalyzer())
    signals = [
        {
            "value": 4.0,
            "text": "Good",
            "dimension": "quality",
            "source_stats": {
                "review_count": 10,
                "review_diversity": 0.5,
                "accuracy": 0.7,
                "account_age_days": 90,
                "flagged_ratio": 0.0,
            },
            "created_at": datetime.now(timezone.utc) - timedelta(days=1),
        }
    ]
    result = pipeline.score(signals)
    breakdown = result["breakdown"]
    assert "base_prior" in breakdown
    assert "final" in breakdown


def test_pipeline_counterfactual_keys():
    pipeline = ScoringPipeline(analyzer=MockAnalyzer())
    signals = [
        {
            "value": 4.0,
            "text": "Good",
            "dimension": "quality",
            "source_stats": {
                "review_count": 10,
                "review_diversity": 0.5,
                "accuracy": 0.7,
                "account_age_days": 90,
                "flagged_ratio": 0.0,
            },
            "created_at": datetime.now(timezone.utc) - timedelta(days=1),
        }
    ]
    result = pipeline.score(signals)
    cf = result["counterfactual"]
    assert "without_dampened" in cf
    assert "without_trust_bonus" in cf
