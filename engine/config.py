from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://trustrank:trustrank@localhost:5432/trustrank"
    redis_url: str = "redis://localhost:6379/0"
    anthropic_api_key: str = ""
    analyzer_backend: str = "mock"
    decay_half_life_days: float = 90.0
    velocity_window_days: int = 7
    velocity_boost: float = 1.1
    trend_window_days: int = 30
    trend_threshold: float = 0.02
    cusum_allowance_factor: float = 0.5
    cusum_threshold_factor: float = 5.0
    coordination_similarity_threshold: float = 0.85
    coordination_cv_threshold: float = 0.5
    reciprocal_jaccard_threshold: float = 0.3
    reciprocal_sentiment_threshold: float = 0.7
    trust_damping: float = 0.7
    trust_max_hops: int = 2
    trust_score_weight: float = 0.15
    tier_platinum_score: float = 4.5
    tier_platinum_signals: int = 100
    tier_platinum_confidence: float = 0.85
    tier_gold_score: float = 4.0
    tier_gold_signals: int = 50
    tier_gold_confidence: float = 0.75
    tier_silver_score: float = 3.5
    tier_silver_signals: int = 20
    tier_silver_confidence: float = 0.60
    tier_bronze_score: float = 2.5
    tier_bronze_signals: int = 5
    tier_bronze_confidence: float = 0.40
    default_dimension_weights: dict = {
        "quality": 0.35, "reliability": 0.25, "responsiveness": 0.20, "trust": 0.20,
    }
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()
