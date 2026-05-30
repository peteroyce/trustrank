import json
from engine.analysis.base import BaseAnalyzer, AnalysisResult
from engine.config import settings

class ClaudeAnalyzer(BaseAnalyzer):
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def analyze(self, text: str, context: dict | None = None) -> AnalysisResult:
        if not text or not text.strip(): return AnalysisResult()
        prompt = f'Analyze this review and return JSON with: sentiment (float -1 to 1), fake_probability (float 0 to 1), tags (list of 3-5 strings), summary (one sentence). Review: "{text}". Return ONLY valid JSON.'
        response = self.client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=200, messages=[{"role":"user","content":prompt}])
        try:
            data = json.loads(response.content[0].text)
            return AnalysisResult(sentiment=float(data.get("sentiment",0)), fake_probability=float(data.get("fake_probability",0)), tags=data.get("tags",[]), summary=data.get("summary",""))
        except (json.JSONDecodeError, KeyError, IndexError):
            return AnalysisResult()

def get_analyzer() -> BaseAnalyzer:
    from engine.analysis.mock import MockAnalyzer
    if settings.analyzer_backend == "claude" and settings.anthropic_api_key:
        return ClaudeAnalyzer()
    return MockAnalyzer()
