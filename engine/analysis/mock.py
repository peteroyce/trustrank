import re
from collections import Counter
from engine.analysis.base import BaseAnalyzer, AnalysisResult

_POSITIVE = {"good","great","excellent","amazing","wonderful","love","loved","best","fantastic","awesome","perfect","delicious","fast","fresh","quality","beautiful","outstanding","incredible","superb","happy","recommend"}
_NEGATIVE = {"bad","terrible","horrible","worst","hate","awful","poor","slow","broken","waste","disappointing","disgusting","rude","expensive","overpriced","dirty","cold","damaged","useless","never"}

class MockAnalyzer(BaseAnalyzer):
    def analyze(self, text: str, context: dict | None = None) -> AnalysisResult:
        if not text or not text.strip(): return AnalysisResult()
        lower = text.lower()
        words = re.findall(r'\b[a-z]+\b', lower)
        pos = sum(1 for w in words if w in _POSITIVE)
        neg = sum(1 for w in words if w in _NEGATIVE)
        total = pos + neg
        sentiment = (pos - neg) / max(total, 1)
        sentiment = max(-1.0, min(1.0, sentiment))
        exclamation_ratio = text.count("!") / max(len(text), 1)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        short_text = len(words) < 5
        generic_phrases = sum(1 for p in ["buy now","best ever","highly recommend","must buy"] if p in lower)
        fake_prob = min(1.0, exclamation_ratio*5 + caps_ratio*2 + generic_phrases*0.15 + (0.2 if short_text else 0))
        stopwords = {"the","a","an","is","it","to","and","of","in","for","was","with","on","at","this","that","i","my","we","but","not","very","so","just"}
        filtered = [w for w in words if w not in stopwords and len(w) > 2]
        counts = Counter(filtered)
        tags = [word for word, _ in counts.most_common(5)]
        summary = text[:100] + "..." if len(text) > 100 else text
        return AnalysisResult(sentiment=sentiment, fake_probability=fake_prob, tags=tags, summary=summary)
