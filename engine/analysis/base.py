from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class AnalysisResult:
    sentiment: float = 0.0
    fake_probability: float = 0.0
    tags: list[str] = field(default_factory=list)
    summary: str = ""

class BaseAnalyzer(ABC):
    @abstractmethod
    def analyze(self, text: str, context: dict | None = None) -> AnalysisResult:
        pass
