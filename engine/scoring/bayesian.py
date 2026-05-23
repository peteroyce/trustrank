class BetaBinomial:
    def __init__(self, alpha: float = 2.0, beta: float = 2.0):
        self.alpha = alpha
        self.beta = beta

    def update(self, successes: int, failures: int) -> None:
        self.alpha += successes
        self.beta += failures

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def confidence(self) -> float:
        total = self.alpha + self.beta - 4.0
        return 1.0 - 1.0 / (1.0 + total / 20.0)

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta, "mean": self.mean(), "confidence": self.confidence()}


class DirichletMultinomial:
    def __init__(self, alphas: list[float] | None = None):
        self.alphas = list(alphas) if alphas else [1.0, 1.0, 1.0, 1.0, 1.0]

    def update(self, counts: list[int]) -> None:
        for i in range(5):
            self.alphas[i] += counts[i]

    def mean_distribution(self) -> list[float]:
        total = sum(self.alphas)
        return [a / total for a in self.alphas]

    def weighted_mean(self) -> float:
        dist = self.mean_distribution()
        return sum((i + 1) * p for i, p in enumerate(dist))

    def variance(self) -> float:
        dist = self.mean_distribution()
        mean = self.weighted_mean()
        return sum(p * (i + 1 - mean) ** 2 for i, p in enumerate(dist))

    def confidence(self) -> float:
        total = sum(self.alphas) - 5.0
        return 1.0 - 1.0 / (1.0 + total / 30.0)

    def to_dict(self) -> dict:
        return {"alphas": self.alphas, "mean": self.weighted_mean(), "variance": self.variance(), "confidence": self.confidence()}
