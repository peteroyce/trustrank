class ScoreTracer:
    def __init__(self):
        self._factors: dict[str, float] = {}
        self._alerts: list[dict] = []

    def add(self, name: str, value: float) -> None:
        self._factors[name] = value

    def add_alert(self, alert_type: str, detail: str, severity: str = "medium") -> None:
        self._alerts.append({"type": alert_type, "detail": detail, "severity": severity})

    def finalize(self, final_score: float) -> None:
        self._factors["final"] = final_score

    def get_breakdown(self) -> dict:
        return dict(self._factors)

    def get_alerts(self) -> list[dict]:
        return list(self._alerts)
