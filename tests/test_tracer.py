from engine.explainability.tracer import ScoreTracer

def test_tracer_records_factors():
    tracer = ScoreTracer()
    tracer.add("base_prior", 3.0)
    tracer.add("bayesian_update", 1.2)
    breakdown = tracer.get_breakdown()
    assert breakdown["base_prior"] == 3.0
    assert "final" not in breakdown

def test_tracer_finalize():
    tracer = ScoreTracer()
    tracer.add("base_prior", 3.0)
    tracer.finalize(4.0)
    assert tracer.get_breakdown()["final"] == 4.0

def test_tracer_alerts():
    tracer = ScoreTracer()
    tracer.add_alert("burst_detected", "7 signals in 2h", "medium")
    alerts = tracer.get_alerts()
    assert len(alerts) == 1 and alerts[0]["type"] == "burst_detected"
