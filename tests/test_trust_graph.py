from engine.graph.trust import TrustGraph

def test_direct_trust():
    g = TrustGraph(damping=0.7, max_hops=2)
    g.add_edge("A", "B", weight=0.9, category="general")
    assert g.direct_trust("A", "B") == 0.9

def test_transitive_trust_one_hop():
    g = TrustGraph(damping=0.7, max_hops=2)
    g.add_edge("A", "B", weight=0.9, category="general")
    g.add_edge("B", "C", weight=0.8, category="general")
    trust = g.indirect_trust("A", "C", category="general")
    assert abs(trust - 0.9 * 0.8 * 0.7) < 0.01

def test_transitive_trust_max_hops():
    g = TrustGraph(damping=0.7, max_hops=2)
    g.add_edge("A", "B", weight=0.9, category="general")
    g.add_edge("B", "C", weight=0.8, category="general")
    g.add_edge("C", "D", weight=0.7, category="general")
    trust = g.indirect_trust("A", "D", category="general")
    assert trust == 0.0

def test_multiple_paths_takes_max():
    g = TrustGraph(damping=0.7, max_hops=2)
    g.add_edge("A", "B", weight=0.5, category="general")
    g.add_edge("A", "C", weight=0.9, category="general")
    g.add_edge("B", "D", weight=0.8, category="general")
    g.add_edge("C", "D", weight=0.8, category="general")
    trust = g.indirect_trust("A", "D", category="general")
    assert abs(trust - max(0.5*0.8*0.7, 0.9*0.8*0.7)) < 0.01

def test_category_filtering():
    g = TrustGraph(damping=0.7, max_hops=2)
    g.add_edge("A", "B", weight=0.9, category="food")
    g.add_edge("B", "C", weight=0.8, category="tech")
    trust = g.indirect_trust("A", "C", category="food")
    assert trust == 0.0

def test_trust_bonus():
    g = TrustGraph(damping=0.7, max_hops=2)
    g.add_edge("X", "A", weight=0.9, category="general")
    g.add_edge("Y", "A", weight=0.8, category="general")
    bonus = g.compute_trust_bonus("A", weight=0.15)
    assert bonus > 0
