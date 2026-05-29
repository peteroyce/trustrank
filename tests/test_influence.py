import networkx as nx
from engine.graph.influence import katz_centrality

def test_hub_has_highest_influence():
    G = nx.DiGraph()
    for i in range(10):
        G.add_edge("hub", f"node{i}", weight=0.8)
    scores = katz_centrality(G, alpha=0.1)
    assert scores["hub"] == max(scores.values())

def test_isolated_node_low_influence():
    G = nx.DiGraph()
    G.add_node("isolated")
    G.add_edge("A", "B", weight=0.5)
    scores = katz_centrality(G, alpha=0.1)
    assert scores.get("isolated", 0) <= scores.get("A", 0)

def test_empty_graph():
    G = nx.DiGraph()
    scores = katz_centrality(G, alpha=0.1)
    assert scores == {}
