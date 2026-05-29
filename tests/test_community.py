from engine.graph.community import detect_communities, are_same_community
import networkx as nx

def test_two_clear_communities():
    G = nx.Graph()
    for i in range(5):
        for j in range(i + 1, 5):
            G.add_edge(f"a{i}", f"a{j}", weight=0.9)
    for i in range(5):
        for j in range(i + 1, 5):
            G.add_edge(f"b{i}", f"b{j}", weight=0.9)
    G.add_edge("a0", "b0", weight=0.1)
    communities = detect_communities(G)
    assert len(set(communities.values())) >= 2

def test_single_node():
    G = nx.Graph()
    G.add_node("alone")
    communities = detect_communities(G)
    assert "alone" in communities

def test_same_community_flag():
    communities = {"A": 0, "B": 0, "C": 1}
    assert are_same_community("A", "B", communities)
    assert not are_same_community("A", "C", communities)
