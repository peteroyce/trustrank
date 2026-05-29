import networkx as nx
import community as community_louvain

def detect_communities(G: nx.Graph) -> dict[str, int]:
    if len(G.nodes) == 0: return {}
    if len(G.nodes) == 1: return {list(G.nodes)[0]: 0}
    if len(G.edges) == 0: return {n: i for i, n in enumerate(G.nodes)}
    return community_louvain.best_partition(G, weight="weight")

def are_same_community(a: str, b: str, communities: dict[str, int]) -> bool:
    return communities.get(a) == communities.get(b) and a in communities
