import networkx as nx

def katz_centrality(G: nx.DiGraph, alpha: float = 0.1, max_iter: int = 100, tol: float = 1e-6) -> dict[str, float]:
    """Compute Katz centrality on the reversed graph so that nodes with many outgoing
    edges (i.e. nodes that influence many others) receive higher scores."""
    if len(G.nodes) == 0: return {}
    # Reverse graph so that outgoing influence is captured as incoming paths in Katz
    G_rev = G.reverse(copy=False)
    try:
        return nx.katz_centrality(G_rev, alpha=alpha, max_iter=max_iter, tol=tol, weight="weight")
    except nx.NetworkXError:
        return nx.katz_centrality(G_rev, alpha=alpha * 0.5, max_iter=max_iter, tol=tol, weight="weight")
