import networkx as nx

class TrustGraph:
    def __init__(self, damping: float = 0.7, max_hops: int = 2):
        self.graph = nx.DiGraph()
        self.damping = damping
        self.max_hops = max_hops

    def add_edge(self, source: str, target: str, weight: float, category: str = "general") -> None:
        self.graph.add_edge(source, target, weight=weight, category=category)

    def direct_trust(self, source: str, target: str) -> float:
        if self.graph.has_edge(source, target):
            return self.graph[source][target]["weight"]
        return 0.0

    def indirect_trust(self, source: str, target: str, category: str = "general") -> float:
        if source == target: return 1.0
        if source not in self.graph or target not in self.graph: return 0.0
        max_trust = 0.0
        # queue entries: (current_node, accumulated_trust, hops_taken)
        # acc_trust at start is 1.0 (no damping yet); first hop multiplies edge weight only,
        # subsequent hops multiply damping factor as well.
        queue = [(source, 1.0, 0)]
        visited = {source}
        while queue:
            current, acc_trust, hops = queue.pop(0)
            if hops >= self.max_hops: continue
            for neighbor in self.graph.successors(current):
                edge = self.graph[current][neighbor]
                edge_cat = edge.get("category", "general")
                if edge_cat != "general" and edge_cat != category: continue
                # First hop: no damping applied (direct edge weight only)
                # Subsequent hops: apply damping to represent distance decay
                if hops == 0:
                    new_trust = acc_trust * edge["weight"]
                else:
                    new_trust = acc_trust * edge["weight"] * self.damping
                if neighbor == target:
                    max_trust = max(max_trust, new_trust)
                elif neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, new_trust, hops + 1))
        return max_trust

    def compute_trust_bonus(self, entity_id: str, weight: float = 0.15) -> float:
        if entity_id not in self.graph: return 0.0
        incoming = [self.graph[pred][entity_id]["weight"] for pred in self.graph.predecessors(entity_id)]
        if not incoming: return 0.0
        return (sum(incoming) / len(incoming)) * weight

    def get_subgraph(self, entity_id: str, hops: int = 2) -> dict:
        if entity_id not in self.graph: return {"nodes": [], "edges": []}
        nodes = {entity_id}
        edges = []
        frontier = {entity_id}
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                for succ in self.graph.successors(node):
                    nodes.add(succ)
                    edges.append({"source": node, "target": succ, **self.graph[node][succ]})
                    next_frontier.add(succ)
                for pred in self.graph.predecessors(node):
                    nodes.add(pred)
                    edges.append({"source": pred, "target": node, **self.graph[pred][node]})
                    next_frontier.add(pred)
            frontier = next_frontier
        return {"nodes": list(nodes), "edges": edges}
