import random
import networkx as nx
import matplotlib.pyplot as plt
# import numpy as np

MAX_EDGE_WEIGHT = 100
MAX_EDGE_LENGTH = 100
CONVEX_BUDGET_FACTOR = 0.2
print("testinstance")


class MSTKPInstance:
    """
    Represents an instance of the MST problem with an extra knapsack constraint.
    Each edge is encoded as a tuple (u, v, w, l) where u and v are the nodes connected by the edge, w is the weight of the edge, and l is the length of the edge.
    The graph is encoded as a list of edges.
    The graph is connected.
    In order to compute the budget, we compute two MSTs:
    - One with the weight of the edges as weights, called Tw
    - One with the length of the edges as weights, called Tl
    The budget is taken a weighted average between the weight of Tw and the weight (not length) of Tl.
    """
    def __init__(self, num_nodes, density):
        self.num_nodes = num_nodes
        self.budget = None
        self.edges = []

        self.density = density
        self.graph = None
        self.tw = None
        self.tl = None

        self.compute_instance()

    def compute_instance(self):
        # Step 1: Start with a spanning tree to ensure connectivity
        self.graph = nx.Graph()
        nodes = list(range(self.num_nodes))
        random.shuffle(nodes)

        # Create a random spanning tree
        for i in range(self.num_nodes - 1):
            u, v = nodes[i], nodes[i + 1]
            weight = random.randint(1, MAX_EDGE_WEIGHT)
            length = random.randint(1, MAX_EDGE_LENGTH)
            self.graph.add_edge(u, v, weight=weight, length=length)

        # Step 2: Add extra edges based on density
        total_possible_edges = (self.num_nodes * (self.num_nodes - 1)) // 2  # Complete graph edges
        target_edges = max(self.num_nodes - 1, round(self.density * total_possible_edges))  # Ensure at least the tree edges
        edges_added = set(self.graph.edges)

        while len(edges_added) < target_edges:
            u, v = random.sample(nodes, 2)
            if (u, v) not in edges_added and (v, u) not in edges_added:
                weight = random.randint(1, MAX_EDGE_WEIGHT)
                length = random.randint(1, MAX_EDGE_LENGTH)
                self.graph.add_edge(u, v, weight=weight, length=length)
                edges_added.add((u, v))

        # Store edges in list format
        # self.edges = [(u, v, d["weight"], d["length"]) for u, v, d in self.graph.edges(data=True)]
        self.edges = [(min(u, v), max(u, v), d["weight"], d["length"]) for u, v, d in self.graph.edges(data=True)]        


        # Step 3: Compute the two MSTs
        self.tw = nx.minimum_spanning_tree(self.graph, weight="weight")
        self.tl = nx.minimum_spanning_tree(self.graph, weight="length")

        # Step 4: Compute budget
        total_tw_weight = sum(d["weight"] for _, _, d in self.tw.edges(data=True))
        total_tl_weight = sum(d["weight"] for _, _, d in self.tl.edges(data=True))

        assert 0 <= CONVEX_BUDGET_FACTOR <= 1
        self.budget = int(CONVEX_BUDGET_FACTOR * total_tw_weight + (1 - CONVEX_BUDGET_FACTOR) * total_tl_weight)
        # self.budget = self.num_nodes * 20 -20
        # self.budget=900]
        assert total_tw_weight <= self.budget <= total_tl_weight
        del self.graph

    def print_all_edges(self):
        print("All edges in the graph (u, v, weight, length):")
        for u, v, w, l in self.edges:
            print(f"Edge ({u}, {v}): Weight = {w}, Length = {l}")

    def plot(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(u, v, w) for u, v, w, l in self.edges])
        pos = nx.spring_layout(G)
        edge_labels = {(u, v): f"{w}, {l}" for u, v, w, l in self.edges}
        nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_weight="bold")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.axis("off")
        plt.savefig("mstkp_instance.png", format="PNG")
        plt.show()

if __name__ == "__main__":
    instance = MSTKPInstance(100, 0.3)  # More edges with higher density
    instance.plot()
    print(f"Budget: {instance.budget}")
    print(f"Edges: {instance.edges}")
    print(f"Total edges in the graph: {len(instance.edges)}")
    print(f"Expected number of edges: {round(0.6 * ((10 * 9) // 2))}")  # Expected based on density
    print(f"Tw: {instance.tw.edges(data=True)}")
    print(f"Tl: {instance.tl.edges(data=True)}")
    print(f"Total Tw weight: {sum([d['weight'] for _, _, d in instance.tw.edges(data=True)])}")
    print(f"Total Tl weight: {sum([d['weight'] for _, _, d in instance.tl.edges(data=True)])}")
    print(f"Total Tw length: {sum([d['length'] for _, _, d in instance.tw.edges(data=True)])}")
    print(f"Total Tl length: {sum([d['length'] for _, _, d in instance.tl.edges(data=True)])}")

