class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, point):
        self.nodes.append(point)

    def add_edge(self, p1, p2):
        self.edges.append((p1, p2))

    def geometry_type(self):
        return "graph"
