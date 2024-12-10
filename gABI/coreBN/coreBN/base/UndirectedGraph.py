#!/usr/bin/env python3

import itertools

import networkx as nx


class UndirectedGraph(nx.Graph):
    """
    Base class for all the Undirected Graphical models.
    Each node in the graph can represent either a random variable, `Factor`,
    or a cluster of random variables. Edges in the graph are interactions
    between the nodes.
    Parameters
    ----------
    data : input graph, based on networkx Graph class
        The data can be an edge list or any Networkx graph object.
        If data=None (default) an empty graph is created. 
   
    """

    def __init__(self, ebunch=None):
        super(UndirectedGraph, self).__init__(ebunch)

    def add_node(self, node, weight=None):
        """
        Add a single node to the Graph.
        Parameters
        ----------
        node: str, int, or any hashable python object.
            The node to add to the graph.
        weight: int, float
            The weight of the node.
        Examples
        --------
        >>> from coreBN.base import UndirectedGraph
        >>> G = UndirectedGraph()
        >>> G.add_node(node='A')
        >>> G.nodes()
        NodeView(('A',))
        Adding a node with some weight.
        >>> G.add_node(node='B', weight=0.3)
        The weight of these nodes can be accessed as:
        >>> G.nodes['B']
        {'weight': 0.3}
       
        """
        # Check for networkx 2.0 syntax
        if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], dict):
            node, attrs = node
            if attrs.get("weight", None) is not None:
                attrs["weight"] = weight
        else:
            attrs = {"weight": weight}
        super(UndirectedGraph, self).add_node(node, weight=weight)

    def add_nodes_from(self, nodes, weights=None):
        """
        Add multiple nodes to the graph.
       
        Parameters
        ----------
        nodes: iterable container
            A container of nodes (list, dict, set, or any hashable python
            object).
        weights: list, tuple (default=None)
            A container of weights (int, float). The weight value at index i
            is associated with the variable at index i.

        Usage example
        --------
        >>> from coreBN.base import UndirectedGraph
        >>> G = UndirectedGraph()
        >>> G.add_nodes_from(nodes=['A', 'B', 'C'])
        >>> G.nodes()
        NodeView(('A', 'B', 'C'))
        Adding nodes with weights:
        >>> G.add_nodes_from(nodes=['D', 'E'], weights=[0.3, 0.6])
        
        """
        nodes = list(nodes)

        if weights:
            if len(nodes) != len(weights):
                raise ValueError(
                    "The number of elements in nodes and weights" "should be equal."
                )
            for index in range(len(nodes)):
                self.add_node(node=nodes[index], weight=weights[index])
        else:
            for node in nodes:
                self.add_node(node=node)

    def add_edge(self, u, v, weight=None):
        """
        Add an edge between u and v.
        The nodes u and v will be automatically added if they are
        not already in the graph.
        Parameters
        ----------
        u, v : nodes
            Nodes can be any hashable Python object.
        weight: int, float (default=None)
            The weight of the edge.
        Examples
        --------
        >>> from coreBN.base import UndirectedGraph
        >>> G = UndirectedGraph()
        >>> G.add_nodes_from(nodes=['Alice', 'Bob', 'Charles'])
        >>> G.add_edge(u='Alice', v='Bob')
        >>> G.nodes()
        NodeView(('Alice', 'Bob', 'Charles'))
        >>> G.edges()
        EdgeView([('Alice', 'Bob')])
    
        """
        super(UndirectedGraph, self).add_edge(u, v, weight=weight)

    def add_edges_from(self, ebunch, weights=None):
        """
        Add all the edges in edge_bunch.
        If nodes referred in the ebunch are not already present, they
        will be automatically added. Node names can be any hashable python
        object.
        
        Parameters
        ----------
        ebunch : edge container 
            Each edge given in the container will be added to the graph.
            The edges must be given as 2-tuples (u, v).
        weights: list, tuple (default=None)
            A container of weights (int, float). The weight value at index i
            is associated with the edge at index i.
        Examples
        --------
        >>> from coreBN.base import UndirectedGraph
        >>> G = UndirectedGraph()
        >>> G.add_nodes_from(nodes=['Alice', 'Bob', 'Charles'])
        >>> G.add_edges_from(ebunch=[('Alice', 'Bob'), ('Bob', 'Charles')]
        >>>#Adding edges with weights:
        >>> G.add_edges_from([('Ankur', 'Maria'), ('Maria', 'Mason')], weights=[0.3, 0.5])
        >>> G.edge['Ankur']['Maria']
        {'weight': 0.3}
   
        """
        ebunch = list(ebunch)

        if weights:
            if len(ebunch) != len(weights):
                raise ValueError("The number of elements in ebunch and weights" "should be equal")
            for index in range(len(ebunch)):
                self.add_edge(ebunch[index][0], ebunch[index][1], weight=weights[index])
        else:
            for edge in ebunch:
                self.add_edge(edge[0], edge[1])

    def is_clique(self, nodes):
        """
        Check if the given nodes form a clique.
        Parameters
        ----------
        nodes: list, array-like
            List of nodes to check if they are a part of any clique.
        Examples
        --------
        >>> from coreBN.base import UndirectedGraph
        >>> G = UndirectedGraph(ebunch=[('A', 'B'), ('C', 'B'), ('B', 'D'),
                                        ('B', 'E'), ('D', 'E'), ('E', 'F'),
                                        ('D', 'F'), ('B', 'F')])
        >>> G.is_clique(nodes=['A', 'B', 'C', 'D'])
        False
        >>> G.is_clique(nodes=['B', 'D', 'E', 'F'])
        True
        """
        for node1, node2 in itertools.combinations(nodes, 2):
            if not self.has_edge(node1, node2):
                return False
        return True

    def is_triangulated(self):
        """
        Checks whether the undirected graph is triangulated (also known as chordal) or not.
        A chordal graph is one in which all cycles of four or more vertices have a chord.
        Examples
        --------
        >>> from coreBN.base import UndirectedGraph
        >>> G = UndirectedGraph()
        >>> G.add_edges_from(ebunch=[('x1', 'x2'), ('x1', 'x3'),('x2', 'x4'), ('x3', 'x4')])                       
        >>> G.is_triangulated()
        False
        >>> G.add_edge(u='x1', v='x4')
        >>> G.is_triangulated()
        True
        """
        return nx.is_chordal(self)



