#!/usr/bin/env python3

import networkx as nx
from typing import List, Tuple, Any
import itertools
import matplotlib.pyplot as plt
import numpy as np

def cartesian_product_n(graph: nx.Graph, n: int) -> nx.Graph:
    """
    Compute the Cartesian product of n copies of the input graph.
    
    Args:
        graph (nx.Graph): The input graph
        n (int): Number of copies to take the Cartesian product of
        
    Returns:
        nx.Graph: The resulting Cartesian product graph
        
    Example:
        If G is a path graph with 2 nodes, and n=2, the result will be a 2x2 grid graph
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    
    # For n=1, just return a copy of the original graph
    if n == 1:
        return graph.copy()
    
    # Get the nodes and edges of the original graph
    original_nodes = list(graph.nodes())
    original_edges = list(graph.edges())
    
    # Create all possible n-tuples of nodes from the original graph
    product_nodes = list(itertools.product(original_nodes, repeat=n))
    
    # Create the new graph
    product_graph = nx.Graph()
    
    # Add all nodes to the new graph
    for node_tuple in product_nodes:
        product_graph.add_node(node_tuple)
    
    # Add edges between nodes that differ in exactly one position
    # and where that difference corresponds to an edge in the original graph
    for i in range(len(product_nodes)):
        for j in range(i + 1, len(product_nodes)):
            node1 = product_nodes[i]
            node2 = product_nodes[j]
            
            # Check if nodes differ in exactly one position
            differences = [(pos, node1[pos], node2[pos]) 
                          for pos in range(n) 
                          if node1[pos] != node2[pos]]
            
            if len(differences) == 1:
                pos, val1, val2 = differences[0]
                # Check if the differing values form an edge in the original graph
                if graph.has_edge(val1, val2):
                    product_graph.add_edge(node1, node2)
    
    return product_graph

def cartesian_product_n_with_weights(graph: nx.Graph, n: int) -> nx.Graph:
    """
    Compute the Cartesian product of n copies of the input graph, preserving edge weights.
    
    Args:
        graph (nx.Graph): The input graph with edge weights
        n (int): Number of copies to take the Cartesian product of
        
    Returns:
        nx.Graph: The resulting Cartesian product graph with edge weights
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    
    # For n=1, just return a copy of the original graph
    if n == 1:
        return graph.copy()
    
    # Get the nodes and edges of the original graph
    original_nodes = list(graph.nodes())
    original_edges = list(graph.edges(data=True))
    
    # Create all possible n-tuples of nodes from the original graph
    product_nodes = list(itertools.product(original_nodes, repeat=n))
    
    # Create the new graph
    product_graph = nx.Graph()
    
    # Add all nodes to the new graph
    for node_tuple in product_nodes:
        product_graph.add_node(node_tuple)
    
    # Add edges between nodes that differ in exactly one position
    # and where that difference corresponds to an edge in the original graph
    for i in range(len(product_nodes)):
        for j in range(i + 1, len(product_nodes)):
            node1 = product_nodes[i]
            node2 = product_nodes[j]
            
            # Check if nodes differ in exactly one position
            differences = [(pos, node1[pos], node2[pos]) 
                          for pos in range(n) 
                          if node1[pos] != node2[pos]]
            
            if len(differences) == 1:
                pos, val1, val2 = differences[0]
                # Check if the differing values form an edge in the original graph
                if graph.has_edge(val1, val2):
                    # Copy the edge attributes from the original graph
                    edge_data = graph.get_edge_data(val1, val2)
                    product_graph.add_edge(node1, node2, **edge_data)
    
    return product_graph

def visualize_graph(graph: nx.Graph, title: str = "Graph Visualization", pos=None):
    """
    Visualize a graph using matplotlib.
    
    Args:
        graph (nx.Graph): The graph to visualize
        title (str): Title for the plot
        pos (dict, optional): Node positions for layout
    """
    plt.figure(figsize=(10, 8))
    
    # If no positions provided, compute spring layout
    if pos is None:
        pos = nx.spring_layout(graph)
    
    # Draw the graph
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=10, font_weight='bold')
    
    plt.title(title)
    plt.show()

def visualize_grid_product(graph: nx.Graph, n: int, title: str = None):
    """
    Visualize the Cartesian product of n copies of a graph in a grid layout.
    
    Args:
        graph (nx.Graph): The original graph
        n (int): Number of copies
        title (str, optional): Title for the plot
    """
    product_graph = cartesian_product_n(graph, n)
    
    # Create a grid layout for the nodes
    pos = {}
    num_nodes = len(graph.nodes())
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_nodes ** n)))
    
    # Assign positions in a grid
    for i, node in enumerate(product_graph.nodes()):
        row = i // grid_size
        col = i % grid_size
        pos[node] = (col, -row)  # Negative row for top-to-bottom layout
    
    if title is None:
        title = f"Cartesian Product of {n} copies"
    
    visualize_graph(product_graph, title, pos)

def create_example_graphs():
    """
    Create and return a dictionary of example graphs.
    """
    graphs = {}
    
    # Path graph (linear)
    graphs['path'] = nx.path_graph(3)
    
    # Cycle graph
    graphs['cycle'] = nx.cycle_graph(4)
    
    # Star graph
    graphs['star'] = nx.star_graph(3)
    
    # Complete graph
    graphs['complete'] = nx.complete_graph(3)
    
    return graphs

# Example usage
if __name__ == "__main__":
    # Create example graphs
    example_graphs = create_example_graphs()
    
    # Example 1: Path graph product
    print("\nExample 1: Path Graph Product")
    G_path = example_graphs['path']
    G_path_product = cartesian_product_n(G_path, 2)
    print(f"Original path graph: {G_path.number_of_nodes()} nodes, {G_path.number_of_edges()} edges")
    print(f"Product graph: {G_path_product.number_of_nodes()} nodes, {G_path_product.number_of_edges()} edges")
    visualize_grid_product(G_path, 2, "Path Graph Product (2 copies)")
    
    # Example 2: Cycle graph product
    print("\nExample 2: Cycle Graph Product")
    G_cycle = example_graphs['cycle']
    G_cycle_product = cartesian_product_n(G_cycle, 2)
    print(f"Original cycle graph: {G_cycle.number_of_nodes()} nodes, {G_cycle.number_of_edges()} edges")
    print(f"Product graph: {G_cycle_product.number_of_nodes()} nodes, {G_cycle_product.number_of_edges()} edges")
    visualize_grid_product(G_cycle, 2, "Cycle Graph Product (2 copies)")
    
    # Example 3: Star graph product
    print("\nExample 3: Star Graph Product")
    G_star = example_graphs['star']
    G_star_product = cartesian_product_n(G_star, 2)
    print(f"Original star graph: {G_star.number_of_nodes()} nodes, {G_star.number_of_edges()} edges")
    print(f"Product graph: {G_star_product.number_of_nodes()} nodes, {G_star_product.number_of_edges()} edges")
    visualize_grid_product(G_star, 2, "Star Graph Product (2 copies)")
    
    # Example 4: Complete graph product
    print("\nExample 4: Complete Graph Product")
    G_complete = example_graphs['complete']
    G_complete_product = cartesian_product_n(G_complete, 2)
    print(f"Original complete graph: {G_complete.number_of_nodes()} nodes, {G_complete.number_of_edges()} edges")
    print(f"Product graph: {G_complete_product.number_of_nodes()} nodes, {G_complete_product.number_of_edges()} edges")
    visualize_grid_product(G_complete, 2, "Complete Graph Product (2 copies)") 