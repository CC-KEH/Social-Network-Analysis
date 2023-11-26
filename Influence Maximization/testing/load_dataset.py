import networkx as nx
import random
import networkx as nx
import random
import numpy as np

def load_weighted_graph(file_path):
    """
    Load a graph from a text file and assign 'transition_proba' and 'log_transition_proba' attributes to all edges.
    
    The file should be in the following format:
    # Directed graph (each unordered pair of nodes is saved once): Wiki-Vote.txt 
    # Wikipedia voting on promotion to administratorship (till January 2008). Directed edge A->B means user A voted on B becoming Wikipedia administrator.
    # Nodes: 7115 Edges: 103689
    # FromNodeId	ToNodeId
    30	1412
    30	3352
    30	5254
    3	28
    3	30
    3	39
    3	54

    Args:
        file_path (str): The path to the file.
        
    Returns:
        G (networkx.DiGraph): A directed graph with 'transition_proba' and 'log_transition_proba' attributes assigned to all edges.
    """
    # Create an empty directed graph
    G = nx.DiGraph()

    # Open the file and read the lines
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Iterate over the lines and add edges to the graph
    for line in lines:
        # Ignore lines starting with '#'
        if not line.startswith('#'):
            # Split the line into a list of strings
            nodes = line.split()
            # Add an edge between the nodes
            G.add_edge(nodes[0], nodes[1])

    # Assign 'transition_proba' and 'log_transition_proba' attributes to each edge in the graph
    for (u, v) in G.edges():
        transition_proba = random.random()
        G.edges[u,v]['transition_proba'] = transition_proba
        G.edges[u,v]['log_transition_proba'] = -np.log(transition_proba)

    return G


def save_weighted_graph(G, file_path):
    """
    Save a weighted graph to a text file.
    
    Args:
        G (networkx.DiGraph): A directed graph with 'transition_proba' and 'log_transition_proba' attributes assigned to all edges.
        file_path (str): The path to the file.
    """
    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Write the edges to the file
        for u, v, data in G.edges(data=True):
            # Write the edge as a line in the format: 'FromNodeId ToNodeId transition_proba log_transition_proba'
            file.write(f'{u} {v} {data["transition_proba"]} {data["log_transition_proba"]}\n')


filepath = 'Wiki-Vote.txt'
G = load_weighted_graph(filepath)

save_weighted_graph(G, 'dataset.txt')