import matplotlib.pyplot as plt
import random
import time
from igraph import *
from icm import IC
import numpy as np

def visualize_graph(g, seed_nodes, spread_nodes):
    """
    Visualize the graph with seed nodes and spread nodes highlighted.
    """
    layout = g.layout("fr")  # You can choose a layout algorithm that suits your graph better
    visual_style = {}
    visual_style["vertex_size"] = 20  # Adjust the size of vertices as needed
    visual_style["vertex_color"] = ["red" if v in seed_nodes else "blue" for v in range(g.vcount())]

    # Add labels to nodes
    labels = [str(i) for i in range(g.vcount())]
    visual_style["vertex_label"] = labels

    # Set edge color
    visual_style["edge_color"] = "#B3CDE3"  # You can change the edge color as needed

    # Plot the graph
    plot(g, layout=layout, **visual_style)
    plt.show()


def random_subgraph(g, p):
    """
    Generate a random subgraph using edge probabilities.
    Input:  graph object, edge activation probability
    Output: random subgraph as an adjacency matrix
    """
    random_graph = g.copy()
    edges_to_delete = [edge.index for edge in random_graph.es if random.random() < p]
    random_graph.delete_edges(edges_to_delete)
    return random_graph

def IC(g,S,p=0.5,mc=1000):
    """
    Input:  graph object, set of seed nodes, propagation probability
            and the number of Monte-Carlo simulations
    Output: average number of nodes influenced by the seed nodes
    """
    
    # Loop over the Monte-Carlo Simulations
    spread = []
    for i in range(mc):
        
        # Simulate propagation process      
        new_active, A = S[:], S[:]
        while new_active:

            # For each newly active node, find its neighbors that become activated
            new_ones = []
            for node in new_active:
                
                # Determine neighbors that become infected
                np.random.seed(i)
                success = np.random.uniform(0,1,len(g.neighbors(node,mode="out"))) < p
                new_ones += list(np.extract(success, g.neighbors(node,mode="out")))

            new_active = list(set(new_ones) - set(A))
            
            # Add newly activated nodes to the set of activated nodes
            A += new_active
            
        spread.append(len(A))
        
    return(np.mean(spread))

def mixed_greedy(g, k, p=0.1, mc=1000):
    """
    Input:  graph object, number of seed nodes, edge activation probability, Monte Carlo simulations
    Output: optimal seed set, resulting spread
    """
    start_time = time.time()
    subgraph = random_subgraph(g, p)

    # Initialize the sorted list with node indices and their estimated influence spread
    marg_gain = [(node, IC(subgraph, [node], p, mc)) for node in range(g.vcount())]
    Q = sorted(marg_gain, key=lambda x: x[1], reverse=True)

    # Select the first node and remove from candidate list
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q, LOOKUPS,timelapse = Q[1:],[g.vcount()], [time.time() - start_time]

    affected_nodes = set(g.neighbors(S[0]))  # Initialize affected_nodes with neighbors of the first node

    for _ in range(k - 1):
        check, node_lookup = False, 0

        while not check:
            # Count the number of times the spread is computed
            node_lookup += 1
            # Recalculate spread of top node
            current = Q[0][0]

            # Recalculate influence spread and store the marginal gain in the list
            Q[0] = (current, IC(subgraph, S + [current], p, mc) - spread)

            # Re-sort the list
            Q = sorted(Q, key=lambda x: x[1], reverse=True)

            # Check if the previous top node stayed on top after the sort
            check = (Q[0][0] == current)

        # Select the next node
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)
        LOOKUPS.append(node_lookup)
        timelapse.append(time.time() - start_time)

        # Remove the selected node from the list
        Q = Q[1:]

        # Update affected_nodes with neighbors of the newly added node
        affected_nodes.update(set(g.neighbors(current)) - set(S))

    return S, SPREAD, timelapse


if __name__ == "__main__":
    # Create a simple network
    source = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5]
    target = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 6, 7, 8, 9]

    g = Graph(directed=True)
    g.add_vertices(range(10))
    g.add_edges(zip(source,target))

    # Plot graph
    g.vs["label"], g.es["color"], g.vs["color"] = range(10), "#B3CDE3", "#FBB4AE"
    plot(g,bbox = (200,200),margin = 20,layout = g.layout("kk"))
    
    # Run algorithms
    mgc_output   = mixed_greedy(g,2,p = 0.2,mc = 1000)
    print("MIXEDGREEDY output:", mgc_output[0])