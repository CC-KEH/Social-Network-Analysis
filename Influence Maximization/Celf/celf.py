import matplotlib.pyplot as plt
from random import uniform, seed
import numpy as np
import time
from igraph import *
from icm import IC


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

def celf(g,k,p=0.1,mc=1000):  
    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """
      
    # --------------------
    # Find the first node with greedy algorithm
    # --------------------
    
    # Calculate the first iteration sorted list
    start_time = time.time() 
    marg_gain = [IC(g,[node],p,mc) for node in range(g.vcount())]
    # Create the sorted list of nodes and their marginal gain 
    Q = sorted(zip(range(g.vcount()),marg_gain), key=lambda x: x[1],reverse=True)

    # Select the first node and remove from candidate list
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q, LOOKUPS, timelapse = Q[1:], [g.vcount()], [time.time()-start_time]
    
    # --------------------
    # Find the next k-1 nodes using the list-sorting procedure
    # --------------------
    
    for _ in range(k-1):    

        check, node_lookup = False, 0
        
        while not check:
            
            # Count the number of times the spread is computed
            node_lookup += 1
            
            # Recalculate spread of top node
            current = Q[0][0]
            
            # Evaluate the spread function and store the marginal gain in the list
            Q[0] = (current,IC(g,S+[current],p,mc) - spread)

            # Re-sort the list
            Q = sorted(Q, key = lambda x: x[1], reverse = True)

            # Check if previous top node stayed on top after the sort
            check = (Q[0][0] == current)

        # Select the next node
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)
        LOOKUPS.append(node_lookup)
        timelapse.append(time.time() - start_time)

        # Remove the selected node from the list
        Q = Q[1:]
    
    return(S,SPREAD,timelapse,LOOKUPS)


if __name__ == "__main__":
    # Create simple network with 0 and 1 as the influential nodes
    source = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,3,4,5]
    target = [2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,6,7,8,9]
    g = Graph(directed=True)
    g.add_vertices(range(10))
    g.add_edges(zip(source,target))
    # Plot graph
    g.vs["label"], g.es["color"], g.vs["color"] = range(10), "#B3CDE3", "#FBB4AE"
    plot(g,bbox = (200,200),margin = 20,layout = g.layout("kk"))
    
    # Run algorithms
    celf_output   = celf(g,2,p = 0.2,mc = 1000)
    # Print results
    print("celf output:   " + str(celf_output[0]))