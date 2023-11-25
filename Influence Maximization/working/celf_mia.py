from random import uniform, seed
import numpy as np
import time
from igraph import *
import networkx as nx
import itertools
def generate_graph(rand=None,
                   num_nodes_min_max=[10, 11],
                   rate=0.4,
                   weight_min_max=[0, 0.1],
                   directed=False):
    """Creates a connected graph.

    Args:
        rand: A random seed for the graph generator. Default= None.
        num_nodes_min_max: A sequence [lower, upper) number of nodes per graph.
        weight_min_max: A sequence [lower, upper) transition probabilities for the 
            edges in the graph

    Returns:
        The graph.
    """
    
    if rand is None:
        seed = 2
        rand = np.random
        rand.seed(seed)
        
    
    # Sample num_nodes.
    num_nodes = rand.randint(*num_nodes_min_max)

    # Create geographic threshold graph.
    rand_graph = nx.fast_gnp_random_graph(num_nodes, rate, directed=directed)
    weights = np.random.uniform(weight_min_max[0], 
                                weight_min_max[1], 
                                rand_graph.number_of_edges())
    weights_dict = dict(zip(rand_graph.edges, 
                            weights))
    log_weights_dict = dict(zip(rand_graph.edges, 
                                -np.log(weights)))
    
    nx.set_edge_attributes(rand_graph, weights_dict, 'transition_proba')
    nx.set_edge_attributes(rand_graph, log_weights_dict, 'log_transition_proba')
    
    return rand_graph


def mip(u, v, grph):
    """ Maximum influence path function (or shortest
    path in the graph)
    
    Returns:
        - path as list of node index
        - path length in term of propagation probability
    """
    path = nx.dijkstra_path(grph, u, v, weight='log_transition_probability')
    a, b = itertools.tee(path)
    next(b, None)
    return path, np.exp(-sum(grph.edges[s]['log_transition_proba'] for s in zip(a,b)))

def pp(u, v, grph):
    """ Propagation probability of an edge (u, v).
    Product of all probabilities of the shortest path
    between u and v.
    """
    return mip(u, v, grph)[1]

def miia(v, theta, grph):
    """ Maximum Influence In-Arborescence funtion
    All the paths to v with propagation probability
    above theta
    """
    u_list = []
    for u in range(grph.number_of_nodes()):
        if v == u: continue
        path, score = mip(u, v, grph)
        if score > theta:
            u_list.append(path)
    return u_list

def mioa(u, theta, grph):
    """ Maximum Influence Out-Arborescence funtion
    All the paths from u with propagation probability
    above theta
    """
    v_list = []
    for v in range(grph.number_of_nodes()):
        if v == u: continue
        path, score = mip(u, v, grph)
        if score > theta:
            v_list.append(path)
    return v_list

def in_neighbors(u, miia):
    """ Compute in-neighbors of u in miia
    """
    result_set = []
    for path in miia:
        if u in path and path[0] != u:
            result_set.append(path[path.index(u) - 1])
    return result_set

def hsh(i, lst):
    return int(str(hash(repr(lst))) + str(hash(i)))
    
def ap(u, S, miia_v_theta, grph, cache={}, cnt=0):
    """ Compute activation probability of node 
    u, from set S and maximum influence in-arboresence
    miia_v_theta.
    """
    n_in = in_neighbors(u, miia_v_theta)
    if u in S:
        return 1.
    elif not len(n_in):
        return 0.
    else:
        base = 1
        for in_neighbor in n_in:
            if hsh(in_neighbor, miia_v_theta) in cache:
                p = cache[hsh(in_neighbor, miia_v_theta)]
            else:
                p =  ap(in_neighbor, S, miia_v_theta, grph, cache)
                cache[hsh(in_neighbor, miia_v_theta)] = p
            base *= (1 - p*pp(in_neighbor, u, grph))
        return 1 - base
 


def celf(g,k):  
    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """
      
    # --------------------
    # Find the first node with greedy algorithm
    # --------------------
    
    # Calculate the first iteration sorted list
    start_time = time.time() 
    marg_gain = []
    shared_cache = {}
    for node in range(g.number_of_nodes()):
        gain = sum(ap(i,[node], miia(i, 0.001, g), g, cache=shared_cache) for i in range(g.number_of_nodes()))
        marg_gain.append(gain)
    # Create the sorted list of nodes and their marginal gain 
    Q = sorted(zip(range(g.number_of_nodes()),marg_gain), key=lambda x: x[1],reverse=True)

    # Select the first node and remove from candidate list
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q, LOOKUPS, timelapse = Q[1:], [g.number_of_nodes()], [time.time()-start_time]
    
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
            Q[0] = (current, sum(ap(i,S+[current], miia(i, 0.001, g), g, cache=shared_cache) for i in range(g.number_of_nodes())) - spread)
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
    G = generate_graph(directed=True)
    k = 10
    theta = 0.001
    
    # Run algorithms
    celf_output   = celf(G,k)
    # Print results
    print("celf output:   " + str(celf_output[0]))
    