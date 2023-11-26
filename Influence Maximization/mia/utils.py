import itertools
import networkx as nx
import numpy as np

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
