import networkx as nx
import numpy as np
import itertools
import time
from heapdict import heapdict

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



#**********************************************************************************************************************
#* PMIA
#**********************************************************************************************************************


""" All the acronyms refer to quantities defined in [1].
This is an implementation of the PMIA algorithm [1] inspired by Github user
nd7141's implementation.

[1] -- Scalable Influence Maximization for Prevalent Viral Marketing in
Large-Scale Social Networks.
"""


ALPHA_ASSERT = "node u=%s must have exactly one neighbor, got %s instead"

def update_ap(ap, S, pmiia, pmiia_mip):
    ''' Assumption: PMIIAv is a directed tree, which is a subgraph of general G.
    PMIIA_MIPv -- dictionary of MIP from nodes in PMIIA
    PMIIAv is rooted at v.
    '''
    # going from leaves to root
    sorted_mips = sorted(pmiia_mip.items(), 
                         key = lambda x: len(x[1]), 
                         reverse = True)
    for u, _ in sorted_mips:
        if u in S:
            ap[(u, pmiia)] = 1
        elif not pmiia.in_edges([u]):
            ap[(u, pmiia)] = 0
        else:
            in_edges = pmiia.in_edges([u])
            prod = 1
            for w, _ in in_edges:
                p = pmiia.edges[(w,u)]['transition_proba']
                prod *= 1 - ap[(w, pmiia)]*p
            ap[(u, pmiia)] = 1 - prod

def update_alpha(alpha, node, S, pmiia, pmiia_mip, ap):
    # going from root to leaves
    sorted_mips = sorted(pmiia_mip.items(), key=lambda x: len(x[1]))
    for u, _ in sorted_mips:
        if u == node:
            alpha[(pmiia, u)] = 1
        else:
            out_edges = list(pmiia.out_edges([u]))
            assert len(out_edges) == 1, ALPHA_ASSERT % (u, len(out_edges))
            w = out_edges[0][1]
            if w in S:
                alpha[(pmiia, u)] = 0
            else:
                in_edges = pmiia.in_edges([w])
                prod = 1
                for up, _ in in_edges:
                    if up != u:
                        pp_up = pmiia.edges[up, w]['transition_proba']
                        prod *= (1 - ap[(up, pmiia)] * pp_up)
                pp = pmiia.edges[u, w]['transition_proba']
                alpha[(pmiia, u)] = alpha[(pmiia, w)] * pp * prod


def compute_pmiia(graph, inactive_seeds, node, theta, S):

    # initialize PMIIA
    pmiia = nx.DiGraph()
    pmiia.add_node(node)
    pmiia_mip = {node: [node]} # MIP(u,v) for u in PMIIA

    crossing_edges = set([in_edge for in_edge in graph.in_edges([node]) 
                          if in_edge[0] not in inactive_seeds + [node]])
    edge_weights = dict()
    dist = {node: 0} # shortest paths from the root u

    # grow PMIIA
    while crossing_edges:
        # Dijkstra's greedy criteria
        min_dist = np.Inf
        sorted_crossing_edges = sorted(crossing_edges) # to break ties consistently
        for edge in sorted_crossing_edges:
            if edge not in edge_weights:
                edge_weights[edge] = graph.edges[edge]['log_transition_proba']
        
            edge_weight = edge_weights[edge]
            if dist[edge[1]] + edge_weight < min_dist:
                min_dist = dist[edge[1]] + edge_weight
                min_edge = edge
        # check stopping criteria
        if min_dist < -np.log(theta):
            dist[min_edge[0]] = min_dist
            pmiia.add_edge(
                min_edge[0], 
                min_edge[1],
                log_transition_proba=min_dist,
                transition_proba=np.exp(-min_dist))

            pmiia_mip[min_edge[0]] = pmiia_mip[min_edge[1]] + [min_edge[0]]
            # update crossing edges
            crossing_edges.difference_update(graph.out_edges(min_edge[0]))
            if min_edge[0] not in S:
                crossing_edges.update([in_edge for in_edge in graph.in_edges(min_edge[0])
                                       if (in_edge[0] not in pmiia) and 
                                       (in_edge[0] not in inactive_seeds)])
        else:
            break
    return pmiia, pmiia_mip


def compute_pmioa(graph, node, theta, S):
    """
     Compute PMIOA -- subgraph of G that's rooted at u.
     Uses Dijkstra's algorithm until length of path doesn't exceed -log(theta)
     or no more nodes can be reached.
    """
    # initialize PMIOA
    pmioa = nx.DiGraph()
    pmioa.add_node(node)
    pmioa_mip = {node: [node]} # MIP(u,v) for v in PMIOA

    crossing_edges = set([out_edge for out_edge in graph.out_edges([node]) 
                          if out_edge[1] not in S + [node]])
    edge_weights = dict()
    dist = {node: 0} # shortest paths from the root u

    # grow PMIOA
    while crossing_edges:
        # Dijkstra's greedy criteria
        min_dist = np.inf
        # break ties consistently with the sort
        sorted_crossing_edges = sorted(crossing_edges) 
        for edge in sorted_crossing_edges:
            if edge not in edge_weights:
                edge_weights[edge] = graph.edges[edge]['log_transition_proba']
            edge_weight = edge_weights[edge]
            if dist[edge[0]] + edge_weight < min_dist:
                min_dist = dist[edge[0]] + edge_weight
                min_edge = edge
        # check stopping criteria
        if min_dist < -np.log(theta):
            dist[min_edge[1]] = min_dist
            pmioa.add_edge(
                min_edge[0], 
                min_edge[1],
                log_transition_proba=min_dist,
                transition_proba=np.exp(-min_dist))

            pmioa_mip[min_edge[1]] = pmioa_mip[min_edge[0]] + [min_edge[1]]
            # update crossing edges
            crossing_edges.difference_update(graph.in_edges(min_edge[1]))
            crossing_edges.update([
                out_edge for out_edge in graph.out_edges(min_edge[1])
                if (out_edge[1] not in pmioa) 
                and (out_edge[1] not in S)])
        else:
            break
    return pmioa, pmioa_mip

def update_inactive_seeds(inactive_seeds, S, node, pmioa, pmiia):
    for v in pmioa[node]:
        for si in S:
            # if seed node is effective and it's blocked by u
            # then it becomes ineffective
            if ((si in pmiia[v]) and 
                (si not in inactive_seeds[v]) and 
                (node in pmiia[v][si])):
                    inactive_seeds[v].append(si)


def pmia(graph, k, theta):
    # initialization
    S = []
    inc_inf = dict(zip(graph.nodes, [0] * len(graph)))
    pmiia = dict() # node to tree
    pmioa = dict()
    pmiia_mip = dict() # node to MIPs (dict)
    pmioa_mip = dict()
    ap = dict()
    alpha = dict()
    inactive_seeds = dict()
    
    # Initialization
    for node in graph:
        inactive_seeds[node] = []
        pmiia[node], pmiia_mip[node] = compute_pmiia(
            graph, 
            inactive_seeds[node], 
            node, 
            theta, 
            S)
        
        for u in pmiia[node]:
            ap[(u, pmiia[node])] = 0 # ap of u node in PMIIA[v]
        update_alpha(alpha, node, S, pmiia[node], pmiia_mip[node], ap)
        for u in pmiia[node]:
            inc_inf[u] += alpha[(pmiia[node], u)]*(1 - ap[(u, pmiia[node])])
    
    # Main Loop
    for _ in range(k):
        node, _ = max(inc_inf.items(), key = lambda x: x[1])
        inc_inf.pop(node) # exclude node u for next iterations

        pmioa[node], pmioa_mip[node] = compute_pmioa(graph, node, theta, S)
        for v in pmioa[node]:
            for w in pmiia[v]:
                if w not in S + [node]:
                    inc_inf[w] -= alpha[(pmiia[v],w)] * (1 - ap[(w, pmiia[v])])

        update_inactive_seeds(inactive_seeds, S, node, pmioa_mip, pmiia_mip)
        S.append(node)

        for v in pmioa[node]:
            if v != u:
                pmiia[v], pmiia_mip[v] = compute_pmiia(
                    graph, inactive_seeds[v], v, theta, S)
                update_ap(ap, S, pmiia[v], pmiia_mip[v])
                update_alpha(alpha, v, S, pmiia[v], pmiia_mip[v], ap)
                # add new incremental influence
                for w in pmiia[v]:
                    if w not in S:
                        inc_inf[w] += alpha[(pmiia[v], w)]*(1 - ap[(w, pmiia[v])])

    return S








#**********************************************************************************************************************
#* GREEDY-MIA
#**********************************************************************************************************************

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
    
def naive_greedy_algorithm(n_source, grph):
    s = []

    for _ in range(n_source):
        max_influence = 0
        max_node = 0
        shared_cache = {}
        for node in range(grph.number_of_nodes()):
            if not node in s:
                influence = sum(ap(i, s + [node], miia(i, 0.001, grph), grph, cache=shared_cache) for i in range(grph.number_of_nodes()))
        
                if influence > max_influence:
                    max_node = node
                    max_influence = influence
        s.append(max_node)
    
    return s

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


def update_cur_best(Q, cur_best, u, u_mg):
    """
    Update cur_best based on the marginal gain of the given node u.
    """
    if cur_best is None or u_mg > Q[0][1]:
        cur_best = u
    return cur_best

def heapify(Q):
    """
    Heapify the queue Q after an element has been updated.
    """
    Q.sort(key=lambda x: x[1], reverse=True)


class Node(object):
    def __init__(self, node):
        self.node = node
        self.mg1 = 0
        self.prev_best = None
        self.mg2 = 0
        self.flag = None
        self.list_index = 0

def celfpp(g, k):
    S = set()
    # Note that heapdict is min heap and hence add negative priorities for
    # it to work.
    Q = heapdict()
    last_seed = None
    cur_best = None
    node_data_list = []
    shared_cache = {}
    start_time = time.time()
    spread_computations = 0  # Track the number of spread computations
    
    for node in range(g.number_of_nodes()):
        node_data = Node(node)
        node_data.mg1 = sum(ap(i,[node], miia(i, 0.001, g), g, cache=shared_cache) for i in range(g.number_of_nodes()))
        node_data.prev_best = cur_best
        node_data.mg2 = sum(ap(i,[node, cur_best.node], miia(i, 0.001, g), g, cache=shared_cache) for i in range(g.number_of_nodes())) if cur_best else node_data.mg1
        node_data.flag = 0
        cur_best = cur_best if cur_best and cur_best.mg1 > node_data.mg1 else node_data
        g.nodes[node]['node_data'] = node_data
        node_data_list.append(node_data)
        node_data.list_index = len(node_data_list) - 1
        Q[node_data.list_index] = - node_data.mg1

    
    timelapse = [time.time() - start_time]
    LOOKUPS = [g.number_of_nodes()]
    SPREAD = [Q.peekitem()[1]]
    
    
    while len(S) < k:
        node_idx, _ = Q.peekitem()
        node_data = node_data_list[node_idx]
        if node_data.flag == len(S):
            S.add(node_data.node)
            del Q[node_idx]
            last_seed = node_data
            continue
        elif node_data.prev_best == last_seed:
            node_data.mg1 = node_data.mg2
        else:
            before = sum(ap(i,S, miia(i, 0.001, g), g, cache=shared_cache) for i in range(g.number_of_nodes()))
            S.add(node_data.node)
            after = sum(ap(i,S, miia(i, 0.001, g), g, cache=shared_cache) for i in range(g.number_of_nodes()))
            S.remove(node_data.node)
            node_data.mg1 = after - before
            node_data.prev_best = cur_best
            S.add(cur_best.node)
            before = sum(ap(i,S, miia(i, 0.001, g), g, cache=shared_cache) for i in range(g.number_of_nodes()))
            S.add(node_data.node)
            after = sum(ap(i,S, miia(i, 0.001, g), g, cache=shared_cache) for i in range(g.number_of_nodes()))
            S.remove(cur_best.node)
            if node_data.node != cur_best.node: S.remove(node_data.node)
            node_data.mg2 = after - before

        if cur_best and cur_best.mg1 < node_data.mg1:
            cur_best = node_data

        node_data.flag = len(S)
        Q[node_idx] = - node_data.mg1
        
        # Increment spread computations counter
        timelapse.append(time.time() - start_time)
        spread_computations += 1
        LOOKUPS.append(spread_computations)
        SPREAD.append(SPREAD[-1] + node_data.mg1)
        
    return(S,SPREAD,timelapse,LOOKUPS)

G = generate_graph(num_nodes_min_max=[100,110],directed=True)
k = 10
theta = 0.001
pmia_output = pmia(G, k, theta)
greedy_output = naive_greedy_algorithm(k, G)
celf_output = celf(G,k)
celfpp_output = celfpp(G,k)

print("PMIA output:       ", pmia_output)
print("Greedy_MIA output: ", greedy_output)
print("CELF_MIA output:   ", celf_output[0])
print("CELF++_MIA output: ", celf_output[0])
