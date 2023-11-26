import diffusion
from heapdict import heapdict
import time

def celfpp(g, k, p=0.1, mc=1000):  
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
    Q = sorted([(node, gain, 0, None) for node, gain in zip(range(g.vcount()), marg_gain)], key=lambda x: x[1], reverse=True)

    # Select the first node and remove from candidate list
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q, LOOKUPS, timelapse = Q[1:], [g.vcount()], [time.time()-start_time]
    last_seed = None
    cur_best = None
    
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
            
            # Check if the previous top node stayed on top after the sort
            if Q[0][2] == len(S):
                check = True
            else:
                # Evaluate the spread function and store the marginal gain in the list
                if Q[0][3] == last_seed and Q[0][2] == len(S) - 1:
                    Q[0] = (current, Q[0][1], len(S), last_seed)
                else:
                    Q[0] = (current, IC(g, S+[current], p, mc) - spread, len(S), cur_best)

            # Re-sort the list
            Q = sorted(Q, key = lambda x: x[1], reverse = True)

        # Select the next node
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)
        LOOKUPS.append(node_lookup)
        timelapse.append(time.time() - start_time)
        last_seed = Q[0][0]

        # Remove the selected node from the list
        Q = Q[1:]
    
    return(S,SPREAD,timelapse,LOOKUPS)


class Node(object):
    def __init__(self, node):
        self.node = node
        self.mg1 = 0
        self.prev_best = None
        self.mg2 = 0
        self.flag = None
        self.list_index = 0

def celfpp(graph, diffuse, k):
    S = set()
    # Note that heapdict is min heap and hence add negative priorities for
    # it to work.
    Q = heapdict()
    last_seed = None
    cur_best = None
    node_data_list = []

    for node in graph.nodes:
        node_data = Node(node)
        node_data.mg1 = diffuse.diffuse_mc([node])
        node_data.prev_best = cur_best
        node_data.mg2 = diffuse.diffuse_mc([node, cur_best.node]) if cur_best else node_data.mg1
        node_data.flag = 0
        cur_best = cur_best if cur_best and cur_best.mg1 > node_data.mg1 else node_data
        graph.nodes[node]['node_data'] = node_data
        node_data_list.append(node_data)
        node_data.list_index = len(node_data_list) - 1
        Q[node_data.list_index] = - node_data.mg1

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
            before = diffuse.diffuse_mc(S)
            S.add(node_data.node)
            after = diffuse.diffuse_mc(S)
            S.remove(node_data.node)
            node_data.mg1 = after - before
            node_data.prev_best = cur_best
            S.add(cur_best.node)
            before = diffuse.diffuse_mc(S)
            S.add(node_data.node)
            after = diffuse.diffuse_mc(S)
            S.remove(cur_best.node)
            if node_data.node != cur_best.node: S.remove(node_data.node)
            node_data.mg2 = after - before

        if cur_best and cur_best.mg1 < node_data.mg1:
            cur_best = node_data

        node_data.flag = len(S)
        Q[node_idx] = - node_data.mg1

    return S