import matplotlib.pyplot as plt
import random
import numpy as np
import time
from igraph import *
import networkx as nx
import itertools

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



def celfpp(g, k, p=0.1, mc=1000):  
    start_time = time.time() 
    
    marg_gain = [IC(g,[node],p,mc) for node in range(g.vcount())]
    
    Q = sorted([(node, gain, 0, gain, 0) for node, gain in zip(range(g.vcount()), marg_gain)], key=lambda x: x[1], reverse=True)
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    
    Q, LOOKUPS, timelapse = Q[1:], [g.vcount()], [time.time()-start_time]
    last_seed = None
    cur_best = None
    
    for _ in range(k-1):    
        check, node_lookup = False, 0
        while not check:
            node_lookup += 1
            current = Q[0][0]
            if Q[0][4] == 1 and Q[0][2] == len(S):
                check = True
            else:
                if Q[0][3] == last_seed:
                    Q[0] = (current, Q[0][1], len(S), Q[0][1], 1)
                else:
                    spread_gain = IC(g, S+[current], p, mc) - spread
                    Q[0] = (current, spread_gain, len(S), spread_gain, 0)
            Q = sorted(Q, key = lambda x: x[1], reverse = True)
            cur_best = Q[0][0]
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)
        LOOKUPS.append(node_lookup)
        timelapse.append(time.time() - start_time)
        last_seed = Q[0][0]
        Q = Q[1:]
    
    return(S,SPREAD,timelapse,LOOKUPS)



# Create simple network with 0 and 1 as the influential nodes
source = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
target = [2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9]

g = Graph(directed=True)
g.add_vertices(range(10))
g.add_edges(zip(source,target))

celfpp_output = celfpp(g,2,p=0.2,mc=1000)
print("celf++ output: " + str(celfpp_output))