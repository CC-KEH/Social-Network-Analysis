def celfpp_mia(g, k,theta=0.1):  
    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """
      
    # --------------------
    # Find the first node with greedy algorithm
    # --------------------
    
    # Calculate the first iteration sorted list
    start_time = time.time() 

    # Initialize
    IncInf = {v: 0 for v in g.vs}
    MIIA = {v: compute_MIIA(v, theta) for v in g.vs}
    MIOA = {v: compute_MIOA(v, theta) for v in g.vs}

    for v in g.vs:
        ap = {u: 0 for u in MIIA[v]}
        alpha = compute_alpha(v, MIIA[v])
        for u in MIIA[v]:
            IncInf[u] += alpha[u] * (1 - ap[u])

    # Create the sorted list of nodes and their marginal gain 
    Q = sorted([(node, gain, 0, None) for node, gain in zip(range(g.vcount()), [IncInf[v] for v in g.vs])], key=lambda x: x[1], reverse=True)

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
                    Q[0] = (current, IncInf[current] - spread, len(S), cur_best)

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