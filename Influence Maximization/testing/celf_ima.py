def celf_mia(g, k, theta=0.1):
    """
    Input:  graph object, number of seed nodes, influence threshold
    Output: optimal seed set, resulting spread, time for each iteration
    """
    
    # Calculate the first iteration sorted list
    start_time = time.time() 

    
    # Initialize
    S = set()
    IncInf = {v: 0 for v in g.vs}
    MIIA = {v: compute_MIIA(v, theta) for v in g.vs}
    MIOA = {v: compute_MIOA(v, theta) for v in g.vs}
    start_time = time.time()
    
    for v in g.vs:
        ap = {u: 0 for u in MIIA[v]}
        alpha = compute_alpha(v, MIIA[v])
        for u in MIIA[v]:
            IncInf[u] += alpha[u] * (1 - ap[u])

    # Create the sorted list of nodes and their marginal gain 
    Q = sorted(zip(range(g.vcount()), [IncInf[v] for v in g.vs]), key=lambda x: x[1], reverse=True)

    # Select the first node and remove from candidate list
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q, LOOKUPS, timelapse = Q[1:], [g.vcount()], [time.time()-start_time]

    # Find the next k-1 nodes using the list-sorting procedure
    while len(S) < k:
        check, node_lookup = False, 0
        while not check:
            # Count the number of times the spread is computed
            node_lookup += 1
            # Recalculate spread of top node
            current = Q[0][0]
            # Evaluate the spread function and store the marginal gain in the list
            Q[0] = (current, IncInf.get(current, 0) - spread, len(S), None)
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