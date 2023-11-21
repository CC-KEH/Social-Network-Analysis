import utils
import time

def greedy_mia(graph, k, theta):
    """
    Input:  graph object, number of seed nodes, influence threshold
    Output: optimal seed set
    """
    
    # Initialize
    S = set()
    last_seed = None
    cur_best = None

    mg1 = {node: compute_MIIA(node, theta) for node in graph.nodes}
    prev_best = {node: None for node in graph.nodes}
    mg2 = {node: compute_MIIA(node, theta) - compute_MIIA(prev_best[node], theta) if prev_best[node] else mg1[node] for node in graph.nodes}

    for node in graph.nodes:
        if not cur_best or mg1[cur_best] < mg1[node]:
            cur_best = node

    while len(S) < k:
        S.add(cur_best)
        last_seed = cur_best
        cur_best = None
        for node in graph.nodes:
            if node in S:
                continue
            before = compute_MIIA(S, theta)
            S.add(node)
            after = compute_MIIA(S, theta)
            S.remove(node)
            mg1[node] = after - before
            prev_best[node] = cur_best
            S.add(cur_best)
            before = compute_MIIA(S, theta)
            S.add(node)
            after = compute_MIIA(S, theta)
            S.remove(cur_best)
            if node != cur_best: S.remove(node)
            mg2[node] = after - before
            if not cur_best or mg1[cur_best] < mg1[node]:
                cur_best = node

    return S
