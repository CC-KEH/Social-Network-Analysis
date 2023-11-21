import numpy as np
def compute_MIIA(v, theta):
    # This function should return the Maximum Influence In-Arborescence of a node v
    # For simplicity, let's return all nodes that have a direct edge to v with weight >= theta
    return {u for u in g.vs if g[u, v] >= theta}

def compute_MIOA(v, theta):
    # This function should return the Maximum Influence Out-Arborescence of a node v
    # For simplicity, let's return all nodes that v has a direct edge to with weight >= theta
    return {u for u in g.vs if g[v, u] >= theta}

def compute_alpha(v, MIIA_v):
    # This function should compute the alpha values for v and each node in MIIA(v)
    # For simplicity, let's return the edge weights from v to each node in MIIA(v)
    return {u: g[v, u] for u in MIIA_v}

def compute_ap(w, S, MIIA_v):
    # This function should compute the activation probability of node w given seed set S and MIIA(v)
    # For simplicity, let's assume the activation probability is the product of edge weights from S to w
    return np.prod([g[s, w] for s in S if s in MIIA_v])


def MIA_IC(g, k, theta):
    """
    Input:  graph object, set of seed nodes, influence threshold
    Output: Seed set S
    """
    
    # Initialize
    S = set()
    IncInf = {v: 0 for v in g.vs}
    MIIA = {v: compute_MIIA(v, theta) for v in g.vs}
    MIOA = {v: compute_MIOA(v, theta) for v in g.vs}

    for v in g.vs:
        ap = {u: 0 for u in MIIA[v]}
        alpha = compute_alpha(v, MIIA[v])
        for u in MIIA[v]:
            IncInf[u] += alpha[u] * (1 - ap[u])

    while len(S) <= k:
        u = max(g.vs, key=lambda v: IncInf[v])
        for v in MIOA[u] - S:
            for w in MIIA[v] - S:
                IncInf[w] -= alpha[v] * (1 - ap[w])
        S.add(u)
        for v in MIOA[u]:
            ap = compute_ap(w, S, MIIA[v])
            alpha = compute_alpha(v, MIIA[v])
            for w in MIIA[v] - S:
                IncInf[w] += alpha[v] * (1 - ap[w])

    return S

