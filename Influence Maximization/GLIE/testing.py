import itertools
import networkx as nx
import numpy as np
import random
def mip(u, v, grph):
    path = nx.dijkstra_path(grph, u, v, weight='log_transition_probability')
    a, b = itertools.tee(path)
    next(b, None)
    return path, np.exp(-sum(grph.edges[s]['log_transition_probability'] for s in zip(a, b)))

def pp(u, v, grph):
    return mip(u, v, grph)[1]

def ap(u, S, MIIA):
    if u in S:
        return 1
    elif not MIIA.predecessors(u):
        return 0
    else:
        product = 1
        for w in MIIA.predecessors(u):
            product *= (1 - ap(w, S, MIIA) * pp(w, u, MIIA))
        return 1 - product


def miia(v, theta, grph):
    u_list = []
    for u in range(grph.number_of_nodes()):
        if v == u:
            continue
        path, score = mip(u, v, grph)
        if score > theta:
            u_list.append(path)
    return u_list

def mioa(u, theta, grph):
    v_list = []
    for v in range(grph.number_of_nodes()):
        if v == u:
            continue
        path, score = mip(u, v, grph)
        if score > theta:
            v_list.append(path)
    return v_list

def in_neighbors(u, miia):
    result_set = []
    for path in miia:
        if u in path and path[0] != u:
            result_set.append(path[path.index(u) - 1])
    return result_set

def compute_alpha(v, u, S, miia):
    if u == v:
        return 1
    else:
        w = list(miia.successors(u))[0]
        if w in S:
            return 0
        else:
            product = compute_alpha(v, w, S, miia) * pp(u, w, G)
            for u0 in set(miia.predecessors(w)) - {u}:
                product *= (1 - ap(u0, S, G) * pp(u0, w, G))
            return product

def MIA(grph, k, theta):
    S = set()
    IncInf = {v: 0 for v in grph.nodes()}

    for v in grph.nodes():
        miia_v = miia(v, theta, grph)
        mioa_v = mioa(v, theta, grph)

        for u in miia_v:
            ap_u = 0
            alpha_u = compute_alpha(v, u, S, grph.reverse())
            for u0 in miia_v:
                IncInf[u0] += alpha_u * (1 - ap(u0, S, grph.reverse()))

    for i in range(k):
        u = max(set(grph.nodes()) - S, key=lambda v: IncInf[v])

        for v in set(mioa[u]) - S:
            for w in set(miia[v]) - S:
                IncInf[w] -= compute_alpha(v, w, S, grph.reverse()) * (1 - ap(w, S, grph.reverse()))

        S.add(u)

        for v in set(mioa[u]) - S:
            for w in set(miia[v]) - S:
                ap_w = ap(w, S, grph.reverse())
                alpha_w = compute_alpha(v, w, S, grph.reverse())
                IncInf[w] += alpha_w * (1 - ap_w)

    return S

# Example Usage
# Assuming you have a directed graph 'grph' with appropriate edge weights and probabilities
# Set theta - replace this with the actual threshold
theta = 0.5


# Generate a directed graph with 5 nodes
G = nx.DiGraph()
G.add_nodes_from(range(1, 6))

# Add random edges with weights and probabilities
for u, v in G.edges():
    G[u][v]['weight'] = random.uniform(0.1, 1.0)
    G[u][v]['log_transition_probability'] = -1 * random.uniform(0.1, 1.0)  # Assuming log probabilities

# Print edge data
for edge in G.edges(data=True):
    print(f"Edge {edge[0]} -> {edge[1]}: Weight = {edge[2]['weight']}, Log Probability = {edge[2]['log_transition_probability']}")

# Visualize the graph
nx.draw(G, with_labels=True, font_weight='bold', arrowsize=20, node_size=700, font_size=10)

# Run the MIA algorithm
seed_set = MIA(G, 2, theta)

print("Seed Set:", seed_set)
