import networkx as nx
import numpy as np
import itertools
import time

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
                # success = np.random.uniform(0,1,len(g.neighbors(node,mode="out"))) < p
                # new_ones += list(np.extract(success, g.neighbors(node,mode="out")))
                outgoing_neighbors = list(nx.neighbors(g, node))
                success = np.random.uniform(0, 1, len(outgoing_neighbors)) < p
                new_ones += list(np.extract(success, outgoing_neighbors))
            
            new_active = list(set(new_ones) - set(A))
            
            # Add newly activated nodes to the set of activated nodes
            A += new_active
            
        spread.append(len(A))
        
    return(np.mean(spread))

def lie_function(G, S):
    # Constants for the LIE model
    p_star = 0.5  # Placeholder value, replace with actual value
    d_star = 1  # Placeholder value, replace with actual value

    # Get one-hop and two-hop neighbors
    N1_S = set()
    N2_S = set()
    for node in S:
        N1_S.update(G.neighbors(node))
        N2_S.update(nx.single_source_shortest_path_length(G, node, cutoff=2).keys())
    
    N1_S.difference_update(S)  # Remove nodes in the seed set
    N2_S.difference_update(S)  # Remove nodes in the seed set and one-hop neighbors

    # Calculate the LIE function based on Equation (4)
    lie_value = len(S) + sum(
        1 + 1 / len(N1_S.union(S)) * sum(p_star * d_star for u in N2_S.difference(S))
        for x in N1_S.difference(S)
    )

    return lie_value

def DBATM(G, k, max_iter):
    # Initialize parameters
    n = G.number_of_nodes()
    alpha = 0.9  # Velocity update parameter
    beta = 0.5  # Frequency update parameter
    F_max = 0
    S_max = []

    # Initialize population and velocity
    population = np.random.randint(0, 2, (n, k))
    velocity = np.zeros((n, k))

    # Initialize worst experience memory
    worst_experience = np.zeros((n, k))

    for iteration in range(max_iter):
        # Calculate fitness values using the LIE function
        fitness = np.zeros(population.shape[0])
        for i in range(population.shape[0]):
            S = population[i, :]
            F = lie_function(G, S)  # Replace with your actual LIE function
            fitness[i] = F

        # Update best solution
        if np.max(fitness) > F_max:
            F_max = np.max(fitness)
            S_max = population[np.argmax(fitness), :]

        # Update velocity
        best_position = population[np.argmax(fitness), :]
        velocity = (0.9 * velocity + alpha * (best_position - population)) * beta

        # Update position
        for i in range(population.shape[0]):
            for j in range(k):
                if np.random.rand() < 0.5:
                    if population[i, j] == 0:
                        population[i, j] = 1
                        worst_experience[i, j] = np.random.rand()
                    else:
                        population[i, j] = 0
                        worst_experience[i, j] = 1 - np.random.rand()

        # Apply local search
        for i in range(population.shape[0]):
            S = population[i, :]
            new_S = local_search(G, S, worst_experience[i, :])
            new_F = lie_function(G, new_S)  # Replace with your actual LIE function

            if new_F > fitness[i]:
                population[i, :] = new_S
                worst_experience[i, :] = np.zeros(k)

    return S_max, F_max

def local_search(G, current_seed_set, worst_experience):
    current_lie_value = lie_function(G, current_seed_set)

    for i in range(len(current_seed_set)):
        # Try removing each node from the seed set and evaluate the LIE function
        candidate_seed_set = current_seed_set[:i] + current_seed_set[i+1:]
        candidate_lie_value = lie_function(G, candidate_seed_set)

        # If the candidate solution is better, update the seed set and LIE value
        if candidate_lie_value > current_lie_value:
            current_seed_set = candidate_seed_set
            current_lie_value = candidate_lie_value

    # Check if the worst experience is worse than the current solution
    if np.any(worst_experience > current_lie_value):
        # Replace the worst experience with the current solution
        worst_experience = np.maximum(worst_experience, current_lie_value)

    return current_seed_set, current_lie_value, worst_experience



if __name__ == "__main__":
    # Load social network data
    # G = nx.read_gml('data.gml')
    G = generate_graph(directed=True)
    # Set parameters
    k = 10  # Number of influential nodes
    max_iter = 100  # Maximum number of iterations

    # Run DBATM with enhanced features
    S_max, F_max = DBATM(G, k, max_iter)

    print("Maximum influence:", F_max)
    print("Selected influential nodes:", S_max)






#* Rough Implementaion of DBATM
# import networkx as nx
# import numpy as np

# def DBATM(G, k, max_iter):
#     # Initialize parameters
#     n = G.number_of_nodes()
#     alpha = 0.9  # Velocity update parameter
#     beta = 0.5  # Frequency update parameter
#     F_max = 0
#     S_max = []

#     # Initialize population
#     population = np.random.randint(0, 2, (n, k))

#     # Initialize worst experience memory
#     worst_experience = np.zeros((n, k))

#     for iter in range(max_iter):
#         # Calculate fitness values
#         fitness = np.zeros(population.shape[0])
#         for i in range(population.shape[0]):
#             S = population[i, :]
#             F = estimate_influence(G, S)
#             fitness[i] = F

#         # Update best solution
#         if np.max(fitness) > F_max:
#             F_max = np.max(fitness)
#             S_max = population[np.argmax(fitness), :]

#         # Update velocity
#         best_position = population[np.argmax(fitness), :]
#         velocity = (0.9 * velocity + alpha * (best_position - population)) * beta

#         # Update position
#         for i in range(population.shape[0]):
#             for j in range(k):
#                 if np.random.rand() < 0.5:
#                     if population[i, j] == 0:
#                         population[i, j] = 1
#                         worst_experience[i, j] = np.random.rand()
#                     else:
#                         population[i, j] = 0
#                         worst_experience[i, j] = 1 - np.random.rand()

#         # Apply local search
#         for i in range(population.shape[0]):
#             S = population[i, :]
#             new_S = local_search(G, S, worst_experience[i, :])
#             new_F = estimate_influence(G, new_S)

#             if new_F > fitness[i]:
#                 population[i, :] = new_S
#                 worst_experience[i, :] = np.zeros(k)

#     return S_max, F_max

# def local_search(G, S, worst_experience):
#     # Implement local search with removal and swap operators
#     new_S = S.copy()
#     F_new = estimate_influence(G, new_S)

#     # Removal search
#     for i in range(S.shape[0]):
#         if S[i] == 1:
#             new_S[i] = 0
#             new_F = estimate_influence(G, new_S)

#             if new_F > F_new:
#                 F_new = new_F

#             new_S[i] = 1

#     # Swap search
#     for i in range(S.shape[0]):
#         if S[i] == 1:
#             for j in range(G.number_of_nodes()):
#                 if S[j] == 0:
#                     new_S[i] = 0
#                     new_S[j] = 1
#                     new_F = estimate_influence(G, new_S)

#                     if new_F > F_new:
#                         F_new = new_F
#                         new_S[i] = 1
#                         new_S[j] = 0

#     return new_S

# def estimate_influence(G, S):
#     # Implement influence estimation algorithm here (e.g., IC or LT model)
#     ...

# if __name__ == "__main__":
#     # Load social network data
#     G = nx.read_gml('data.gml')

#     # Set parameters
#     k = 10  # Number of influential nodes
#     max_iter = 100  # Maximum number of iterations

#     # Run DBATM with enhanced features
#     S_max, F_max = DBATM(G, k, max_iter)

#     print("Maximum influence:", F_max)
#     print("Selected influential nodes:", S_max)
