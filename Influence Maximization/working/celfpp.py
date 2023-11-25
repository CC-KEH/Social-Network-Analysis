def update_cur_best(Q, cur_best, u, u_mg):
    """
    Update cur_best based on the marginal gain of the given node u.
    """
    if cur_best is None or u_mg > Q[0][1]:
        cur_best[0] = u
    return cur_best

def heapify(Q):
    """
    Heapify the queue Q after an element has been updated.
    """
    Q.sort(key=lambda x: x[1], reverse=True)

def celfpp(graph, diffuse, k):
    S, Q, last_seed, cur_best = [], [], [], []
    SPREAD, timelapse, LOOKUPS = [], [], []

    # Initialization: Compute initial marginal gains and update heap Q
    for u in graph.nodes:
        u_mg1 = diffuse.diffuse_mc([u])  # Calculate u_mg1
        u_prev_best = cur_best[0]
        u_mg2 = diffuse.diffuse_mc([u, u_prev_best]) - u_mg1 if u_prev_best is not None else u_mg1
        u_flag = 0
        Q.append([u, u_mg1, u_prev_best, u_mg2, u_flag])
        cur_best[0] = update_cur_best(Q, cur_best, u, u_mg1)

    # Iteratively select seeds until |S| reaches k
    while len(S) < k:
        u = Q[0]  # root element in Q
        if u[4] == len(S):
            S.append(u[0])
            Q.pop(0)
            last_seed[0] = u[0]
            cur_best[0] = None
            SPREAD.append(u[1])
            LOOKUPS.append(u[4])
            timelapse.append(time.time() - start_time)
            continue
        elif u[2] == last_seed[0] and u[4] == len(S) - 1:
            u[1] = u[3]
        else:
            u[1] = diffuse.diffuse_mc(S + [u[0]]) - diffuse.diffuse_mc(S)
            u[2] = cur_best[0]
            u[3] = diffuse.diffuse_mc(S + [u[0], u[2]]) - diffuse.diffuse_mc(S + [u[2]])

        u[4] = len(S)
        cur_best[0] = update_cur_best(Q, cur_best, u, u[1])
        heapify(Q)

        SPREAD.append(u[1])
        LOOKUPS.append(u[4])
        timelapse.append(time.time() - start_time)

    return (S, SPREAD, timelapse, LOOKUPS)
