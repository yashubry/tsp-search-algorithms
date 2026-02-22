import heapq  # min-heap / priority queue -- A* needs to always expand the cheapest node first
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree  # scipy's built-in MST so we dont have to implement prim's ourselves
from scipy.sparse import csr_matrix  # minimum_spanning_tree() needs a sparse matrix format, so we convert to this first


def mst_heuristic(mat, current_city, unvisited):
    # this is h(n) -- the heuristic function for A*
    # it gives a lower bound on the cost to finish the tour from current_city,
    # visiting everything in unvisited, and returning to city 0
    #
    # why is it admissible (never overestimates)?
    # any valid tour completion has to do at least these 3 things:
    #   (1) connect current_city into the unvisited cities somehow
    #   (2) link all the unvisited cities together
    #   (3) eventually return to city 0
    # MST is the cheapest possible way to connect a set of cities,
    # so MST + cheapest entry edge + cheapest return edge is always <= real remaining cost

    unvisited_list = list(unvisited)  # convert set to list so we can index into it

    # base case: no unvisited cities left, just need to go home
    if len(unvisited_list) == 0:
        return mat[current_city, 0]  # just the direct edge back to start

    # --- part 1: MST over just the unvisited cities ---
    # np.ix_ lets us slice a 2D array by two index lists at once
    # so sub is the submatrix of distances between only the unvisited cities
    sub = mat[np.ix_(unvisited_list, unvisited_list)]

    # convert to sparse format bc that's what scipy's MST function wants
    sparse_sub = csr_matrix(sub)

    # compute MST -- cheapest set of edges that connects all unvisited cities
    mst = minimum_spanning_tree(sparse_sub)

    # convert back to dense array and sum up all the edge weights
    mst_cost = mst.toarray().sum()

    # --- part 2: cheapest edge FROM current_city INTO the unvisited set ---
    # we have to enter the unvisited cities somehow, this is the cheapest possible way to do it
    cheapest_entry = min(mat[current_city, j] for j in unvisited_list)

    # --- part 3: cheapest edge FROM any unvisited city BACK to city 0 ---
    # the tour has to close eventually, this is the cheapest that return trip could possibly be
    cheapest_return = min(mat[j, 0] for j in unvisited_list)

    # add all 3 parts together to get the full heuristic value
    return mst_cost + cheapest_entry + cheapest_return


def astar(mat):
    # A* search for TSP, always starting and ending at city 0
    #
    # the key idea: in normal A* (like pathfinding on a map), a state is just a location.
    # for TSP, a state is the ENTIRE PARTIAL TOUR so far -- e.g. (0, 3, 1) means
    # we visited city 0, then 3, then 1, in that order.
    # we need this because which city to visit next depends on what we've already visited
    #
    # f(n) = g(n) + h(n)
    # g(n) = actual cost of the partial tour so far (real edge weights added up)
    # h(n) = MST heuristic -- lower bound on the cost to finish
    #
    # returns the complete route as a list like [0, 2, 3, 1, 0]
    # and nodes_expanded which we need for the part 2 experiment plot

    n = mat.shape[0]            # number of cities
    all_cities = set(range(n))  # {0, 1, 2, ..., n-1}

    # initial state: we've only visited city 0 so far
    start_state = (0,)

    # everything except city 0 is unvisited at the start
    unvisited_start = all_cities - {0}

    # compute h for the very first state
    h0 = mst_heuristic(mat, 0, unvisited_start)

    # priority queue holds tuples of (f_value, g_value, state_tuple)
    # heapq always pops the smallest f first, which is exactly what A* needs
    # we keep g in there too so we can use it when we pop a node
    frontier = [(h0, 0.0, start_state)]

    # best_g tracks the cheapest g we've found for each state so far
    # this is how we avoid re-expanding states we already have a better path to
    # kind of like a visited set, but for partial tours that can be reached multiple ways
    best_g = {start_state: 0.0}

    nodes_expanded = 0  # this is what we plot in part 2

    while frontier:
        # pop whatever has the lowest f = g + h
        f, g, state = heapq.heappop(frontier)

        nodes_expanded += 1  # just expanded this node

        # pruning: if we already found a cheaper way to this exact state, skip it
        # this happens bc we dont update heap entries, we just push new ones
        if g > best_g.get(state, float("inf")):
            continue

        current_city = state[-1]             # last city we visited
        unvisited = all_cities - set(state)  # cities not in our tour yet

        # goal check: if nothing is unvisited, we've been everywhere
        # just need to close the loop back to city 0
        if len(unvisited) == 0:
            route = list(state) + [0]  # tack on the return to start
            return route, nodes_expanded

        # expansion: try adding each unvisited city as the next stop
        for next_city in unvisited:
            # new g = old g + cost of edge to next_city
            new_g = g + mat[current_city, next_city]
            new_state = state + (next_city,)  # extend the partial tour

            # only bother if this is the best path to new_state we've seen
            if new_g < best_g.get(new_state, float("inf")):
                best_g[new_state] = new_g

                new_unvisited = unvisited - {next_city}
                h = mst_heuristic(mat, next_city, new_unvisited)

                new_f = new_g + h
                heapq.heappush(frontier, (new_f, new_g, new_state))

    # shouldnt ever get here on a fully connected graph, but just in case
    return None, nodes_expanded