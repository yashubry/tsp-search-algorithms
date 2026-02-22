#yo, what is astar 

# search algorithm that finds the guaranteed optimal solution. Unlike the greedy algorithms which just make quick decisions, 
# A* explores the search space systematically using a priority queue. It always expands the most promising path first based on

#ok so f(n) = g(n) + h(n)
# g(n) is the cost from the start node to the current node n, and h(n) is the heuristic estimate of the cost from n to the goal.

#ALLEGEDLY ITS IS THE ONLY SEARCH ALGO THAT IS GUARENTTED TO FOND THE MOST OPTIMAL TOUR
#the one downside is that it will be extremely slow with the larger inputs 
#it will get slower because for n siticies there are n! possible tours 

import heapq  # need this for the priority queue -- heapq always pops the smallest element first
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree  # using scipy for MST so i dont have to implement prim's myself
from scipy.sparse import csr_matrix  # scipy's MST function needs sparse matrix format, so i have to convert first


def mst_heuristic(mat, current_city, unvisited):
    #this is h(n) -- the heuristic that estimates how much it costs to finish the tour
    #it needs to be admissible, meaning it can never OVERestimate the real remaining cost
    #if it overestimates,A* might skip the optimal solution thinking its too expensive

    unvisited_list = list(unvisited)  # convert set to list so i can index into it

    # base case: no cities left to visit, just go home
    if len(unvisited_list) == 0:
        return mat[current_city, 0]

    # first attempt: i tried just using the MST cost alone as the heuristic
    # but that underestimates too much because it ignores the entry and return edges
    # sub = mat[np.ix_(unvisited_list, unvisited_list)]
    # return minimum_spanning_tree(csr_matrix(sub)).toarray().sum()

    # better approach: MST + cheapest way in + cheapest way back to start
    # this is still admissible because any tour completion must do at least these 3 things

    # part 1: build submatrix of just the unvisited cities
    # np.ix_ lets me slice a 2D array with two index lists at once
    sub = mat[np.ix_(unvisited_list, unvisited_list)]

    # convert to sparse format bc that's what scipy wants
    sparse_sub = csr_matrix(sub)

    # compute the MST -- cheapest way to connect all unvisited cities
    mst = minimum_spanning_tree(sparse_sub)

    # sum up all the MST edge weights
    mst_cost = mst.toarray().sum()

    # part 2: cheapest edge from current city into the unvisited set
    # i have to enter the unvisited cities somehow, this is the cheapest possible way
    cheapest_entry = min(mat[current_city, j] for j in unvisited_list)

    # part 3: cheapest edge from any unvisited city back to city 0
    # the tour has to end at city 0, so someone has to make that return trip
    cheapest_return = min(mat[j, 0] for j in unvisited_list)

    # add all 3 parts together
    return mst_cost + cheapest_entry + cheapest_return


def astar(mat):
    # A* for TSP -- guaranteed to find the optimal tour
    #
    # the key insight: a STATE is not just a city, its the entire partial tour so far
    # e.g. (0, 3, 1) means we visited city 0, then 3, then 1 in that order
    # we need this because which city to go to next depends on what we already visited
    #
    # f(n) = g(n) + h(n)
    # g(n) = real cost of the partial tour so far
    # h(n) = MST heuristic -- lower bound on cost to finish
    #
    # returns the complete route like [0, 2, 3, 1, 0] and nodes_expanded for the plot

    n = mat.shape[0]            # number of cities
    all_cities = set(range(n))  # {0, 1, 2, ..., n-1}

    # start state: only visited city 0 so far
    start_state = (0,)

    # everything except 0 is unvisited at the start
    unvisited_start = all_cities - {0}

    # compute initial heuristic
    h0 = mst_heuristic(mat, 0, unvisited_start)

    # priority queue holds (f, g, state) tuples
    # heapq pops the smallest f first, which is exactly what A* needs
    # i store g separately so i can use it after popping
    frontier = [(h0, 0.0, start_state)]

    # first attempt: i tried using a simple visited set like in regular BFS
    # but that doesnt work for TSP because the same city can appear in different partial tours
    # visited = set()

    # better approach: track the best g value seen for each state
    # only re-add a state to the frontier if we found a cheaper path to it
    best_g = {start_state: 0.0}

    nodes_expanded = 0  # tracking this for the part 2 experiment plot

    while frontier:
        # pop the state with the lowest f value
        f, g, state = heapq.heappop(frontier)

        nodes_expanded += 1  # expanding this node now

        # pruning: skip if we already found a cheaper path to this state
        # this happens because we add new entries to the heap instead of updating old ones
        if g > best_g.get(state, float("inf")):
            continue

        current_city = state[-1]             # last city we visited
        unvisited = all_cities - set(state)  # cities not yet in the tour

        # goal check: if nothing is left to visit, close the loop and return
        if len(unvisited) == 0:
            route = list(state) + [0]  # add the return to start
            return route, nodes_expanded

        # expansion: try adding each unvisited city as the next stop
        for next_city in unvisited:
            new_g = g + mat[current_city, next_city]  # real cost to get to next_city
            new_state = state + (next_city,)           # extend the partial tour

            # only worth exploring if this is the cheapest path to new_state we've seen
            if new_g < best_g.get(new_state, float("inf")):
                best_g[new_state] = new_g

                new_unvisited = unvisited - {next_city}
                h = mst_heuristic(mat, next_city, new_unvisited)

                new_f = new_g + h
                heapq.heappush(frontier, (new_f, new_g, new_state))

    # shouldnt ever get here on a fully connected graph
    return None, nodes_expanded