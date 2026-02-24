# src/greedy.py

#i am using numpy because i think that the adjacency matrix is stored as a 2D numpy array.
#i need this for things like mat.shape and mat[i, j]
import numpy as np

# need random for RRNN (Repeated Random Nearest Neighbor),
import random

#wrote helper functions in utils.py:
from src.utils import return_to_start, get_cost

#ook from now on im going to include the things in part 1 one by one 

def nearest_neighbor(mat, start=0): # start is the city we begin at (i think its important to set default to 0)
    # mat is the adjacency matrix (n x n)

# Idea: Start at some city (defa 0). and then repeatedly go to the closest city you haven't visited yet.
# hen all cities are visited, return to the starting city.

# it hink that tt's fast and simple, but it can get stuck making short-sighted choices,so it's not guaranteed to be optimal.

    #firsy imma gonn init all the starting vars before moving inot the loop 

    n = mat.shape[0] # n is the number of cities and since nxn we cna get it from .shape() function 
    # now i have to keep track of cities we still need to visit
    # i am gonna be using a set makes it easy to remove cities
    unvisited = set(range(n))
    # we are starting at 'start', so remove it from unvisited
    unvisited.remove(start)
    route =[start] # here, i made route which stores the order we visit cities, starts at zero obv!
    cur =start     # cur is the current city

    while unvisited:
        # choose the unvisited city with the smallest distance from curr
        nxt= min(unvisited, key=lambda j: mat[cur, j])
        route.append(nxt) #add to route to keep track!! very import its what we return 
        unvisited.remove(nxt)
        cur =nxt #this is how me update 
    route =return_to_start(route)  #finally, return to the start city to complete the cycle I FORGOT THIS AT FIRST  BUT ITS VERY IMPORTANT WOOWW

    return route #voila! gold medal yashu

##############################################################################################################################

# 2-opt(local improvement)

#after readin the wiki 2-opt page, i gotta:
# pick two edges, remove them, and reconnect in a way that reverses a segment.
# The main practical effect is it can remove "crossing" edges in the tour.
#
# Wikipedia describes a 2optSwap(route, v1, v2) that:
# 1) keeps route[start..v1] in the same order
# 2) reverses route[v1+1..v2]
# 3) keeps route[v2+1..end] in the same order
#
# NOTE: I am using a simple implementation where I rebuild the route and
# recompute the full cost to check if it improved.
# Wikipedia also mentions a faster O(1) delta trick (using just 4 edges),
# but I am keeping it simple first.

def two_opt_swap(route_no_end, i, k):
    # route_no_end is a tour WITHOUT the repeated start at the end
    # example: [0, 2, 5, 3] (not [0, 2, 5, 3, 0])

    # This matches the swap steps from the Wikipedia pseudocode:
    # new_route = route[0..i] + reverse(route[i+1..k]) + route[k+1..end]
    new_route = route_no_end[:i+1]
    new_route += route_no_end[i+1:k+1][::-1]
    new_route += route_no_end[k+1:]
    return new_route


def two_opt(mat, route):
    # route is expected to be a cycle like [0, ..., 0]
    # I will work with a version without the last repeated 0 while swapping
    route_no_end = route[:-1]

    best_route = route_no_end[:]
    best_cost = get_cost(return_to_start(best_route), mat)

    improved = True
    while improved:
        improved = False

        # Like Wikipedia says, we keep trying swaps until no improvement is made.
        # Also, to keep the start city fixed (the "depot" idea),
        # we do not allow i = 0, because that can mess up the start.
        #
        # If n is small, this O(n^2) search is fine.
        n = len(best_route)

        for i in range(1, n - 1):
            for k in range(i + 1, n):
                candidate = two_opt_swap(best_route, i, k)
                candidate_cost = get_cost(return_to_start(candidate), mat)

                if candidate_cost < best_cost:
                    best_route = candidate
                    best_cost = candidate_cost

                    # This is basically the "goto start_again" idea from Wikipedia:
                    # once we find a better route, accept it and restart the search.
                    improved = True
                    break
            if improved:
                break

    return return_to_start(best_route)


#NN + 2-opt (run NN first, then improve it with 2-opt)
def nearest_neighbor_2opt(mat, start=0):
    route = nearest_neighbor(mat, start=start)
    route = two_opt(mat, route)
    return route

##############################################################################################################################

# Repeated Random Nearest Neighbor (RRNN)
# Idea:similar to NN, but instead of always picking the 1 closest city,
#pick randomly from the k closest cities.
# Repeat num_repeats times and keep the best route found.
#
#This uses randomness to explore different tours.

def rrnn(mat, k=3, num_repeats=50, start=0, seed=0):
    # seed makes results reproducible (same random choices each run)
    random.seed(seed)

    n = mat.shape[0]

    # keep track of the best route across all repeats
    best_route = None
    best_cost = float("inf")

    #repeat the randomized NN process a bunch of times
    for _ in range(num_repeats):
        unvisited = set(range(n))
        unvisited.remove(start)

        route = [start]
        cur = start

        while len(unvisited) > 0:
            #sort unvisited cities by distance from cur
            ordered = sorted(unvisited, key=lambda j: mat[cur, j])

            #choose randomly among the k closest
            #if there are fewer than k cities left, just use all of them
            choices = ordered[:k]

            nxt = random.choice(choices)

            route.append(nxt)
            unvisited.remove(nxt)
            cur = nxt

        #close the tour
        route = return_to_start(route)

        #optional but usually helps: improve each candidate with 2-opt
        route = two_opt(mat, route)

        #compare and keep the best
        c = get_cost(route, mat)
        if c < best_cost:
            best_cost = c
            best_route = route

    return best_route