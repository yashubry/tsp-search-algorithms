# okay now this is for part3! 
# where ill use the local search argorithms for the TSP 

import os
import numpy as np #we need this for the adjacency matrix and things like mat.shape and mat[i, j]
from src.utils import return_to_start, get_cost


def random_tour(n):
    #i need a random complete tour to start from for all three algorithms
    # i shuffle cities 1 through n-1 randomly and wrap with 0 at start and end
    inner_cities = list(range(1, n))       # cities 1 through n-1
    np.random.shuffle(inner_cities)        # shuffle them randomly
    return [0] + inner_cities + [0]        # wrap with 0 at start and end

#okay ! now to start hill climbing! wookoo
def hill_climbing(mat, num_restarts):
    # i'm doing randomly restarting hill climbing... the idea is: start from a random tour, keep swapping pairs of cities
    #if a swap makes the tour cheaper i keep it, otherwise i throw it away
    # i do this num_restarts times and return the best tour i found overall

    num_cities = mat.shape[0]
    top_route = None
    top_cost = float("inf")
    history = []  # i'm tracking this for the iterations plot

    for attempt in range(num_restarts):
        # start from a new random tour each restart
        curr_route = random_tour(num_cities)
        curr_cost = get_cost(curr_route, mat)

        # keep swapping until no improvement is found
        got_better = True
        while got_better:
            got_better = False
            # try every pair of cities in the route (excluding the start/end city 0)
            for i in range(1, num_cities):
                for j in range(i + 1, num_cities):
                    # swap cities at positions i and j
                    candidate = curr_route[:]
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    candidate_cost = get_cost(candidate, mat)

                    # only keep the swap if it actually improves the cost
                    if candidate_cost < curr_cost:
                        curr_route = candidate
                        curr_cost = candidate_cost
                        got_better = True  # found an improvement so keep going

        # track the best cost found at each restart for the plot
        history.append(curr_cost)

        # update overall best if this restart found something better
        if curr_cost < top_cost:
            top_cost = curr_cost
            top_route = curr_route

    return top_route, history #yassssssss#slaayyyyyyy#wowowowoghgogodqpquhfinf13

#ok now im gonna go sim annealing i guess

def simulated_annealing(mat, alpha, initial_temp, max_iterations):
    # simulated annealing is like hill climbing but smarter
    # i sometimes accept WORSE solutions early on to escape local minima
    # the trick is a temperature variable that starts high and cools down over time
    # when temp is high i accept bad solutions more freely
    # when temp is low i'm basically just doing hill climbing
    # this lets me explore more of the search space early on

    num_cities = mat.shape[0]
    curr_route = random_tour(num_cities)
    curr_cost = get_cost(curr_route, mat)

    top_route = curr_route[:]
    top_cost = curr_cost

    heat = initial_temp  # yashu's name for temperature lol
    history = []  # tracking this for the iterations plot

    for step in range(max_iterations):
        # pick two random cities to swap (not city 0 since that's fixed)
        idx1, idx2 = np.random.choice(range(1, num_cities), size=2, replace=False)
        candidate = curr_route[:]
        candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
        candidate_cost = get_cost(candidate, mat)

        # always accept if the new route is better
        # sometimes accept if its worse, based on temperature
        delta = candidate_cost - curr_cost
        if delta < 0 or np.random.random() < np.exp(-delta / heat):
            curr_route = candidate
            curr_cost = candidate_cost

        # update best if i found something better
        if curr_cost < top_cost:
            top_cost = curr_cost
            top_route = curr_route[:]

        # cool down the temperature by alpha each step
        heat *= alpha

        # track best cost found so far at each step
        history.append(top_cost)

    return top_route, history

def genetic_algorithm(mat, mutation_chance, population_size, num_generations):
    #ok so the genetic algorithm is honestly the coolest one lol
    #i start with a bunch of random tours and keep combining the best ones
    #to create new tours that hopefully inherit good traits from their parents
    #its basically survival of the fittest but for TSP haha

    num_cities = mat.shape[0]

    # step 1: create initial population of random tours
    pool = [random_tour(num_cities) for _ in range(population_size)]
    gen_history = []  # gotta track this for the plot!!

    def score(route):
        # helper to get the cost of a tour, i use this a lot lol
        return get_cost(route, mat)

    def pmx_crossover(mom, dad):
        # partially mapped crossover (PMX) -- this is the hard part ngl
        # i take a random slice from mom and fill the rest from dad
        # this makes sure every city appears exactly once in the kid
        # took me a while to get this right lol
        inner_size = num_cities - 1  # excluding the start/end city 0
        mom_inner = mom[1:-1]  # strip the 0s at start and end
        dad_inner = dad[1:-1]

        # pick two random crossover points
        cut1, cut2 = sorted(np.random.choice(range(inner_size), size=2, replace=False))

        # start kid with the slice from mom
        kid = [None] * inner_size
        kid[cut1:cut2] = mom_inner[cut1:cut2]

        # fill remaining spots with cities from dad in order
        # skipping cities already in the kid so i dont get duplicates
        dad_leftovers = [c for c in dad_inner if c not in kid]
        fill_idx = 0
        for spot in range(inner_size):
            if kid[spot] is None:
                kid[spot] = dad_leftovers[fill_idx]
                fill_idx += 1

        return [0] + kid + [0]

    def maybe_mutate(route):
        # randomly swap two cities with mutation_chance probability
        # this keeps the pool diverse so i dont get stuck lol
        if np.random.random() < mutation_chance:
            idx1, idx2 = np.random.choice(range(1, num_cities), size=2, replace=False)
            route[idx1], route[idx2] = route[idx2], route[idx1]
        return route

    for gen in range(num_generations):
        # sort pool by cost -- best tours first obvs
        pool = sorted(pool, key=score)

        # track best cost this generation for the plot
        gen_history.append(score(pool[0]))

        # create kids by crossing over pairs of parents
        # i only use the top half of the pool as parents (elitism)
        # the bottom half gets yeeted lol
        kids = []
        while len(kids) < population_size:
            # pick two random parents from the top half
            mom = pool[np.random.randint(0, population_size // 2)]
            dad = pool[np.random.randint(0, population_size // 2)]
            kid = pmx_crossover(mom, dad)
            kid = maybe_mutate(kid)
            kids.append(kid)

        # combine parents and kids, keep the best population_size tours
        # this is the elitism part -- survival of the fittest!!
        pool = sorted(pool + kids, key=score)[:population_size]

    # return the best tour found and the cost history
    winner = pool[0]
    return winner, gen_history