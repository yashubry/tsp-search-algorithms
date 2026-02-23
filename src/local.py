# okay now this is for part3! 
# where ill use the local search argorithms for the TSP 

import os
import numpy as np #we need this for the adjacency matrix and things like mat.shape and mat[i, j]
from src.utils import return_to_start, get_cost


def random_tour(n):
    #i need a random complete tour to start from for all three algorithms
    # i shuffle cities 1 through n-1 randomly and wrap with 0 at start and end
    cities = list(range(1, n))       # cities 1 through n-1
    np.random.shuffle(cities)        # shuffle them randomly
    return [0] + cities + [0]        # wrap with 0 at start and end

#okay ! now to start hill climbing! wookoo
def hill_climbing(mat, num_restarts):
    # i'm doing randomly restarting hill climbing... the idea is: start from a random tour, keep swapping pairs of cities
    #if a swap makes the tour cheaper i keep it, otherwise i throw it away
    # i do this num_restarts times and return the best tour i found overall

    #uhh how do i do the restarts... um 

    n = mat.shape[0]
    best_route = None
    best_cost = float("inf")
    cost_per_restart = []  # i'm tracking this for the iterations plot

    for restart in range(num_restarts):
        # start from a new random tour each restart
        route = random_tour(n)
        current_cost = get_cost(route, mat)

        # keep swapping until no improvement is found
        improved = True
        while improved:
            improved = False
            # try every pair of cities in the route (excluding the start/end city 0)
            for i in range(1, n):
                for j in range(i + 1, n):
                    # swap cities at positions i and j
                    new_route = route[:]
                    new_route[i], new_route[j] = new_route[j], new_route[i]
                    new_cost = get_cost(new_route, mat)

                    # only keep the swap if it actually improves the cost
                    if new_cost < current_cost:
                        route = new_route
                        current_cost = new_cost
                        improved = True  # found an improvement so keep going

        #track the best cost found at each restart for the plot
        cost_per_restart.append(current_cost)

        #update overall best if this restart found something better
        if current_cost < best_cost:
            best_cost = current_cost
            best_route = route

    return best_route, cost_per_restart #yassssssss#slaayyyyyyy#wowowowoghgogodqpquhfinf13

#ok now im gonna go sim annealing i guess

def simulated_annealing(mat, alpha, initial_temp, max_iterations):
    # simulated annealing is like hill climbing but smarter
    # i sometimes accept WORSE solutions early on to escape local minima
    # the trick is a temperature variable that starts high and cools down over time
    # when temp is high i accept bad solutions more freely
    # when temp is low i'm basically just doing hill climbing
    # this lets me explore more of the search space early on

    n = mat.shape[0]
    route = random_tour(n)
    current_cost = get_cost(route, mat)

    best_route = route[:]
    best_cost = current_cost

    temp = initial_temp
    cost_per_iteration = []  # tracking this for the iterations plot

    for iteration in range(max_iterations):
        # pick two random cities to swap (not city 0 since that's fixed)
        i, j = np.random.choice(range(1, n), size=2, replace=False)
        new_route = route[:]
        new_route[i], new_route[j] = new_route[j], new_route[i]
        new_cost = get_cost(new_route, mat)

        # always accept if the new route is better
        # sometimes accept if its worse, based on temperature
        diff = new_cost - current_cost
        if diff < 0 or np.random.random() < np.exp(-diff / temp):
            route = new_route
            current_cost = new_cost

        # update best if i found something better
        if current_cost < best_cost:
            best_cost = current_cost
            best_route = route[:]

        # cool down the temperature by alpha each iteration
        temp *= alpha

        # track best cost found so far at each iteration
        cost_per_iteration.append(best_cost)

    return best_route, cost_per_iteration

def genetic_algorithm(mat, mutation_chance, population_size, num_generations):
    #ok so the genetic algorithm is honestly the coolest one lol
    #i start with a bunch of random tours and keep combining the best ones
    #to create new tours that hopefully inherit good traits from their parents
    #its basically survival of the fittest but for TSP haha

    n = mat.shape[0]

    # step 1: create initial population of random tours
    # just a bunch of random tours to start with
    population = [random_tour(n) for _ in range(population_size)]
    best_cost_per_generation = []  # gotta track this for the plot!!

    def tour_cost(route):
        # helper to get the cost of a tour, i use this a lot lol
        return get_cost(route, mat)

    def pmx_crossover(parent1, parent2):
        # partially mapped crossover (PMX) -- this is the hard part ngl
        # i take a random slice from parent1 and fill the rest from parent2
        # this makes sure every city appears exactly once in the child
        # took me a while to get this right lol
        size = n - 1  # excluding the start/end city 0
        p1 = parent1[1:-1]  # strip the 0s at start and end
        p2 = parent2[1:-1]

        # pick two random crossover points
        cx1, cx2 = sorted(np.random.choice(range(size), size=2, replace=False))

        # start child with the slice from parent1
        child = [None] * size
        child[cx1:cx2] = p1[cx1:cx2]

        # fill remaining positions with cities from parent2 in order
        # skipping cities already in the child so i dont get duplicates
        p2_remaining = [c for c in p2 if c not in child]
        j = 0
        for i in range(size):
            if child[i] is None:
                child[i] = p2_remaining[j]
                j += 1

        return [0] + child + [0]

    def mutate(route):
        # randomly swap two cities with mutation_chance probability
        # this keeps the population diverse so i dont get stuck lol
        if np.random.random() < mutation_chance:
            i, j = np.random.choice(range(1, n), size=2, replace=False)
            route[i], route[j] = route[j], route[i]
        return route

    for generation in range(num_generations):
        # sort population by cost -- best tours first obvs
        population = sorted(population, key=tour_cost)

        # track best cost this generation for the plot
        best_cost_per_generation.append(tour_cost(population[0]))

        # create children by crossing over pairs of parents
        # i only use the top half of the population as parents (elitism)
        # the bottom half gets yeeted lol
        children = []
        while len(children) < population_size:
            # pick two random parents from the top half
            p1 = population[np.random.randint(0, population_size // 2)]
            p2 = population[np.random.randint(0, population_size // 2)]
            child = pmx_crossover(p1, p2)
            child = mutate(child)
            children.append(child)

        # combine parents and children, keep the best population_size tours
        # this is the elitism part -- survival of the fittest!!
        population = sorted(population + children, key=tour_cost)[:population_size]

    # return the best tour found and the cost history
    best_route = population[0]
    return best_route, best_cost_per_generation
