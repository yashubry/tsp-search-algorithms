import argparse
import time

from src.utils import load_matrix, get_cost
from src.greedy import nearest_neighbor, nearest_neighbor_2opt, rrnn
from src.astar import astar
from src.local import hill_climbing, simulated_annealing, genetic_algorithm

# ok so this file is basically my command line runner for all 7 algorithms
# i need this for the screen recording and also so the TAs can run my code
# usage: python run_cli.py --algo nn --file path/to/matrix.txt

def run_and_clock(fn, *args, fix_zero_cpu=True, **kwargs):
    # runs any function and times it
    # returns whatever the function returns plus wall time and cpu time
    # i wrote this helper bc i was tired of copy pasting timing code everywhere lol

    wall_start = time.time_ns()
    cpu_start = time.process_time_ns()
    output = fn(*args, **kwargs)
    cpu_end = time.process_time_ns()
    wall_end = time.time_ns()

    wall_took = wall_end - wall_start
    cpu_took = cpu_end - cpu_start

    # my laptop is so fast that cpu time sometimes registers as 0
    # so i run it 50 times and average if that happens lol
    if fix_zero_cpu and cpu_took == 0:
        num_reps = 50
        cpu_start = time.process_time_ns()
        for _ in range(num_reps):
            fn(*args, **kwargs)
        cpu_end = time.process_time_ns()
        cpu_took = (cpu_end - cpu_start) // num_reps

    return output, wall_took, cpu_took


def main():
    # set up the argument parser so i can run this from the command line
    parser = argparse.ArgumentParser(description="run one TSP algorithm on one matrix file")
    parser.add_argument("--file", required=True, help="path to the adjacency matrix .txt file")
    parser.add_argument("--algo", required=True, choices=[
        "nn", "nn2opt", "rrnn", "astar",
        "hill_climbing", "simulated_annealing", "genetic_algorithm"
    ])
    # optional hyperparams for each algorithm
    parser.add_argument("--start", type=int, default=0)          # starting city for greedy
    parser.add_argument("--k", type=int, default=3)               # k for rrnn
    parser.add_argument("--repeats", type=int, default=50)        # num_repeats for rrnn
    parser.add_argument("--seed", type=int, default=0)            # random seed for rrnn
    parser.add_argument("--restarts", type=int, default=25)       # num_restarts for hill climbing
    parser.add_argument("--alpha", type=float, default=0.99)      # cooling rate for SA
    parser.add_argument("--temp", type=float, default=100.0)      # initial temp for SA
    parser.add_argument("--iterations", type=int, default=1000)   # max iterations for SA
    parser.add_argument("--mutation", type=float, default=0.1)    # mutation chance for GA
    parser.add_argument("--popsize", type=int, default=50)        # population size for GA
    parser.add_argument("--generations", type=int, default=100)   # num generations for GA
    parsed = parser.parse_args()

    # load the matrix from the file
    adj = load_matrix(parsed.file)
    num_cities = adj.shape[0]

    # ok now run whichever algorithm was requested
    # each one returns slightly different things so i have to unpack carefully

    num_visited = None  # only used for astar

    if parsed.algo == "nn":
        best_path, wall_took, cpu_took = run_and_clock(
            nearest_neighbor, adj, start=parsed.start)

    elif parsed.algo == "nn2opt":
        best_path, wall_took, cpu_took = run_and_clock(
            nearest_neighbor_2opt, adj, start=parsed.start)

    elif parsed.algo == "rrnn":
        best_path, wall_took, cpu_took = run_and_clock(
            rrnn, adj, k=parsed.k, num_repeats=parsed.repeats,
            start=parsed.start, seed=parsed.seed)

    elif parsed.algo == "astar":
        # astar returns (route, nodes_expanded) so i gotta unpack the tuple
        (best_path, num_visited), wall_took, cpu_took = run_and_clock(astar, adj)

    elif parsed.algo == "hill_climbing":
        # hill climbing returns (route, cost_history) -- i only need the route
        (best_path, _), wall_took, cpu_took = run_and_clock(
            hill_climbing, adj, num_restarts=parsed.restarts)

    elif parsed.algo == "simulated_annealing":
        # same deal -- returns (route, history)
        (best_path, _), wall_took, cpu_took = run_and_clock(
            simulated_annealing, adj,
            alpha=parsed.alpha, initial_temp=parsed.temp,
            max_iterations=parsed.iterations)

    elif parsed.algo == "genetic_algorithm":
        # same deal -- returns (route, gen_history)
        (best_path, _), wall_took, cpu_took = run_and_clock(
            genetic_algorithm, adj,
            mutation_chance=parsed.mutation,
            population_size=parsed.popsize,
            num_generations=parsed.generations)

    # compute the tour cost
    tour_cost = get_cost(best_path, adj)

    # print everything out nicely
    print(f"algo:       {parsed.algo}")
    print(f"n:          {num_cities}")
    print(f"route:      {best_path}")
    print(f"cost:       {tour_cost}")
    print(f"runtime_ns: {wall_took}")
    print(f"cpu_ns:     {cpu_took}")
    if parsed.algo == "astar":
        print(f"nodes_expanded: {num_visited}")


if __name__ == "__main__":
    main()