from __future__ import annotations

import argparse
import time

from src.utils import load_matrix, route_cost
from src.greedy import nearest_neighbor, nearest_neighbor_2opt, rrnn


def time_one_call(fn, *args, cpu_repeat_if_zero: bool = True, **kwargs):
    t0 = time.time_ns()
    c0 = time.process_time_ns()
    route = fn(*args, **kwargs)
    c1 = time.process_time_ns()
    t1 = time.time_ns()

    runtime_ns = t1 - t0
    cpu_ns = c1 - c0

    if cpu_repeat_if_zero and cpu_ns == 0:
        reps = 50
        c0 = time.process_time_ns()
        for _ in range(reps):
            _ = fn(*args, **kwargs)
        c1 = time.process_time_ns()
        cpu_ns = (c1 - c0) // reps

    return route, runtime_ns, cpu_ns


def main():
    parser = argparse.ArgumentParser(description="Run one TSP algorithm on one adjacency matrix.")
    parser.add_argument("--file", required=True, help="Path to adjacency matrix .txt file")
    parser.add_argument("--algo", required=True, choices=["nn", "nn2opt", "rrnn"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mat = load_matrix(args.file)

    if args.algo == "nn":
        route, runtime_ns, cpu_ns = time_one_call(nearest_neighbor, mat, start=args.start)
    elif args.algo == "nn2opt":
        route, runtime_ns, cpu_ns = time_one_call(nearest_neighbor_2opt, mat, start=args.start)
    else:
        route, runtime_ns, cpu_ns = time_one_call(
            rrnn, mat, k=args.k, num_repeats=args.repeats, start=args.start, seed=args.seed
        )

    cost = route_cost(route, mat)

    print(f"algo: {args.algo}")
    print(f"n: {mat.shape[0]}")
    print(f"route: {route}")
    print(f"cost: {cost}")
    print(f"runtime_ns: {runtime_ns}")
    print(f"cpu_ns: {cpu_ns}")


if __name__ == "__main__":
    main()