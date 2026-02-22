from __future__ import annotations

import glob
import os
import csv
import re
import time
import statistics

from src.utils import load_matrix, get_cost
from src.greedy import nearest_neighbor, nearest_neighbor_2opt, rrnn


def time_one_call(fn, *args, cpu_repeat_if_zero=True, **kwargs):
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


def parse_n(filename):
    # Example: 7_random_adj_mat_4.txt
    m = re.match(r"(\d+)_random_adj_mat_\d+\.txt", os.path.basename(filename))
    return int(m.group(1))


def main():
    files = sorted(glob.glob("data/matrices/*_random_adj_mat_*.txt"))

    if not files:
        print("No matrix files found.")
        return

    os.makedirs("experiments", exist_ok=True)

    with open("experiments/greedy_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "file", "algorithm", "runtime_ns", "cpu_ns", "cost"])

        for file in files:
            print(f"Running on {file}")
            n = parse_n(file)
            mat = load_matrix(file)

            # NN
            r, rt, cpu = time_one_call(nearest_neighbor, mat, start=0)
            writer.writerow([n, os.path.basename(file), "nn", rt, cpu, route_cost(r, mat)])

            # NN + 2opt
            r, rt, cpu = time_one_call(nearest_neighbor_2opt, mat, start=0)
            writer.writerow([n, os.path.basename(file), "nn2opt", rt, cpu, route_cost(r, mat)])

            # RRNN
            r, rt, cpu = time_one_call(rrnn, mat, k=3, num_repeats=50, start=0, seed=0)
            writer.writerow([n, os.path.basename(file), "rrnn", rt, cpu, route_cost(r, mat)])

    print("Done. Results saved to experiments/greedy_results.csv")


if __name__ == "__main__":
    main()