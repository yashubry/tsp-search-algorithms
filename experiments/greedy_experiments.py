import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing my greedy-style algorithms
from src.greedy import nearest_neighbor, nearest_neighbor_2opt, rrnn
from src.utils import load_matrix, get_cost

# this is the folder where all the adjacency matrix .txt files are stored
MATRICES_DIR = "data/matrices/Project_1_-_matrices"

# these are the sizes we were given in the dataset
SIZES = [5, 10, 15, 20, 25, 30]


def load_matrices_by_size(data_dir):
    # group matrices by their size (number of cities)
    # key = size, value = list of matrices of that size
    matrices = {s: [] for s in SIZES}

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(data_dir, fname)
        mat = load_matrix(fpath)
        n = mat.shape[0]  # n x n matrix so shape[0] gives number of cities
        if n in matrices:
            matrices[n].append(mat)

    return matrices


def time_algorithm(fn, *args, **kwargs):
    # records both wall time and CPU time for any algorithm
    # *args and **kwargs let this work for any function without rewriting it

    t0 = time.time_ns()
    c0 = time.process_time_ns()
    result = fn(*args, **kwargs)
    c1 = time.process_time_ns()
    t1 = time.time_ns()

    runtime_ns = t1 - t0
    cpu_ns = c1 - c0

    # if cpu time is 0, the run was too fast for the clock to catch
    # so we re-run 50 times and average -- same trick the spec mentions
    if cpu_ns == 0:
        reps = 50
        c0 = time.process_time_ns()
        for _ in range(reps):
            fn(*args, **kwargs)
        c1 = time.process_time_ns()
        cpu_ns = (c1 - c0) // reps

    return result, runtime_ns, cpu_ns


def run_rrnn_tuning(matrices):
    # spec says to find optimal hyperparameters for RRNN before comparing algorithms
    # we vary one parameter at a time and keep the other fixed
    # that way we can actually tell which parameter is causing changes in cost

    # --- vary k, hold num_repeats fixed at 50 ---
    # k=1 is basically regular NN, higher k = more randomness
    k_values = [1, 2, 3, 5, 7]
    fixed_repeats = 50

    k_results = []
    print("tuning k...")

    for k in k_values:
        for size in SIZES:
            costs = []
            for mat in matrices[size]:
                route, _, _ = time_algorithm(rrnn, mat, k=k, num_repeats=fixed_repeats)
                costs.append(get_cost(route, mat))
            # median is more robust than mean if one matrix happens to be weird
            k_results.append({"k": k, "size": size, "median_cost": np.median(costs)})

    df_k = pd.DataFrame(k_results)
    df_k.to_csv("experiments/rrnn_k_tuning.csv", index=False)
    print("saved rrnn_k_tuning.csv")

    # --- vary num_repeats, hold k fixed at 3 ---
    # more repeats = better chance of finding a good tour but slower
    repeat_values = [10, 25, 50, 100, 200]
    fixed_k = 3

    repeat_results = []
    print("tuning num_repeats...")

    for repeats in repeat_values:
        for size in SIZES:
            costs = []
            for mat in matrices[size]:
                route, _, _ = time_algorithm(rrnn, mat, k=fixed_k, num_repeats=repeats)
                costs.append(get_cost(route, mat))
            repeat_results.append({"num_repeats": repeats, "size": size, "median_cost": np.median(costs)})

    df_repeats = pd.DataFrame(repeat_results)
    df_repeats.to_csv("experiments/rrnn_repeats_tuning.csv", index=False)
    print("saved rrnn_repeats_tuning.csv")

    return df_k, df_repeats


def run_comparison(matrices, best_k=3, best_repeats=50):
    # run all 3 algorithms on every matrix
    # spec asks for median real runtime, median CPU time, and median cost
    # we store every trial so we can compute medians after

    results = []

    algorithms = {
        "nn":     lambda mat: time_algorithm(nearest_neighbor, mat),
        "nn2opt": lambda mat: time_algorithm(nearest_neighbor_2opt, mat),
        "rrnn":   lambda mat: time_algorithm(rrnn, mat, k=best_k, num_repeats=best_repeats),
    }

    for algo_name, algo_fn in algorithms.items():
        print(f"running {algo_name}...")
        for size in SIZES:
            for i, mat in enumerate(matrices[size]):
                route, runtime_ns, cpu_ns = algo_fn(mat)
                cost = get_cost(route, mat)
                results.append({
                    "algo": algo_name,
                    "size": size,
                    "trial": i,
                    "runtime_ns": runtime_ns,
                    "cpu_ns": cpu_ns,
                    "cost": cost,
                })

    df = pd.DataFrame(results)
    df.to_csv("experiments/greedy_comparison.csv", index=False)
    print("saved greedy_comparison.csv")
    return df


def run_consistency_experiment(matrices, best_k=3, best_repeats=50):
    # BONUS EXPERIMENT: how consistent is each algorithm across different random matrices?
    #
    # this is genuinely interesting because two algorithms can have the same MEDIAN cost
    # but one might be all over the place while the other is reliable
    # we measure this using the coefficient of variation (std / mean)
    # a higher value means the algorithm is more unpredictable
    #
    # this adds real depth to the report -- it's one thing to say "RRNN finds better tours"
    # but another to say "and it's also more consistent than plain NN"

    results = []

    algorithms = {
        "nn":     lambda mat: get_cost(nearest_neighbor(mat), mat),
        "nn2opt": lambda mat: get_cost(nearest_neighbor_2opt(mat), mat),
        "rrnn":   lambda mat: get_cost(rrnn(mat, k=best_k, num_repeats=best_repeats), mat),
    }

    for algo_name, algo_fn in algorithms.items():
        for size in SIZES:
            costs = [algo_fn(mat) for mat in matrices[size]]
            results.append({
                "algo": algo_name,
                "size": size,
                "mean_cost": np.mean(costs),
                "std_cost": np.std(costs),
                # coefficient of variation = how much variability relative to the mean
                # multiply by 100 to express as a percentage
                "cv": (np.std(costs) / np.mean(costs)) * 100,
            })

    df = pd.DataFrame(results)
    df.to_csv("experiments/greedy_consistency.csv", index=False)
    print("saved greedy_consistency.csv")
    return df


def plot_rrnn_tuning(df_k, df_repeats):
    os.makedirs("experiments/plots", exist_ok=True)

    # plot 1: k on x axis, median cost on y axis (spec requirement)
    plt.figure()
    for size in SIZES:
        subset = df_k[df_k["size"] == size]
        plt.plot(subset["k"], subset["median_cost"], marker="o", label=f"n={size}")
    plt.xlabel("k (number of random choices)")
    plt.ylabel("median cost")
    plt.title("RRNN: Effect of k on Solution Cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiments/plots/rrnn_k_tuning.png")
    plt.close()
    print("saved rrnn_k_tuning.png")

    # plot 2: num_repeats on x axis, median cost on y axis (spec requirement)
    plt.figure()
    for size in SIZES:
        subset = df_repeats[df_repeats["size"] == size]
        plt.plot(subset["num_repeats"], subset["median_cost"], marker="o", label=f"n={size}")
    plt.xlabel("num_repeats")
    plt.ylabel("median cost")
    plt.title("RRNN: Effect of num_repeats on Solution Cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiments/plots/rrnn_repeats_tuning.png")
    plt.close()
    print("saved rrnn_repeats_tuning.png")


def plot_comparison(df):
    os.makedirs("experiments/plots", exist_ok=True)

    # collapse 10 trials per size down to one median per algorithm per size
    # spec specifically asks for median, not mean
    summary = (
        df.groupby(["algo", "size"])
          .median(numeric_only=True)
          .reset_index()
    )

    # consistent colors across all 3 plots so the report looks cohesive
    colors = {"nn": "blue", "nn2opt": "orange", "rrnn": "green"}
    # x axis ticks match exactly what the spec asks for: 5, 10, 15, 20, 25, 30
    xticks = SIZES

    # plot 3: total runtime (spec requirement)
    plt.figure()
    for algo in ["nn", "nn2opt", "rrnn"]:
        subset = summary[summary["algo"] == algo]
        plt.plot(subset["size"], subset["runtime_ns"] / 1e6, marker="o", label=algo, color=colors[algo])
    plt.xlabel("number of cities (n)")
    plt.ylabel("median runtime (ms)")
    plt.title("Total Runtime vs Problem Size")
    plt.xticks(xticks)
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiments/plots/greedy_runtime.png")
    plt.close()
    print("saved greedy_runtime.png")

    # plot 4: CPU time (spec requirement)
    plt.figure()
    for algo in ["nn", "nn2opt", "rrnn"]:
        subset = summary[summary["algo"] == algo]
        plt.plot(subset["size"], subset["cpu_ns"] / 1e6, marker="o", label=algo, color=colors[algo])
    plt.xlabel("number of cities (n)")
    plt.ylabel("median CPU time (ms)")
    plt.title("CPU Time vs Problem Size")
    plt.xticks(xticks)
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiments/plots/greedy_cpu.png")
    plt.close()
    print("saved greedy_cpu.png")

    # plot 5: solution cost (spec requirement)
    plt.figure()
    for algo in ["nn", "nn2opt", "rrnn"]:
        subset = summary[summary["algo"] == algo]
        plt.plot(subset["size"], subset["cost"], marker="o", label=algo, color=colors[algo])
    plt.xlabel("number of cities (n)")
    plt.ylabel("median cost")
    plt.title("Solution Cost vs Problem Size")
    plt.xticks(xticks)
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiments/plots/greedy_cost.png")
    plt.close()
    print("saved greedy_cost.png")


def plot_consistency(df):
    os.makedirs("experiments/plots", exist_ok=True)

    # BONUS PLOT: coefficient of variation per algorithm per size
    # this shows how predictable each algorithm is, not just how good it is on average
    # a lower CV means the algorithm gives similar results regardless of which matrix you give it
    colors = {"nn": "blue", "nn2opt": "orange", "rrnn": "green"}

    plt.figure()
    for algo in ["nn", "nn2opt", "rrnn"]:
        subset = df[df["algo"] == algo]
        plt.plot(subset["size"], subset["cv"], marker="o", label=algo, color=colors[algo])
    plt.xlabel("number of cities (n)")
    plt.ylabel("coefficient of variation (%)")
    plt.title("Solution Consistency Across Matrices (Lower = More Consistent)")
    plt.xticks(SIZES)
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiments/plots/greedy_consistency.png")
    plt.close()
    print("saved greedy_consistency.png")


if __name__ == "__main__":
    print("loading matrices...")
    matrices = load_matrices_by_size(MATRICES_DIR)

    for size in SIZES:
        print(f"  n={size}: {len(matrices[size])} matrices loaded")

    # step 1: tune RRNN hyperparameters (spec requirement)
    df_k, df_repeats = run_rrnn_tuning(matrices)

    # step 2: run full comparison (spec requirement)
    # update best_k and best_repeats after looking at the tuning plots!
    df_comparison = run_comparison(matrices, best_k=3, best_repeats=50)

    # bonus: consistency experiment
    df_consistency = run_consistency_experiment(matrices, best_k=3, best_repeats=50)

    # step 3: generate all required plots + bonus
    plot_rrnn_tuning(df_k, df_repeats)
    plot_comparison(df_comparison)
    plot_consistency(df_consistency)

    print("all done! check experiments/plots/ for your figures")