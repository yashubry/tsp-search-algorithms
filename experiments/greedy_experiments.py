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

# pink color palette :)
COLORS = {
    "nn":     "hotpink",
    "nn2opt": "deeppink",
    "rrnn":   "lightpink"
}

# consistent marker styles so plots are readable even in grayscale
MARKERS = {
    "nn":     "o",
    "nn2opt": "s",
    "rrnn":   "^"
}


def load_matrices_by_size(data_dir):
    # group matrices by their size (number of cities)
    # key = size, value = list of matrices of that size
    matrices = {s: [] for s in SIZES}

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".txt"):
            continue  # skip anything that isnt a matrix file
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
            k_results.append({"k": k, "size": size, "median_cost": np.median(costs)})

    df_k = pd.DataFrame(k_results)
    df_k.to_csv("experiments/rrnn_k_tuning.csv", index=False)
    print("saved rrnn_k_tuning.csv")

    # --- vary num_repeats, hold k fixed at 3 ---
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
    # store every single trial so we can compute medians after

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
    # bonus experiment: how consistent is each algorithm across different random matrices?
    # two algorithms can have the same median cost but one might be all over the place
    # we measure this using coefficient of variation (std / mean * 100)
    # lower CV = more predictable and reliable across different inputs

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
                "cv": (np.std(costs) / np.mean(costs)) * 100,
            })

    df = pd.DataFrame(results)
    df.to_csv("experiments/greedy_consistency.csv", index=False)
    print("saved greedy_consistency.csv")
    return df


def style_plot(title, xlabel, ylabel):
    # helper to apply consistent styling to every plot
    # calling this after plt.figure() sets up the background and font stuff
    plt.title(title, fontsize=13, fontweight="bold", color="deeppink", pad=12)
    plt.xlabel(xlabel, fontsize=11, color="#333333")
    plt.ylabel(ylabel, fontsize=11, color="#333333")
    plt.xticks(SIZES)
    plt.gca().set_facecolor("#fff0f5")   # blush pink background inside the plot
    plt.gcf().set_facecolor("#ffffff")   # white outer background
    plt.gca().spines["top"].set_visible(False)     # remove top border
    plt.gca().spines["right"].set_visible(False)   # remove right border
    plt.gca().spines["left"].set_color("lightpink")
    plt.gca().spines["bottom"].set_color("lightpink")
    plt.gca().tick_params(colors="#555555")
    plt.grid(True, color="lightpink", linestyle="--", linewidth=0.6, alpha=0.7)
    plt.legend(fontsize=9)
    plt.tight_layout()


def plot_rrnn_tuning(df_k, df_repeats):
    os.makedirs("experiments/plots", exist_ok=True)

    # plot 1: k on x axis, median cost on y axis
    plt.figure(figsize=(8, 5))
    for size in SIZES:
        subset = df_k[df_k["size"] == size]
        plt.plot(subset["k"], subset["median_cost"],
                 marker="o", label=f"n={size}",
                 color=plt.cm.RdPu(size / 35))
    style_plot(
        title="RRNN: Effect of k on Solution Cost",
        xlabel="k (number of random choices)",
        ylabel="median cost"
    )
    plt.xticks([1, 2, 3, 5, 7])  # actual k values we tested
    plt.savefig("experiments/plots/rrnn_k_tuning.png", dpi=150)
    plt.close()
    print("saved rrnn_k_tuning.png")

    # plot 2: num_repeats on x axis, median cost on y axis
    plt.figure(figsize=(8, 5))
    for size in SIZES:
        subset = df_repeats[df_repeats["size"] == size]
        plt.plot(subset["num_repeats"], subset["median_cost"],
                 marker="o", label=f"n={size}",
                 color=plt.cm.RdPu(size / 35))
    style_plot(
        title="RRNN: Effect of num_repeats on Solution Cost",
        xlabel="num_repeats",
        ylabel="median cost"
    )
    plt.xticks([10, 25, 50, 100, 200])  # actual repeat values we tested
    plt.savefig("experiments/plots/rrnn_repeats_tuning.png", dpi=150)
    plt.close()
    print("saved rrnn_repeats_tuning.png")


def plot_comparison(df):
    os.makedirs("experiments/plots", exist_ok=True)

    # collapse 10 trials per size down to one median per algorithm per size
    summary = (
        df.groupby(["algo", "size"])
          .median(numeric_only=True)
          .reset_index()
    )

    # plot 3: total runtime
    plt.figure(figsize=(8, 5))
    for algo in ["nn", "nn2opt", "rrnn"]:
        subset = summary[summary["algo"] == algo]
        plt.plot(subset["size"], subset["runtime_ns"] / 1e6,
                 marker=MARKERS[algo], label=algo, color=COLORS[algo], linewidth=2)
    style_plot(
        title="Total Runtime vs Problem Size",
        xlabel="number of cities (n)",
        ylabel="median runtime (ms)"
    )
    plt.savefig("experiments/plots/greedy_runtime.png", dpi=150)
    plt.close()
    print("saved greedy_runtime.png")

    # plot 4: CPU time
    plt.figure(figsize=(8, 5))
    for algo in ["nn", "nn2opt", "rrnn"]:
        subset = summary[summary["algo"] == algo]
        plt.plot(subset["size"], subset["cpu_ns"] / 1e6,
                 marker=MARKERS[algo], label=algo, color=COLORS[algo], linewidth=2)
    style_plot(
        title="CPU Time vs Problem Size",
        xlabel="number of cities (n)",
        ylabel="median CPU time (ms)"
    )
    plt.savefig("experiments/plots/greedy_cpu.png", dpi=150)
    plt.close()
    print("saved greedy_cpu.png")

    # plot 5: solution cost
    plt.figure(figsize=(8, 5))
    for algo in ["nn", "nn2opt", "rrnn"]:
        subset = summary[summary["algo"] == algo]
        plt.plot(subset["size"], subset["cost"],
                 marker=MARKERS[algo], label=algo, color=COLORS[algo], linewidth=2)
    style_plot(
        title="Solution Cost vs Problem Size",
        xlabel="number of cities (n)",
        ylabel="median cost"
    )
    plt.savefig("experiments/plots/greedy_cost.png", dpi=150)
    plt.close()
    print("saved greedy_cost.png")


def plot_consistency(df):
    os.makedirs("experiments/plots", exist_ok=True)

    plt.figure(figsize=(8, 5))
    for algo in ["nn", "nn2opt", "rrnn"]:
        subset = df[df["algo"] == algo]
        plt.plot(subset["size"], subset["cv"],
                 marker=MARKERS[algo], label=algo, color=COLORS[algo], linewidth=2)
    style_plot(
        title="Solution Consistency Across Matrices (Lower = More Consistent)",
        xlabel="number of cities (n)",
        ylabel="coefficient of variation (%)"
    )
    plt.savefig("experiments/plots/greedy_consistency.png", dpi=150)
    plt.close()
    print("saved greedy_consistency.png")


if __name__ == "__main__":
    # load existing CSVs from previous run so we dont have to redo everything
    print("loading existing results...")
    df_k = pd.read_csv("experiments/rrnn_k_tuning.csv")
    df_repeats = pd.read_csv("experiments/rrnn_repeats_tuning.csv")
    df_comparison = pd.read_csv("experiments/greedy_comparison.csv")
    print("loaded!")

    # run only the new consistency experiment
    print("loading matrices for consistency experiment...")
    matrices = load_matrices_by_size(MATRICES_DIR)
    for size in SIZES:
        print(f"  n={size}: {len(matrices[size])} matrices loaded")

    df_consistency = run_consistency_experiment(matrices, best_k=3, best_repeats=50)

    # regenerate all plots with the new pink styling
    print("generating plots...")
    plot_rrnn_tuning(df_k, df_repeats)
    plot_comparison(df_comparison)
    plot_consistency(df_consistency)

    print("all done!! check experiments/plots/ :)")