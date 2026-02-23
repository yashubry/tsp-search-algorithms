#ok lets make the experiments now woohoohooooo haha

#import time!

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#my own A* implementation
from src.astar import astar, mst_heuristic
from src.greedy import nearest_neighbor, nearest_neighbor_2opt, rrnn
from src.utils import load_matrix, get_cost

#folder where all the matrix files live
MATRICES_DIR = "data/matrices/Project_1_-_matrices"

# there are gonna beall sizes in the dataset
SIZES = [5, 6, 7, 8, 9, 10, 15, 20, 25, 30]

#A* can only apparentely only handle small matrices so my labtop does not time out!
#we'll try n=5 and n=10, anything bigger will likely time out
#after 10 we see a really wierd dip so im going to make it keep going up to 20 just to see what happens, but anything bigger than 20 

ASTAR_SIZES = [5, 6, 7, 8, 9, 10, 15, 20]
# all green color palette for the A* experiment plots LOLOL NOT PINK GREEN NOW SO FAIRLY OFF PARENTS
COLORS = {
    "nn":     "lightgreen",
    "nn2opt": "mediumseagreen",
    "rrnn":   "seagreen",
    "astar":  "darkgreen"
}

MARKERS = {
    "nn":     "o",
    "nn2opt": "s",
    "rrnn":   "^",
    "astar":  "D"
}

#ok os next i gotta at the metrix lepder and helper functions, going to do that now i think !
def load_matrices_by_size(data_dir):
    #same loader as greedy_experiments.py
    # groups matrices into a dict by city count
    matrices = {s: [] for s in SIZES}

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(data_dir, fname)
        mat = load_matrix(fpath)
        n = mat.shape[0]
        if n in matrices:
            matrices[n].append(mat)

    return matrices

def time_algorithm(fn, *args, **kwargs):
    #same timing helper as greedy_experiments.py
    #so yrecords wall time and CPU time for any function

    t0 = time.time_ns()
    c0 = time.process_time_ns()
    result = fn(*args, **kwargs)
    c1 = time.process_time_ns()
    t1 = time.time_ns()

    runtime_ns = t1 - t0
    cpu_ns = c1 - c0

    #if cpu time is 0, we wanna run 50 times and average
    if cpu_ns == 0:
        reps = 50
        c0 = time.process_time_ns()
        for _ in range(reps):
            fn(*args, **kwargs)
        c1 = time.process_time_ns()
        cpu_ns = (c1 - c0) // reps

    return result, runtime_ns, cpu_ns

#ALRIGHT! NOW TIME TO RUN TEH A START EXPERIMENTS
def run_astar_experiments(matrices):
    # A* is the big one -- guaranteed optimal but soooo slow on larger inputs
    # thats why we only run it on ASTAR_SIZES (n=5 and n=10)
    # anything bigger will probably just hang forever lol

    results = []

    print("running astar...")
    for size in ASTAR_SIZES:
        print(f"  running n={size}...")  # printing progress bc A* can feel frozen
        for i, mat in enumerate(matrices[size]):

            # cant use the time_algorithm helper here bc astar returns TWO things
            # (route AND nodes_expanded) so i have to time it manually
            t0 = time.time_ns()
            c0 = time.process_time_ns()
            route, nodes_expanded = astar(mat)  # this is the slow part lol
            c1 = time.process_time_ns()
            t1 = time.time_ns()

            runtime_ns = t1 - t0
            cpu_ns = c1 - c0

            # compute the actual tour cost
            cost = get_cost(route, mat)

            # saving nodes_expanded too -- we need it for the part 2 plot
            results.append({
                "algo": "astar",
                "size": size,
                "trial": i,
                "runtime_ns": runtime_ns,
                "cpu_ns": cpu_ns,
                "cost": cost,
                "nodes_expanded": nodes_expanded,  # how many states A* had to explore
            })

            # printing this so i can see progress trial by trial
            # without this it just looks frozen which is terrifying
            print(f"    trial {i} done -- nodes expanded: {nodes_expanded}")

    df = pd.DataFrame(results)
    df.to_csv("experiments/astar_results.csv", index=False)
    print("saved astar_results.csv")
    return df

# ok now im going to make the greedy algo 
def run_greedy_on_astar_sizes(matrices, best_k=3, best_repeats=50):
    # we need to run the greedy algorithms on the SAME matrices as A*
    # so we can divide their results by A*'s results to normalize
    # we only run on ASTAR_SIZES since thats all A* can handle

    results = []

    algorithms = {
        "nn":     lambda mat: time_algorithm(nearest_neighbor, mat),
        "nn2opt": lambda mat: time_algorithm(nearest_neighbor_2opt, mat),
        "rrnn":   lambda mat: time_algorithm(rrnn, mat, k=best_k, num_repeats=best_repeats),
    }

    for algo_name, algo_fn in algorithms.items():
        print(f"running {algo_name} on astar sizes...")
        for size in ASTAR_SIZES:  # only the small sizes!
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
                    "nodes_expanded": 0,  # greedy algorithms dont expand nodes so just use 0
                })

    df = pd.DataFrame(results)
    df.to_csv("experiments/greedy_on_astar_sizes.csv", index=False)
    print("saved greedy_on_astar_sizes.csv")
    return df

#now i have to make the normalization function 
# i think this is the most important one 
#im going to make it so that any value until 1.0 means that algo is faster/cheaper than a*
def normalize_against_astar(df_greedy, df_astar):
    # the spec asks us to divide each algorithm's results by A*'s results
    # so if A* took 10ms and NN took 2ms, NN gets plotted as 0.2
    # this shows how much faster/slower/better each algorithm is RELATIVE to A*

    # first compute medians for each algorithm and size
    astar_summary = (
        df_astar.groupby("size")
                .median(numeric_only=True)
                .reset_index()
    )

    greedy_summary = (
        df_greedy.groupby(["algo", "size"])
                 .median(numeric_only=True)
                 .reset_index()
    )

    normalized_rows = []

    for size in ASTAR_SIZES:
        # get A*'s median values for this size
        astar_row = astar_summary[astar_summary["size"] == size].iloc[0]
        astar_runtime = astar_row["runtime_ns"]
        astar_cpu = astar_row["cpu_ns"]
        astar_cost = astar_row["cost"]

        for algo in ["nn", "nn2opt", "rrnn"]:
            greedy_row = greedy_summary[
                (greedy_summary["algo"] == algo) &
                (greedy_summary["size"] == size)
            ].iloc[0]

            normalized_rows.append({
                "algo": algo,
                "size": size,
                # divide greedy value by A* value to get the ratio
                "runtime_ratio": greedy_row["runtime_ns"] / astar_runtime,
                "cpu_ratio":     greedy_row["cpu_ns"] / astar_cpu,
                "cost_ratio":    greedy_row["cost"] / astar_cost,
            })

    df = pd.DataFrame(normalized_rows)
    df.to_csv("experiments/normalized_results.csv", index=False)
    print("saved normalized_results.csv")
    return df

#YAYYAYAYA NOW THE FUN PART I CAN WORK ON THE PLOTTING FUCTIONS FOR THE A* EXPERIMENTS
# i will make them green again to match 
def style_plot_green(title, xlabel, ylabel):
    # same style helper as greedy but green themed!
    plt.title(title, fontsize=13, fontweight="bold", color="seagreen", pad=12)
    plt.xlabel(xlabel, fontsize=11, color="#333333")
    plt.ylabel(ylabel, fontsize=11, color="#333333")
    plt.xticks(ASTAR_SIZES)
    plt.gca().set_facecolor("#f0fff0")   # honeydew green background
    plt.gcf().set_facecolor("#ffffff")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_color("lightgreen")
    plt.gca().spines["bottom"].set_color("lightgreen")
    plt.gca().tick_params(colors="#555555")
    plt.grid(True, color="lightgreen", linestyle="--", linewidth=0.6, alpha=0.7)
    plt.legend(fontsize=9)
    plt.tight_layout()


def plot_normalized(df):
    os.makedirs("experiments/plots", exist_ok=True)

    # plot 1: normalized runtime
    plt.figure(figsize=(8, 5))
    for algo in ["nn", "nn2opt", "rrnn"]:
        subset = df[df["algo"] == algo]
        plt.plot(subset["size"], subset["runtime_ratio"],
                 marker=MARKERS[algo], label=algo, color=COLORS[algo], linewidth=2)
    # draw a horizontal line at y=1 -- this is where A* sits
    # anything below this line is faster than A*
    plt.axhline(y=1.0, color="darkgreen", linestyle="--", linewidth=1.5, label="A* baseline")
    style_plot_green(
        title="Runtime Relative to A*",
        xlabel="number of cities (n)",
        ylabel="runtime / A* runtime"
    )
    plt.savefig("experiments/plots/normalized_runtime.png", dpi=150)
    plt.close()
    print("saved normalized_runtime.png")

    # plot 2: normalized CPU time
    plt.figure(figsize=(8, 5))
    for algo in ["nn", "nn2opt", "rrnn"]:
        subset = df[df["algo"] == algo]
        plt.plot(subset["size"], subset["cpu_ratio"],
                 marker=MARKERS[algo], label=algo, color=COLORS[algo], linewidth=2)
    plt.axhline(y=1.0, color="darkgreen", linestyle="--", linewidth=1.5, label="A* baseline")
    style_plot_green(
        title="CPU Time Relative to A*",
        xlabel="number of cities (n)",
        ylabel="CPU time / A* CPU time"
    )
    plt.savefig("experiments/plots/normalized_cpu.png", dpi=150)
    plt.close()
    print("saved normalized_cpu.png")

    # plot 3: normalized cost
    plt.figure(figsize=(8, 5))
    for algo in ["nn", "nn2opt", "rrnn"]:
        subset = df[df["algo"] == algo]
        plt.plot(subset["size"], subset["cost_ratio"],
                 marker=MARKERS[algo], label=algo, color=COLORS[algo], linewidth=2)
    plt.axhline(y=1.0, color="darkgreen", linestyle="--", linewidth=1.5, label="A* baseline")
    style_plot_green(
        title="Solution Cost Relative to A*",
        xlabel="number of cities (n)",
        ylabel="cost / A* cost"
    )
    plt.savefig("experiments/plots/normalized_cost.png", dpi=150)
    plt.close()
    print("saved normalized_cost.png")


def plot_nodes_expanded(df_astar):
    os.makedirs("experiments/plots", exist_ok=True)

    # compute median nodes expanded per size
    summary = (
        df_astar.groupby("size")
                .mean(numeric_only=True)
                .reset_index()
    )

    # this plot shows how fast the search space explodes as n increases
    # the spec says nodes expanded should grow faster than n itself
    plt.figure(figsize=(8, 5))
    plt.plot(summary["size"], summary["nodes_expanded"],
             marker="D", color="darkgreen", linewidth=2, label="A*")
    style_plot_green(
        title="Nodes Expanded by A* vs Problem Size",
        xlabel="number of cities (n)",
        ylabel="median nodes expanded"
    )
    plt.savefig("experiments/plots/astar_nodes.png", dpi=150)
    plt.close()
    print("saved astar_nodes.png")


## so now i will make the main block that will run everyhting 
if __name__ == "__main__":
    print("loading matrices...")
    matrices = load_matrices_by_size(MATRICES_DIR)

    # sanity check
    for size in SIZES:
        print(f"  n={size}: {len(matrices[size])} matrices loaded")

    #step 1:run a* on small matrices
    #this is the slow part -- go get a snack lol
    df_astar = run_astar_experiments(matrices)

    # 2:run greedy algorithms on the same sizes so we can normalize
    df_greedy = run_greedy_on_astar_sizes(matrices)

    #3: normalize greedy results against A*
    df_normalized = normalize_against_astar(df_greedy, df_astar)

    #4:generate all 4 plots
    plot_normalized(df_normalized)
    plot_nodes_expanded(df_astar)

    print("all done bruh!! the plots should be in experiments-plots :)")