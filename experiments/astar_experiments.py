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
DATA_FOLDER = "data/matrices/Project_1_-_matrices"

# there are gonna be all sizes in the dataset
ALL_SIZES = [5, 6, 7, 8, 9, 10, 15, 20, 25, 30]

#A* can apparently only handle small matrices so my laptop does not time out!
#we'll try n=5 and n=10, anything bigger will likely time out
#after 10 we see a really weird dip so im going to make it keep going up to 20 just to see what happens
SMALL_SIZES = [5, 6, 7, 8, 9, 10, 15, 20]

# all green color palette for the A* experiment plots LOLOL NOT PINK GREEN NOW SO FAIRLY OFF PARENTS
COLOR_MAP = {
    "nn":     "lightgreen",
    "nn2opt": "mediumseagreen",
    "rrnn":   "seagreen",
    "astar":  "darkgreen"
}

SHAPE_MAP = {
    "nn":     "o",
    "nn2opt": "s",
    "rrnn":   "^",
    "astar":  "D"
}

#ok so next i gotta make the matrix loader and helper functions, going to do that now i think!
def group_matrices_by_n(data_folder):
    # groups matrices into a dict by city count
    # same idea as greedy_experiments.py
    grouped = {s: [] for s in ALL_SIZES}

    for fname in sorted(os.listdir(data_folder)):
        if not fname.endswith(".txt"):
            continue
        full_path = os.path.join(data_folder, fname)
        adj = load_matrix(full_path)
        num_cities = adj.shape[0]
        if num_cities in grouped:
            grouped[num_cities].append(adj)

    return grouped

def clock_it(fn, *args, **kwargs):
    # times any function and spits back wall time + cpu time
    # i wrote this bc i was tired of copy pasting the timing code everywhere lol

    wall_start = time.time_ns()
    cpu_start = time.process_time_ns()
    output = fn(*args, **kwargs)
    cpu_end = time.process_time_ns()
    wall_end = time.time_ns()

    wall_elapsed = wall_end - wall_start
    cpu_elapsed = cpu_end - cpu_start

    # if cpu time is 0 my laptop is too fast so run 50 times and average
    if cpu_elapsed == 0:
        num_reps = 50
        cpu_start = time.process_time_ns()
        for _ in range(num_reps):
            fn(*args, **kwargs)
        cpu_end = time.process_time_ns()
        cpu_elapsed = (cpu_end - cpu_start) // num_reps

    return output, wall_elapsed, cpu_elapsed

#ALRIGHT! NOW TIME TO RUN THE A STAR EXPERIMENTS
def collect_astar_data(grouped):
    # A* is the big one -- guaranteed optimal but soooo slow on larger inputs
    # thats why we only run it on SMALL_SIZES
    # anything bigger will probably just hang forever lol

    all_rows = []

    print("running astar...")
    for num_cities in SMALL_SIZES:
        print(f"  running n={num_cities}...")  # printing progress bc A* can feel frozen
        for trial_num, adj in enumerate(grouped[num_cities]):

            # cant use clock_it here bc astar returns TWO things
            # (route AND nodes_expanded) so i time it manually
            wall_start = time.time_ns()
            cpu_start = time.process_time_ns()
            best_path, states_visited = astar(adj)  # this is the slow part lol
            cpu_end = time.process_time_ns()
            wall_end = time.time_ns()

            wall_elapsed = wall_end - wall_start
            cpu_elapsed = cpu_end - cpu_start

            # compute the actual tour cost
            tour_cost = get_cost(best_path, adj)

            all_rows.append({
                "algo": "astar",
                "size": num_cities,
                "trial": trial_num,
                "runtime_ns": wall_elapsed,
                "cpu_ns": cpu_elapsed,
                "cost": tour_cost,
                "nodes_expanded": states_visited,
            })

            # printing this so i can see progress trial by trial
            # without this it just looks frozen which is terrifying lol
            print(f"    trial {trial_num} done -- states visited: {states_visited}")

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv("experiments/astar_results.csv", index=False)
    print("saved astar_results.csv")
    return df_out

# ok now im going to collect greedy results on the same small sizes
def collect_greedy_data(grouped, chosen_k=3, chosen_repeats=50):
    # need to run greedy on the SAME matrices as A*
    # so i can divide their results by A*'s results later
    # only running on SMALL_SIZES since thats all A* can handle

    all_rows = []

    algo_lookup = {
        "nn":     lambda adj: clock_it(nearest_neighbor, adj),
        "nn2opt": lambda adj: clock_it(nearest_neighbor_2opt, adj),
        "rrnn":   lambda adj: clock_it(rrnn, adj, k=chosen_k, num_repeats=chosen_repeats),
    }

    for algo_name, run_fn in algo_lookup.items():
        print(f"running {algo_name} on small sizes...")
        for num_cities in SMALL_SIZES:
            for trial_num, adj in enumerate(grouped[num_cities]):
                best_path, wall_elapsed, cpu_elapsed = run_fn(adj)
                tour_cost = get_cost(best_path, adj)
                all_rows.append({
                    "algo": algo_name,
                    "size": num_cities,
                    "trial": trial_num,
                    "runtime_ns": wall_elapsed,
                    "cpu_ns": cpu_elapsed,
                    "cost": tour_cost,
                    "nodes_expanded": 0,  # greedy algos dont expand nodes so just use 0
                })

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv("experiments/greedy_on_astar_sizes.csv", index=False)
    print("saved greedy_on_astar_sizes.csv")
    return df_out

# now i have to make the normalization function
# i think this is the most important one
# any value under 1.0 means that algo is faster/cheaper than a*
def compute_ratios(df_greedy, df_astar):
    # divide each algorithm's results by A*'s results
    # so if A* took 10ms and NN took 2ms, NN gets plotted as 0.2
    # this shows how much faster/slower/better each algorithm is RELATIVE to A*

    astar_medians = (
        df_astar.groupby("size")
                .median(numeric_only=True)
                .reset_index()
    )

    greedy_medians = (
        df_greedy.groupby(["algo", "size"])
                 .median(numeric_only=True)
                 .reset_index()
    )

    ratio_rows = []

    for num_cities in SMALL_SIZES:
        astar_row = astar_medians[astar_medians["size"] == num_cities].iloc[0]
        base_wall = astar_row["runtime_ns"]
        base_cpu = astar_row["cpu_ns"]
        base_cost = astar_row["cost"]

        for algo_name in ["nn", "nn2opt", "rrnn"]:
            greedy_row = greedy_medians[
                (greedy_medians["algo"] == algo_name) &
                (greedy_medians["size"] == num_cities)
            ].iloc[0]

            ratio_rows.append({
                "algo": algo_name,
                "size": num_cities,
                "runtime_ratio": greedy_row["runtime_ns"] / base_wall,
                "cpu_ratio":     greedy_row["cpu_ns"] / base_cpu,
                "cost_ratio":    greedy_row["cost"] / base_cost,
            })

    df_out = pd.DataFrame(ratio_rows)
    df_out.to_csv("experiments/normalized_results.csv", index=False)
    print("saved normalized_results.csv")
    return df_out

#YAYYAYAYA NOW THE FUN PART -- THE PLOTTING FUNCTIONS FOR THE A* EXPERIMENTS
# green themed to match part 2 vibes
def style_green(plot_title, x_label, y_label):
    plt.title(plot_title, fontsize=13, fontweight="bold", color="seagreen", pad=12)
    plt.xlabel(x_label, fontsize=11, color="#333333")
    plt.ylabel(y_label, fontsize=11, color="#333333")
    plt.xticks(SMALL_SIZES)
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


def make_ratio_plots(df_ratios):
    os.makedirs("experiments/plots", exist_ok=True)

    # plot 1: normalized runtime
    plt.figure(figsize=(8, 5))
    for algo_name in ["nn", "nn2opt", "rrnn"]:
        chunk = df_ratios[df_ratios["algo"] == algo_name]
        plt.plot(chunk["size"], chunk["runtime_ratio"],
                 marker=SHAPE_MAP[algo_name], label=algo_name,
                 color=COLOR_MAP[algo_name], linewidth=2)
    # horizontal line at y=1 -- this is where A* sits
    # anything below = faster than A*
    plt.axhline(y=1.0, color="darkgreen", linestyle="--", linewidth=1.5, label="A* baseline")
    style_green(
        plot_title="Runtime Relative to A*",
        x_label="number of cities (n)",
        y_label="runtime / A* runtime"
    )
    plt.savefig("experiments/plots/normalized_runtime.png", dpi=150)
    plt.close()
    print("saved normalized_runtime.png")

    # plot 2: normalized CPU time
    plt.figure(figsize=(8, 5))
    for algo_name in ["nn", "nn2opt", "rrnn"]:
        chunk = df_ratios[df_ratios["algo"] == algo_name]
        plt.plot(chunk["size"], chunk["cpu_ratio"],
                 marker=SHAPE_MAP[algo_name], label=algo_name,
                 color=COLOR_MAP[algo_name], linewidth=2)
    plt.axhline(y=1.0, color="darkgreen", linestyle="--", linewidth=1.5, label="A* baseline")
    style_green(
        plot_title="CPU Time Relative to A*",
        x_label="number of cities (n)",
        y_label="CPU time / A* CPU time"
    )
    plt.savefig("experiments/plots/normalized_cpu.png", dpi=150)
    plt.close()
    print("saved normalized_cpu.png")

    # plot 3: normalized cost
    plt.figure(figsize=(8, 5))
    for algo_name in ["nn", "nn2opt", "rrnn"]:
        chunk = df_ratios[df_ratios["algo"] == algo_name]
        plt.plot(chunk["size"], chunk["cost_ratio"],
                 marker=SHAPE_MAP[algo_name], label=algo_name,
                 color=COLOR_MAP[algo_name], linewidth=2)
    plt.axhline(y=1.0, color="darkgreen", linestyle="--", linewidth=1.5, label="A* baseline")
    style_green(
        plot_title="Solution Cost Relative to A*",
        x_label="number of cities (n)",
        y_label="cost / A* cost"
    )
    plt.savefig("experiments/plots/normalized_cost.png", dpi=150)
    plt.close()
    print("saved normalized_cost.png")


def make_nodes_plot(df_astar):
    os.makedirs("experiments/plots", exist_ok=True)

    # average nodes expanded per size
    # using mean here bc i want to show the general trend
    per_size = (
        df_astar.groupby("size")
                .mean(numeric_only=True)
                .reset_index()
    )

    # this plot shows how fast the search space explodes as n increases
    plt.figure(figsize=(8, 5))
    plt.plot(per_size["size"], per_size["nodes_expanded"],
             marker="D", color="darkgreen", linewidth=2, label="A*")
    style_green(
        plot_title="Nodes Expanded by A* vs Problem Size",
        x_label="number of cities (n)",
        y_label="median nodes expanded"
    )
    plt.savefig("experiments/plots/astar_nodes.png", dpi=150)
    plt.close()
    print("saved astar_nodes.png")


## so now i will make the main block that will run everything
if __name__ == "__main__":
    print("loading matrices...")
    grouped = group_matrices_by_n(DATA_FOLDER)

    # sanity check -- make sure i got the right number of matrices per size
    for sz in ALL_SIZES:
        print(f"  n={sz}: {len(grouped[sz])} matrices loaded")

    # step 1: run a* on small matrices
    # this is the slow part -- go get a snack lol
    df_astar = collect_astar_data(grouped)

    # step 2: run greedy algorithms on the same sizes so we can normalize
    df_greedy = collect_greedy_data(grouped)

    # step 3: compute ratios against A*
    df_ratios = compute_ratios(df_greedy, df_astar)

    # step 4: generate all 4 plots
    make_ratio_plots(df_ratios)
    make_nodes_plot(df_astar)

    print("all done bruh!! the plots should be in experiments/plots :)")