# YAYAYAYAYYAYAYAYYAY THIS IS THE LAST OF THE EXPERIMENT FILES..
# for now... dun dun dunnnnnnn

# gonna make this purple themed for the giggles! oh em gee 

# again the imorts ineteesting ohemgee
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.local import hill_climbing, simulated_annealing, genetic_algorithm
from src.greedy import nearest_neighbor, nearest_neighbor_2opt, rrnn
from src.utils import load_matrix, get_cost

# ok in terms of all the sizes, im going to keep to similar to what we did for a* so that there are parallel comparisons 
MATRICES_DIR = "data/matrices/Project_1_-_matrices"
SIZES = [5, 10, 15, 20, 25, 30]
ASTAR_SIZES = [5, 6, 7, 8, 9, 10, 15, 20]  # sizes A* was run on

# purple color palette with accents
COLORS = {
    "hill_climbing": "#9b59b6",  
    "simulated_annealing": "#6c3483", 
    "genetic_algorithm":  "#d7bde2",
    "nn":    "#f1948a",  
    "nn2opt":"#85c1e9", 
    "rrnn":"#82e0aa",  
    "astar":  "#f0b27a",   # soft orange for A* baseline
}

MARKERS = {
    "hill_climbing":        "o",
    "simulated_annealing":  "s",
    "genetic_algorithm":    "^",
}

#again, going to use the matrix loader
def load_matrices_by_size(data_dir):
    # same loader as the other experiment files
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
    # ok i literally copied this from greedy_experiments.py haha
    # it times any function and returns wall time and cpu time
    t0 = time.time_ns()
    c0 = time.process_time_ns()
    result = fn(*args, **kwargs)
    c1 = time.process_time_ns()
    t1 = time.time_ns()

    runtime_ns = t1 - t0
    cpu_ns = c1 - c0

    # if cpu time is 0 i run it 50 times and average bc my laptop is too fast lol
    if cpu_ns == 0:
        reps = 50
        c0 = time.process_time_ns()
        for _ in range(reps):
            fn(*args, **kwargs)
        c1 = time.process_time_ns()
        cpu_ns = (c1 - c0) // reps

    return result, runtime_ns, cpu_ns


def style_plot_purple(title, xlabel, ylabel):
    # ok i made this purple themed bc pink was part 1 and green was part 2
    # purple is giving royalty vibes for part 3 lol
    plt.title(title, fontsize=13, fontweight="bold", color="#6c3483", pad=12)
    plt.xlabel(xlabel, fontsize=11, color="#333333")
    plt.ylabel(ylabel, fontsize=11, color="#333333")
    plt.gca().set_facecolor("#f9f0ff")   # light lavender background yasss
    plt.gcf().set_facecolor("#ffffff")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_color("#d7bde2")
    plt.gca().spines["bottom"].set_color("#d7bde2")
    plt.gca().tick_params(colors="#555555")
    plt.grid(True, color="#d7bde2", linestyle="--", linewidth=0.6, alpha=0.7)
    plt.legend(fontsize=9)
    plt.tight_layout()

def tune_hill_climbing(matrices):
    # ok so i need to figure out the best num_restarts for hill climbing
    # i want to see how many restarts actually helps before it stops mattering

    # first i tried just doing like 3 values: 5, 10, 50
    # restart_values = [5, 10, 50]
    # but then i realized i needed more granularity to see the trend clearly
    # so i added more values lol
    restart_values = [5, 10, 25, 50, 100]
    results = []

    print("tuning hill climbing... this might take a sec lol")
    for num_restarts in restart_values:
        for size in SIZES:
            costs = []
            for mat in matrices[size]:
                # i first tried returning just the route here
                # route = hill_climbing(mat, num_restarts=num_restarts)
                # but then i remembered it returns a tuple haha oops
                route, _ = hill_climbing(mat, num_restarts=num_restarts)
                costs.append(get_cost(route, mat))

            # i tried mean first but median is more robust to outliers
            # median_cost = np.mean(costs)
            results.append({
                "num_restarts": num_restarts,
                "size": size,
                "median_cost": np.median(costs)  # median not mean!!
            })
            print(f"  num_restarts={num_restarts}, n={size} done!")

    df = pd.DataFrame(results)
    df.to_csv("experiments/hc_tuning.csv", index=False)
    print("saved hc_tuning.csv yayyy")
    return df


def tune_simulated_annealing(matrices):
    #tuning alpha for simulated annealing
    #alpha controls how fast the temperature cools down
    #i had no idea what values to try at first lol
    #first attempt: i tried really extreme values
    #alpha_values = [0.5, 0.99]
    #but 0.5 was way too fast and 0.99 took forever so i picked values in between
    alpha_values = [0.90, 0.95, 0.99, 0.999]
    results = []

    print("tuning simulated annealing... go get a snack lol")
    for alpha in alpha_values:
        for size in SIZES:
            costs = []
            for mat in matrices[size]:
                # i forgot to pass initial_temp the first time and got an error haha
                # route, _ = simulated_annealing(mat, alpha=alpha)
                route, _ = simulated_annealing(
                    mat, alpha=alpha, initial_temp=100, max_iterations=1000)
                costs.append(get_cost(route, mat))

            results.append({
                "alpha": alpha,
                "size": size,
                "median_cost": np.median(costs)
            })
            print(f"  alpha={alpha}, n={size} done!")

    df = pd.DataFrame(results)
    df.to_csv("experiments/sa_tuning.csv", index=False)
    print("saved sa_tuning.csv!!")
    return df


def tune_genetic_algorithm(matrices):
    #tuning num_generations for the genetic algorithm
    #more generations = better solutions but wayyyy slower
    #i want to find the sweet spot where it stops improving

    #first i tried tuning population_size instead
    #pop_values = [10, 50, 100]
    #but then i realized num_generations was more interesting to tune
    #bc population_size just makes it slower without much benefit after a point
    generation_values = [10, 25, 50, 100, 200]
    results = []

    print("tuning genetic algorithm... this one is slow ngl")
    for num_generations in generation_values:
        for size in SIZES:
            costs = []
            for mat in matrices[size]:
                # i originally forgot mutation_chance and got a TypeError lol
                # route, _ = genetic_algorithm(mat, population_size=50, num_generations=num_generations)
                route, _ = genetic_algorithm(
                    mat,
                    mutation_chance=0.1,# 10% chance of mutation
                    population_size=50,#50 tours in the population
                    num_generations=num_generations)
                costs.append(get_cost(route, mat))

            results.append({
                "num_generations": num_generations,
                "size": size,
                "median_cost": np.median(costs)
            })
            print(f"  num_generations={num_generations}, n={size} done!")

    df = pd.DataFrame(results)
    df.to_csv("experiments/ga_tuning.csv", index=False)
    print("saved ga_tuning.csv woooo")
    return df

def plot_tuning(df_hc, df_sa, df_ga):
    # ok now i gotta plot all the tuning results
    # one plot per algorithm showing how the hyperparameter affects cost
    os.makedirs("experiments/plots", exist_ok=True)

    # plot 1: hill climbing -- num_restarts vs median cost
    # i first tried plotting all sizes on one plot but it was too cluttered
    # so i just plot each size as a separate line like i did in part 1
    plt.figure(figsize=(8, 5))
    for size in SIZES:
        subset = df_hc[df_hc["size"] == size]
        plt.plot(subset["num_restarts"], subset["median_cost"],
                 marker="o", label=f"n={size}",
                 color=plt.cm.Purples(size / 35))
    style_plot_purple(
        title="Hill Climbing: Effect of num_restarts on Solution Cost",
        xlabel="num_restarts",
        ylabel="median cost"
    )
    plt.xticks([5, 10, 25, 50, 100])
    plt.savefig("experiments/plots/hc_tuning.png", dpi=150)
    plt.close()
    print("saved hc_tuning.png!!")

    # plot 2: simulated annealing -- alpha vs median cost
    # alpha is tricky to plot bc the values are so close together
    # i tried a log x axis first but it looked weird lol
    # plt.xscale("log")
    plt.figure(figsize=(8, 5))
    for size in SIZES:
        subset = df_sa[df_sa["size"] == size]
        plt.plot(subset["alpha"], subset["median_cost"],
                 marker="s", label=f"n={size}",
                 color=plt.cm.Purples(size / 35))
    style_plot_purple(
        title="Simulated Annealing: Effect of Alpha on Solution Cost",
        xlabel="alpha (cooling rate)",
        ylabel="median cost"
    )
    plt.xticks([0.90, 0.95, 0.99, 0.999])
    plt.savefig("experiments/plots/sa_tuning.png", dpi=150)
    plt.close()
    print("saved sa_tuning.png!!")

    # plot 3: genetic algorithm -- num_generations vs median cost
    plt.figure(figsize=(8, 5))
    for size in SIZES:
        subset = df_ga[df_ga["size"] == size]
        plt.plot(subset["num_generations"], subset["median_cost"],
                 marker="^", label=f"n={size}",
                 color=plt.cm.Purples(size / 35))
    style_plot_purple(
        title="Genetic Algorithm: Effect of num_generations on Solution Cost",
        xlabel="num_generations",
        ylabel="median cost"
    )
    plt.xticks([10, 25, 50, 100, 200])
    plt.savefig("experiments/plots/ga_tuning.png", dpi=150)
    plt.close()
    print("saved ga_tuning.png!!")

def plot_cost_over_iterations(matrices, best_restarts=25, best_alpha=0.99, best_generations=100):
    # ok this is the fun part -- i get to see how each algorithm improves over time
    # i run each algorithm once on a medium sized matrix and plot the cost history
    # i picked n=20 bc its big enough to be interesting but not too slow

    # i first tried running on n=30 but it took forever lol
    # sample_size = 30
    sample_size = 20
    os.makedirs("experiments/plots", exist_ok=True)

    # just grab the first matrix of that size
    mat = matrices[sample_size][0]

    # plot 1: hill climbing -- best cost found after each restart
    _, cost_per_restart = hill_climbing(mat, num_restarts=best_restarts)

    # i tried plotting raw cost per restart first
    # but then i realized i should plot the RUNNING MINIMUM
    # bc the whole point is to track the best found so far
    # cost_per_restart is already the cost at each restart so i take cumulative min
    running_best = np.minimum.accumulate(cost_per_restart)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(running_best) + 1), running_best,
             color="#9b59b6", linewidth=2, marker="o", markersize=4)
    style_plot_purple(
        title="Hill Climbing: Best Cost per Restart (n=20)",
        xlabel="restart number",
        ylabel="best cost found so far"
    )
    plt.savefig("experiments/plots/hc_iterations.png", dpi=150)
    plt.close()
    print("saved hc_iterations.png!!")

    # plot 2: simulated annealing -- best cost found at each iteration
    _, cost_per_iteration = simulated_annealing(
        mat, alpha=best_alpha, initial_temp=100, max_iterations=1000)

    # sa already tracks running best inside the function so i can plot directly
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cost_per_iteration) + 1), cost_per_iteration,
             color="#6c3483", linewidth=2)
    style_plot_purple(
        title="Simulated Annealing: Best Cost per Iteration (n=20)",
        xlabel="iteration",
        ylabel="best cost found so far"
    )
    plt.savefig("experiments/plots/sa_iterations.png", dpi=150)
    plt.close()
    print("saved sa_iterations.png!!")

    # plot 3: genetic algorithm -- best cost found per generation
    _, cost_per_generation = genetic_algorithm(
        mat,
        mutation_chance=0.1,
        population_size=50,
        num_generations=best_generations)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cost_per_generation) + 1), cost_per_generation,
             color="#d7bde2", linewidth=2, marker="^", markersize=4)
    style_plot_purple(
        title="Genetic Algorithm: Best Cost per Generation (n=20)",
        xlabel="generation",
        ylabel="best cost found so far"
    )
    plt.savefig("experiments/plots/ga_iterations.png", dpi=150)
    plt.close()
    print("saved ga_iterations.png!!")

def run_local_experiments(matrices, best_restarts=25, best_alpha=0.99, best_generations=100):
    # ok now i run all three algorithms on all sizes and record everything
    # this is the main data collection function woooo

    results = []

    algorithms = {
        # i first tried using lambda here but it got confusing lol
        # so i just wrote it out explicitly
        "hill_climbing": lambda mat: time_algorithm(
            hill_climbing, mat, num_restarts=best_restarts),
        "simulated_annealing": lambda mat: time_algorithm(
            simulated_annealing, mat, alpha=best_alpha,
            initial_temp=100, max_iterations=1000),
        "genetic_algorithm": lambda mat: time_algorithm(
            genetic_algorithm, mat, mutation_chance=0.1,
            population_size=50, num_generations=best_generations),
    }

    for algo_name, algo_fn in algorithms.items():
        print(f"running {algo_name}...")
        for size in SIZES:
            costs = []
            runtimes = []
            cpus = []
            for mat in matrices[size]:
                # time_algorithm returns (result, runtime, cpu)
                # but result here is a tuple (route, cost_history) so i unpack carefully
                (route, _), runtime_ns, cpu_ns = algo_fn(mat)
                cost = get_cost(route, mat)
                costs.append(cost)
                runtimes.append(runtime_ns)
                cpus.append(cpu_ns)
                
            results.append({
                "algo": algo_name,
                "size": size,
                "median_cost": np.median(costs),
                "median_runtime_ns": np.median(runtimes),
                "median_cpu_ns": np.median(cpus),
            })
            print(f"  n={size} done!")

    df = pd.DataFrame(results)
    df.to_csv("experiments/local_results.csv", index=False)
    print("saved local_results.csv!!")
    return df


def normalize_against_astar_local(df_local):
    # same normalization as part 2 -- divide everything by A*'s values
    # i load the A* results csv that i already generated in astar_experiments.py
    # i first tried rerunning A* here but that would take forever lol
    # so i just load the csv instead!!

    print("loading astar results from csv...")
    df_astar = pd.read_csv("experiments/astar_results.csv")

    # compute median A* values per size
    astar_summary = (
        df_astar.groupby("size")
                .median(numeric_only=True)
                .reset_index()
    )

    # only normalize for sizes that A* actually ran on
    common_sizes = [s for s in ASTAR_SIZES if s in SIZES]

    normalized_rows = []
    for size in common_sizes:
        astar_row = astar_summary[astar_summary["size"] == size]
        if len(astar_row) == 0:
            continue  # skip if A* didnt run on this size
        astar_row = astar_row.iloc[0]

        for algo in ["hill_climbing", "simulated_annealing", "genetic_algorithm"]:
            local_row = df_local[
                (df_local["algo"] == algo) &
                (df_local["size"] == size)
            ]
            if len(local_row) == 0:
                continue
            local_row = local_row.iloc[0]

            normalized_rows.append({
                "algo": algo,
                "size": size,
                "runtime_ratio": local_row["median_runtime_ns"] / astar_row["runtime_ns"],
                "cpu_ratio": local_row["median_cpu_ns"] / astar_row["cpu_ns"],
                "cost_ratio": local_row["median_cost"] / astar_row["cost"],
            })

    df = pd.DataFrame(normalized_rows)
    df.to_csv("experiments/local_normalized.csv", index=False)
    print("saved local_normalized.csv!!")
    return df


def plot_normalized_local(df):
    # same normalized plots as part 2 but for local search algorithms
    # purple themed obvsss
    os.makedirs("experiments/plots", exist_ok=True)

    sizes = sorted(df["size"].unique())

    # plot 1: normalized runtime
    plt.figure(figsize=(8, 5))
    for algo in ["hill_climbing", "simulated_annealing", "genetic_algorithm"]:
        subset = df[df["algo"] == algo]
        plt.plot(subset["size"], subset["runtime_ratio"],
                 marker=MARKERS[algo], label=algo,
                 color=COLORS[algo], linewidth=2)
    plt.axhline(y=1.0, color="#f0b27a", linestyle="--", linewidth=1.5, label="A* baseline")
    style_plot_purple(
        title="Runtime Relative to A*",
        xlabel="number of cities (n)",
        ylabel="runtime / A* runtime"
    )
    plt.xticks(sizes)
    plt.savefig("experiments/plots/local_normalized_runtime.png", dpi=150)
    plt.close()
    print("saved local_normalized_runtime.png!!")

    # plot 2: normalized CPU time
    # plot 2: normalized CPU time
    plt.figure(figsize=(8, 5))
    for algo in ["hill_climbing", "simulated_annealing", "genetic_algorithm"]:
        subset = df[df["algo"] == algo]
        plt.plot(subset["size"], subset["cpu_ratio"],
                 marker=MARKERS[algo], label=algo,
                 color=COLORS[algo], linewidth=2)
    plt.axhline(y=1.0, color="#f0b27a", linestyle="--", linewidth=1.5, label="A* baseline")
    style_plot_purple(
        title="CPU Time Relative to A*",
        xlabel="number of cities (n)",
        ylabel="CPU time / A* CPU time"
    )
    plt.xticks(sizes)
    plt.savefig("experiments/plots/local_normalized_cpu.png", dpi=150)
    plt.close()
    print("saved local_normalized_cpu.png!!")

    # plot 3: normalized cost
    plt.figure(figsize=(8, 5))
    for algo in ["hill_climbing", "simulated_annealing", "genetic_algorithm"]:
        subset = df[df["algo"] == algo]
        plt.plot(subset["size"], subset["cost_ratio"],
                 marker=MARKERS[algo], label=algo,
                 color=COLORS[algo], linewidth=2)
    plt.axhline(y=1.0, color="#f0b27a", linestyle="--", linewidth=1.5, label="A* baseline")
    style_plot_purple(
        title="Solution Cost Relative to A*",
        xlabel="number of cities (n)",
        ylabel="cost / A* cost"
    )
    plt.xticks(sizes)
    plt.savefig("experiments/plots/local_normalized_cost.png", dpi=150)
    plt.close()
    print("saved local_normalized_cost.png!!")
    
if __name__ == "__main__":
    print("loading matrices... here we go!!")
    matrices = load_matrices_by_size(MATRICES_DIR)

    # sanity check -- make sure i loaded the right number of matrices
    for size in SIZES:
        print(f"  n={size}: {len(matrices[size])} matrices loaded")

    # step 1: hyperparameter tuning
    # i run these first so i know the best params before the main experiments
    print("\nstarting hyperparameter tuning wooooo")
    df_hc = tune_hill_climbing(matrices)
    df_sa = tune_simulated_annealing(matrices)
    df_ga = tune_genetic_algorithm(matrices)

    # step 2: plot the tuning results
    print("\nplotting tuning results...")
    plot_tuning(df_hc, df_sa, df_ga)

    # step 3: plot cost over iterations for each algorithm
    # i do this before the main experiments bc it only needs one matrix
    print("\nplotting cost over iterations...")
    plot_cost_over_iterations(matrices)

    # step 4: run main experiments on all sizes
    print("\nrunning main local search experiments...")
    df_local = run_local_experiments(matrices)

    # step 5: normalize against A* and plot
    print("\nnormalizing against A*...")
    df_normalized = normalize_against_astar_local(df_local)
    plot_normalized_local(df_normalized)

    print("\nALL DONE YAYYYYY!! check experiments/plots/ for all the figures :)")
