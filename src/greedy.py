from __future__ import annotations
import numpy as np
import random
from typing import List, Optional

from src.utils import ensure_cycle, route_cost


def nearest_neighbor(mat: np.ndarray, start: int = 0) -> List[int]:
    n = mat.shape[0]
    unvisited = set(range(n))
    unvisited.remove(start)

    route = [start]
    cur = start

    while unvisited:
        nxt = min(unvisited, key=lambda j: mat[cur, j])
        route.append(nxt)
        unvisited.remove(nxt)
        cur = nxt

    return ensure_cycle(route)


def two_opt(mat: np.ndarray, route: List[int]) -> List[int]:
    """
    Standard 2-opt improvement: repeatedly reverse segments that improve the tour.
    Input route should be a cycle (start == end). Output is also a cycle.
    """
    best = route[:]
    best_cost = route_cost(best, mat)
    n = len(best) - 1  # last city equals first

    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                if k - i <= 1:
                    continue
                candidate = best[:i] + best[i:k][::-1] + best[k:]
                candidate[-1] = candidate[0]
                c_cost = route_cost(candidate, mat)
                if c_cost + 1e-12 < best_cost:
                    best, best_cost = candidate, c_cost
                    improved = True
                    break
            if improved:
                break

    return best


def nearest_neighbor_2opt(mat: np.ndarray, start: int = 0) -> List[int]:
    return two_opt(mat, nearest_neighbor(mat, start=start))


def rrnn(
    mat: np.ndarray,
    k: int,
    num_repeats: int,
    start: int = 0,
    seed: Optional[int] = None,
) -> List[int]:
    """
    Repeated Random Nearest Neighbor:
    - At each step, randomly choose the next city from the k closest unvisited cities.
    - Repeat num_repeats times, apply 2-opt, return best tour found.
    """
    n = mat.shape[0]
    rng = random.Random(seed)

    k = max(1, min(k, n - 1))
    num_repeats = max(1, num_repeats)

    best_route: Optional[List[int]] = None
    best_cost = float("inf")

    for _ in range(num_repeats):
        unvisited = set(range(n))
        unvisited.remove(start)

        route = [start]
        cur = start

        while unvisited:
            ordered = sorted(unvisited, key=lambda j: mat[cur, j])
            choices = ordered[:k]
            nxt = rng.choice(choices)
            route.append(nxt)
            unvisited.remove(nxt)
            cur = nxt

        route = ensure_cycle(route)
        route = two_opt(mat, route)
        c = route_cost(route, mat)

        if c < best_cost:
            best_cost = c
            best_route = route

    assert best_route is not None
    return best_route