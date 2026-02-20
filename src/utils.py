from __future__ import annotations
import numpy as np
from typing import List


def load_matrix(path: str) -> np.ndarray:
    """Load adjacency matrix from whitespace-delimited file."""
    return np.loadtxt(path)


def ensure_cycle(route: List[int]) -> List[int]:
    if not route:
        return route
    return route if route[0] == route[-1] else route + [route[0]]


def route_cost(route: List[int], mat: np.ndarray) -> float:
    cost = 0.0
    for i in range(len(route) - 1):
        cost += mat[route[i], route[i + 1]]
    return float(cost)