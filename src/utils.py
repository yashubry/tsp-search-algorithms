from __future__ import annotations
import numpy as np
from typing import List

##hiii codereviewer! this file here are basically the helper functions that i made to help me thruout this project :)
#i also included the helpers that i made but were faulty or that i didnt reLLY use, just to show my thought process and how i iterated on my code
#i hope this is helpful for you to understand how i approached the problem and how i debugged my code.
def load_matrix(path: str) -> np.ndarray:
    """Load adjacency matrix from whitespace-delimited file."""
    return np.loadtxt(path)


# makes sure the route ends where it started
def return_to_start(route):
    # if route is empty, just return it
    if len(route) == 0:
        return route

    # if it does not already end at the first city,
    # add the first city to the end
    if route[-1] != route[0]:
        route= route +[route[0]]

    return route


# ok this func basically adds up the distances along the route using the matrix :D
def get_cost(route,mat):
    total =0.0 #i was advised to make this a float sice the distances in the matrix are floats, and we wanna float result no rouding ahah help me :()
    for i in range(len(route) -1):
        total += mat[route[i],route[i+ 1]]
    return total