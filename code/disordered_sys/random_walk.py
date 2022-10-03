# Heiko Menzler
# heikogeorg.menzler@stud.uni-goettingen.de
#
# Date: 16.09.2022
"""
This module implements a self-avoiding random walk on a graph.

Note to Grader:
Only the self_avoiding_random_walk_backtracking function is used in the 
algorithm but the other functions highlight how the algorithms were
developed and are set up.
"""

from typing import Iterable, Optional

import numpy as np
import networkx as nx

def random_walk(
    g: nx.Graph,
    steps: int,
    initial: Optional[int] = None,
) -> Iterable[int]:
    """Generate a random walk over a graph."""
    # if no initial node is given generate one automatically
    if initial is None:
        initial = np.random.choice(list(nx.nodes(g)))
    yield initial  # type: ignore

    node_idx = initial
    for _ in range(steps):
        current_node = g[node_idx]
        available_nodes = list(current_node.keys())
        node_idx = np.random.choice(available_nodes)
        yield node_idx

def self_avoiding_random_walk(
    g: nx.Graph,
    initial: Optional[int] = None,
) -> Iterable[int]:
    """Generate a random walk over a graph, avoiding nodes already visited."""
    if initial is None:
        all_nodes = list(nx.nodes(g))
        initial = np.random.choice(all_nodes)
    yield initial  # type: ignore

    visited = set()

    node_idx = initial
    while True:

        current_node = g[node_idx]
        available_nodes = set(current_node.keys())
        available_nodes -= visited

        if not available_nodes: break

        node_idx = np.random.choice(list(available_nodes))
        visited.add(node_idx)

        yield node_idx

def self_avoiding_random_walk_backtracking(
    g: nx.Graph,
    initial: Optional[int] = None,
) -> Iterable[int]:
    """Generate a random walk over all nodes of a graph, avoiding nodes already visited."""
    if initial is None:
        all_nodes = list(nx.nodes(g))
        initial = np.random.choice(all_nodes)
    yield initial  # type: ignore

    visited = {initial}
    path = [initial]

    node_idx = initial
    while len(path) > 0:

        current_node = g[node_idx]
        available_nodes = set(current_node.keys())
        available_nodes -= visited

        if not available_nodes:
            node_idx = path.pop()
            continue

        node_idx = np.random.choice(list(available_nodes))
        visited.add(node_idx)
        path.append(node_idx)

        yield node_idx

# the rest is just debugging
def _main():
    for n in self_avoiding_random_walk_backtracking(
        nx.random_regular_graph(3, 10)
    ):
        print(n)

if __name__ == "__main__":
    _main()
