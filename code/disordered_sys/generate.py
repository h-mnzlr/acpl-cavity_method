# Heiko Menzler
# heikogeorg.menzler@stud.uni-goettingen.de
#
# Date: 20.06.2022

import numpy as np
import networkx as nx

from numpy.typing import NDArray

def generate_j(c: int, N: int) -> NDArray[np.float64]:
    """Generate a J matrix as specified in exercise (2.1)."""
    random_nums = np.random.normal(
        0, 1 / np.sqrt(c), size=int(N * (N + 1) / 2)
    )
    j = np.zeros(shape=(N, N))
    for i in range(N):
        t = i * (i + 1) // 2
        j[i, 0:i + 1] = random_nums[t:t + i + 1]
        j[0:i, i] = random_nums[t:t + i]
    return j

def generate_m(c: int, N: int) -> NDArray[np.float64]:
    """Generate a M matrix as specified in exercise (2.1)."""
    j = generate_j(c, N)

    graph = nx.random_regular_graph(c, N)
    a = nx.to_numpy_array(graph)
    return (a * j).astype(np.float64)

def generate_m_full(N: int) -> NDArray[np.float64]:
    """Generate a fully connected M matrix as specified in exercise (2.1)."""
    return generate_j(N - 1, N)
