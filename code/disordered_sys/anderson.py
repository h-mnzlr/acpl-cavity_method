# Heiko Menzler
# heikogeorg.menzler@stud.uni-goettingen.de
#
# Date: 07.07.2022
import numpy as np
import networkx as nx

from numpy.typing import NDArray

def anderson_graph(conn: int = 3, size: int = 2**10, seed=None) -> nx.Graph:
    """Create an infinite dimensional Anderson Graph."""
    if seed is None:
        seed = np.random.default_rng()
    return nx.random_regular_graph(conn, size, seed=seed)

def anderson_hamiltonian(
    conn_graph: nx.Graph,
    w: float = .3,
) -> NDArray[np.float64]:
    """Create an Anderson Hamiltonian matrix from a graph."""
    adj_matrix = nx.adjacency_matrix(conn_graph)
    anderson_noise = np.random.uniform(low=-w, high=w, size=len(conn_graph))

    return -adj_matrix + anderson_noise * np.identity(len(conn_graph))

def anderson_cavity_eqn(
    samples: NDArray[np.float64] | np.float64,
    eigv: float,
    impurity_w: float,
    epsilon: float,
    rng: np.random.Generator = np.random.default_rng()
):
    """Implements the cavity equation for the Anderson model."""
    e = rng.uniform(low=-impurity_w, high=impurity_w)
    return 1j * (eigv - 1j * epsilon - e) + np.sum(1 / samples)
