# Heiko Menzler
# heikogeorg.menzler@stud.uni-goettingen.de
#
# Date: 07.07.2022
import numpy as np
import networkx as nx

from numpy.typing import NDArray

def anderson_graph(conn: int = 3, size: int = 2**10) -> nx.Graph:
    """Create an infinite dimensional Anderson Graph."""
    return nx.random_regular_graph(conn, size)

def anderson_hamiltonian(conn_graph: nx.Graph, w: float = .3) -> NDArray:
    """Create an Anderson Hamiltonian matrix from a graph."""
    adj_matrix = nx.adjacency_matrix(conn_graph)
    anderson_noise = np.random.uniform(low=-w, high=w, size=len(conn_graph))

    return -adj_matrix + anderson_noise * np.identity(len(conn_graph))
