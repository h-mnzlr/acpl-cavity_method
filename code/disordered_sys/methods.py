# Heiko Menzler
# heikogeorg.menzler@stud.uni-goettingen.de
#
# Date: 12.07.2022

import numba
import numpy as np

from numpy.typing import NDArray
from typing import Callable, Iterable, Any

# Type definitions
ProgressBar = Callable[[Iterable], Iterable]

Configuration = Any
ForwardStepper = Callable[[Configuration], Configuration]

Population = NDArray[np.float64]
PopulationEnsemble = list[Population]
UpdateEqn = Callable[[Population], Population]
PopulationStepper = Callable[[Population], Population | None]

identity_iterator = lambda x: (yield from x)

def forward_step_runner(
    initial_conf: Configuration,
    forward: ForwardStepper,
    max_steps: int,
    pbar: None | ProgressBar = None
) -> PopulationEnsemble:
    """Main function of a simple simulation consisting of only forward steps."""
    if pbar is None:
        pbar = identity_iterator

    configuration = initial_conf
    conf_ensemble = []

    for _ in pbar(range(max_steps)):
        forward(configuration)
        conf_ensemble.append(configuration)
    return conf_ensemble

@numba.njit
def population_step(
    update_eqn: UpdateEqn,
    pop: Population,
    c: int,
    in_place: bool = True,
    rng: np.random.Generator = np.random.default_rng()
) -> Population:
    """General stepper for the population step method."""
    if not in_place:
        pop = pop.copy()
    n_p = len(pop)
    population_idxs = rng.integers(low=0, high=n_p, size=c - 1)
    change_idx = rng.integers(low=0, high=n_p)
    sample_values = pop.take(population_idxs[:-1])
    pop[change_idx] = update_eqn(sample_values)
    return pop
