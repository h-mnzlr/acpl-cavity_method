# Heiko Menzler
# heikogeorg.menzler@stud.uni-goettingen.de
#
# Date: 12.07.2022
from functools import partial

import numpy as np

from numpy.typing import NDArray
from typing import Callable, Iterable, Any

# Type definitions
ProgressBar = Callable[[Iterable], Iterable]

Configuration = Any
ForwardStepper = Callable[[Configuration], Configuration]

Population = NDArray[np.complex64]
PopulationEnsemble = list[Population]
UpdateEqn = Callable[[Population], Population]
PopulationStepper = Callable[[Population], Population | None]

identity_iterator = lambda x: (yield from x)

def forward_step_runner(
    initial_conf: Configuration,
    forward: ForwardStepper,
    max_steps: int,
    callback: None | Callable[[int, Configuration], None] = None,
    pbar: None | ProgressBar = None
) -> PopulationEnsemble:
    """Main function of a simple simulation consisting of only forward steps."""
    if pbar is None:
        pbar = identity_iterator

    configuration = initial_conf
    conf_ensemble = []

    for step in pbar(range(max_steps)):
        forward(configuration)
        conf_ensemble.append(configuration)
        if callback is not None:
            callback(step, configuration)
    return conf_ensemble

def population_step(
    pop: Population,
    update_eqn: UpdateEqn,
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

def spectral_density(
    marginal_precisions: list[NDArray[np.float64]]
) -> list[np.float64]:
    """Calculate spectral density from distributions of marginal distributions."""
    spectral_densities = []
    for marginals in marginal_precisions:
        G_ii = 1j / marginals
        rho_lambda = G_ii.imag.mean() / np.pi
        spectral_densities.append(rho_lambda)
    return spectral_densities

def full_population_marginal(
    initial_pop: Population,
    update_eqn: UpdateEqn,
    c: int,
    rng: np.random.Generator = np.random.default_rng()
) -> PopulationEnsemble:
    cavity_pop_stepper = partial(
        population_step, update_eqn=update_eqn, c=c - 1, rng=rng
    )
    cavity_pop_ens = forward_step_runner(
        initial_pop, cavity_pop_stepper, max_steps=100_000
    )

    marginals_pop_stepper = partial(
        population_step, update_eqn=update_eqn, c=c, rng=rng
    )
    marginals_pop_ens = forward_step_runner(
        cavity_pop_ens[-1], marginals_pop_stepper, max_steps=12_000
    )
    return marginals_pop_ens
