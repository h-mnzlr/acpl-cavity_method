# Heiko Menzler
# heikogeorg.menzler@stud.uni-goettingen.de
#
# Date: 12.07.2022
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import ArrayLike
from disordered_sys.methods import Population

def plot_limits(
    vals: ArrayLike,
    margin: float = .05,
    quant_margin: float = .005
) -> tuple[np.number, np.number]:
    vals = np.array(vals)
    lower, upper = np.quantile(vals, quant_margin), np.quantile(vals, 1 - quant_margin)
    length = upper - lower
    if length <= 0:
        larger_abs = max(np.abs(upper), np.abs(lower))
        length = larger_abs * .05
    return lower - length * margin, upper + length * margin

def show_complex_hist(vals: ArrayLike, axs=None):
    if axs is None:
        _, axs = plt.subplots(ncols=2)
    vals = np.array(vals)
    ax1, ax2 = axs
    ax1.set_xlim(*plot_limits(vals.real))
    ax2.set_xlim(*plot_limits(vals.imag))
    ax1.hist(vals.real, bins=int(len(vals) / 10), stacked=True, density=True)
    ax2.hist(vals.imag, bins=int(len(vals) / 10), stacked=True, density=True)

@dataclass
class InteractiveSimulationPlotter:
    update_every: int
    subplots: tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes]]

    def _draw(self, step: int, pop: Population) -> None:
        fig, (ax1, ax2, ax3) = self.subplots

        ax1.lines[0].set_ydata(self.diffs)
        ax1.lines[0].set_xdata(self.steps)
        ax1.set_ylim(plot_limits(self.diffs), emit=True)
        ax1.set_xlim(plot_limits(self.steps), emit=True)

        ax2.clear()
        ax3.clear()
        ax2.set_title("Population values (real)")
        ax3.set_title("Population values (imag)")
        show_complex_hist(pop, (ax2, ax3))

        fig.canvas.draw()
        fig.canvas.flush_events()

    def _first_call(self, pop: Population) -> None:
        self.real_lims = plot_limits(pop.real)
        self.imag_lims = plot_limits(pop.imag)
        self.old_pop = pop.copy()

    def call(self, step: int, pop: Population) -> None:
        if step % self.update_every != 0:
            return
        if step == 0:
            self._first_call(pop)
            return
        diff = np.sum(np.abs(pop - self.old_pop)
                      ) / self.update_every / len(pop)
        self.diffs.append(diff)
        self.steps.append(step)
        if step % self.plot_every == 0:
            self._draw(step, pop)
        self.old_pop = pop.copy()

    def __call__(self, step: int, pop: Population) -> None:
        self.call(step, pop)

    def __post_init__(self):
        _, (ax1, _, _) = self.subplots
        ax1.plot([1], [1], "r-")

        self.diffs: list[Population] = []
        self.steps: list[int] = []
        self.plot_every = self.update_every * 50
