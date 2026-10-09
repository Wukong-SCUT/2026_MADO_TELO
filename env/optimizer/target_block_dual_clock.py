"""Fixed generation-synchronous dual-clock plan for NT074.

The slow clock is one objective-split environment action.  Its requested
SepCMAES budget is decomposed into complete-generation fast ticks.  Every fast
tick advances each persistent local optimizer by exactly one generation and
then runs one target-block communication/commit event.
"""

from dataclasses import dataclass

import numpy as np


TARGET_BLOCK_DUAL_CLOCK_STATE_VERSION = 3


@dataclass(frozen=True)
class TargetBlockDualClockPlan:
    microcycles: int
    requested_fes_per_agent: np.ndarray
    generation_fes_per_agent: np.ndarray


def build_target_block_dual_clock_plan(
    *,
    requested_fes_per_agent,
    population_per_agent,
) -> TargetBlockDualClockPlan:
    """Build the fixed NT074 plan or fail closed.

    The first implementation deliberately exposes no cadence parameter.  A
    slow action must contain at least two complete generations for every
    agent, and all agents must have the same generation count so that graph
    observation and cooperative commit remain synchronous.
    """

    requested = np.asarray(
        requested_fes_per_agent,
        dtype=np.int64,
    ).reshape(-1)
    populations = np.asarray(
        population_per_agent,
        dtype=np.int64,
    ).reshape(-1)
    if requested.size == 0 or populations.shape != requested.shape:
        raise ValueError(
            "Dual-clock requested budgets and populations must be aligned "
            "non-empty per-agent vectors."
        )
    if np.any(requested <= 0) or np.any(populations <= 0):
        raise ValueError(
            "Dual-clock budgets and population sizes must be positive."
        )
    if np.any(requested % populations != 0):
        raise ValueError(
            "Dual-clock local budgets must contain complete optimizer "
            "generations."
        )
    generations = requested // populations
    if np.any(generations != generations[0]):
        raise ValueError(
            "Dual-clock agents must expose the same number of complete "
            "generations per slow action."
        )
    microcycles = int(generations[0])
    if microcycles < 2:
        raise ValueError(
            "Dual-clock mode requires at least two complete generations per "
            "slow action; one generation is the NT073 single-clock case."
        )
    return TargetBlockDualClockPlan(
        microcycles=microcycles,
        requested_fes_per_agent=requested.copy(),
        generation_fes_per_agent=populations.copy(),
    )
