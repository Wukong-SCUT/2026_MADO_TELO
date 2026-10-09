"""Generation-boundary pause/resume adapter for the local SepCMAES core.

This module is intentionally separate from ``SepCMAESOpt``.  The historical
wrapper remains a one-shot optimizer, while this adapter exposes the smallest
state contract needed to test persistent experts without changing the default
objective-split environment.
"""

from __future__ import annotations

import copy
from collections import defaultdict
import time
from typing import Dict

import numpy as np

from optimizers.cmaes.sepcmaes import SEPCMAES


class StatefulSepCMAES:
    """Advance SepCMAES by complete generations and resume it exactly."""

    STATE_SCHEMA_VERSION = 2

    def __init__(self, problem: Dict, options: Dict):
        self.problem = dict(problem)
        self.options = dict(options)
        if bool(self.options.get("is_restart", False)):
            raise ValueError("StatefulSepCMAES requires is_restart=False.")
        if self.options.get("mean") is None:
            raise ValueError("StatefulSepCMAES requires an explicit initial mean.")

        core_options = dict(self.options)
        core_options["is_restart"] = False
        # Tranche boundaries are controlled by advance_evaluations().  Keeping
        # the core limit infinite prevents a boundary from discarding the final
        # generation's distribution update.
        core_options["max_function_evaluations"] = np.inf
        core_options["max_runtime"] = np.inf
        self.core = SEPCMAES(self.problem, core_options)
        self._evaluate_initial_mean = bool(
            self.options.get("stateful_evaluate_initial_mean", False)
        )
        self._initial_mean_evaluated = False
        self._initialized = False
        self._z = None
        self._x = None
        self._mean = None
        self._s = None
        self._p = None
        self._c = None
        self._d = None
        self._y = None
        self._last_advance_best_x = None
        self._last_advance_best_y = np.inf

    @property
    def n_individuals(self) -> int:
        return int(self.core.n_individuals)

    @property
    def n_function_evaluations(self) -> int:
        return int(self.core.n_function_evaluations)

    @property
    def n_generations(self) -> int:
        return int(self.core._n_generations)

    def _initialize_core(self, evaluate_mean: bool) -> None:
        self.core.start_time = time.time()
        self.core.c_s = self.core.options.get(
            "c_s",
            (self.core._mu_eff + 2.0)
            / (
                self.core.ndim_problem
                + self.core._mu_eff
                + 3.0
            ),
        )
        self.core.c_cov = self.core.options.get(
            "c_cov", self.core._set_c_cov()
        )
        self.core.d_sigma = self.core.options.get(
            "d_sigma", self.core._set_d_sigma()
        )
        self.core._s_1 = 1.0 - self.core.c_s
        self.core._s_2 = np.sqrt(
            self.core._mu_eff
            * self.core.c_s
            * (2.0 - self.core.c_s)
        )

        lam = int(self.core.n_individuals)
        dim = int(self.core.ndim_problem)
        self._z = np.empty((lam, dim), dtype=np.float64)
        self._x = np.empty((lam, dim), dtype=np.float64)
        self._mean = self.core._initialize_mean(False)
        self._x[:] = self._mean
        self._s = np.zeros((dim,), dtype=np.float64)
        self._p = np.zeros((dim,), dtype=np.float64)
        self._c = np.asarray(
            self.core.options.get("c", np.ones((dim,))),
            dtype=np.float64,
        ).reshape(dim).copy()
        self._d = np.sqrt(self._c)
        self._y = np.empty((lam,), dtype=np.float64)
        if evaluate_mean:
            self._y[:] = self.core._evaluate_fitness(self._mean)
        else:
            self._y.fill(np.inf)
        self.core._list_initial_mean = [self._mean.copy()]
        self.core._n_generations = 0
        self.core.termination_signal = self.core.Terminations.NO_TERMINATION
        self._initial_mean_evaluated = bool(evaluate_mean)
        self._initialized = True

    def initialize(self) -> None:
        """Initialize distribution state under the configured FEs contract."""
        if self._initialized:
            return
        self._initialize_core(self._evaluate_initial_mean)

    def advance_generations(self, generations: int) -> Dict:
        """Advance by exactly ``generations`` complete population updates."""
        count = int(generations)
        if count < 0:
            raise ValueError("generations must be non-negative.")
        self.initialize()
        advance_best_x = None
        advance_best_y = np.inf
        for _ in range(count):
            self.core.start_time = time.time()
            self.core.termination_signal = self.core.Terminations.NO_TERMINATION
            self._z, self._x, self._y = self.core.iterate(
                self._z,
                self._x,
                self._mean,
                self._d,
                self._y,
            )
            if self._y is None:
                raise RuntimeError("SepCMAES stopped before completing a generation.")
            best_index = int(np.argmin(self._y))
            best_y = float(self._y[best_index])
            if best_y < advance_best_y:
                advance_best_y = best_y
                advance_best_x = np.asarray(
                    self._x[best_index], dtype=np.float64
                ).copy()
            (
                self._mean,
                self._s,
                self._p,
                self._c,
                self._d,
            ) = self.core._update_distribution(
                self._z,
                self._x,
                self._s,
                self._p,
                self._c,
                self._d,
                self._y,
                mean_old=self._mean,
            )
            self.core._n_generations += 1
        self._last_advance_best_x = advance_best_x
        self._last_advance_best_y = advance_best_y
        return self.result()

    def advance_evaluations(self, evaluations: int) -> Dict:
        """Advance by a budget divisible by the offspring population size."""
        budget = int(evaluations)
        if budget < 0:
            raise ValueError("evaluations must be non-negative.")
        if budget % self.n_individuals != 0:
            raise ValueError(
                "StatefulSepCMAES accepts only complete generations: "
                f"evaluations={budget}, n_individuals={self.n_individuals}."
            )
        return self.advance_generations(budget // self.n_individuals)

    def export_rng_state(self) -> Dict:
        """Return all random-generator states without other optimizer fields."""
        return {
            "rng": copy.deepcopy(self.core.rng.bit_generator.state),
            "rng_initialization": copy.deepcopy(
                self.core.rng_initialization.bit_generator.state
            ),
            "rng_optimization": copy.deepcopy(
                self.core.rng_optimization.bit_generator.state
            ),
        }

    def import_rng_state(self, state: Dict) -> None:
        """Restore random streams only; useful for a matched-RNG cold arm."""
        if not isinstance(state, dict):
            raise TypeError("RNG state must be a dictionary.")
        for key, generator in (
            ("rng", self.core.rng),
            ("rng_initialization", self.core.rng_initialization),
            ("rng_optimization", self.core.rng_optimization),
        ):
            if key not in state:
                raise ValueError(f"RNG state is missing {key}.")
            generator.bit_generator.state = copy.deepcopy(state[key])

    def recenter(self, target, max_shift: float = np.inf) -> Dict:
        """Move the search mean toward ``target`` without resetting shape/path."""
        self.initialize()
        target_array = self._require_finite(
            "recenter_target",
            target,
            (int(self.core.ndim_problem),),
        )
        target_array = np.clip(
            target_array,
            np.asarray(self.core.lower_boundary, dtype=np.float64),
            np.asarray(self.core.upper_boundary, dtype=np.float64),
        )
        limit = float(max_shift)
        if np.isnan(limit) or limit < 0.0:
            raise ValueError("recenter max_shift must be non-negative.")
        requested = target_array - self._mean
        requested_norm = float(np.linalg.norm(requested))
        applied = requested.copy()
        if np.isfinite(limit) and requested_norm > limit > 0.0:
            applied *= limit / requested_norm
        elif limit == 0.0:
            applied.fill(0.0)
        new_mean = np.clip(
            self._mean + applied,
            np.asarray(self.core.lower_boundary, dtype=np.float64),
            np.asarray(self.core.upper_boundary, dtype=np.float64),
        )
        applied = new_mean - self._mean
        self._mean = new_mean
        self.core.mean = self._mean.copy()
        return {
            "requested_norm": requested_norm,
            "applied_norm": float(np.linalg.norm(applied)),
            "remaining_norm": float(np.linalg.norm(target_array - self._mean)),
        }

    def _metadata(self) -> Dict:
        return {
            "dimension": int(self.core.ndim_problem),
            "n_individuals": int(self.core.n_individuals),
            "n_parents": int(self.core.n_parents),
            "lower_boundary": np.asarray(
                self.core.lower_boundary, dtype=np.float64
            ).copy(),
            "upper_boundary": np.asarray(
                self.core.upper_boundary, dtype=np.float64
            ).copy(),
            "c_c": float(self.core.c_c),
            "c_s": float(self.core.c_s),
            "c_cov": float(self.core.c_cov),
            "d_sigma": float(self.core.d_sigma),
            "optimizer_guide_numeric_guard": bool(
                self.core.optimizer_guide_numeric_guard
            ),
            "optimizer_numeric_counter_enable": bool(
                self.core.numeric_telemetry.counters_enabled
            ),
            "optimizer_guide_sigma_exp_clip": float(
                self.core.optimizer_guide_sigma_exp_clip
            ),
            "optimizer_guide_sigma_clip_ratio": float(
                self.core.optimizer_guide_sigma_clip_ratio
            ),
            "initial_mean_evaluated": bool(
                self._initial_mean_evaluated
            ),
        }

    def _assert_exportable_runtime_state(self) -> None:
        dim = int(self.core.ndim_problem)
        lam = int(self.core.n_individuals)
        for name, value, shape in (
            ("z", self._z, (lam, dim)),
            ("x", self._x, (lam, dim)),
            ("mean", self._mean, (dim,)),
            ("s", self._s, (dim,)),
            ("p", self._p, (dim,)),
            ("c", self._c, (dim,)),
            ("d", self._d, (dim,)),
            ("y", self._y, (lam,)),
        ):
            arr = np.asarray(value, dtype=np.float64)
            if tuple(arr.shape) != tuple(shape) or not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"Cannot export invalid StatefulSepCMAES field {name}."
                )
        sigma = float(self.core.sigma)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(
                "Cannot export StatefulSepCMAES with non-finite or "
                "non-positive sigma."
            )
        if np.any(np.asarray(self._c) <= 0.0) or np.any(
            np.asarray(self._d) <= 0.0
        ):
            raise ValueError(
                "Cannot export StatefulSepCMAES with non-positive "
                "covariance diagonal."
            )
        if self.core.optimizer_guide_numeric_guard:
            span = np.asarray(
                self.core.upper_boundary, dtype=np.float64
            ) - np.asarray(self.core.lower_boundary, dtype=np.float64)
            axis_max = (
                float(self.core.optimizer_guide_sigma_clip_ratio) * span
            )
            effective_axis_std = sigma * np.asarray(
                self._d, dtype=np.float64
            )
            if (
                not np.all(np.isfinite(effective_axis_std))
                or np.any(effective_axis_std > axis_max + 1e-12)
            ):
                raise ValueError(
                    "Cannot export StatefulSepCMAES with an effective "
                    "axis scale above the search-domain cap."
                )

    def export_state(self) -> Dict:
        """Export a deep, runtime-independent generation-boundary snapshot."""
        self.initialize()
        self._assert_exportable_runtime_state()
        return {
            "schema_version": int(self.STATE_SCHEMA_VERSION),
            "metadata": self._metadata(),
            "arrays": {
                "z": np.asarray(self._z, dtype=np.float64).copy(),
                "x": np.asarray(self._x, dtype=np.float64).copy(),
                "mean": np.asarray(self._mean, dtype=np.float64).copy(),
                "s": np.asarray(self._s, dtype=np.float64).copy(),
                "p": np.asarray(self._p, dtype=np.float64).copy(),
                "c": np.asarray(self._c, dtype=np.float64).copy(),
                "d": np.asarray(self._d, dtype=np.float64).copy(),
                "y": np.asarray(self._y, dtype=np.float64).copy(),
            },
            "progress": {
                "n_function_evaluations": int(
                    self.core.n_function_evaluations
                ),
                "termination_signal": int(self.core.termination_signal),
                "time_function_evaluations": float(
                    self.core.time_function_evaluations
                ),
                "n_generations": int(self.core._n_generations),
                "best_so_far_x": (
                    None
                    if self.core.best_so_far_x is None
                    else np.asarray(
                        self.core.best_so_far_x, dtype=np.float64
                    ).copy()
                ),
                "best_so_far_y": float(self.core.best_so_far_y),
                "sigma": float(self.core.sigma),
                "counter_early_stopping": int(
                    self.core._counter_early_stopping
                ),
                "base_early_stopping": float(
                    self.core._base_early_stopping
                ),
                "list_fitness": copy.deepcopy(self.core._list_fitness),
                "list_initial_mean": [
                    np.asarray(value, dtype=np.float64).copy()
                    for value in self.core._list_initial_mean
                ],
                "list_generations": copy.deepcopy(
                    self.core._list_generations
                ),
                "n_restart": int(self.core._n_restart),
                "last_advance_best_x": (
                    None
                    if self._last_advance_best_x is None
                    else np.asarray(
                        self._last_advance_best_x, dtype=np.float64
                    ).copy()
                ),
                "last_advance_best_y": float(self._last_advance_best_y),
            },
            "rng": self.export_rng_state(),
            "numeric_telemetry": {
                "records": copy.deepcopy(self.core.numeric_telemetry.records),
                "counters": dict(self.core.numeric_telemetry.counters),
            },
        }

    @staticmethod
    def _require_finite(name: str, value, shape) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if tuple(arr.shape) != tuple(shape):
            raise ValueError(
                f"State field {name} has shape {arr.shape}, expected {shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"State field {name} contains NaN or infinity.")
        return arr.copy()

    @staticmethod
    def _same_float(left: float, right: float) -> bool:
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-15))

    def import_state(self, state: Dict) -> None:
        """Restore a snapshot without evaluating the objective."""
        if not isinstance(state, dict):
            raise TypeError("Optimizer state must be a dictionary.")
        if int(state.get("schema_version", -1)) != self.STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported StatefulSepCMAES state schema.")

        if not self._initialized:
            self._initialize_core(False)
        metadata = state.get("metadata", {})
        expected = self._metadata()
        for key in ("dimension", "n_individuals", "n_parents"):
            if int(metadata.get(key, -1)) != int(expected[key]):
                raise ValueError(f"State metadata mismatch for {key}.")
        for key in ("lower_boundary", "upper_boundary"):
            restored = np.asarray(metadata.get(key), dtype=np.float64)
            if restored.shape != expected[key].shape or not np.array_equal(
                restored, expected[key]
            ):
                raise ValueError(f"State metadata mismatch for {key}.")
        for key in (
            "c_c",
            "c_s",
            "c_cov",
            "d_sigma",
            "optimizer_guide_sigma_exp_clip",
            "optimizer_guide_sigma_clip_ratio",
        ):
            if key not in metadata or not self._same_float(
                metadata[key], expected[key]
            ):
                raise ValueError(f"State metadata mismatch for {key}.")
        for key in (
            "optimizer_guide_numeric_guard",
            "optimizer_numeric_counter_enable",
        ):
            if key not in metadata or bool(metadata[key]) != bool(expected[key]):
                raise ValueError(f"State metadata mismatch for {key}.")
        state_initial_eval = bool(
            metadata.get("initial_mean_evaluated", False)
        )
        if state_initial_eval != bool(
            self.options.get("stateful_evaluate_initial_mean", False)
        ):
            raise ValueError(
                "State metadata mismatch for initial_mean_evaluated."
            )

        dim = int(self.core.ndim_problem)
        lam = int(self.core.n_individuals)
        arrays = state.get("arrays", {})
        restored_arrays = {
            "z": self._require_finite("z", arrays.get("z"), (lam, dim)),
            "x": self._require_finite("x", arrays.get("x"), (lam, dim)),
            "mean": self._require_finite("mean", arrays.get("mean"), (dim,)),
            "s": self._require_finite("s", arrays.get("s"), (dim,)),
            "p": self._require_finite("p", arrays.get("p"), (dim,)),
            "c": self._require_finite("c", arrays.get("c"), (dim,)),
            "d": self._require_finite("d", arrays.get("d"), (dim,)),
            "y": self._require_finite("y", arrays.get("y"), (lam,)),
        }
        progress = state.get("progress", {})
        sigma = float(progress.get("sigma", np.nan))
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("State sigma must be finite and positive.")
        if np.any(restored_arrays["c"] <= 0.0) or np.any(
            restored_arrays["d"] <= 0.0
        ):
            raise ValueError("State covariance diagonal must be positive.")
        if bool(expected["optimizer_guide_numeric_guard"]):
            span = np.asarray(
                expected["upper_boundary"], dtype=np.float64
            ) - np.asarray(expected["lower_boundary"], dtype=np.float64)
            axis_max = (
                float(expected["optimizer_guide_sigma_clip_ratio"]) * span
            )
            effective_axis_std = sigma * restored_arrays["d"]
            if (
                not np.all(np.isfinite(effective_axis_std))
                or np.any(effective_axis_std > axis_max + 1e-12)
            ):
                raise ValueError(
                    "State effective axis scale exceeds the "
                    "search-domain cap."
                )

        best_x_raw = progress.get("best_so_far_x")
        best_x = None
        if best_x_raw is not None:
            best_x = self._require_finite(
                "best_so_far_x", best_x_raw, (dim,)
            )
        best_y = float(progress.get("best_so_far_y", np.inf))
        if np.isnan(best_y):
            raise ValueError("State best_so_far_y cannot be NaN.")
        last_advance_x_raw = progress.get("last_advance_best_x")
        last_advance_x = None
        if last_advance_x_raw is not None:
            last_advance_x = self._require_finite(
                "last_advance_best_x",
                last_advance_x_raw,
                (dim,),
            )
        last_advance_y = float(
            progress.get("last_advance_best_y", np.inf)
        )
        if np.isnan(last_advance_y):
            raise ValueError("State last_advance_best_y cannot be NaN.")
        if (last_advance_x is None) != (not np.isfinite(last_advance_y)):
            raise ValueError(
                "State last advance best coordinate/value mismatch."
            )
        n_evals = int(progress.get("n_function_evaluations", -1))
        n_generations = int(progress.get("n_generations", -1))
        if n_evals < 0 or n_generations < 0:
            raise ValueError("State progress counters must be non-negative.")
        termination_signal = int(progress.get("termination_signal", -1))
        try:
            restored_termination = self.core.Terminations(
                termination_signal
            )
        except ValueError as exc:
            raise ValueError(
                "State termination signal is invalid."
            ) from exc
        expected_evals = n_generations * lam + int(state_initial_eval)
        if n_evals != expected_evals:
            raise ValueError(
                "State FEs/generation mismatch: "
                f"{n_evals} != {n_generations} * {lam} "
                f"+ {int(state_initial_eval)}."
            )

        self._z = restored_arrays["z"]
        self._x = restored_arrays["x"]
        self._mean = restored_arrays["mean"]
        self._s = restored_arrays["s"]
        self._p = restored_arrays["p"]
        self._c = restored_arrays["c"]
        self._d = restored_arrays["d"]
        self._y = restored_arrays["y"]
        self.core.mean = self._mean.copy()
        self.core.sigma = sigma
        self.core.n_function_evaluations = n_evals
        self.core.termination_signal = restored_termination
        self.core.time_function_evaluations = float(
            progress.get("time_function_evaluations", 0.0)
        )
        self.core._n_generations = n_generations
        self.core.best_so_far_x = None if best_x is None else best_x.copy()
        self.core.best_so_far_y = best_y
        self.core._counter_early_stopping = int(
            progress.get("counter_early_stopping", 0)
        )
        self.core._base_early_stopping = float(
            progress.get("base_early_stopping", best_y)
        )
        self.core._list_fitness = copy.deepcopy(
            progress.get("list_fitness", [best_y])
        )
        self.core._list_initial_mean = [
            self._require_finite(
                "list_initial_mean",
                value,
                (dim,),
            )
            for value in progress.get("list_initial_mean", [self._mean])
        ]
        self.core._list_generations = [
            int(value) for value in progress.get("list_generations", [])
        ]
        self.core._n_restart = int(progress.get("n_restart", 0))
        self._last_advance_best_x = (
            None if last_advance_x is None else last_advance_x.copy()
        )
        self._last_advance_best_y = last_advance_y
        self.import_rng_state(state.get("rng", {}))

        telemetry = state.get("numeric_telemetry", {})
        self.core.numeric_telemetry.records = copy.deepcopy(
            telemetry.get("records", [])
        )
        self.core.numeric_telemetry.counters = defaultdict(
            int,
            {
                str(key): int(value)
                for key, value in telemetry.get("counters", {}).items()
            },
        )
        self.core.runtime = 0.0
        self.core.start_time = time.time()
        self._initial_mean_evaluated = state_initial_eval
        self._initialized = True

    def result(self) -> Dict:
        """Return a detached summary without mutating optimizer state."""
        self.initialize()
        return {
            "best_so_far_x": (
                None
                if self.core.best_so_far_x is None
                else np.asarray(
                    self.core.best_so_far_x, dtype=np.float64
                ).copy()
            ),
            "best_so_far_y": float(self.core.best_so_far_y),
            "advance_best_x": (
                None
                if self._last_advance_best_x is None
                else np.asarray(
                    self._last_advance_best_x, dtype=np.float64
                ).copy()
            ),
            "advance_best_y": float(self._last_advance_best_y),
            "n_function_evaluations": int(
                self.core.n_function_evaluations
            ),
            "n_generations": int(self.core._n_generations),
            "mean": np.asarray(self._mean, dtype=np.float64).copy(),
            "sigma": float(self.core.sigma),
            "s": np.asarray(self._s, dtype=np.float64).copy(),
            "p": np.asarray(self._p, dtype=np.float64).copy(),
            "c": np.asarray(self._c, dtype=np.float64).copy(),
            "d": np.asarray(self._d, dtype=np.float64).copy(),
            "x": np.asarray(self._x, dtype=np.float64).copy(),
            "y": np.asarray(self._y, dtype=np.float64).copy(),
            "time_function_evaluations": float(
                self.core.time_function_evaluations
            ),
        }
