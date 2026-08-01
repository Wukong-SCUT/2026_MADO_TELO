import time
from typing import Dict

import numpy as np

from optimizers.cmaes.sepcmaes import SEPCMAES


class SepCMAESOpt:
    """
    Wrapper that adapts the bundled SepCMAES implementation to unified_opt.
    """

    def __init__(self, problem: Dict, options: Dict):
        self.problem = problem
        self.options = options

        self.fitness_function = problem["fitness_function"]
        self.ndim_problem = int(problem["ndim_problem"])
        self.lower_boundary = np.asarray(problem["lower_boundary"], dtype=np.float64).reshape(-1)
        self.upper_boundary = np.asarray(problem["upper_boundary"], dtype=np.float64).reshape(-1)

        self.max_function_evaluations = int(options.get("max_function_evaluations", 0))
        self.seed_rng = int(options.get("seed_rng", 0))
        self.mean = np.asarray(options["mean"], dtype=np.float64).reshape(-1)
        self.sigma = float(options["sigma"])
        self.n_individuals = int(options.get("n_individuals", 4 + int(3 * np.log(max(2, self.ndim_problem)))))
        self.n_parents = int(options.get("n_parents", max(1, self.n_individuals // 2)))
        self.verbose = bool(options.get("verbose", False))
        self.nonfinite_penalty = float(options.get("nonfinite_penalty", 1e300))
        self.cov_lr_scale = float(options.get("cov_lr_scale", 1.0))
        self.c_s_scale = float(options.get("c_s_scale", 1.0))
        self.use_custom_learning_rates = bool(options.get("use_custom_learning_rates", True))

    @staticmethod
    def _mu_eff(n_individuals: int, n_parents: int) -> float:
        w = np.log((float(n_individuals) + 1.0) / 2.0) - np.log(np.arange(n_individuals, dtype=np.float64) + 1.0)
        w_pos = w[:n_parents]
        return float((np.sum(w_pos) ** 2) / np.sum(np.square(w_pos)))

    def _fitness_batch(self, x_batch: np.ndarray):
        x = np.asarray(x_batch, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x = np.clip(x, self.lower_boundary, self.upper_boundary)
        y = self.fitness_function(x)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        y = np.nan_to_num(
            y,
            nan=self.nonfinite_penalty,
            posinf=self.nonfinite_penalty,
            neginf=-self.nonfinite_penalty,
        )
        return y

    def optimize(self):
        start_time = time.time()
        sep_options = {
            "max_function_evaluations": int(self.max_function_evaluations),
            "seed_rng": int(self.seed_rng),
            "mean": self.mean.copy(),
            "sigma": float(self.sigma),
            "n_individuals": int(self.n_individuals),
            "n_parents": int(self.n_parents),
            "verbose": int(self.verbose),
            "is_restart": False,
        }
        if self.use_custom_learning_rates:
            mu_eff = self._mu_eff(self.n_individuals, self.n_parents)
            c_s_default = float((mu_eff + 2.0) / (self.ndim_problem + mu_eff + 3.0))
            c_cov_default = float(
                ((1.0 / mu_eff) * (2.0 / np.power(self.ndim_problem + np.sqrt(2.0), 2))
                 + (1.0 - 1.0 / mu_eff)
                 * min(1.0, (2.0 * mu_eff - 1.0) / (np.power(self.ndim_problem + 2.0, 2) + mu_eff)))
                * ((self.ndim_problem + 2.0) / 3.0)
            )
            c_cov = float(max(1e-12, c_cov_default * max(1e-6, self.cov_lr_scale)))
            c_s = float(max(1e-12, c_s_default * max(1e-6, self.c_s_scale)))
            sep_options.update({"c_cov": float(c_cov), "c_s": float(c_s)})

        opt = SEPCMAES(
            problem={
                "fitness_function": self._fitness_batch,
                "ndim_problem": int(self.ndim_problem),
                "lower_boundary": self.lower_boundary,
                "upper_boundary": self.upper_boundary,
            },
            options=sep_options,
        )
        res = opt.optimize()
        return {
            "best_so_far_x": np.asarray(res["best_so_far_x"], dtype=np.float64).copy(),
            "best_so_far_y": float(res["best_so_far_y"]),
            "n_function_evaluations": int(min(res["n_function_evaluations"], self.max_function_evaluations)),
            "time_function_evaluations": float(res.get("time_function_evaluations", time.time() - start_time)),
            "x": np.asarray(res.get("best_so_far_x", self.mean), dtype=np.float64).reshape(-1).copy(),
            "y": np.asarray([res.get("best_so_far_y", np.inf)], dtype=np.float64),
            "mean": np.asarray(res.get("best_so_far_x", self.mean), dtype=np.float64).reshape(-1).copy(),
            "sigma": float(self.sigma),
        }
