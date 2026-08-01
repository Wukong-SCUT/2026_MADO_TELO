import time
from typing import Dict

import numpy as np

from optimizers.cmaes.cmaes import CMAES


class CMAESOpt:
    """
    Wrapper that adapts the bundled CMAES implementation to unified_opt.
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
        cma_options = {
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
            c_s_default = float((mu_eff + 2.0) / (self.ndim_problem + mu_eff + 5.0))
            alpha_cov = 2.0
            c_1_default = float(alpha_cov / (np.square(self.ndim_problem + 1.3) + mu_eff))
            c_w_default = float(
                min(
                    1.0 - c_1_default,
                    alpha_cov * (0.25 + mu_eff + 1.0 / mu_eff - 2.0) / (np.square(self.ndim_problem + 2.0) + alpha_cov * mu_eff / 2.0),
                )
            )
            cov_scale = float(max(1e-6, self.cov_lr_scale))
            c_s_scale = float(max(1e-6, self.c_s_scale))
            c_1 = float(c_1_default * cov_scale)
            c_w = float(c_w_default * cov_scale)
            if (c_1 + c_w) > 0.9:
                ratio = 0.9 / max(1e-12, c_1 + c_w)
                c_1 *= ratio
                c_w *= ratio
            c_s = float(c_s_default * c_s_scale)
            cma_options.update({"c_1": float(c_1), "c_w": float(c_w), "c_s": float(c_s)})

        cma = CMAES(
            problem={
                "fitness_function": self._fitness_batch,
                "ndim_problem": int(self.ndim_problem),
                "lower_boundary": self.lower_boundary,
                "upper_boundary": self.upper_boundary,
            },
            options=cma_options,
        )
        res = cma.optimize()
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
