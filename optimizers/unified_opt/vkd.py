import time
from typing import Dict

import numpy as np

from .VkD_CMAES import VkdCma, default_option


class VKD:
    """
    Lightweight wrapper to expose a MMES-like optimize() interface.
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
        self.k_init = int(options.get("k_init", 0))
        self.kmax = int(options.get("kmax", max(0, self.ndim_problem - 1)))
        self.verbose = bool(options.get("verbose", False))

    def _fitness_single(self, x):
        y = self.fitness_function(np.asarray(x, dtype=np.float64).reshape(1, -1))
        return float(np.asarray(y, dtype=np.float64).reshape(-1)[0])

    def optimize(self):
        start_time = time.time()
        np.random.seed(self.seed_rng)

        best_x = self.mean.copy()
        best_y = np.inf
        total_eval = 0

        lam = max(2, int(self.n_individuals))
        restart_id = 0
        last_xmean = self.mean.copy()
        last_sigma = float(self.sigma)
        final_arx = np.tile(self.mean, (lam, 1))
        final_arf = np.full((lam,), np.inf, dtype=np.float64)
        final_lam = int(lam)
        final_k_init = int(max(0, min(self.k_init, self.ndim_problem - 1)))
        final_kmax = int(max(0, min(self.kmax, self.ndim_problem - 1)))
        final_k = int(final_k_init)
        final_k_active = 0

        while total_eval < self.max_function_evaluations:
            remaining = int(self.max_function_evaluations - total_eval)
            if remaining <= 0:
                break

            opts = default_option(
                self.ndim_problem,
                remaining,
                lb=self.lower_boundary,
                ub=self.upper_boundary,
            )
            opts["lam"] = int(lam)
            opts["seed_rng"] = int(self.seed_rng + restart_id)
            opts["batch_evaluation"] = False
            opts["k_init"] = int(max(0, min(self.k_init, self.ndim_problem - 1)))
            opts["kmax"] = int(max(0, min(self.kmax, self.ndim_problem - 1)))
            if opts["k_init"] > opts["kmax"]:
                opts["k_init"] = int(opts["kmax"])
            for key in (
                "kmin",
                "k_inc_cond",
                "k_dec_cond",
                "k_adapt_factor",
                "factor_sigma_slope",
                "factor_diag_slope",
                "cs",
                "ds",
            ):
                if key in self.options:
                    opts[key] = self.options[key]

            vkd = VkdCma(
                self._fitness_single,
                last_xmean,
                last_sigma,
                np.tile(last_xmean, (lam, 1)),
                **opts,
            )

            satisfied = False
            condition = ""
            while not satisfied:
                vkd._onestep()
                satisfied, condition = vkd._check()

            total_eval += int(vkd.neval)
            final_lam = int(lam)
            final_k_init = int(opts["k_init"])
            final_kmax = int(opts["kmax"])
            final_k = int(getattr(vkd, "k", final_k_init))
            final_k_active = int(getattr(vkd, "k_active", 0))
            final_arx = np.copy(vkd.arx)
            final_arf = np.asarray(vkd.arf, dtype=np.float64).reshape(-1)
            last_xmean = np.copy(vkd.xmean)
            last_sigma = float(vkd.sigma)

            idx = int(np.argmin(final_arf))
            if float(final_arf[idx]) < best_y:
                best_y = float(final_arf[idx])
                best_x = np.copy(final_arx[idx])

            if condition == "maxeval":
                break

            lam = min(int(lam * 2), 100)
            restart_id += 1

        if self.verbose:
            print(f"[VKD] fevals={total_eval}, best={best_y:.6e}")

        return {
            "best_so_far_x": best_x,
            "best_so_far_y": float(best_y),
            "x": np.copy(final_arx),
            "y": np.copy(final_arf),
            "n_function_evaluations": int(min(total_eval, self.max_function_evaluations)),
            "time_function_evaluations": float(time.time() - start_time),
            "mean": np.copy(last_xmean),
            "sigma": float(last_sigma),
            "n_individuals": int(final_lam),
            "lam": int(final_lam),
            "k_init": int(final_k_init),
            "kmax": int(final_kmax),
            "k": int(final_k),
            "k_active": int(final_k_active),
        }
