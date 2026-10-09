import time
from typing import Dict

import numpy as np

from optimizers.cmaes.cmaes import CMAES


class CMAESOpt:
    """
    Wrapper that adapts optimizers.cmaes.CMAES to unified_opt interface.
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

    def build_core_options(self) -> Dict:
        """Translate unified wrapper options into native CMAES options."""
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
        for key in (
            "optimizer_guide_enable",
            "optimizer_guide_direction",
            "optimizer_guide_strength",
            "optimizer_guide_injection_pairs",
            "optimizer_guide_use_negative_pair",
            "optimizer_guide_numeric_guard",
            "optimizer_numeric_fail_soft",
            "optimizer_numeric_telemetry_enable",
            "optimizer_numeric_telemetry_dir",
            "optimizer_numeric_telemetry_context",
            "optimizer_numeric_counter_enable",
            "optimizer_numeric_counter_dir",
            "optimizer_guide_sigma_exp_clip",
            "optimizer_guide_sigma_clip_ratio",
            "optimizer_guide_sample_clip_ratio",
            "optimizer_guide_internal_mode",
            "optimizer_guide_internal_mean_lr",
            "optimizer_guide_internal_path_lr",
            "optimizer_guide_internal_cov_lr",
            "optimizer_guide_internal_agree_cos_min",
            "optimizer_guide_internal_max_step_ratio",
            "optimizer_guide_internal_max_rel_step",
            "optimizer_guide_internal_path_max_rel_norm",
            "optimizer_guide_internal_cov_rank1_clip",
            "optimizer_guide_internal_disable_sample_injection",
            "optimizer_anchor_enable",
            "optimizer_anchor_point",
            "optimizer_anchor_strength",
            "optimizer_anchor_mix_strength",
            "optimizer_anchor_sample_ratio",
            "optimizer_anchor_sample_clip_ratio",
            "optimizer_anchor_mean_pull",
            "optimizer_anchor_sample_injection",
        ):
            if key in self.options:
                cma_options[key] = self.options[key]
        return cma_options

    def build_core_problem(self) -> Dict:
        """Return the bounded/non-finite-safe problem used by the native core."""
        return {
            "fitness_function": self._fitness_batch,
            "ndim_problem": int(self.ndim_problem),
            "lower_boundary": self.lower_boundary,
            "upper_boundary": self.upper_boundary,
        }

    def optimize(self):
        start_time = time.time()
        cma_options = self.build_core_options()

        cma = CMAES(
            problem=self.build_core_problem(),
            options=cma_options,
        )
        res = cma.optimize()
        best_x = np.asarray(
            res["best_so_far_x"], dtype=np.float64
        ).reshape(-1)
        if bool(res.get("optimizer_numeric_fail_soft_triggered", False)):
            best_x = np.where(np.isnan(best_x), self.mean, best_x)
            best_x = np.where(np.isposinf(best_x), self.upper_boundary, best_x)
            best_x = np.where(np.isneginf(best_x), self.lower_boundary, best_x)
            best_x = np.clip(
                best_x, self.lower_boundary, self.upper_boundary
            )
        return {
            "best_so_far_x": best_x.copy(),
            "best_so_far_y": float(res["best_so_far_y"]),
            "n_function_evaluations": int(min(res["n_function_evaluations"], self.max_function_evaluations)),
            "time_function_evaluations": float(res.get("time_function_evaluations", time.time() - start_time)),
            "x": best_x.copy(),
            "y": np.asarray([res.get("best_so_far_y", np.inf)], dtype=np.float64),
            "mean": best_x.copy(),
            "sigma": float(self.sigma),
            "optimizer_guide_internal_applied": float(
                res.get("optimizer_guide_internal_applied", 0.0)
            ),
            "optimizer_guide_internal_mean_step_norm": float(
                res.get("optimizer_guide_internal_mean_step_norm", 0.0)
            ),
            "optimizer_guide_internal_alignment": float(
                res.get("optimizer_guide_internal_alignment", 0.0)
            ),
            "optimizer_anchor_applied": float(res.get("optimizer_anchor_applied", 0.0)),
            "optimizer_anchor_mean_step_norm": float(
                res.get("optimizer_anchor_mean_step_norm", 0.0)
            ),
            "optimizer_anchor_sample_applied": float(
                res.get("optimizer_anchor_sample_applied", 0.0)
            ),
            "optimizer_anchor_dist": float(res.get("optimizer_anchor_dist", 0.0)),
            "optimizer_numeric_telemetry": list(
                res.get("optimizer_numeric_telemetry", [])
            ),
            "optimizer_numeric_guard_counters": dict(
                res.get("optimizer_numeric_guard_counters", {})
            ),
            "optimizer_numeric_fail_soft_enabled": bool(
                res.get("optimizer_numeric_fail_soft_enabled", False)
            ),
            "optimizer_numeric_fail_soft_triggered": bool(
                res.get("optimizer_numeric_fail_soft_triggered", False)
            ),
            "optimizer_numeric_fail_soft_reason": str(
                res.get("optimizer_numeric_fail_soft_reason", "")
            ),
            "optimizer_numeric_fail_soft_generation": int(
                res.get("optimizer_numeric_fail_soft_generation", -1)
            ),
            "optimizer_numeric_fail_soft_evaluations": int(
                res.get("optimizer_numeric_fail_soft_evaluations", 0)
            ),
        }
