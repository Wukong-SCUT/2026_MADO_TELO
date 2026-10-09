import numpy as np  # engine for numerical computing
import copy

from .es import ES  # abstract class of all Evolution Strategies (ES) classes
from .numeric_forensics import generation_record, scale_probe
from .numeric_telemetry import OptimizerNumericTelemetry, array_summary

from collections import deque

class SEPCMAES(ES):
    """Separable Covariance Matrix Adaptation Evolution Strategy (SEPCMAES).

    .. note:: `SEPCMAES` learns only the **diagonal** elements of the full covariance matrix explicitly, leading
       to a *linear* time complexity (w.r.t. each sampling) for large-scale black-box optimization. It is **highly
       recommended** to first attempt more advanced ES variants (e.g., `LMCMA`, `LMMAES`) for large-scale black-box
       optimization, since typically the performance of `SEPCMAES` deteriorates significantly on **non-separable,
       ill-conditioned** fitness landscape.

    Parameters
    ----------
    problem : dict
              problem arguments with the following common settings (`keys`):
                * 'fitness_function' - objective function to be **minimized** (`func`),
                * 'ndim_problem'     - number of dimensionality (`int`),
                * 'upper_boundary'   - upper boundary of search range (`array_like`),
                * 'lower_boundary'   - lower boundary of search range (`array_like`).
    options : dict
              optimizer options with the following common settings (`keys`):
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'sigma'    - initial global step-size, aka mutation strength (`float`),
                * 'mean'     - initial (starting) point, aka mean of Gaussian search distribution (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'n_individuals' - number of offspring, aka offspring population size (`int`, default:
                  `4 + int(3*np.log(options['ndim_problem']))`),
                * 'n_parents'     - number of parents, aka parental population size (`int`, default:
                  `int(options['n_individuals']/2)`),
                * 'c_c'           - learning rate of evolution path update (`float`, default:
                  `4.0/(options['ndim_problem'] + 4.0)`).

    Examples
    --------
    Use the black-box optimizer `SEPCMAES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy  # engine for numerical computing
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.sepcmaes import SEPCMAES
       >>> problem = {'fitness_function': rosenbrock,  # to define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5.0*numpy.ones((2,)),
       ...            'upper_boundary': 5.0*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # to set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3.0*numpy.ones((2,)),
       ...            'sigma': 3.0}  # global step-size may need to be fine-tuned for better performance
       >>> sepcmaes = SEPCMAES(problem, options)  # to initialize the optimizer class
       >>> results = sepcmaes.optimize()  # to run the optimization/evolution process
       >>> print(f"SEPCMAES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       SEPCMAES: 5000, 0.0093

    For its correctness checking of Python coding, please refer to `this code-based repeatability report
    <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/_repeat_sepcmaes.py>`_
    for all details. For *pytest*-based automatic testing, please see `test_sepcmaes.py
    <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/test_sepcmaes.py>`_.

    Attributes
    ----------
    c_c           : `float`
                    learning rate of evolution path update.
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search distribution.
    n_individuals : `int`
                    number of offspring, aka offspring population size.
    n_parents     : `int`
                    number of parents, aka parental population size.
    sigma         : `float`
                    final global step-size, aka mutation strength.

    References
    ----------
    Ros, R. and Hansen, N., 2008, September.
    `A simple modification in CMA-ES achieving linear time and space complexity.
    <https://link.springer.com/chapter/10.1007/978-3-540-87700-4_30>`_
    In International Conference on Parallel Problem Solving from Nature (pp. 296-305).
    Springer, Berlin, Heidelberg.
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.options = options
        self.c_c = options.get('c_c', 4.0/(self.ndim_problem + 4.0))
        self.c_s, self.c_cov = None, None
        self.d_sigma = None
        self.numeric_telemetry = OptimizerNumericTelemetry("sepcmaes", options)
        # Diagnostic-only forensic recorder; None disables every forensic code path
        # below, so the default run is unchanged.  Attached by the event-slot session
        # when --objective_split_optimizer_numeric_forensics is enabled.
        self._forensics = None
        self._forensics_pending = None
        self._s_1, self._s_2 = None, None
        self.optimizer_guide_enable = bool(options.get("optimizer_guide_enable", False))
        self.optimizer_guide_strength = float(max(0.0, options.get("optimizer_guide_strength", 0.0)))
        self.optimizer_guide_injection_pairs = int(
            max(0, options.get("optimizer_guide_injection_pairs", 1))
        )
        self.optimizer_guide_use_negative_pair = bool(
            options.get("optimizer_guide_use_negative_pair", True)
        )
        self.optimizer_guide_numeric_guard = bool(
            options.get("optimizer_guide_numeric_guard", False)
        )
        self.optimizer_guide_sigma_exp_clip = float(
            max(0.0, options.get("optimizer_guide_sigma_exp_clip", 20.0))
        )
        self.optimizer_guide_sigma_clip_ratio = float(
            max(0.0, options.get("optimizer_guide_sigma_clip_ratio", 0.5))
        )
        self.optimizer_guide_sample_clip_ratio = float(
            max(0.0, options.get("optimizer_guide_sample_clip_ratio", 0.0))
        )
        self.optimizer_guide_direction = self._resolve_optimizer_guide(
            options.get("optimizer_guide_direction", None)
        )
        self.optimizer_guide_internal_mode = str(
            options.get("optimizer_guide_internal_mode", "off")
        ).lower()
        if self.optimizer_guide_internal_mode not in {
            "off",
            "mean",
            "mean_path",
            "mean_path_covdiag",
        }:
            self.optimizer_guide_internal_mode = "off"
        self.optimizer_guide_internal_mean_lr = float(
            max(0.0, options.get("optimizer_guide_internal_mean_lr", 0.0))
        )
        self.optimizer_guide_internal_agree_cos_min = float(
            np.clip(options.get("optimizer_guide_internal_agree_cos_min", -0.25), -1.0, 1.0)
        )
        self.optimizer_guide_internal_max_step_ratio = float(
            max(0.0, options.get("optimizer_guide_internal_max_step_ratio", 0.05))
        )
        self.optimizer_guide_internal_max_rel_step = float(
            max(0.0, options.get("optimizer_guide_internal_max_rel_step", 0.5))
        )
        self.optimizer_guide_internal_disable_sample_injection = bool(
            options.get("optimizer_guide_internal_disable_sample_injection", False)
        )
        self.optimizer_guide_internal_applied = 0.0
        self.optimizer_guide_internal_mean_step_norm = 0.0
        self.optimizer_guide_internal_alignment = 0.0
        self.optimizer_anchor_enable = bool(options.get("optimizer_anchor_enable", False))
        self.optimizer_anchor_strength = float(max(0.0, options.get("optimizer_anchor_strength", 0.0)))
        self.optimizer_anchor_mix_strength = float(
            max(0.0, options.get("optimizer_anchor_mix_strength", self.optimizer_anchor_strength))
        )
        self.optimizer_anchor_sample_ratio = float(
            max(0.0, options.get("optimizer_anchor_sample_ratio", 0.25))
        )
        self.optimizer_anchor_sample_clip_ratio = float(
            max(0.0, options.get("optimizer_anchor_sample_clip_ratio", 0.25))
        )
        self.optimizer_anchor_mean_pull = bool(options.get("optimizer_anchor_mean_pull", True))
        self.optimizer_anchor_sample_injection = bool(
            options.get("optimizer_anchor_sample_injection", True)
        )
        self.optimizer_anchor_point = self._resolve_optimizer_anchor(
            options.get("optimizer_anchor_point", None)
        )
        self.optimizer_anchor_applied = 0.0
        self.optimizer_anchor_mean_step_norm = 0.0
        self.optimizer_anchor_sample_applied = 0.0
        self.optimizer_anchor_dist = 0.0

    def _resolve_optimizer_guide(self, raw):
        if not self.optimizer_guide_enable:
            return None
        if raw is None:
            return None
        guide = np.asarray(raw, dtype=np.float64).reshape(-1)
        if guide.size != self.ndim_problem:
            return None
        norm = float(np.linalg.norm(guide))
        if (not np.isfinite(norm)) or norm <= 1e-12:
            return None
        return guide / norm

    def _resolve_optimizer_anchor(self, raw):
        if not self.optimizer_anchor_enable:
            return None
        if raw is None:
            return None
        anchor = np.asarray(raw, dtype=np.float64).reshape(-1)
        if anchor.size != self.ndim_problem or not np.all(np.isfinite(anchor)):
            return None
        return self._repair_bounds(anchor)

    def _optimizer_anchor_step_max(self):
        ratio = float(self.optimizer_anchor_sample_clip_ratio)
        if ratio <= 0.0:
            return None
        if (self.lower_boundary is None) or (self.upper_boundary is None):
            return None
        span = np.asarray(self.upper_boundary, dtype=np.float64) - np.asarray(
            self.lower_boundary, dtype=np.float64
        )
        span = span[np.isfinite(span) & (span > 0.0)]
        if span.size == 0:
            return None
        return float(ratio * np.mean(span))

    def _anchor_direction_from(self, mean):
        anchor = self.optimizer_anchor_point
        if anchor is None:
            return None, 0.0
        direction = np.asarray(anchor, dtype=np.float64).reshape(-1) - np.asarray(
            mean, dtype=np.float64
        ).reshape(-1)
        dist = float(np.linalg.norm(direction))
        if (not np.isfinite(dist)) or dist <= 1e-12:
            return None, 0.0
        return direction, dist

    def _bounded_optimizer_anchor_candidate(self, mean, d, alpha):
        direction, dist = self._anchor_direction_from(mean)
        if direction is None:
            return None, None
        step = float(alpha) * direction
        step_max = self._optimizer_anchor_step_max()
        step_norm = float(np.linalg.norm(step))
        if step_max is not None and step_max > 0.0 and step_norm > step_max:
            step *= float(step_max / max(step_norm, 1e-12))
        candidate_x = self._repair_bounds(np.asarray(mean, dtype=np.float64) + step)
        if not np.all(np.isfinite(candidate_x)):
            return None, None
        sigma = float(self.sigma)
        if (not np.isfinite(sigma)) or sigma <= 0.0:
            sigma = float(getattr(self, "_sigma_bak", 1.0))
        denom = np.asarray(d, dtype=np.float64).reshape(-1)
        denom = np.where(np.abs(denom) > 1e-12, denom, 1.0)
        candidate_z = (candidate_x - mean) / (max(sigma, 1e-12) * denom)
        if not np.all(np.isfinite(candidate_z)):
            return None, None
        self.optimizer_anchor_dist = dist
        return candidate_x, candidate_z

    def _inject_optimizer_anchor_samples(self, z, x, mean, d):
        if (
            not self.optimizer_anchor_sample_injection
            or self.optimizer_anchor_point is None
            or self.optimizer_anchor_sample_ratio <= 0.0
        ):
            return z, x
        n_anchor = int(round(float(self.n_individuals) * float(self.optimizer_anchor_sample_ratio)))
        n_anchor = int(np.clip(n_anchor, 1, max(1, self.n_individuals)))
        applied = 0
        for cursor in range(n_anchor):
            frac = float(cursor + 1) / float(n_anchor)
            alpha = float(np.clip(self.optimizer_anchor_mix_strength * frac, 0.0, 1.0))
            if alpha <= 0.0:
                continue
            candidate_x, candidate_z = self._bounded_optimizer_anchor_candidate(mean, d, alpha)
            if candidate_x is None or candidate_z is None:
                continue
            x[cursor] = candidate_x
            z[cursor] = candidate_z
            applied += 1
        if applied:
            self.optimizer_anchor_sample_applied = 1.0
            self.optimizer_anchor_applied = 1.0
        return z, x

    def _optimizer_guide_sigma_max(self):
        ratio = float(self.optimizer_guide_sigma_clip_ratio)
        if ratio <= 0.0:
            return None
        if (self.lower_boundary is None) or (self.upper_boundary is None):
            return None
        span = np.asarray(self.upper_boundary, dtype=np.float64) - np.asarray(
            self.lower_boundary, dtype=np.float64
        )
        span = span[np.isfinite(span) & (span > 0.0)]
        if span.size == 0:
            return None
        return float(ratio * np.mean(span))

    def _optimizer_guide_sample_step_max(self):
        ratio = float(self.optimizer_guide_sample_clip_ratio)
        if ratio <= 0.0:
            return None
        if (self.lower_boundary is None) or (self.upper_boundary is None):
            return None
        span = np.asarray(self.upper_boundary, dtype=np.float64) - np.asarray(
            self.lower_boundary, dtype=np.float64
        )
        span = span[np.isfinite(span) & (span > 0.0)]
        if span.size == 0:
            return None
        return float(ratio * np.mean(span))

    def _bounded_optimizer_guide_candidate(self, mean, d, signed_scale):
        sigma = float(self.sigma)
        if (not np.isfinite(sigma)) or sigma <= 0.0:
            sigma = float(getattr(self, "_sigma_bak", 1.0))
        step = sigma * float(signed_scale) * self.optimizer_guide_direction
        step_max = self._optimizer_guide_sample_step_max()
        step_norm = float(np.linalg.norm(step))
        if step_max is not None and step_max > 0.0 and step_norm > step_max:
            step *= float(step_max / max(step_norm, 1e-12))
        candidate_x = mean + step
        denom = np.asarray(d, dtype=np.float64).reshape(-1)
        denom = np.where(np.abs(denom) > 1e-12, denom, 1.0)
        if self.optimizer_guide_sample_clip_ratio > 0.0:
            if not np.all(np.isfinite(candidate_x)):
                return None, None
            candidate_x = self._repair_bounds(candidate_x)
            if not np.all(np.isfinite(candidate_x)):
                return None, None
            candidate_z = (candidate_x - mean) / (max(sigma, 1e-12) * denom)
            if not np.all(np.isfinite(candidate_z)):
                return None, None
            return candidate_x, candidate_z
        return candidate_x, float(signed_scale) * self.optimizer_guide_direction / denom

    def _clip_optimizer_guide_sigma(self, sigma, fallback=None):
        if not self.optimizer_guide_numeric_guard:
            return sigma
        sigma = float(sigma)
        if not np.isfinite(sigma):
            self.numeric_telemetry.count("sigma_nonfinite_repair")
            if fallback is not None and np.isfinite(float(fallback)):
                sigma = float(fallback)
            else:
                sigma = float(getattr(self, "_sigma_bak", 1.0))
        sigma_floor = float(getattr(self, "sigma_threshold", 1e-12))
        if sigma < sigma_floor:
            self.numeric_telemetry.count("sigma_floor_clip")
        sigma = max(sigma_floor, sigma)
        sigma_max = self._optimizer_guide_sigma_max()
        if sigma_max is not None and sigma_max > 0.0:
            if sigma > sigma_max:
                self.numeric_telemetry.count("sigma_max_clip")
            sigma = min(sigma, sigma_max)
        return sigma

    def _optimizer_guide_exp_arg(self, exp_arg):
        if not self.optimizer_guide_numeric_guard:
            return exp_arg
        raw_exp_arg = float(exp_arg)
        exp_arg = float(np.nan_to_num(exp_arg, nan=0.0, posinf=0.0, neginf=0.0))
        if not np.isfinite(raw_exp_arg):
            self.numeric_telemetry.count("exp_nonfinite_repair")
        clip = float(self.optimizer_guide_sigma_exp_clip)
        if clip > 0.0:
            if exp_arg < -clip or exp_arg > clip:
                self.numeric_telemetry.count("exp_arg_clip")
            exp_arg = float(np.clip(exp_arg, -clip, clip))
        return exp_arg

    def _sanitize_optimizer_guide_covariance_diag(self, c, fallback_c):
        arr = np.asarray(c, dtype=np.float64).reshape(self.ndim_problem).copy()
        fallback = np.asarray(
            fallback_c, dtype=np.float64
        ).reshape(self.ndim_problem).copy()
        valid_fallback = np.isfinite(fallback) & (fallback > 0.0)
        fallback = np.where(valid_fallback, fallback, 1.0)
        nonfinite = ~np.isfinite(arr)
        nonpositive = np.isfinite(arr) & (arr <= 0.0)
        if np.any(nonfinite):
            self.numeric_telemetry.count(
                "covariance_nonfinite_repair",
                int(np.count_nonzero(nonfinite)),
            )
        if np.any(nonpositive):
            self.numeric_telemetry.count(
                "nonpositive_covariance_repair",
                int(np.count_nonzero(nonpositive)),
            )
        arr = np.where(nonfinite | nonpositive, fallback, arr)
        arr = np.maximum(arr, np.finfo(np.float64).tiny)
        d = np.sqrt(arr)

        sigma = float(self._clip_optimizer_guide_sigma(self.sigma))
        span = np.asarray(self.upper_boundary, dtype=np.float64) - np.asarray(
            self.lower_boundary, dtype=np.float64
        )
        effective_axis_max = (
            float(self.optimizer_guide_sigma_clip_ratio) * span
        )
        d_max = effective_axis_max / max(
            sigma, float(getattr(self, "sigma_threshold", 1e-12))
        )
        d_max = np.maximum(
            np.nan_to_num(
                d_max,
                nan=1.0,
                posinf=np.sqrt(np.finfo(np.float64).max),
                neginf=1.0,
            ),
            np.sqrt(np.finfo(np.float64).tiny),
        )
        clipped = d > d_max
        if np.any(clipped):
            self.numeric_telemetry.count(
                "covariance_axis_std_clip",
                int(np.count_nonzero(clipped)),
            )
        d = np.minimum(d, d_max)
        c_sanitized = np.square(d)
        return c_sanitized, d

    def _inject_one_optimizer_guide_sample(self, z, x, mean, d, cursor, signed_scale):
        old_z = np.copy(z[cursor])
        old_x = np.copy(x[cursor])
        candidate_x, candidate_z = self._bounded_optimizer_guide_candidate(
            mean, d, signed_scale
        )
        if candidate_x is None or candidate_z is None:
            self.numeric_telemetry.count("guided_sample_reject")
            return z, x
        if self.optimizer_guide_numeric_guard:
            if not np.all(np.isfinite(candidate_x)):
                self.numeric_telemetry.count("guided_sample_reject")
                return z, x
            before_repair = np.copy(candidate_x)
            candidate_x = self._repair_bounds(candidate_x)
            if not np.array_equal(before_repair, candidate_x):
                self.numeric_telemetry.count("guided_sample_bound_repair")
            if not np.all(np.isfinite(candidate_x)):
                self.numeric_telemetry.count("guided_sample_reject")
                return z, x
            sigma = self._clip_optimizer_guide_sigma(self.sigma)
            if sigma <= 0.0 or not np.isfinite(sigma):
                self.numeric_telemetry.count("guided_sample_reject")
                return z, x
            denom = np.asarray(d, dtype=np.float64).reshape(-1)
            denom = np.where(np.abs(denom) > 1e-12, denom, 1.0)
            candidate_z = (candidate_x - mean) / (sigma * denom)
            if not np.all(np.isfinite(candidate_z)):
                self.numeric_telemetry.count("guided_sample_reject")
                return z, x
        z[cursor] = candidate_z
        x[cursor] = candidate_x
        if self.optimizer_guide_numeric_guard and (
            not np.all(np.isfinite(z[cursor])) or not np.all(np.isfinite(x[cursor]))
        ):
            self.numeric_telemetry.count("guided_sample_rollback")
            z[cursor] = old_z
            x[cursor] = old_x
        return z, x

    def _inject_optimizer_guide_samples(self, z, x, mean, d):
        guide = self.optimizer_guide_direction
        alpha = float(self.optimizer_guide_strength)
        if (
            self.optimizer_guide_internal_disable_sample_injection
            and self.optimizer_guide_internal_mode != "off"
        ):
            return z, x
        if guide is None or alpha <= 0.0 or self.optimizer_guide_injection_pairs <= 0:
            return z, x
        cursor = 0
        for pair_idx in range(self.optimizer_guide_injection_pairs):
            scale = alpha * float(pair_idx + 1)
            if cursor < self.n_individuals:
                if self.optimizer_guide_numeric_guard:
                    z, x = self._inject_one_optimizer_guide_sample(z, x, mean, d, cursor, scale)
                else:
                    candidate_x, candidate_z = self._bounded_optimizer_guide_candidate(
                        mean, d, scale
                    )
                    if candidate_x is not None and candidate_z is not None:
                        x[cursor] = candidate_x
                        z[cursor] = candidate_z
                cursor += 1
            if self.optimizer_guide_use_negative_pair and cursor < self.n_individuals:
                if self.optimizer_guide_numeric_guard:
                    z, x = self._inject_one_optimizer_guide_sample(z, x, mean, d, cursor, -scale)
                else:
                    candidate_x, candidate_z = self._bounded_optimizer_guide_candidate(
                        mean, d, -scale
                    )
                    if candidate_x is not None and candidate_z is not None:
                        x[cursor] = candidate_x
                        z[cursor] = candidate_z
                cursor += 1
        return z, x

    @staticmethod
    def _clip_vector_norm(vec, max_norm):
        arr = np.asarray(vec, dtype=np.float64)
        limit = float(max_norm)
        if limit <= 0.0 or not np.isfinite(limit):
            return np.zeros_like(arr)
        norm = float(np.linalg.norm(arr))
        if (not np.isfinite(norm)) or norm <= 1e-12:
            return np.zeros_like(arr)
        if norm > limit:
            arr = arr * float(limit / max(norm, 1e-12))
        return arr

    def _search_span_mean(self):
        if (self.lower_boundary is None) or (self.upper_boundary is None):
            return 1.0
        span = np.asarray(self.upper_boundary, dtype=np.float64) - np.asarray(
            self.lower_boundary, dtype=np.float64
        )
        span = span[np.isfinite(span) & (span > 0.0)]
        if span.size == 0:
            return 1.0
        return float(np.mean(span))

    def _apply_optimizer_guide_internal_mean(self, mean_old, mean_new, z_w):
        self.optimizer_guide_internal_applied = 0.0
        self.optimizer_guide_internal_mean_step_norm = 0.0
        self.optimizer_guide_internal_alignment = 0.0
        guide = self.optimizer_guide_direction
        alpha = float(self.optimizer_guide_strength)
        if (
            self.optimizer_guide_internal_mode == "off"
            or guide is None
            or alpha <= 0.0
            or self.optimizer_guide_internal_mean_lr <= 0.0
            or mean_old is None
        ):
            return mean_new
        z_w = np.asarray(z_w, dtype=np.float64).reshape(-1)
        z_norm = float(np.linalg.norm(z_w))
        alignment = 0.0
        if z_norm > 1e-12 and np.isfinite(z_norm):
            alignment = float(np.dot(guide, z_w / z_norm))
            if alignment < float(self.optimizer_guide_internal_agree_cos_min):
                self.optimizer_guide_internal_alignment = alignment
                return mean_new
        mean_old = np.asarray(mean_old, dtype=np.float64).reshape(-1)
        mean_new = np.asarray(mean_new, dtype=np.float64).reshape(-1)
        own_step = mean_new - mean_old
        sigma = float(self.sigma)
        if (not np.isfinite(sigma)) or sigma <= 0.0:
            sigma = float(getattr(self, "_sigma_bak", 1.0))
        raw_step = (
            sigma
            * alpha
            * float(self.optimizer_guide_internal_mean_lr)
            * guide
        )
        abs_cap = float(self.optimizer_guide_internal_max_step_ratio) * self._search_span_mean()
        rel_base = max(float(np.linalg.norm(own_step)), abs(sigma) * 1e-3, 1e-12)
        rel_cap = float(self.optimizer_guide_internal_max_rel_step) * rel_base
        caps = [x for x in (abs_cap, rel_cap) if np.isfinite(x) and x > 0.0]
        if not caps:
            return mean_new
        guide_step = self._clip_vector_norm(raw_step, min(caps))
        step_norm = float(np.linalg.norm(guide_step))
        if (not np.isfinite(step_norm)) or step_norm <= 0.0:
            return mean_new
        guided_mean = self._repair_bounds(mean_new + guide_step)
        if not np.all(np.isfinite(guided_mean)):
            return mean_new
        self.optimizer_guide_internal_applied = 1.0
        self.optimizer_guide_internal_mean_step_norm = step_norm
        self.optimizer_guide_internal_alignment = alignment
        return guided_mean

    def _apply_optimizer_anchor_mean_pull(self, mean_old, mean_new):
        self.optimizer_anchor_mean_step_norm = 0.0
        if (
            not self.optimizer_anchor_mean_pull
            or self.optimizer_anchor_point is None
            or self.optimizer_anchor_strength <= 0.0
            or mean_old is None
        ):
            return mean_new
        mean_old = np.asarray(mean_old, dtype=np.float64).reshape(-1)
        mean_new = np.asarray(mean_new, dtype=np.float64).reshape(-1)
        direction = self.optimizer_anchor_point - mean_new
        dist = float(np.linalg.norm(direction))
        if (not np.isfinite(dist)) or dist <= 1e-12:
            return mean_new
        own_step = mean_new - mean_old
        raw_step = float(np.clip(self.optimizer_anchor_strength, 0.0, 1.0)) * direction
        abs_cap = self._optimizer_anchor_step_max()
        rel_base = max(float(np.linalg.norm(own_step)), abs(float(self.sigma)) * 1e-3, 1e-12)
        rel_cap = 0.5 * rel_base
        caps = [x for x in (abs_cap, rel_cap) if x is not None and np.isfinite(x) and x > 0.0]
        if not caps:
            return mean_new
        step = self._clip_vector_norm(raw_step, min(caps))
        step_norm = float(np.linalg.norm(step))
        if (not np.isfinite(step_norm)) or step_norm <= 0.0:
            return mean_new
        guided_mean = self._repair_bounds(mean_new + step)
        if not np.all(np.isfinite(guided_mean)):
            return mean_new
        self.optimizer_anchor_applied = 1.0
        self.optimizer_anchor_mean_step_norm = step_norm
        self.optimizer_anchor_dist = dist
        return guided_mean

    def _set_c_cov(self):
        c_cov = (1.0/self._mu_eff)*(2.0/np.power(self.ndim_problem + np.sqrt(2.0), 2)) + (
            (1.0 - 1.0/self._mu_eff)*np.minimum(1.0, (2.0*self._mu_eff - 1.0)/(
                np.power(self.ndim_problem + 2.0, 2) + self._mu_eff)))
        c_cov *= (self.ndim_problem + 2.0)/3.0  # for faster adaptation
        return c_cov

    def _set_d_sigma(self):
        d_sigma = np.maximum((self._mu_eff - 1.0)/(self.ndim_problem + 1.0) - 1.0, 0.0)
        return 1.0 + self.c_s + 2.0*np.sqrt(d_sigma)

    def replay_history(self, history, args=None):
        """
        根据 (x, y) 的历史记录重放以恢复优化器参数.
        history: list of dict, 每个元素是 {"x": ndarray, "y": ndarray}
        """
        # 初始化参数
        z = np.empty((self.n_individuals, self.ndim_problem))
        x = np.empty((self.n_individuals, self.ndim_problem))
        s = np.zeros((self.ndim_problem,))
        p = np.zeros((self.ndim_problem,))
        c = self.options.get('c', np.ones((self.ndim_problem,)))
        d = np.ones((self.ndim_problem,))
        mean = self._initialize_mean(False)
        y = np.full((self.n_individuals,), self._evaluate_fitness(mean, args))

        # 重放历史
        for record in history:
            x = record["x"]
            y_bak = np.copy(y)
            y = record["y"]

            # 近似恢复扰动 z
            mean = np.mean(x, axis=0)
            z = (x - mean) / self.sigma

            mean, s, p, c, d = self._update_distribution(z, x, s, p, c, d, y)
            self._n_generations += 1

        return z, x, mean, s, p, c, d, y

    def initialize(self, is_restart=False, xs=None, history=None):
        self.c_s = self.options.get('c_s', (self._mu_eff + 2.0) / (self.ndim_problem + self._mu_eff + 3.0))
        self.c_cov = self.options.get('c_cov', self._set_c_cov())
        self.d_sigma = self.options.get('d_sigma', self._set_d_sigma())
        self._s_1 = 1.0 - self.c_s
        self._s_2 = np.sqrt(self._mu_eff * self.c_s * (2.0 - self.c_s))

        z = np.empty((self.n_individuals, self.ndim_problem))
        x = np.empty((self.n_individuals, self.ndim_problem))
        s = np.zeros((self.ndim_problem,))
        p = np.zeros((self.ndim_problem,))
        c = self.options.get('c', np.ones((self.ndim_problem,)))
        d = np.ones((self.ndim_problem,))
        y = np.empty((self.n_individuals,))

        if history is not None:
            return self.replay_history(history)

        if xs is not None:
            assert xs.shape == (self.n_individuals, self.ndim_problem)
            x = self._repair_bounds(np.copy(xs))
            mean = np.mean(x, axis=0)
            y = self._evaluate_fitness(x)
            z = (x - mean) / self.sigma  # 近似反推扰动（如果需要更精确，可加上 inv(diag(d)））
        else:
            mean = self._initialize_mean(is_restart)
            x[:] = mean
            y[:] = self._evaluate_fitness(mean)

        self._list_initial_mean.append(np.copy(mean))
        self._n_generations = 0

        return z, x, mean, s, p, c, d, y

    def iterate(self, z=None, x=None, mean=None, d=None, y=None, args=None):
        # Step 1: 检查是否终止
        if self._check_terminations():
            return z, x, y

        # Step 2: 批量生成 z 矩阵
        z = self.rng_optimization.standard_normal((self.n_individuals, self.ndim_problem))

        # Step 3: 批量生成 x 矩阵
        # 如果 d 是标量，广播到矩阵操作；如果是向量/矩阵，直接计算
        if self.optimizer_guide_numeric_guard:
            self.sigma = self._clip_optimizer_guide_sigma(self.sigma)
        x = mean + self.sigma * d * z
        forensics = self._forensics if getattr(self._forensics, "enabled", False) else None
        if forensics is not None:
            # Diagnostic snapshot only; unused when forensics is disabled.
            self._forensics_pending = {
                "x_before_injection": np.copy(x),
                "counters_before_injection": dict(self.numeric_telemetry.counters),
            }
        z, x = self._inject_optimizer_guide_samples(z, x, mean, d)
        z, x = self._inject_optimizer_anchor_samples(z, x, mean, d)

        if getattr(self, "_boundary_capture", False):
            self._boundary_raw = np.copy(x)
        x = self._repair_bounds(x)
        if forensics is not None and self._forensics_pending is not None:
            pending = self._forensics_pending
            changed = np.any(x != pending["x_before_injection"], axis=1)
            pending["changed_rows"] = [int(i) for i in np.nonzero(changed)[0]]
            after = dict(self.numeric_telemetry.counters)
            before = pending["counters_before_injection"]
            pending["counter_delta"] = {
                str(key): int(after.get(key, 0) - before.get(key, 0))
                for key in set(after) | set(before)
                if int(after.get(key, 0)) != int(before.get(key, 0))
            }
            pending.pop("x_before_injection", None)
            pending.pop("counters_before_injection", None)

        # Step 4: 批量计算目标函数值 y
        y = self._evaluate_fitness(x, args)

        return z, x, y


    def _update_distribution(self, z=None, x=None, s=None, p=None, c=None, d=None, y=None, mean_old=None):
        forensics = self._forensics if getattr(self._forensics, "enabled", False) else None
        if forensics is not None:
            # Diagnostic copies of the pre-update state; the update below is unchanged.
            _f_s_before = np.copy(s)
            _f_p_before = np.copy(p)
            _f_c_before = np.copy(c)
            _f_d_before = np.copy(d)
            _f_pending = self._forensics_pending
            self._forensics_pending = None
        order = np.argsort(y)
        zeros = np.zeros((self.ndim_problem,))
        z_w, mean, dz_w = np.copy(zeros), np.copy(zeros), np.copy(zeros)
        for k in range(self.n_parents):
            z_w += self._w[k]*z[order[k]]
            mean += self._w[k]*x[order[k]]  # update distribution mean
            dz = d*z[order[k]]
            dz_w += self._w[k]*dz*dz
        s = self._s_1*s + self._s_2*z_w
        if (np.linalg.norm(s)/np.sqrt(1.0 - np.power(1.0 - self.c_s, 2.0*(self._n_generations + 1)))) < (
                (1.4 + 2.0/(self.ndim_problem + 1.0))*self._e_chi):
            h = np.sqrt(self.c_c*(2.0 - self.c_c))*np.sqrt(self._mu_eff)*d*z_w
        else:
            h = 0
        p = (1.0 - self.c_c)*p + h
        old_c = np.copy(c)
        if self.optimizer_guide_numeric_guard:
            with np.errstate(over="ignore", invalid="ignore"):
                c = (1.0 - self.c_cov)*c + (1.0/self._mu_eff)*self.c_cov*p*p + (
                        self.c_cov*(1.0 - 1.0/self._mu_eff)*dz_w)
        else:
            c = (1.0 - self.c_cov)*c + (1.0/self._mu_eff)*self.c_cov*p*p + (
                    self.c_cov*(1.0 - 1.0/self._mu_eff)*dz_w)
        old_sigma = float(self.sigma)
        exp_arg = self.c_s/self.d_sigma*(np.linalg.norm(s)/self._e_chi - 1.0)
        exp_arg_used = self._optimizer_guide_exp_arg(exp_arg)
        self.sigma *= np.exp(exp_arg_used)
        sigma_after_exp = float(self.sigma)
        self.sigma = self._clip_optimizer_guide_sigma(self.sigma, fallback=old_sigma)
        if self.optimizer_guide_numeric_guard:
            c, d = self._sanitize_optimizer_guide_covariance_diag(c, old_c)
        elif np.any(c <= 0):  # undefined in the original paper
            self.numeric_telemetry.count(
                "nonpositive_covariance_repair",
                int(np.count_nonzero(c <= 0)),
            )
            cc = np.copy(c)
            cc[cc <= 0] = 1.0
            d = np.sqrt(cc)
        else:
            d = np.sqrt(c)
        if self.numeric_telemetry.enabled:
            self.numeric_telemetry.emit(
                "generation_update",
                self._n_generations,
                sigma_before=old_sigma,
                sigma_after_exp=sigma_after_exp,
                sigma_after_guard=float(self.sigma),
                exp_arg_raw=float(exp_arg),
                exp_arg_used=float(exp_arg_used),
                **array_summary("covariance_diag", c),
                **array_summary("axis_std", d),
                c_s=float(self.c_s),
                d_sigma=float(self.d_sigma),
                mu_eff=float(self._mu_eff),
                e_chi=float(self._e_chi),
                n_parents=int(self.n_parents),
                n_individuals=int(self.n_individuals),
                exp_arg_clipped=bool(float(exp_arg) != float(exp_arg_used)),
                exp_arg_repair_nonfinite=bool(not np.isfinite(float(exp_arg))),
                s_stable=scale_probe(s),
                p_stable=scale_probe(p),
                wd_stable=scale_probe(z_w),
                axes_after_min_axis=int(np.argmin(d)),
                axes_after_min_value=float(np.min(d)),
                axes_after_max_axis=int(np.argmax(d)),
                axes_after_max_value=float(np.max(d)),
                ranking_order=[int(x) for x in np.asarray(order).reshape(-1)],
                fitness_used_for_ranking=[
                    float(x) for x in np.asarray(y, dtype=np.float64).reshape(-1)
                ],
                guide_enable=bool(self.optimizer_guide_enable),
                guide_direction_present=bool(
                    getattr(self, "optimizer_guide_direction", None) is not None
                ),
            )
        self.optimizer_anchor_mean_step_norm = 0.0
        mean = self._apply_optimizer_guide_internal_mean(mean_old, mean, z_w)
        mean = self._apply_optimizer_anchor_mean_pull(mean_old, mean)
        if forensics is not None:
            try:
                _f_order = np.asarray(order).reshape(-1)
                _f_fitness = np.asarray(y, dtype=np.float64).reshape(-1)
                _f_rows = list((_f_pending or {}).get("changed_rows", []))
                _f_parent_rows = []
                for _f_row in _f_rows:
                    _f_pos = np.nonzero(_f_order == int(_f_row))[0]
                    if _f_pos.size and int(_f_pos[0]) < int(self.n_parents):
                        _f_rank = int(_f_pos[0])
                        _f_parent_rows.append({
                            "row": int(_f_row),
                            "parent_rank": _f_rank,
                            "weight": float(self._w[_f_rank]),
                            "fitness": float(_f_fitness[int(_f_row)]),
                        })
                forensics.record_generation(generation_record(
                    family="sepcmaes",
                    generation=int(self._n_generations),
                    sigma_before=old_sigma,
                    sigma_after_exp=sigma_after_exp,
                    sigma_after_guard=float(self.sigma),
                    exp_arg_raw=exp_arg,
                    exp_arg_used=exp_arg_used,
                    c_s=self.c_s,
                    d_sigma=self.d_sigma,
                    mu_eff=self._mu_eff,
                    e_chi=self._e_chi,
                    n_parents=int(self.n_parents),
                    n_individuals=int(self.n_individuals),
                    path_before=_f_s_before,
                    path_after=s,
                    path_persist_factor=self._s_1,
                    wd=z_w,
                    cov_before=_f_c_before,
                    cov_after=c,
                    axes_before=_f_d_before,
                    axes_after=d,
                    fitness=_f_fitness,
                    order=_f_order,
                    guide={
                        "changed_rows": _f_rows,
                        "rows_in_parents": _f_parent_rows,
                        "counter_delta": (_f_pending or {}).get("counter_delta", {}),
                        "guide_direction_present": bool(
                            getattr(self, "optimizer_guide_direction", None) is not None
                        ),
                        "guide_enable": bool(self.optimizer_guide_enable),
                        "anchor_applied": float(
                            getattr(self, "optimizer_anchor_applied", 0.0)
                        ),
                        "anchor_sample_applied": float(
                            getattr(self, "optimizer_anchor_sample_applied", 0.0)
                        ),
                    },
                    extra={
                        "path_p_before": scale_probe(_f_p_before),
                        "path_p_after": scale_probe(p),
                        "axis_equal_one_count": int(np.count_nonzero(
                            np.asarray(d, dtype=np.float64).reshape(-1) == 1.0
                        )),
                        "covariance_diag_repair_counter": int(
                            self.numeric_telemetry.counters.get(
                                "nonpositive_covariance_repair", 0
                            )
                        ),
                        "eigen_replacement_field_applicable": False,
                    },
                ))
            except Exception:
                # Observation only: a forensic failure must not change the run.
                pass
        return mean, s, p, c, d

    def restart_reinitialize(self, z=None, x=None, mean=None, s=None, p=None, c=None, d=None, y=None):
        is_restart = ES.restart_reinitialize(self, y)
        if is_restart:
            z, x, mean, s, p, c, d, y = self.initialize(is_restart)
        return z, x, mean, s, p, c, d, y

    def optimize(self,xs=None, fitness_function=None, history=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        z, x, mean, s, p, c, d, y = self.initialize(xs=xs, history=history)
        fitness_history = []  # 用于记录每一代的评估值
        self.history = deque(maxlen=300)  # 保存最后 300 代

        while not self.termination_signal:
            # sample and evaluate offspring population
            z, x, y = self.iterate(z, x, mean, d, y, args)

            self.history.append({"x": np.copy(x), "y": np.copy(y)})
            fitness_history.append(np.copy(y))  # 保存当前代评估值
            if self._check_terminations():
                break
            self._print_verbose_info(fitness, y)
            mean, s, p, c, d = self._update_distribution(
                z, x, s, p, c, d, y, mean_old=mean
            )
            self._n_generations += 1
            if self.is_restart:
                z, x, mean, s, p, c, d, y = self.restart_reinitialize(z, x, mean, s, p, c, d, y)
        results = self._collect(fitness, y, mean)
        results['s'] = s
        results['p'] = p
        results['d'] = d
        results['x'] = x
        results['y'] = y
        results['optimizer_guide_internal_applied'] = float(
            self.optimizer_guide_internal_applied
        )
        results['optimizer_guide_internal_mean_step_norm'] = float(
            self.optimizer_guide_internal_mean_step_norm
        )
        results['optimizer_guide_internal_alignment'] = float(
            self.optimizer_guide_internal_alignment
        )
        results['optimizer_anchor_applied'] = float(self.optimizer_anchor_applied)
        results['optimizer_anchor_mean_step_norm'] = float(
            self.optimizer_anchor_mean_step_norm
        )
        results['optimizer_anchor_sample_applied'] = float(
            self.optimizer_anchor_sample_applied
        )
        results['optimizer_anchor_dist'] = float(self.optimizer_anchor_dist)
        results['optimizer_numeric_telemetry'] = self.numeric_telemetry.result_records()
        results['optimizer_numeric_guard_counters'] = self.numeric_telemetry.result_counters()
        results['History'] = list(self.history)
        results['FitnessHistory'] = fitness_history  # 每代评估值历史
        return results
