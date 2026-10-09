import numpy as np  # engine for numerical computing

from optimizers.cmaes.es import ES   # abstract class of all Evolution Strategies (ES) classes
from optimizers.cmaes.numeric_forensics import (
    direction_probe,
    generation_record,
    scale_probe,
)
from optimizers.cmaes.numeric_telemetry import OptimizerNumericTelemetry, array_summary


class CMAESNumericFailure(RuntimeError):
    """Internal signal used to stop one numerically invalid CMAES block."""


class CMAES(ES):
    """Covariance Matrix Adaptation Evolution Strategy (CMAES).

    .. note:: `CMAES` is widely recognized as one of **State-Of-The-Art (SOTA)** evolutionary
       algorithms for continuous black-box optimization (BBO), according to the well-recognized
       `Nature <https://doi.org/10.1038/nature14544>`_ review of **Evolutionary Computation**.

       For some (rather all) interesting applications of `CMA-ES`, please refer to e.g.,
       `[SIMULIA > CST Studio Suite > Automatic Optimization (Dassault Systèmes)]
       <https://www.3ds.com/products/simulia/cst-studio-suite/automatic-optimization>`_,
       `[PNAS-2024] <https://doi.org/10.1073/pnas.2318641121>`_,
       `[Nature Communications-2024] <https://www.nature.com/articles/s41467-024-45882-z>`_,
       `[AIAA Journal-2024] <https://arc.aiaa.org/doi/full/10.2514/1.J063251>`_,
       `[NeurIPS-2024 Spotlight] <https://openreview.net/forum?id=79q206xswc>`_,
       `[ICLR-2024 Spotlight] <https://openreview.net/forum?id=KsUh8MMFKQ>`_,
       `[TMRB-2024] <https://ieeexplore.ieee.org/document/10302449>`_,
       `[LWC-2024] <https://ieeexplore.ieee.org/abstract/document/10531788>`_,
       `[RSIF-2024] <https://royalsocietypublishing.org/doi/10.1098/rsif.2024.0141>`_,
       `[MNRAS-2024] <https://academic.oup.com/mnras/article/530/1/947/7643636>`_,
       `[Medical Physics-2024] <https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.16962>`_,
       `[Wolff, 2024]
       <https://ieeexplore.ieee.org/abstract/document/10464875>`_, `[Jankowski et al., 2024]
       <https://arxiv.org/abs/2404.02795>`_, `[Martin, 2024, Ph.D. Dissertation (Harvard University)]
       <https://dash.harvard.edu/handle/1/37378922>`_, `[Milekovic et al., 2023, Nature Medicine]
       <https://doi.org/10.1038/s41591-023-02584-1>`_, `[Chen et al., 2023, Science Robotics]
       <https://www.science.org/doi/10.1126/scirobotics.adc9244>`_, `[Falk et al., 2023, PNAS]
       <https://www.pnas.org/doi/abs/10.1073/pnas.2219558120>`_, `[Thamm&Rosenow, 2023, PRL]
       <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.130.116202>`_, `[Brea et al., 2023, Nature
       Communications] <https://www.nature.com/articles/s41467-023-38570-x>`_, `[Ghafouri&Biros, 2023]
       <https://link.springer.com/chapter/10.1007/978-3-031-45087-7_6>`_, `[Barral, 2023, Ph.D. Dissertation (University of Oxford)]
       <https://ora.ox.ac.uk/objects/uuid:8d96225a-71c6-4649-814e-608c213c8a14/files/dr494vk723>`_, `[Slade et al., 2022, Nature]
       <https://www.nature.com/articles/s41586-022-05191-1>`_, `[Croon et al., 2022, Nature]
       <https://www.nature.com/articles/s41586-022-05182-2>`_, `[Rudolph et al., 2022, Nature Communications]
       <https://www.nature.com/articles/s41467-023-43908-6>`_, `[Cazenille et al., 2022, Bioinspiration & Biomimetics]
       <https://iopscience.iop.org/article/10.1088/1748-3190/ac7fd1>`_, `[Franks et al., 2021]
       <https://www.biorxiv.org/content/10.1101/2021.09.13.460170v1.abstract>`_, `[Yuan et al., 2021, MNRAS]
       <https://academic.oup.com/mnras/article/502/3/3582/6122578>`_, `[Löffler et al., 2021, Nature Communications]
       <https://www.nature.com/articles/s41467-021-22017-2>`_, `[Papadopoulou et al., 2021, JPCB]
       <https://pubs.acs.org/doi/10.1021/acs.jpcb.1c07562>`_, `[Schmucker et al., 2021, PLoS Comput Biol]
       <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009689>`_,
       `[Barkley, 2021, Ph.D. Dissertation (Harvard University)]
       <https://dash.harvard.edu/handle/1/37368472>`_, `[Fernandes, 2021, Ph.D. Dissertation (Harvard University)]
       <https://dash.harvard.edu/handle/1/37370084>`_, `[Quinlivan, 2021, Ph.D. Dissertation (Harvard University)]
       <https://dash.harvard.edu/handle/1/37369463>`_, `[Vasios et al., 2020, Soft Robotics]
       <https://www.liebertpub.com/doi/full/10.1089/soro.2018.0149>`_, `[Pal et al., 2020]
       <https://iopscience.iop.org/article/10.1088/1361-665X/abbd1d>`_, `[Lei, 2020, Ph.D. Dissertation (University of Oxford)]
       <https://tinyurl.com/yzkjwr34>`_, `[Pisaroni et al., 2019, Journal of Aircraft]
       <https://arc.aiaa.org/doi/10.2514/1.C035054>`_, `[Yang et al., 2019, Journal of Aircraft]
       <https://arc.aiaa.org/doi/full/10.2514/1.C034873>`_, `[Ong et al., 2019, PLOS Computational Biology]
       <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006993>`_, `[Zhang et al., 2017, Science]
       <https://www.science.org/doi/full/10.1126/science.aal5054>`_, `[Wei&Mahadevan, 2016, Soft Matter]
       <https://pubs.rsc.org/en/content/articlehtml/2016/sm/c5sm01597a>`_, `[Loshchilov&Hutter, 2016]
       <https://arxiv.org/abs/1604.07269>`_, `[Molinari et al., 2014, AIAAJ]
       <https://arc.aiaa.org/doi/full/10.2514/1.J052715>`_, `[Melton, 2014, Acta Astronautica]
       <https://www.sciencedirect.com/science/article/pii/S0094576514002318>`_, `[Khaira et al., 2014, ACS Macro Lett.]
       <https://pubs.acs.org/doi/full/10.1021/mz5002349>`_, `[Otake et al., 2013, Phys. Med. Biol.]
       <https://iopscience.iop.org/article/10.1088/0031-9155/58/23/8535>`_, `[Wang et al., 2010, TOG]
       <https://dl.acm.org/doi/10.1145/1778765.1778810>`_, `[Wampler&Popović, 2009, TOG]
       <https://dl.acm.org/doi/10.1145/1531326.1531366>`_
       `RoboCup <https://doi.org/10.1007/s10458-024-09642-z>`_, 2014 3D Simulation League Competition Champions,
       `[Muller et al., 2001, AIAAJ] <https://arc.aiaa.org/doi/abs/10.2514/2.1342>`_,
       to name a few.

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
                * 'sigma'         - initial global step-size, aka mutation strength (`float`),
                * 'mean'          - initial (starting) point, aka mean of Gaussian search distribution (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'n_individuals' - number of offspring, aka offspring population size (`int`, default:
                  `4 + int(3*np.log(problem['ndim_problem']))`),
                * 'n_parents'     - number of parents, aka parental population size (`int`, default:
                  `int(options['n_individuals']/2)`).

    Examples
    --------
    Use the black-box optimizer `CMAES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy  # engine for numerical computing
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.cmaes import CMAES
       >>> problem = {'fitness_function': rosenbrock,  # to define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5.0*numpy.ones((2,)),
       ...            'upper_boundary': 5.0*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # to set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3.0*numpy.ones((2,)),
       ...            'sigma': 3.0}  # global step-size may need to be fine-tuned for better performance
       >>> cmaes = CMAES(problem, options)  # to initialize the optimizer class
       >>> results = cmaes.optimize()  # to run the optimization/evolution process
       >>> print(f"CMAES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       CMAES: 5000, 0.0017

    For its correctness checking of Python coding, please refer to `this code-based repeatability report
    <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/_repeat_cmaes.py>`_
    for all details. For *pytest*-based automatic testing, please see `test_cmaes.py
    <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/test_cmaes.py>`_.

    Attributes
    ----------
    best_so_far_x : `array_like`
                    final best-so-far solution found during entire optimization.
    best_so_far_y : `array_like`
                    final best-so-far fitness found during entire optimization.
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search distribution.
    n_individuals : `int`
                    number of offspring, aka offspring population size / sample size.
    n_parents     : `int`
                    number of parents, aka parental population size / number of positively selected search points.
    sigma         : `float`
                    final global step-size, aka mutation strength (updated during optimization).

    References
    ----------
    https://cma-es.github.io/

    `Hansen, N. <http://www.cmap.polytechnique.fr/~nikolaus.hansen/>`_, 2023.
    `The CMA evolution strategy: A tutorial.
    <https://arxiv.org/abs/1604.00772>`_
    arXiv preprint arXiv:1604.00772.

    Ollivier, Y., Arnold, L., Auger, A. and Hansen, N., 2017.
    `Information-geometric optimization algorithms: A unifying picture via invariance principles.
    <https://jmlr.org/papers/v18/14-467.html>`_
    Journal of Machine Learning Research, 18(18), pp.1-65.

    Hansen, N., Atamna, A. and Auger, A., 2014, September.
    `How to assess step-size adaptation mechanisms in randomised search.
    <https://link.springer.com/chapter/10.1007/978-3-319-10762-2_6>`_
    In International Conference on Parallel Problem Solving From Nature (pp. 60-69). Springer, Cham.

    Kern, S., Müller, S.D., Hansen, N., Büche, D., Ocenasek, J. and Koumoutsakos, P., 2004.
    `Learning probability distributions in continuous evolutionary algorithms–a comparative review.
    <https://link.springer.com/article/10.1023/B:NACO.0000023416.59689.4e>`_
    Natural Computing, 3, pp.77-112.

    Hansen, N., Müller, S.D. and Koumoutsakos, P., 2003.
    `Reducing the time complexity of the derandomized evolution strategy with covariance matrix adaptation (CMA-ES).
    <https://direct.mit.edu/evco/article-abstract/11/1/1/1139/Reducing-the-Time-Complexity-of-the-Derandomized>`_
    Evolutionary Computation, 11(1), pp.1-18.

    Hansen, N. and Ostermeier, A., 2001.
    `Completely derandomized self-adaptation in evolution strategies.
    <https://direct.mit.edu/evco/article-abstract/9/2/159/892/Completely-Derandomized-Self-Adaptation-in>`_
    Evolutionary Computation, 9(2), pp.159-195.

    Hansen, N. and Ostermeier, A., 1996, May.
    `Adapting arbitrary normal mutation distributions in evolution strategies: The covariance matrix adaptation.
    <https://ieeexplore.ieee.org/abstract/document/542381>`_
    In Proceedings of IEEE International Conference on Evolutionary Computation (pp. 312-317). IEEE.

    Please refer to its *lightweight* Python implementation from `cyberagent.ai
    <https://cyberagent.ai/>`_:
    https://github.com/CyberAgentAILab/cmaes

    Please refer to its *official* Python implementation from `Hansen, N.
    <http://www.cmap.polytechnique.fr/~nikolaus.hansen/>`_:
    https://github.com/CMA-ES/pycma
    """
    def __init__(self, problem, options):
        self.options = options
        ES.__init__(self, problem, options)
        assert self.n_individuals >= 2
        self._w, self._mu_eff, self._mu_eff_minus = None, None, None  # variance effective selection mass
        # c_s (c_σ) -> decay rate for the cumulating path for the step-size control
        self.c_s, self.d_sigma = None, None  # for cumulative step-length adaptation (CSA)
        self._p_s_1, self._p_s_2 = None, None  # for evolution path update of CSA
        self._p_c_1, self._p_c_2 = None, None  # for evolution path update of CMA
        # c_c -> decay rate for cumulating path for the rank-one update of CMA
        # c_1 -> learning rate for the rank-one update of CMA
        # c_w (c_μ) -> learning rate for the rank-µ update of CMA
        self.c_c, self.c_1, self.c_w, self._alpha_cov = None, None, None, 2.0  # for CMA (c_w -> c_μ)
        self._save_eig = options.get('_save_eig', False)  # whether or not save eigenvalues and eigenvectors
        self.numeric_telemetry = OptimizerNumericTelemetry("cmaes", options)
        # Diagnostic-only forensic recorder; attached by the event-slot session when
        # --objective_split_optimizer_numeric_forensics is enabled.  None disables
        # every forensic code path below, so the default run is unchanged.
        self._forensics = None
        self._forensics_pending = None
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
        self.optimizer_numeric_fail_soft = bool(
            options.get("optimizer_numeric_fail_soft", False)
        )
        self.optimizer_numeric_fail_soft_triggered = False
        self.optimizer_numeric_fail_soft_reason = ""
        self.optimizer_numeric_fail_soft_generation = -1
        self.optimizer_numeric_fail_soft_evaluations = 0
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
        self.optimizer_guide_internal_path_lr = float(
            max(0.0, options.get("optimizer_guide_internal_path_lr", 0.0))
        )
        self.optimizer_guide_internal_cov_lr = float(
            max(0.0, options.get("optimizer_guide_internal_cov_lr", 0.0))
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
        self.optimizer_guide_internal_path_max_rel_norm = float(
            max(0.0, options.get("optimizer_guide_internal_path_max_rel_norm", 0.5))
        )
        self.optimizer_guide_internal_cov_rank1_clip = float(
            max(0.0, options.get("optimizer_guide_internal_cov_rank1_clip", 0.02))
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

    def _bounded_optimizer_anchor_candidate(self, mean, alpha):
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
        candidate_d = (candidate_x - mean) / max(sigma, 1e-12)
        if not np.all(np.isfinite(candidate_d)):
            return None, None
        self.optimizer_anchor_dist = dist
        return candidate_x, candidate_d

    def _inject_optimizer_anchor_samples(self, x, d, mean):
        if (
            not self.optimizer_anchor_sample_injection
            or self.optimizer_anchor_point is None
            or self.optimizer_anchor_sample_ratio <= 0.0
        ):
            return x, d
        n_anchor = int(round(float(self.n_individuals) * float(self.optimizer_anchor_sample_ratio)))
        n_anchor = int(np.clip(n_anchor, 1, max(1, self.n_individuals)))
        applied = 0
        for cursor in range(n_anchor):
            frac = float(cursor + 1) / float(n_anchor)
            alpha = float(np.clip(self.optimizer_anchor_mix_strength * frac, 0.0, 1.0))
            if alpha <= 0.0:
                continue
            candidate_x, candidate_d = self._bounded_optimizer_anchor_candidate(mean, alpha)
            if candidate_x is None or candidate_d is None:
                continue
            x[cursor] = candidate_x
            d[cursor] = candidate_d
            applied += 1
        if applied:
            self.optimizer_anchor_sample_applied = 1.0
            self.optimizer_anchor_applied = 1.0
        return x, d

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

    def _bounded_optimizer_guide_candidate(self, mean, signed_scale):
        sigma = float(self.sigma)
        if (not np.isfinite(sigma)) or sigma <= 0.0:
            sigma = float(getattr(self, "_sigma_bak", 1.0))
        step = sigma * float(signed_scale) * self.optimizer_guide_direction
        step_max = self._optimizer_guide_sample_step_max()
        step_norm = float(np.linalg.norm(step))
        if step_max is not None and step_max > 0.0 and step_norm > step_max:
            step *= float(step_max / max(step_norm, 1e-12))
        candidate_x = mean + step
        if self.optimizer_guide_sample_clip_ratio > 0.0:
            if not np.all(np.isfinite(candidate_x)):
                return None, None
            candidate_x = self._repair_bounds(candidate_x)
            if not np.all(np.isfinite(candidate_x)):
                return None, None
            candidate_d = (candidate_x - mean) / max(sigma, 1e-12)
            if not np.all(np.isfinite(candidate_d)):
                return None, None
            return candidate_x, candidate_d
        return candidate_x, float(signed_scale) * self.optimizer_guide_direction

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

    def _inject_one_optimizer_guide_sample(self, x, d, mean, cursor, signed_scale):
        old_x = np.copy(x[cursor])
        old_d = np.copy(d[cursor])
        candidate_x, candidate_d = self._bounded_optimizer_guide_candidate(mean, signed_scale)
        if candidate_x is None or candidate_d is None:
            self.numeric_telemetry.count("guided_sample_reject")
            return x, d
        if self.optimizer_guide_numeric_guard:
            if not np.all(np.isfinite(candidate_x)):
                self.numeric_telemetry.count("guided_sample_reject")
                return x, d
            before_repair = np.copy(candidate_x)
            candidate_x = self._repair_bounds(candidate_x)
            if not np.array_equal(before_repair, candidate_x):
                self.numeric_telemetry.count("guided_sample_bound_repair")
            if not np.all(np.isfinite(candidate_x)):
                self.numeric_telemetry.count("guided_sample_reject")
                return x, d
            sigma = self._clip_optimizer_guide_sigma(self.sigma)
            if sigma <= 0.0 or not np.isfinite(sigma):
                self.numeric_telemetry.count("guided_sample_reject")
                return x, d
            candidate_d = (candidate_x - mean) / sigma
            if not np.all(np.isfinite(candidate_d)):
                self.numeric_telemetry.count("guided_sample_reject")
                return x, d
        x[cursor] = candidate_x
        d[cursor] = candidate_d
        if self.optimizer_guide_numeric_guard and (
            not np.all(np.isfinite(x[cursor])) or not np.all(np.isfinite(d[cursor]))
        ):
            self.numeric_telemetry.count("guided_sample_rollback")
            x[cursor] = old_x
            d[cursor] = old_d
        return x, d

    def _inject_optimizer_guide_samples(self, x, d, mean):
        guide = self.optimizer_guide_direction
        alpha = float(self.optimizer_guide_strength)
        if (
            self.optimizer_guide_internal_disable_sample_injection
            and self.optimizer_guide_internal_mode != "off"
        ):
            return x, d
        if guide is None or alpha <= 0.0 or self.optimizer_guide_injection_pairs <= 0:
            return x, d
        cursor = 0
        for pair_idx in range(self.optimizer_guide_injection_pairs):
            scale = alpha * float(pair_idx + 1)
            if cursor < self.n_individuals:
                if self.optimizer_guide_numeric_guard:
                    x, d = self._inject_one_optimizer_guide_sample(x, d, mean, cursor, scale)
                else:
                    candidate_x, candidate_d = self._bounded_optimizer_guide_candidate(mean, scale)
                    if candidate_x is not None and candidate_d is not None:
                        d[cursor] = candidate_d
                        x[cursor] = candidate_x
                cursor += 1
            if self.optimizer_guide_use_negative_pair and cursor < self.n_individuals:
                if self.optimizer_guide_numeric_guard:
                    x, d = self._inject_one_optimizer_guide_sample(x, d, mean, cursor, -scale)
                else:
                    candidate_x, candidate_d = self._bounded_optimizer_guide_candidate(mean, -scale)
                    if candidate_x is not None and candidate_d is not None:
                        d[cursor] = candidate_d
                        x[cursor] = candidate_x
                cursor += 1
        if self.optimizer_guide_numeric_guard:
            x = self._repair_bounds(x)
        return x, d

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

    def _optimizer_guide_internal_ready(self, wd):
        guide = self.optimizer_guide_direction
        alpha = float(self.optimizer_guide_strength)
        if self.optimizer_guide_internal_mode == "off" or guide is None or alpha <= 0.0:
            return False, 0.0
        wd = np.asarray(wd, dtype=np.float64).reshape(-1)
        wd_norm = float(np.linalg.norm(wd))
        alignment = 0.0
        if wd_norm > 1e-12 and np.isfinite(wd_norm):
            alignment = float(np.dot(guide, wd / wd_norm))
            if alignment < float(self.optimizer_guide_internal_agree_cos_min):
                self.optimizer_guide_internal_alignment = alignment
                return False, alignment
        self.optimizer_guide_internal_alignment = float(alignment)
        return True, float(alignment)

    def _apply_optimizer_guide_internal_mean(self, mean_old, mean_new, wd):
        ready, alignment = self._optimizer_guide_internal_ready(wd)
        if not ready or self.optimizer_guide_internal_mean_lr <= 0.0 or mean_old is None:
            return mean_new
        mean_old = np.asarray(mean_old, dtype=np.float64).reshape(-1)
        mean_new = np.asarray(mean_new, dtype=np.float64).reshape(-1)
        own_step = mean_new - mean_old
        sigma = float(self.sigma)
        if (not np.isfinite(sigma)) or sigma <= 0.0:
            sigma = float(getattr(self, "_sigma_bak", 1.0))
        raw_step = (
            sigma
            * float(self.optimizer_guide_strength)
            * float(self.optimizer_guide_internal_mean_lr)
            * self.optimizer_guide_direction
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

    def _optimizer_guide_internal_path_wd(self, wd):
        if (
            self.optimizer_guide_internal_mode not in {"mean_path", "mean_path_covdiag"}
            or self.optimizer_guide_internal_path_lr <= 0.0
            or self.optimizer_guide_direction is None
            or self.optimizer_guide_strength <= 0.0
        ):
            return wd
        wd = np.asarray(wd, dtype=np.float64).reshape(-1)
        wd_norm = float(np.linalg.norm(wd))
        if wd_norm > 1e-12 and np.isfinite(wd_norm):
            alignment = float(np.dot(self.optimizer_guide_direction, wd / wd_norm))
            if alignment < float(self.optimizer_guide_internal_agree_cos_min):
                self.optimizer_guide_internal_alignment = alignment
                return wd
        cap = float(self.optimizer_guide_internal_path_max_rel_norm) * max(wd_norm, 1e-12)
        raw = (
            float(self.optimizer_guide_strength)
            * float(self.optimizer_guide_internal_path_lr)
            * self.optimizer_guide_direction
        )
        guide_path = self._clip_vector_norm(raw, cap)
        if float(np.linalg.norm(guide_path)) > 1e-12:
            self.optimizer_guide_internal_applied = 1.0
        return wd + guide_path

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

    def _sanitize_covariance(self, cm, fallback_cm=None):
        if self.optimizer_guide_internal_mode == "off" and not self.optimizer_guide_numeric_guard:
            return cm
        fallback = (
            np.eye(self.ndim_problem)
            if fallback_cm is None
            else np.asarray(fallback_cm, dtype=np.float64)
        )
        arr = np.asarray(cm, dtype=np.float64)
        if arr.shape != (self.ndim_problem, self.ndim_problem) or not np.all(np.isfinite(arr)):
            self.numeric_telemetry.count("covariance_nonfinite_repair")
            arr = fallback.copy()
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = (arr + arr.T) / 2.0
        try:
            vals, vecs = np.linalg.eigh(arr)
        except np.linalg.LinAlgError:
            self.numeric_telemetry.count("covariance_eigh_recovery")
            diag_scale = 1.0
            try:
                diag_vals = np.diag(fallback)
                diag_vals = diag_vals[np.isfinite(diag_vals) & (diag_vals > 0.0)]
                if diag_vals.size:
                    diag_scale = float(np.mean(diag_vals))
            except Exception:
                diag_scale = 1.0
            arr = np.eye(self.ndim_problem) * max(1e-8, diag_scale)
            vals, vecs = np.linalg.eigh(arr)
        vals_sanitized = np.nan_to_num(vals, nan=1e-8, posinf=1e8, neginf=1e-8)
        if not np.array_equal(vals, vals_sanitized):
            self.numeric_telemetry.count("covariance_eigen_nonfinite_repair")
        vals_clipped = np.clip(vals_sanitized, 1e-8, 1e8)
        if not np.array_equal(vals_sanitized, vals_clipped):
            self.numeric_telemetry.count("covariance_eigen_clip")
        vals = vals_clipped
        return vecs @ np.diag(vals) @ vecs.T

    def _require_finite_for_fail_soft(self, stage, **values):
        """Reject an invalid update without modifying any finite CMAES state."""
        if not self.optimizer_numeric_fail_soft:
            return
        for name, value in values.items():
            arr = np.asarray(value, dtype=np.float64)
            if not np.all(np.isfinite(arr)):
                raise CMAESNumericFailure(f"{stage}:{name}_nonfinite")

    def _set_c_c(self):
        """Set decay rate of evolution path for the rank-one update of CMA.
        """
        return (4.0 + self._mu_eff / self.ndim_problem) / (
                self.ndim_problem + 4.0 + 2.0 * self._mu_eff / self.ndim_problem)

    def _set_c_w(self):
        return np.minimum(1.0 - self.c_1, self._alpha_cov*(1.0/4.0 + self._mu_eff + 1.0/self._mu_eff - 2.0) /
                          (np.square(self.ndim_problem + 2.0) + self._alpha_cov*self._mu_eff/2.0))

    def _set_d_sigma(self):
        return 1.0 + 2.0*np.maximum(0.0, np.sqrt((self._mu_eff - 1.0)/(self.ndim_problem + 1.0)) - 1.0) + self.c_s

    def initialize(self, is_restart=False):
        w_a = np.log((self.n_individuals + 1.0)/2.0) - np.log(np.arange(self.n_individuals) + 1.0)  # w_apostrophe
        self._mu_eff = np.square(np.sum(w_a[:self.n_parents]))/np.sum(np.square(w_a[:self.n_parents]))
        self._mu_eff_minus = np.square(np.sum(w_a[self.n_parents:]))/np.sum(np.square(w_a[self.n_parents:]))
        self.c_s = self.options.get('c_s', (self._mu_eff + 2.0)/(self.ndim_problem + self._mu_eff + 5.0))
        self.d_sigma = self.options.get('d_sigma', self._set_d_sigma())
        self.c_c = self.options.get('c_c', self._set_c_c())
        self.c_1 = self.options.get('c_1', self._alpha_cov/(np.square(self.ndim_problem + 1.3) + self._mu_eff))
        self.c_w = self.options.get('c_w', self._set_c_w())
        w_min = np.min([1.0 + self.c_1/self.c_w, 1.0 + 2.0*self._mu_eff_minus/(self._mu_eff + 2.0),
                        (1.0 - self.c_1 - self.c_w)/(self.ndim_problem*self.c_w)])
        self._w = np.where(w_a >= 0, 1.0/np.sum(w_a[w_a > 0])*w_a, w_min/(-np.sum(w_a[w_a < 0]))*w_a)
        self._p_s_1, self._p_s_2 = 1.0 - self.c_s, np.sqrt(self.c_s*(2.0 - self.c_s)*self._mu_eff)
        self._p_c_1, self._p_c_2 = 1.0 - self.c_c, np.sqrt(self.c_c*(2.0 - self.c_c)*self._mu_eff)
        x = np.empty((self.n_individuals, self.ndim_problem))  # a population of search points (individuals, offspring)
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        p_s = np.zeros((self.ndim_problem,))  # evolution path (p_σ) for cumulative step-length adaptation (CSA)
        p_c = np.zeros((self.ndim_problem,))  # evolution path for covariance matrix adaptation (CMA)
        cm = np.eye(self.ndim_problem)  # covariance matrix of Gaussian search distribution
        e_ve = np.eye(self.ndim_problem)  # eigenvectors of `cm` (orthogonal matrix)
        e_va = np.ones((self.ndim_problem,))  # square roots of eigenvalues of `cm` (in diagonal rather matrix form)
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        d = np.empty((self.n_individuals, self.ndim_problem))
        self._list_initial_mean.append(np.copy(mean))
        return x, mean, p_s, p_c, cm, e_ve, e_va, y, d

    def iterate(self, x=None, mean=None, e_ve=None, e_va=None, y=None, d=None, args=None):
        if self._check_terminations():
            return x, y, d

        # 批量生成高斯噪声
        z = self.rng_optimization.standard_normal((self.n_individuals, self.ndim_problem))  # shape: (n_individuals, ndim_problem)

        # 批量计算方向向量 d
        d = np.dot(z, np.dot(np.diag(e_va), e_ve.T))  # shape: (n_individuals, ndim_problem)

        # 批量生成后代个体
        if self.optimizer_guide_numeric_guard:
            self.sigma = self._clip_optimizer_guide_sigma(self.sigma)
        x = mean + self.sigma * d  # shape: (n_individuals, ndim_problem)
        forensics = self._forensics if getattr(self._forensics, "enabled", False) else None
        if forensics is not None:
            # Diagnostic snapshot only: taken before injection, never fed back into
            # the control path, and unused when forensics is disabled.
            self._forensics_pending = {
                "x_before_injection": np.copy(x),
                "counters_before_injection": dict(self.numeric_telemetry.counters),
            }
        x, d = self._inject_optimizer_guide_samples(x, d, mean)
        x, d = self._inject_optimizer_anchor_samples(x, d, mean)
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

        # 并行评估适应度
        y = self._evaluate_fitness(x, args)  # 假设 _evaluate_fitness 支持批量化输入

        return x, y, d

    def update_distribution(self, x=None, p_s=None, p_c=None, cm=None, e_ve=None, e_va=None, y=None, d=None, mean_old=None):
        self.optimizer_guide_internal_applied = 0.0
        self.optimizer_guide_internal_mean_step_norm = 0.0
        self.optimizer_guide_internal_alignment = 0.0
        self.optimizer_anchor_mean_step_norm = 0.0
        forensics = self._forensics if getattr(self._forensics, "enabled", False) else None
        if forensics is not None:
            # Diagnostic copies of the pre-update state; the update below is unchanged.
            _f_p_s_before = np.copy(p_s)
            _f_p_c_before = np.copy(p_c)
            _f_cm_before = np.copy(cm)
            _f_axes_before = np.copy(e_va)
            _f_pending = self._forensics_pending
            self._forensics_pending = None
            # Candidates are scored at the clipped positions (see CMAESOpt._fitness_batch);
            # the internal update uses the raw positions below.  Both are kept so the
            # first sigma jump can be recomputed item by item.
            _f_x_raw = np.copy(np.asarray(x, dtype=np.float64))
            _f_lower = getattr(self, "lower_boundary", None)
            _f_upper = getattr(self, "upper_boundary", None)
            if _f_lower is not None and _f_upper is not None:
                _f_x_scored = np.clip(
                    _f_x_raw,
                    np.asarray(_f_lower, dtype=np.float64),
                    np.asarray(_f_upper, dtype=np.float64),
                )
            else:
                _f_x_scored = np.copy(_f_x_raw)
            _f_mean_before = (
                None if mean_old is None
                else np.copy(np.asarray(mean_old, dtype=np.float64))
            )
        order = np.argsort(y)  # to rank all offspring individuals
        wd = np.dot(self._w[:self.n_parents], d[order[:self.n_parents]])
        # update distribution mean via weighted recombination
        mean = np.dot(self._w[:self.n_parents], x[order[:self.n_parents]])
        # update global step-size: cumulative path length control / cumulative step-size control /
        #   cumulative step length adaptation (CSA)
        cm_minus_half = e_ve @ np.diag(1.0/e_va) @ e_ve.T
        p_s = self._p_s_1*p_s + self._p_s_2*np.dot(cm_minus_half, wd)
        old_sigma = float(self.sigma)
        exp_arg = self.c_s/self.d_sigma*(np.linalg.norm(p_s)/self._e_chi - 1.0)
        exp_arg_used = self._optimizer_guide_exp_arg(exp_arg)
        self.sigma *= np.exp(exp_arg_used)
        sigma_after_exp = float(self.sigma)
        self.sigma = self._clip_optimizer_guide_sigma(self.sigma, fallback=old_sigma)
        if (self.numeric_telemetry.counters_enabled and old_sigma > 0.0 and self.sigma > 0.0
                and np.isfinite(old_sigma) and np.isfinite(self.sigma)
                and np.log(self.sigma) - np.log(old_sigma) >= np.log(100.0)):
            self.numeric_telemetry.count("sigma_growth_ge_100x")
        self._require_finite_for_fail_soft(
            "sigma_update",
            sigma=self.sigma,
            exp_arg=exp_arg_used,
            p_s=p_s,
        )
        # update covariance matrix (CMA)
        h_s = 1.0 if np.linalg.norm(p_s)/np.sqrt(1.0 - np.power(1.0 - self.c_s, 2*(self._n_generations + 1))) < (
                1.4 + 2.0/(self.ndim_problem + 1.0))*self._e_chi else 0.0
        wd_path = self._optimizer_guide_internal_path_wd(wd)
        p_c = self._p_c_1*p_c + h_s*self._p_c_2*wd_path
        w_o = self._w*np.where(self._w >= 0, 1.0, self.ndim_problem/(np.square(
            np.linalg.norm(cm_minus_half @ d[order].T, axis=0)) + 1e-8))
        cm = ((1.0 + self.c_1*(1.0 - h_s)*self.c_c*(2.0 - self.c_c) - self.c_1 - self.c_w*np.sum(self._w))*cm +
              self.c_1*np.outer(p_c, p_c))  # rank-one update
        
        # 提取需要计算的向量和权重
        sorted_indices = order[:self.n_individuals] # 尽管公式通常只用前 mu 个，但原代码似乎遍历了所有？需确认原逻辑意图。
        # 假设原逻辑意图是加权求和（通常只有正权重的部分才参与 Rank-mu 更新，这里遵循你的代码逻辑）

        d_sorted = d[sorted_indices]
        w_sorted = w_o[:self.n_individuals] # 注意 w_o 的维度对齐

        # 利用矩阵乘法替代循环: C = C + c_w * (D.T * w) @ D
        # shape: (ndim, ndim) += scalar * (ndim, n_ind) @ (n_ind, ndim)
        weighted_d = d_sorted.T * w_sorted # 广播乘法
        cm += self.c_w * np.dot(weighted_d, d_sorted)

        # do eigen-decomposition and return both eigenvalues and eigenvectors
        cm = self._sanitize_covariance(cm, fallback_cm=cm)
        self._require_finite_for_fail_soft(
            "covariance_update",
            covariance=cm,
            p_c=p_c,
        )
        cm = (cm + np.transpose(cm))/2.0  # to ensure symmetry of covariance matrix
        if self.numeric_telemetry.enabled:
            self.numeric_telemetry.emit(
                "pre_eigh",
                self._n_generations,
                sigma_before=old_sigma,
                sigma_after_exp=sigma_after_exp,
                sigma_after_guard=float(self.sigma),
                exp_arg_raw=float(exp_arg),
                exp_arg_used=float(exp_arg_used),
                **array_summary("covariance", cm),
            )
        # return eigenvalues and eigenvectors of a symmetric matrix
        if forensics is not None:
            # Diagnostic copy of the updated covariance BEFORE eigen-decomposition.
            _f_cm_updated = np.copy(cm)
        try:
            e_va, e_ve = np.linalg.eigh(cm)  # e_va -> eigenvalues, e_ve -> eigenvectors
        except np.linalg.LinAlgError as exc:
            if self.optimizer_numeric_fail_soft:
                raise CMAESNumericFailure("covariance_eigh_nonconvergence") from exc
            raise
        self._require_finite_for_fail_soft(
            "covariance_eigh",
            eigenvalues=e_va,
            eigenvectors=e_ve,
        )
        raw_eigenvalues = None
        if self.numeric_telemetry.enabled or self.numeric_telemetry.counters_enabled:
            raw_eigenvalues = np.copy(e_va)
            if np.any(raw_eigenvalues < 0.0):
                self.numeric_telemetry.count("negative_eigenvalue_replace")
        if forensics is not None:
            # Diagnostic copy of the eigenvalues BEFORE the negative-value floor.
            _f_raw_eigenvalues = np.copy(e_va)
        e_va = np.sqrt(np.where(e_va < 0.0, 1e-8, e_va))  # to avoid negative eigenvalues
        # e_va: squared root of eigenvalues -> interpreted as individual step-sizes and its diagonal entries are
        #       standard deviations of different components (from Nikolaus Hansen, 2023)
        cm = e_ve @ np.diag(np.square(e_va)) @ np.transpose(e_ve)  # to recover covariance matrix
        if self.numeric_telemetry.enabled:
            self.numeric_telemetry.emit(
                "generation_update",
                self._n_generations,
                sigma_before=old_sigma,
                sigma_after_exp=sigma_after_exp,
                sigma_after_guard=float(self.sigma),
                exp_arg_raw=float(exp_arg),
                exp_arg_used=float(exp_arg_used),
                **array_summary("covariance", cm),
                **array_summary("eigenvalue_raw", raw_eigenvalues),
                **array_summary("eigen_std", e_va),
                c_s=float(self.c_s),
                d_sigma=float(self.d_sigma),
                mu_eff=float(self._mu_eff),
                e_chi=float(self._e_chi),
                n_parents=int(self.n_parents),
                n_individuals=int(self.n_individuals),
                h_s=float(h_s),
                negative_eigen_count=int(np.count_nonzero(raw_eigenvalues < 0.0)),
                eigen_replacement_occurred=bool(np.any(raw_eigenvalues < 0.0)),
                exp_arg_clipped=bool(float(exp_arg) != float(exp_arg_used)),
                exp_arg_repair_nonfinite=bool(not np.isfinite(float(exp_arg))),
                p_s_stable=scale_probe(p_s),
                p_c_stable=scale_probe(p_c),
                wd_stable=scale_probe(wd),
                wd_direction=direction_probe(wd),
                axes_after_min_axis=int(np.argmin(e_va)),
                axes_after_min_value=float(np.min(e_va)),
                axes_after_max_axis=int(np.argmax(e_va)),
                axes_after_max_value=float(np.max(e_va)),
                ranking_order=[int(x) for x in np.asarray(order).reshape(-1)],
                fitness_used_for_ranking=[
                    float(x) for x in np.asarray(y, dtype=np.float64).reshape(-1)
                ],
                guide_enable=bool(self.optimizer_guide_enable),
                guide_direction_present=bool(
                    getattr(self, "optimizer_guide_direction", None) is not None
                ),
                guide_internal_applied=float(self.optimizer_guide_internal_applied),
                anchor_applied=float(getattr(self, "optimizer_anchor_applied", 0.0)),
            )
        mean = self._apply_optimizer_guide_internal_mean(mean_old, mean, wd)
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
                    family="cmaes",
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
                    path_before=_f_p_s_before,
                    path_after=p_s,
                    path_persist_factor=self._p_s_1,
                    wd=wd,
                    d_samples=d,
                    cov_before=_f_cm_before,
                    cov_after=cm,
                    axes_before=_f_axes_before,
                    axes_after=e_va,
                    negative_eigen_count=int(np.count_nonzero(e_va == 1e-4)),
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
                        "path_p_c_before": scale_probe(_f_p_c_before),
                        "path_p_c_after": scale_probe(p_c),
                        "h_s": float(h_s),
                    },
                ))
                _jump_log10 = (
                    float(np.log10(float(self.sigma)) - np.log10(old_sigma))
                    if old_sigma > 0.0 and float(self.sigma) > 0.0
                    else None
                )
                _jump_hit = bool(
                    _jump_log10 is not None
                    and _jump_log10 > 0.0
                    and _jump_log10 >= float(forensics.jump.threshold_log10)
                ) or bool(np.any(_f_raw_eigenvalues < 0.0))
                forensics.note_jump(
                    key=id(self),
                    payload={
                        "generation": int(self._n_generations),
                        "identity": dict(forensics.context),
                        "collection": {
                            "phase": "cmaes_update_distribution_after_sigma_update",
                            "generation_index_after_increment": int(self._n_generations),
                            "physical_fes": int(self.n_function_evaluations),
                        },
                        "n_individuals": int(self.n_individuals),
                        "n_parents": int(self.n_parents),
                        "sigma_before": float(old_sigma),
                        "sigma_exp_arg_raw": (
                            float(exp_arg) if np.isfinite(float(exp_arg)) else str(exp_arg)
                        ),
                        "sigma_exp_arg_used": float(exp_arg_used),
                        "sigma_after": float(self.sigma),
                        "sigma_log10_growth": _jump_log10,
                        "x_raw": _f_x_raw.tolist(),
                        "x_scored": _f_x_scored.tolist(),
                        "clip_distance_norm": [
                            float(np.linalg.norm(a - b))
                            for a, b in zip(_f_x_raw, _f_x_scored)
                        ],
                        "fitness_used_for_ranking": [
                            float(v) for v in np.asarray(_f_fitness).reshape(-1)
                        ],
                        "ranking_order": [int(v) for v in _f_order.tolist()],
                        "parent_weights": self._w.tolist(),
                        "effective_weights_w_o": w_o.tolist(),
                        # Legacy field name kept for existing readers; w_o is now
                        # indexed by fitness rank, matching d_sorted_used.
                        "w_o_original": np.asarray(w_o, dtype=np.float64).tolist(),
                        "d_original": np.asarray(d, dtype=np.float64).tolist(),
                        "d_sorted_used": np.asarray(d_sorted, dtype=np.float64).tolist(),
                        "w_sorted_used": np.asarray(w_sorted, dtype=np.float64).tolist(),
                        "pairing_used": (
                            "cm += c_w * (d[order[:n]].T * w_o[:n]) @ d[order[:n]]"
                        ),
                        "negative_weight_scale_index_space": "fitness_rank",
                        "pairing_correct_reference": (
                            "cm += c_w * sum_k w_rank[k] * scale(d_original[order[k]]) "
                            "* outer(d_original[order[k]], d_original[order[k]])"
                        ),
                        "parent_weights_index_space": "fitness_rank",
                        "guide": {
                            "changed_rows": list(_f_rows),
                            "rows_in_parents": list(_f_parent_rows),
                            "counter_delta": dict(
                                (_f_pending or {}).get("counter_delta", {})
                            ),
                        },
                        "mean_before": (
                            None if _f_mean_before is None else _f_mean_before.tolist()
                        ),
                        "mean_after": np.asarray(mean, dtype=np.float64).tolist(),
                        "wd": np.asarray(wd, dtype=np.float64).tolist(),
                        "whitened_wd": np.asarray(
                            cm_minus_half @ np.asarray(wd, dtype=np.float64),
                            dtype=np.float64,
                        ).tolist(),
                        "path_p_s_before": _f_p_s_before.tolist(),
                        "path_p_s_after": np.asarray(p_s, dtype=np.float64).tolist(),
                        "path_p_c_before": _f_p_c_before.tolist(),
                        "path_p_c_after": np.asarray(p_c, dtype=np.float64).tolist(),
                        "h_s": float(h_s),
                        "covariance_before": _f_cm_before.tolist(),
                        "covariance_updated_pre_eigh": _f_cm_updated.tolist(),
                        "raw_eigenvalues": _f_raw_eigenvalues.tolist(),
                        "eigenvalues_after_floor": np.asarray(
                            e_va, dtype=np.float64
                        ).tolist(),
                        "covariance_after_rebuilt": np.asarray(
                            cm, dtype=np.float64
                        ).tolist(),
                        "coefficients": {
                            "c_1": float(self.c_1),
                            "c_w": float(self.c_w),
                            "c_c": float(self.c_c),
                            "c_s": float(self.c_s),
                            "d_sigma": float(self.d_sigma),
                            "mu_eff": float(self._mu_eff),
                            "e_chi": float(self._e_chi),
                            "sum_weights": float(np.sum(self._w)),
                        },
                    },
                    jump=_jump_hit,
                    growth_log10=_jump_log10,
                )
            except Exception:
                # Observation only: a forensic failure must not change the run.
                pass
        return mean, p_s, p_c, cm, e_ve, e_va

    def restart_reinitialize(self, x=None, mean=None, p_s=None, p_c=None,
                             cm=None, e_ve=None, e_va=None, y=None, d=None):
        if ES.restart_reinitialize(self, y):
            x, mean, p_s, p_c, cm, e_ve, e_va, y, d = self.initialize(True)
        return x, mean, p_s, p_c, cm, e_ve, e_va, y, d

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, p_s, p_c, cm, e_ve, e_va, y, d = self.initialize()
        while True:
            # sample and evaluate offspring population
            x, y, d = self.iterate(x, mean, e_ve, e_va, y, d, args)
            if self._check_terminations():
                break
            self._print_verbose_info(fitness, y)
            self._n_generations += 1
            try:
                mean, p_s, p_c, cm, e_ve, e_va = self.update_distribution(
                    x, p_s, p_c, cm, e_ve, e_va, y, d, mean_old=mean
                )
            except CMAESNumericFailure as exc:
                self.optimizer_numeric_fail_soft_triggered = True
                self.optimizer_numeric_fail_soft_reason = str(exc)
                self.optimizer_numeric_fail_soft_generation = int(
                    self._n_generations
                )
                self.optimizer_numeric_fail_soft_evaluations = int(
                    self.n_function_evaluations
                )
                self.numeric_telemetry.count("numeric_fail_soft_termination")
                self.numeric_telemetry.emit(
                    "numeric_fail_soft_termination",
                    self._n_generations,
                    reason=str(exc),
                    n_function_evaluations=int(self.n_function_evaluations),
                )
                context = self.numeric_telemetry.context
                context_text = "; ".join(
                    f"{key}={context[key]}"
                    for key in ("function_id", "env_step", "agent_id")
                    if key in context
                )
                if context_text:
                    context_text += "; "
                print(
                    "[CMAES numeric fail-soft] "
                    f"{context_text}"
                    f"generation={self._n_generations}; "
                    f"fes={self.n_function_evaluations}; reason={exc}",
                    flush=True,
                )
                break
            if self.is_restart:
                x, mean, p_s, p_c, cm, e_ve, e_va, y, d = self.restart_reinitialize(
                    x, mean, p_s, p_c, cm, e_ve, e_va, y, d)
        results = self._collect(fitness, y, mean)
        results['p_s'] = p_s
        results['p_c'] = p_c

        results['c_c'] = self.c_c
        results['c_s'] = self.c_s
        results['d_sigma'] = self.d_sigma
        results['c_1'] = self.c_1
        results['c_w'] = self.c_w
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
        results['optimizer_numeric_fail_soft_enabled'] = bool(
            self.optimizer_numeric_fail_soft
        )
        results['optimizer_numeric_fail_soft_triggered'] = bool(
            self.optimizer_numeric_fail_soft_triggered
        )
        results['optimizer_numeric_fail_soft_reason'] = str(
            self.optimizer_numeric_fail_soft_reason
        )
        results['optimizer_numeric_fail_soft_generation'] = int(
            self.optimizer_numeric_fail_soft_generation
        )
        results['optimizer_numeric_fail_soft_evaluations'] = int(
            self.optimizer_numeric_fail_soft_evaluations
        )

        # by default do *NOT* save eigenvalues and eigenvectors (with *quadratic* space complexity)
        if self._save_eig:
            results['e_va'], results['e_ve'] = e_va, e_ve
        return results
