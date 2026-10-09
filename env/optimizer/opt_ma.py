import math
from array import array
from itertools import islice
from pathlib import Path
from typing import Dict, List, Tuple
import os

import gym
import numpy as np
from gym import spaces

from optimizers.unified_opt import create_optimizer
from benchmark.cdo_bench_f1f14 import Benchmark as CDOBenchF1F14Benchmark
from benchmark.cdo_bench_f1f15 import Benchmark as CDOBenchF1F15Benchmark
from benchmark.wsn_localization import Benchmark as WSNBenchmark
from env.agent.utils.utils import partition_p_and_s
from options import get_options


def _safe_log_improvement(prev_f: float, cur_f: float) -> float:
    if not (np.isfinite(prev_f) and np.isfinite(cur_f)):
        return 0.0
    scale = max(1.0, abs(prev_f), abs(cur_f))
    eps = 1e-8 * scale
    shift = max(0.0, -min(float(prev_f), float(cur_f))) + eps
    num = float(prev_f) + shift
    den = float(cur_f) + shift
    if num <= 0.0 or den <= 0.0:
        return 0.0
    out = float(np.log(num / den))
    return out if np.isfinite(out) else 0.0


class opt_ma(gym.Env):
    """
    Multi-agent CC environment (Phase-1):
    - fixed 20 agents
    - MMES only
    - action in {0,1,2} -> sigma in {0.01, 0.1, 1.0}
    - parallel synchronous commit in each step
    """

    metadata = {"render.modes": []}

    @staticmethod
    def _load_existing_grouping(question: int) -> List[List[int]]:
        """
        Reuse the same grouping rules as env/optimizer/opt.py.
        """
        base_dir = Path(__file__).resolve().parents[2]
        data_files = os.path.join(base_dir, "benchmark", "cec2013lsgo", "cdatafiles")

        if question in [1, 2, 3, 12, 15]:
            grouping_result = [chunk.tolist() for chunk in np.array_split(np.arange(1000), 20)]
        elif question in [13, 14]:
            p_file_path = os.path.join(data_files, f"F{question}-p.txt")
            s_file_path = os.path.join(data_files, f"F{question}-s.txt")
            grouping_result = partition_p_and_s(p_file_path, s_file_path, overlap=5)
        elif question in [4, 5, 6, 7]:
            p_file_path = os.path.join(data_files, f"F{question}-p.txt")
            s_file_path = os.path.join(data_files, f"F{question}-s_.txt")
            grouping_result = partition_p_and_s(p_file_path, s_file_path, overlap=0)
            last_element = grouping_result[-1]
            it = iter(last_element)
            split_parts = [list(islice(it, 50)) for _ in range(0, len(last_element), 50)]
            grouping_result = grouping_result[:-1] + split_parts
        else:
            p_file_path = os.path.join(data_files, f"F{question}-p.txt")
            s_file_path = os.path.join(data_files, f"F{question}-s.txt")
            grouping_result = partition_p_and_s(p_file_path, s_file_path, overlap=0)

        return grouping_result

    @staticmethod
    def _build_equal_grouping(dimension: int, n_groups: int) -> List[List[int]]:
        if n_groups <= 0:
            raise ValueError("n_groups must be positive.")
        d = int(dimension)
        g = int(n_groups)
        if d <= 0:
            raise ValueError("dimension must be positive.")
        # Default split (no overlap) when feasible.
        if g <= d:
            return [chunk.tolist() for chunk in np.array_split(np.arange(d), g)]
        # When agents > dimensions, avoid empty groups by cycling dimensions.
        # This introduces overlap, but keeps every subproblem valid (ndim>=1).
        groups: List[List[int]] = []
        for i in range(g):
            groups.append([int(i % d)])
        return groups

    @staticmethod
    def _build_target_block_grouping(dimension: int, coord_dim: int) -> List[List[int]]:
        if coord_dim <= 0:
            raise ValueError("coord_dim must be positive.")
        if dimension % coord_dim != 0:
            raise ValueError(f"dimension={dimension} is not divisible by coord_dim={coord_dim}.")
        groups = []
        for t in range(dimension // coord_dim):
            l = t * coord_dim
            r = l + coord_dim
            groups.append(list(range(l, r)))
        return groups

    def __init__(self, question: int, opts_in=None):
        super().__init__()
        self.opts = opts_in if opts_in is not None else get_options()
        self.question = int(question)  # external id
        self.divide_method = str(getattr(self.opts, "divide_method", "CEC2013LSGO"))
        self.benchmark_name = str(getattr(self.opts, "benchmark_name", "CEC2013LSGO"))
        self.wsn_id_offset = int(getattr(self.opts, "wsn_id_offset", 100))
        self.masoie_wsn_id_offset = int(getattr(self.opts, "masoie_wsn_id_offset", 200))

        self.problem_family, self.inner_question = self._resolve_problem_identity(self.question)
        if self.problem_family == "WSNLocation":
            self.bench = WSNBenchmark(self.opts)
        elif self.problem_family == "WSNLocationMASOIE":
            raise ValueError("Standalone WSNLocationMASOIE is not included; use CDOBenchF1F15 function 15")
        elif self.problem_family == "DBOF1F10":
            raise ValueError("DBOF1F10 is not included in this package")
        elif self.problem_family == "CDOCompetition":
            raise ValueError("CDOCompetition is not included in this package")
        elif self.problem_family == "CDOBenchF1F14":
            self.bench = CDOBenchF1F14Benchmark(self.opts)
        elif self.problem_family == "CDOBenchF1F15":
            self.bench = CDOBenchF1F15Benchmark(self.opts)
        else:
            raise ValueError("CEC2013LSGO is not included in this package")
        self.info = self.bench.get_info(self.inner_question)
        self.fun = self.bench.get_function(self.inner_question)

        self.D = int(self.info["dimension"])
        self.lb = float(self.info["lower"])
        self.ub = float(self.info["upper"])

        self.n_agents = int(self.opts.fixed_agent_num)
        self.max_fes = int(self.opts.max_fes)
        self.subfes_per_agent = int(self.opts.fixed_subfes_per_agent)
        self.sigma_candidates = list(self.opts.sigma_candidates)
        self.profile_candidates = [str(x).lower() for x in getattr(self.opts, "optimizer_profile_candidates", ["inherit", "conservative", "balanced", "aggressive"])]
        self.mappo_action_arch = str(getattr(self.opts, "mappo_action_arch", "current")).lower()
        self.cfg_param_num = int(getattr(self.opts, "mappo_cfg_param_num", 4))
        if self.mappo_action_arch == "pre_caf4a62":
            self.cfg_param_num = 1
        self.optimizer_candidates = [str(x).lower() for x in getattr(self.opts, "optimizer_candidates", ["mmes", "vkd", "cmaes", "sepcmaes"])]
        self.resource_factors = [float(x) for x in getattr(self.opts, "resource_factors", [0.5, 1.0, 2.0])]
        self.reward_team_weight = float(getattr(self.opts, "mappo_reward_team_weight", 0.8))
        self.reward_local_weight = float(1.0 - self.reward_team_weight)
        self.episode_steps = int(self.opts.episode_steps)
        self.record_eval_individual = bool(getattr(self.opts, "record_eval_individual", 0))
        self.mappo_param_semantics = str(getattr(self.opts, "mappo_param_semantics", "current")).lower()
        if self.mappo_action_arch not in {"current", "pre_caf4a62"}:
            raise ValueError(f"Unsupported mappo_action_arch: {self.mappo_action_arch}")
        if self.mappo_param_semantics not in {"current", "a1"}:
            raise ValueError(f"Unsupported mappo_param_semantics: {self.mappo_param_semantics}")
        self.subopt_manual_override = bool(int(getattr(self.opts, "subopt_manual_override", 0)))
        self.sub_optimizer = str(getattr(self.opts, "sub_optimizer", "mmes")).lower()
        self.max_fes = self._resolve_max_fes_for_problem(self.max_fes)

        if self.n_agents <= 0:
            raise ValueError("fixed_agent_num must be positive.")
        if len(self.profile_candidates) <= 0:
            raise ValueError("optimizer_profile_candidates must be non-empty.")
        if "balanced" not in self.profile_candidates:
            raise ValueError("optimizer_profile_candidates must include 'balanced'.")
        if len(self.optimizer_candidates) <= 0:
            raise ValueError("optimizer_candidates must be non-empty.")
        if len(self.resource_factors) <= 0:
            raise ValueError("resource_factors must be non-empty.")

        # Grouping provider: CEC uses existing partition files; WSN uses configurable split.
        self.grouping_result = self._load_grouping_for_problem()
        if len(self.grouping_result) != self.n_agents:
            raise ValueError(
                f"Grouping size mismatch for problem {self.question} "
                f"(family={self.problem_family}, inner_id={self.inner_question}) under divide_method={self.divide_method}: "
                f"existing grouping has {len(self.grouping_result)} groups, "
                f"but fixed_agent_num={self.n_agents}. "
                f"Please align fixed_agent_num or use function ids with consistent group counts."
            )

        per_agent_nvec_parts = (
            [len(self.optimizer_candidates)]
            + [len(self.profile_candidates)] * self.cfg_param_num
            + [len(self.resource_factors)]
        )
        if bool(int(getattr(self.opts, "objective_split_comm_action_enable", 0))):
            per_agent_nvec_parts.append(
                len(getattr(self.opts, "objective_split_comm_round_candidates", [1]))
            )
        if bool(int(getattr(self.opts, "objective_split_collab_action_enable", 0))):
            per_agent_nvec_parts.append(
                len(getattr(self.opts, "objective_split_collab_modes", ["consensus"]))
            )
        if bool(int(getattr(self.opts, "objective_split_guide_scale_action_enable", 0))):
            per_agent_nvec_parts.append(
                len(getattr(self.opts, "objective_split_guide_scale_candidates", [1.0]))
            )
        if bool(
            int(
                getattr(
                    self.opts,
                    "objective_split_candidate_actuator_action_enable",
                    0,
                )
            )
        ):
            per_agent_nvec_parts.append(
                len(
                    getattr(
                        self.opts,
                        "objective_split_candidate_actuator_candidates",
                        [0.0, 0.25, 0.5],
                    )
                )
            )
        per_agent_nvec = np.asarray(per_agent_nvec_parts, dtype=np.int64)
        # Semantics: action shape is [n_agents, len(per_agent_nvec)].
        self.action_space = spaces.MultiDiscrete(np.tile(per_agent_nvec, (self.n_agents, 1)))
        self.obs_dim = 16
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_agents, self.obs_dim), dtype=np.float32
        )

        self.current_x = np.zeros((self.D,), dtype=np.float64)
        self.current_f = np.inf
        self.gbest_x = np.zeros((self.D,), dtype=np.float64)
        self.gbest_f = np.inf
        self.sum_fes = 0
        self.step_count = 0

        self.last_action_idx = np.zeros((self.n_agents,), dtype=np.int64)      # profile head
        self.last_optimizer_idx = np.zeros((self.n_agents,), dtype=np.int64)   # optimizer head
        self.last_resource_idx = np.zeros((self.n_agents,), dtype=np.int64)    # resource head
        self.last_local_improve = np.zeros((self.n_agents,), dtype=np.float64)
        self.last_team_improve = 0.0
        self.local_improve_hist: List[List[float]] = [[] for _ in range(self.n_agents)]
        # new interaction-aware features
        self.last_commit_success = 0.0  # shared
        self.last_joint_vs_mean_local_gap = 0.0  # shared
        self.last_local_vs_joint_gap = np.zeros((self.n_agents,), dtype=np.float64)  # per-agent
        self.last_local_centered = np.zeros((self.n_agents,), dtype=np.float64)  # per-agent

        # single recorder for current subproblem evaluations (memory-efficient containers)
        # fitness: scalar stream
        self.current_eval_fitness_record = array('d')
        # individual: flattened full-space vectors stream, contiguous float32-like values
        self.current_eval_individual_record = array('f')
        # lightweight parameter cache for profile=inherit
        self.param_state_cache: List[Dict[str, Dict[str, float]]] = [dict() for _ in range(self.n_agents)]

    def _resolve_problem_identity(self, external_question: int):
        q = int(external_question)
        mode = str(self.benchmark_name)
        if mode == "CEC2013LSGO":
            return "CEC2013LSGO", q
        if mode == "DBOF1F10":
            return "DBOF1F10", q
        if mode == "CDOCompetition":
            return "CDOCompetition", q
        if mode == "CDOBenchF1F14":
            return "CDOBenchF1F14", q
        if mode == "CDOBenchF1F15":
            return "CDOBenchF1F15", q
        if mode == "WSNLocation":
            return "WSNLocation", q
        if mode == "WSNLocationMASOIE":
            return "WSNLocationMASOIE", q
        if mode == "Mixed":
            # CEC uses native ids 1..15
            if 1 <= q <= 15:
                return "CEC2013LSGO", q
            # WSN uses shifted ids: offset + k, k starts from 1
            k = q - int(self.wsn_id_offset)
            wsn_k_max = int(len(getattr(self.opts, "wsn_target_num_list", [10, 20, 30, 40, 50])))
            if 1 <= k <= wsn_k_max:
                return "WSNLocation", k
            km = q - int(self.masoie_wsn_id_offset)
            masoie_k_max = int(len(getattr(self.opts, "masoie_wsn_target_num_list", [5])))
            if 1 <= km <= masoie_k_max:
                return "WSNLocationMASOIE", km
            raise ValueError(
                f"Mixed benchmark id {q} is invalid. "
                f"Valid: CEC 1..15 or WSN {self.wsn_id_offset + 1}..{self.wsn_id_offset + wsn_k_max} "
                f"(offset={self.wsn_id_offset}) or MASOIE-WSN "
                f"{self.masoie_wsn_id_offset + 1}..{self.masoie_wsn_id_offset + masoie_k_max} "
                f"(offset={self.masoie_wsn_id_offset})."
            )
        raise ValueError(f"Unsupported benchmark_name: {self.benchmark_name}")

    def _resolve_max_fes_for_problem(self, default_max_fes: int) -> int:
        """
        Resolve per-problem max_fes, mainly for WSN reproduction-style settings.
        """
        if self.problem_family == "WSNLocation":
            lst = list(getattr(self.opts, "wsn_max_fes_list", []))
        elif self.problem_family == "WSNLocationMASOIE":
            lst = list(getattr(self.opts, "masoie_wsn_max_fes_list", []))
        elif self.problem_family == "DBOF1F10":
            lst = list(getattr(self.opts, "dbof1f10_max_fes_list", []))
        elif self.problem_family == "CDOCompetition":
            lst = list(getattr(self.opts, "cdo_max_fes_list", []))
        elif self.problem_family in {"CDOBenchF1F14", "CDOBenchF1F15"}:
            lst = list(getattr(self.opts, "cdo_bench_max_fes_list", []))
        else:
            return int(default_max_fes)
        if len(lst) == 0:
            return int(default_max_fes)
        k = int(self.inner_question) - 1  # WSN inner ids are 1..K
        if 0 <= k < len(lst):
            return int(lst[k])
        return int(default_max_fes)

    def _load_grouping_for_problem(self) -> List[List[int]]:
        if self.problem_family in {"WSNLocation", "WSNLocationMASOIE", "DBOF1F10", "CDOCompetition", "CDOBenchF1F14", "CDOBenchF1F15"}:
            mode = str(getattr(self.opts, "wsn_grouping_mode", "equal_split")).lower()
            if mode == "target_block":
                coord_dim = int(getattr(self.opts, "wsn_coordinate_dim", 3))
                if self.problem_family == "WSNLocationMASOIE":
                    coord_dim = 3
                elif self.problem_family in {"DBOF1F10", "CDOCompetition", "CDOBenchF1F14", "CDOBenchF1F15"}:
                    coord_dim = int(getattr(self.opts, "dbof1f10_dim", 100))
                groups = self._build_target_block_grouping(self.D, coord_dim)
                if len(groups) != self.n_agents:
                    # Keep MAPPO fixed-agent shape compatible by fallback.
                    groups = self._build_equal_grouping(self.D, self.n_agents)
            else:
                groups = self._build_equal_grouping(self.D, self.n_agents)
            return groups
        return self._load_existing_grouping(self.inner_question)

    def _replace_dims(self, base_x: np.ndarray, dims: List[int], sub_x: np.ndarray) -> np.ndarray:
        x_eval = base_x.copy()
        x_eval[np.asarray(dims, dtype=np.int64)] = sub_x
        return x_eval

    def _profile_name(self, profile_idx: int) -> str:
        return str(self.profile_candidates[int(profile_idx)]).lower()

    def _resolve_profile_name(self, agent_id: int, optimizer_name: str, profile_idx: int) -> str:
        p = self._profile_name(profile_idx)
        if p == "numeric_inherit":
            p = "inherit"
        if p != "inherit":
            return p
        cache_i = self.param_state_cache[int(agent_id)]
        if str(optimizer_name).lower() in cache_i:
            return "inherit"
        return "balanced"

    def _resolve_level(self, agent_id: int, optimizer_name: str, level_idx: int, key: str) -> str:
        p = self._profile_name(level_idx)
        if p == "numeric_inherit":
            p = "inherit"
        if p != "inherit":
            return p
        cache_i = self.param_state_cache[int(agent_id)]
        rec = cache_i.get(str(optimizer_name).lower(), None)
        if isinstance(rec, dict) and key in rec:
            return "inherit"
        return "balanced"

    def _sigma_ref(self, agent_id: int, optimizer_name: str) -> float:
        cache_i = self.param_state_cache[int(agent_id)]
        rec = cache_i.get(str(optimizer_name).lower(), None)
        if isinstance(rec, dict) and ("sigma" in rec):
            try:
                s = float(rec.get("sigma"))
                if np.isfinite(s) and s > 0:
                    return s
            except Exception:
                pass
        s0 = float(getattr(self.opts, "sigma", 0.3))
        if not np.isfinite(s0) or s0 <= 0:
            s0 = 0.3
        return float(s0)

    def _sigma_anchor(self) -> float:
        """
        Fixed sigma anchor for explicit non-inherit sigma levels.
        inherit follows cached optimizer state; conservative/balanced/aggressive
        are always mapped from this base so their semantics do not drift.
        """
        s0 = float(getattr(self.opts, "sigma", 0.3))
        if not np.isfinite(s0) or s0 <= 0:
            s0 = 0.3
        return float(s0)

    def _uses_a1_param_semantics(self) -> bool:
        return self.mappo_param_semantics == "a1"

    def _uses_pre_caf4a62_arch(self) -> bool:
        return self.mappo_action_arch == "pre_caf4a62"

    def _explicit_sigma_base(self, sigma_ref: float) -> float:
        return float(sigma_ref if self._uses_a1_param_semantics() else self._sigma_anchor())

    def _sigma_level_map(self) -> Dict[str, float]:
        if self._uses_a1_param_semantics():
            return {"conservative": 0.5, "balanced": 1.0, "aggressive": 2.0}
        return {"conservative": 2.0 / 3.0, "balanced": 1.0, "aggressive": 2.0}

    @staticmethod
    def _level_map_from_values(values, default_values, cast=float) -> Dict[str, float]:
        raw = values if values is not None else default_values
        try:
            vals = list(raw)
        except TypeError:
            vals = list(default_values)
        if len(vals) != 3:
            vals = list(default_values)
        return {
            "conservative": cast(vals[0]),
            "balanced": cast(vals[1]),
            "aggressive": cast(vals[2]),
        }

    def _mmes_ms_level_map(self) -> Dict[str, int]:
        return self._level_map_from_values(
            getattr(self.opts, "subopt_mmes_ms_levels", [2, 4, 6]),
            [2, 4, 6],
            cast=int,
        )

    def _vkd_k_init_level_map(self, dim: int) -> Dict[str, int]:
        limit = int(max(0, dim - 1))
        base = self._level_map_from_values(
            getattr(self.opts, "subopt_vkd_k_init_levels", [0, 2, 4]),
            [0, 2, 4],
            cast=int,
        )
        return {k: int(max(0, min(v, limit))) for k, v in base.items()}

    def _vkd_kmax_level_map(self, dim: int) -> Dict[str, int]:
        limit = int(max(0, dim - 1))
        base = self._level_map_from_values(
            getattr(self.opts, "subopt_vkd_kmax_levels", [8, 32, 64]),
            [8, 32, 64],
            cast=int,
        )
        return {k: int(max(0, min(v, limit))) for k, v in base.items()}

    def _vkd_cs_level_map(self) -> Dict[str, float]:
        scale_map = self._level_map_from_values(
            getattr(self.opts, "subopt_vkd_cs_scale_levels", [0.5, 1.0, 2.0]),
            [0.5, 1.0, 2.0],
            cast=float,
        )
        base_cs = 0.3
        return {k: float(max(1e-12, base_cs * v)) for k, v in scale_map.items()}

    def _vkd_k_inc_cond_level_map(self) -> Dict[str, float]:
        return self._level_map_from_values(
            getattr(self.opts, "subopt_vkd_k_inc_cond_levels", [10.0, 30.0, 60.0]),
            [10.0, 30.0, 60.0],
            cast=float,
        )

    def _vkd_action_param_mode(self) -> str:
        mode = str(getattr(self.opts, "subopt_vkd_action_param_mode", "rank")).lower()
        return mode if mode in {"rank", "tpa_rank"} else "rank"

    def _cma_cov_level_map(self) -> Dict[str, float]:
        return self._level_map_from_values(
            getattr(self.opts, "subopt_cmaes_cov_lr_scale_levels", [0.5, 1.0, 1.5]),
            [0.5, 1.0, 1.5],
            cast=float,
        )

    def _cma_cs_level_map(self) -> Dict[str, float]:
        return self._level_map_from_values(
            getattr(self.opts, "subopt_cmaes_c_s_scale_levels", [0.7, 1.0, 1.3]),
            [0.7, 1.0, 1.3],
            cast=float,
        )

    def _sep_cov_level_map(self) -> Dict[str, float]:
        return self._level_map_from_values(
            getattr(self.opts, "subopt_sepcmaes_cov_lr_scale_levels", [0.5, 1.0, 1.5]),
            [0.5, 1.0, 1.5],
            cast=float,
        )

    def _sep_cs_level_map(self) -> Dict[str, float]:
        return self._level_map_from_values(
            getattr(self.opts, "subopt_sepcmaes_c_s_scale_levels", [0.7, 1.0, 1.3]),
            [0.7, 1.0, 1.3],
            cast=float,
        )

    def _allow_subopt_manual_override(self) -> bool:
        return bool(self.subopt_manual_override or self._uses_a1_param_semantics())

    def _build_optimizer_options_pre_caf4a62(
        self,
        agent_id: int,
        optimizer_name: str,
        profile_idx: int,
        dims: List[int],
        x_base: np.ndarray,
        subfes_i: int,
        seed: int,
    ) -> Dict:
        optimizer_name = str(optimizer_name).lower()
        dim = int(len(dims))
        dims_arr = np.asarray(dims, dtype=np.int64)
        prof = self._resolve_profile_name(agent_id, optimizer_name, profile_idx)
        sigma_ref = self._sigma_ref(agent_id, optimizer_name)

        if prof == "conservative":
            sigma_mul = 0.5
        elif prof == "aggressive":
            sigma_mul = 2.0
        else:
            sigma_mul = 1.0

        options: Dict = {
            "max_function_evaluations": int(subfes_i),
            "mean": x_base[dims_arr].astype(np.float64).copy(),
            "sigma": float(max(1e-12, sigma_ref * sigma_mul)),
            "is_restart": False,
            "verbose": False,
            "seed_rng": int(seed),
            "use_custom_learning_rates": False,
        }

        if optimizer_name == "mmes":
            base_lambda = int(max(4, 4 + int(3 * np.log(max(2, dim)))))
            base_m = int(max(2, 2 * int(np.ceil(np.sqrt(max(1, dim))))))
            ms_map = self._mmes_ms_level_map()
            if prof == "conservative":
                options["n_individuals"] = int(max(4, int(0.75 * base_lambda)))
                options["m"] = int(max(2, int(0.5 * base_m)))
                options["ms"] = int(ms_map["conservative"])
            elif prof == "aggressive":
                options["n_individuals"] = int(max(4, int(1.5 * base_lambda)))
                options["m"] = int(max(2, int(1.5 * base_m)))
                options["ms"] = int(ms_map["aggressive"])
            elif prof == "inherit":
                rec = self.param_state_cache[int(agent_id)].get("mmes", {})
                if isinstance(rec, dict):
                    for k in ("n_individuals", "m", "ms", "c_s", "a_z", "c_a", "gamma"):
                        if k in rec:
                            options[k] = rec[k]
            else:
                options["n_individuals"] = int(base_lambda)
                options["m"] = int(base_m)
                options["ms"] = int(ms_map["balanced"])
            options.setdefault("c_s", float(getattr(self.opts, "subopt_mmes_c_s", 0.3)))
            options.setdefault("a_z", float(getattr(self.opts, "subopt_mmes_a_z", 0.05)))
            c_a_cfg = float(getattr(self.opts, "subopt_mmes_c_a", -1.0))
            if c_a_cfg > 0.0:
                options.setdefault("c_a", c_a_cfg)
            gamma_cfg = float(getattr(self.opts, "subopt_mmes_gamma", -1.0))
            if gamma_cfg > 0.0:
                options.setdefault("gamma", gamma_cfg)
            nind_cfg = int(getattr(self.opts, "subopt_mmes_n_individuals", -1))
            if nind_cfg > 0:
                options["n_individuals"] = int(max(2, nind_cfg))
            m_cfg = int(getattr(self.opts, "subopt_mmes_m", -1))
            if m_cfg > 0:
                options["m"] = int(max(2, m_cfg))
            ms_cfg = int(getattr(self.opts, "subopt_mmes_ms", -1))
            if ms_cfg > 0:
                options["ms"] = int(max(1, ms_cfg))
            for k in ("n_individuals", "m", "ms", "distance"):
                if k in options:
                    options[k] = int(options[k])
        elif optimizer_name == "vkd":
            base_lam = int(max(4, 4 + int(3 * np.log(max(2, dim)))))
            kinit_map = self._vkd_k_init_level_map(dim)
            kmax_map = self._vkd_kmax_level_map(dim)
            if prof == "conservative":
                lam = int(max(4, int(0.75 * base_lam)))
                k_init = int(kinit_map["conservative"])
                kmax = int(kmax_map["conservative"])
            elif prof == "aggressive":
                lam = int(max(4, int(1.5 * base_lam)))
                k_init = int(kinit_map["aggressive"])
                kmax = int(kmax_map["aggressive"])
            elif prof == "inherit":
                rec = self.param_state_cache[int(agent_id)].get("vkd", {})
                lam = int(rec.get("n_individuals", rec.get("lam", base_lam))) if isinstance(rec, dict) else int(base_lam)
                k_init = int(rec.get("k_init", 0)) if isinstance(rec, dict) else 0
                kmax = int(rec.get("kmax", min(max(0, dim - 1), 32))) if isinstance(rec, dict) else int(min(max(0, dim - 1), 32))
            else:
                lam = int(base_lam)
                k_init = int(kinit_map["balanced"])
                kmax = int(kmax_map["balanced"])
            options["n_individuals"] = int(max(2, lam))
            options["k_init"] = int(max(0, k_init))
            options["kmax"] = int(max(0, kmax))
            vkd_nind_cfg = int(getattr(self.opts, "subopt_vkd_n_individuals", -1))
            if vkd_nind_cfg > 0:
                options["n_individuals"] = int(max(2, vkd_nind_cfg))
            vkd_kinit_cfg = int(getattr(self.opts, "subopt_vkd_k_init", -1))
            if vkd_kinit_cfg >= 0:
                options["k_init"] = int(max(0, min(vkd_kinit_cfg, dim - 1)))
            vkd_kmax_cfg = int(getattr(self.opts, "subopt_vkd_kmax", -1))
            if vkd_kmax_cfg >= 0:
                options["kmax"] = int(max(0, min(vkd_kmax_cfg, dim - 1)))
        elif optimizer_name in {"cmaes", "sepcmaes"}:
            base_lam = int(max(4, 4 + int(3 * np.log(max(2, dim)))))
            if prof == "conservative":
                lam = int(max(4, int(0.75 * base_lam)))
            elif prof == "aggressive":
                lam = int(max(4, int(1.5 * base_lam)))
            elif prof == "inherit":
                rec = self.param_state_cache[int(agent_id)].get(optimizer_name, {})
                lam = int(rec.get("n_individuals", base_lam)) if isinstance(rec, dict) else int(base_lam)
            else:
                lam = int(base_lam)
            options["n_individuals"] = int(max(2, lam))
            if optimizer_name == "cmaes":
                cma_nind_cfg = int(getattr(self.opts, "subopt_cmaes_n_individuals", -1))
                if cma_nind_cfg > 0:
                    options["n_individuals"] = int(max(2, cma_nind_cfg))
            else:
                sep_nind_cfg = int(getattr(self.opts, "subopt_sepcmaes_n_individuals", -1))
                if sep_nind_cfg > 0:
                    options["n_individuals"] = int(max(2, sep_nind_cfg))

        return options

    def _build_optimizer_options(
        self,
        agent_id: int,
        optimizer_name: str,
        cfg_levels: List[int],
        dims: List[int],
        x_base: np.ndarray,
        subfes_i: int,
        seed: int,
    ) -> Dict:
        optimizer_name = str(optimizer_name).lower()
        dim = int(len(dims))
        dims_arr = np.asarray(dims, dtype=np.int64)
        sigma_ref = self._sigma_ref(agent_id, optimizer_name)
        explicit_sigma_base = self._explicit_sigma_base(sigma_ref)

        cfg_levels = [int(x) for x in cfg_levels]
        if len(cfg_levels) != self.cfg_param_num:
            raise ValueError(f"cfg_levels length mismatch: expected {self.cfg_param_num}, got {len(cfg_levels)}")
        if self._uses_pre_caf4a62_arch():
            return self._build_optimizer_options_pre_caf4a62(
                agent_id=agent_id,
                optimizer_name=optimizer_name,
                profile_idx=int(cfg_levels[0]),
                dims=dims,
                x_base=x_base,
                subfes_i=subfes_i,
                seed=seed,
            )

        options: Dict = {
            "max_function_evaluations": int(subfes_i),
            "mean": x_base[dims_arr].astype(np.float64).copy(),
            "is_restart": False,
            "verbose": False,
            "seed_rng": int(seed),
        }

        if optimizer_name == "mmes":
            base_lambda = int(max(4, 4 + int(3 * np.log(max(2, dim)))))
            base_m = int(max(2, 2 * int(np.ceil(np.sqrt(max(1, dim))))))
            lv_sigma = self._resolve_level(agent_id, optimizer_name, cfg_levels[0], "sigma")
            lv_lam = self._resolve_level(agent_id, optimizer_name, cfg_levels[1], "n_individuals")
            lv_m = self._resolve_level(agent_id, optimizer_name, cfg_levels[2], "m")
            lv_ms = self._resolve_level(agent_id, optimizer_name, cfg_levels[3], "ms")

            sigma_map = self._sigma_level_map()
            if lv_sigma == "inherit":
                options["sigma"] = float(max(1e-12, sigma_ref))
            else:
                options["sigma"] = float(max(1e-12, explicit_sigma_base * sigma_map.get(lv_sigma, 1.0)))

            lam_map = {
                "conservative": int(max(4, int(0.75 * base_lambda))),
                "balanced": int(base_lambda),
                "aggressive": int(max(4, int(1.5 * base_lambda))),
            }
            m_map = {
                "conservative": int(max(2, int(0.5 * base_m))),
                "balanced": int(base_m),
                "aggressive": int(max(2, int(1.5 * base_m))),
            }
            ms_map = self._mmes_ms_level_map()
            rec = self.param_state_cache[int(agent_id)].get("mmes", {})
            if lv_lam == "inherit" and isinstance(rec, dict) and ("n_individuals" in rec):
                options["n_individuals"] = int(rec["n_individuals"])
            else:
                options["n_individuals"] = int(lam_map.get(lv_lam, lam_map["balanced"]))
            if lv_m == "inherit" and isinstance(rec, dict) and ("m" in rec):
                options["m"] = int(rec["m"])
            else:
                options["m"] = int(m_map.get(lv_m, m_map["balanced"]))
            if lv_ms == "inherit" and isinstance(rec, dict) and ("ms" in rec):
                options["ms"] = int(rec["ms"])
            else:
                options["ms"] = int(ms_map.get(lv_ms, ms_map["balanced"]))
            # keep existing project defaults as stable anchors
            options.setdefault("c_s", float(getattr(self.opts, "subopt_mmes_c_s", 0.3)))
            options.setdefault("a_z", float(getattr(self.opts, "subopt_mmes_a_z", 0.05)))
            if self._allow_subopt_manual_override():
                c_a_cfg = float(getattr(self.opts, "subopt_mmes_c_a", -1.0))
                if c_a_cfg > 0.0:
                    options.setdefault("c_a", c_a_cfg)
                gamma_cfg = float(getattr(self.opts, "subopt_mmes_gamma", -1.0))
                if gamma_cfg > 0.0:
                    options.setdefault("gamma", gamma_cfg)
                nind_cfg = int(getattr(self.opts, "subopt_mmes_n_individuals", -1))
                if nind_cfg > 0:
                    options["n_individuals"] = int(max(2, nind_cfg))
                m_cfg = int(getattr(self.opts, "subopt_mmes_m", -1))
                if m_cfg > 0:
                    options["m"] = int(max(2, m_cfg))
                ms_cfg = int(getattr(self.opts, "subopt_mmes_ms", -1))
                if ms_cfg > 0:
                    options["ms"] = int(max(1, ms_cfg))
            # type safety for MMES integer hyper-parameters
            for k in ("n_individuals", "m", "ms", "distance"):
                if k in options:
                    options[k] = int(options[k])
        elif optimizer_name == "vkd":
            base_lam = int(max(4, 4 + int(3 * np.log(max(2, dim)))))
            vkd_mode = self._vkd_action_param_mode()
            lv_sigma = self._resolve_level(agent_id, optimizer_name, cfg_levels[0], "sigma")
            lv_lam = self._resolve_level(agent_id, optimizer_name, cfg_levels[1], "n_individuals")
            lv_p2_name = "cs" if vkd_mode == "tpa_rank" else "k_init"
            lv_p3_name = "k_inc_cond" if vkd_mode == "tpa_rank" else "kmax"
            lv_p2 = self._resolve_level(agent_id, optimizer_name, cfg_levels[2], lv_p2_name)
            lv_p3 = self._resolve_level(agent_id, optimizer_name, cfg_levels[3], lv_p3_name)

            sigma_map = self._sigma_level_map()
            if lv_sigma == "inherit":
                options["sigma"] = float(max(1e-12, sigma_ref))
            else:
                options["sigma"] = float(max(1e-12, explicit_sigma_base * sigma_map.get(lv_sigma, 1.0)))

            lam_map = {
                "conservative": int(max(4, int(0.75 * base_lam))),
                "balanced": int(base_lam),
                "aggressive": int(max(4, int(1.5 * base_lam))),
            }
            kinit_map = self._vkd_k_init_level_map(dim)
            kmax_map = self._vkd_kmax_level_map(dim)
            rec = self.param_state_cache[int(agent_id)].get("vkd", {})
            if lv_lam == "inherit" and isinstance(rec, dict) and ("n_individuals" in rec):
                options["n_individuals"] = int(rec["n_individuals"])
            else:
                options["n_individuals"] = int(max(2, lam_map.get(lv_lam, lam_map["balanced"])))
            if vkd_mode == "tpa_rank":
                cs_map = self._vkd_cs_level_map()
                k_inc_cond_map = self._vkd_k_inc_cond_level_map()
                options["k_init"] = 0
                options["kmax"] = int(max(0, dim - 1))
                if lv_p2 == "inherit" and isinstance(rec, dict) and ("cs" in rec):
                    options["cs"] = float(rec["cs"])
                else:
                    options["cs"] = float(cs_map.get(lv_p2, cs_map["balanced"]))
                if lv_p3 == "inherit" and isinstance(rec, dict) and ("k_inc_cond" in rec):
                    k_inc_cond = float(rec["k_inc_cond"])
                else:
                    k_inc_cond = float(k_inc_cond_map.get(lv_p3, k_inc_cond_map["balanced"]))
                options["k_inc_cond"] = float(max(1e-12, k_inc_cond))
                options["k_dec_cond"] = float(options["k_inc_cond"])
            else:
                if lv_p2 == "inherit" and isinstance(rec, dict) and ("k_init" in rec):
                    options["k_init"] = int(rec["k_init"])
                else:
                    options["k_init"] = int(max(0, kinit_map.get(lv_p2, kinit_map["balanced"])))
                if lv_p3 == "inherit" and isinstance(rec, dict) and ("kmax" in rec):
                    options["kmax"] = int(rec["kmax"])
                else:
                    options["kmax"] = int(max(0, kmax_map.get(lv_p3, kmax_map["balanced"])))
            if self._allow_subopt_manual_override():
                vkd_nind_cfg = int(getattr(self.opts, "subopt_vkd_n_individuals", -1))
                if vkd_nind_cfg > 0:
                    options["n_individuals"] = int(max(2, vkd_nind_cfg))
                vkd_kinit_cfg = int(getattr(self.opts, "subopt_vkd_k_init", -1))
                if vkd_kinit_cfg >= 0:
                    options["k_init"] = int(max(0, min(vkd_kinit_cfg, dim - 1)))
                vkd_kmax_cfg = int(getattr(self.opts, "subopt_vkd_kmax", -1))
                if vkd_kmax_cfg >= 0:
                    options["kmax"] = int(max(0, min(vkd_kmax_cfg, dim - 1)))
        elif optimizer_name in {"cmaes", "sepcmaes"}:
            base_lam = int(max(4, 4 + int(3 * np.log(max(2, dim)))))
            lv_sigma = self._resolve_level(agent_id, optimizer_name, cfg_levels[0], "sigma")
            lv_lam = self._resolve_level(agent_id, optimizer_name, cfg_levels[1], "n_individuals")
            lv_cov = self._resolve_level(agent_id, optimizer_name, cfg_levels[2], "cov_lr_scale")
            lv_cs = self._resolve_level(agent_id, optimizer_name, cfg_levels[3], "c_s_scale")
            sigma_map = self._sigma_level_map()
            if lv_sigma == "inherit":
                options["sigma"] = float(max(1e-12, sigma_ref))
            else:
                options["sigma"] = float(max(1e-12, explicit_sigma_base * sigma_map.get(lv_sigma, 1.0)))
            rec = self.param_state_cache[int(agent_id)].get(optimizer_name, {})
            if optimizer_name == "cmaes":
                if self._uses_a1_param_semantics():
                    lam_map = {
                        "conservative": int(max(4, int(0.75 * base_lam))),
                        "balanced": int(base_lam),
                        "aggressive": int(max(4, int(1.5 * base_lam))),
                    }
                    cov_scale_map = {"conservative": 0.75, "balanced": 1.0, "aggressive": 1.25}
                    cs_scale_map = {"conservative": 0.75, "balanced": 1.0, "aggressive": 1.25}
                else:
                    # Design doc alignment (full CMAES):
                    # n_individuals: inherit / max(6,0.75*base) / base / 1.25*base
                    lam_map = {
                        "conservative": int(max(6, int(0.75 * base_lam))),
                        "balanced": int(base_lam),
                        "aggressive": int(max(6, int(1.25 * base_lam))),
                    }
                    cov_scale_map = self._cma_cov_level_map()
                    cs_scale_map = self._cma_cs_level_map()

                if lv_lam == "inherit" and isinstance(rec, dict) and ("n_individuals" in rec):
                    options["n_individuals"] = int(rec["n_individuals"])
                else:
                    options["n_individuals"] = int(max(2, lam_map.get(lv_lam, lam_map["balanced"])))
                if lv_cov == "inherit" and isinstance(rec, dict) and ("cov_lr_scale" in rec):
                    options["cov_lr_scale"] = float(rec["cov_lr_scale"])
                else:
                    options["cov_lr_scale"] = float(cov_scale_map.get(lv_cov, 1.0))
                if lv_cs == "inherit" and isinstance(rec, dict) and ("c_s_scale" in rec):
                    options["c_s_scale"] = float(rec["c_s_scale"])
                else:
                    options["c_s_scale"] = float(cs_scale_map.get(lv_cs, 1.0))

                if self._allow_subopt_manual_override():
                    cma_nind_cfg = int(getattr(self.opts, "subopt_cmaes_n_individuals", -1))
                    if cma_nind_cfg > 0:
                        options["n_individuals"] = int(max(2, cma_nind_cfg))
                    cov_override = float(getattr(self.opts, "subopt_cmaes_cov_lr_scale", -1.0))
                    if cov_override > 0.0:
                        options["cov_lr_scale"] = float(cov_override)
                    cs_override = float(getattr(self.opts, "subopt_cmaes_c_s_scale", -1.0))
                    if cs_override > 0.0:
                        options["c_s_scale"] = float(cs_override)
            else:
                # SepCMAES keeps the A1 population map in both modes; current mode
                # differs from A1 in cov/c_s scales and fixed sigma anchoring.
                lam_map = {
                    "conservative": int(max(4, int(0.75 * base_lam))),
                    "balanced": int(base_lam),
                    "aggressive": int(max(4, int(1.5 * base_lam))),
                }
                if self._uses_a1_param_semantics():
                    cov_scale_map = {"conservative": 0.75, "balanced": 1.0, "aggressive": 1.25}
                    cs_scale_map = {"conservative": 0.75, "balanced": 1.0, "aggressive": 1.25}
                else:
                    cov_scale_map = self._sep_cov_level_map()
                    cs_scale_map = self._sep_cs_level_map()

                if lv_lam == "inherit" and isinstance(rec, dict) and ("n_individuals" in rec):
                    options["n_individuals"] = int(rec["n_individuals"])
                else:
                    options["n_individuals"] = int(max(2, lam_map.get(lv_lam, lam_map["balanced"])))
                if lv_cov == "inherit" and isinstance(rec, dict) and ("cov_lr_scale" in rec):
                    options["cov_lr_scale"] = float(rec["cov_lr_scale"])
                else:
                    options["cov_lr_scale"] = float(cov_scale_map.get(lv_cov, 1.0))
                if lv_cs == "inherit" and isinstance(rec, dict) and ("c_s_scale" in rec):
                    options["c_s_scale"] = float(rec["c_s_scale"])
                else:
                    options["c_s_scale"] = float(cs_scale_map.get(lv_cs, 1.0))

                if self._allow_subopt_manual_override():
                    sep_nind_cfg = int(getattr(self.opts, "subopt_sepcmaes_n_individuals", -1))
                    if sep_nind_cfg > 0:
                        options["n_individuals"] = int(max(2, sep_nind_cfg))
                    ccov_override = float(getattr(self.opts, "subopt_sepcmaes_c_cov_scale", -1.0))
                    if ccov_override > 0.0:
                        options["cov_lr_scale"] = float(ccov_override)
                    cs_override = float(getattr(self.opts, "subopt_sepcmaes_c_s_scale", -1.0))
                    if cs_override > 0.0:
                        options["c_s_scale"] = float(cs_override)

        return options

    def _run_subproblem(
        self,
        optimizer_name: str,
        problem: Dict,
        options: Dict,
        x_base: np.ndarray,
        base_f: float,
        dims: List[int],
    ) -> Dict:
        # clear single recorder at run start
        del self.current_eval_fitness_record[:]
        del self.current_eval_individual_record[:]

        optimizer = create_optimizer(optimizer_name, problem, options)
        res = optimizer.optimize()

        best_sub_x = res["best_so_far_x"].copy()
        n_eval = int(res["n_function_evaluations"])
        x_candidate = self._replace_dims(x_base, dims, best_sub_x)
        f_candidate = float(self.fun(x_candidate))

        local_improve = _safe_log_improvement(float(base_f), f_candidate)

        # Convert recorder to compact numpy arrays for downstream usage/serialization.
        sub_fitness_record = np.array(self.current_eval_fitness_record, dtype=np.float64, copy=True)
        if self.record_eval_individual:
            if len(self.current_eval_individual_record) == 0:
                sub_individual_record = np.empty((0, self.D), dtype=np.float32)
            else:
                sub_individual_record = np.array(
                    self.current_eval_individual_record, dtype=np.float32, copy=True
                ).reshape(-1, self.D)
        else:
            sub_individual_record = np.empty((0, self.D), dtype=np.float32)

        # clear single recorder at run end
        del self.current_eval_fitness_record[:]
        del self.current_eval_individual_record[:]

        return {
            "best_sub_x": best_sub_x,
            "x_candidate": x_candidate,
            "f_candidate": f_candidate,
            "local_improve": local_improve,
            "n_eval": n_eval,
            "fitness_record": sub_fitness_record,
            "individual_record": sub_individual_record,
        }

    def _build_obs(self) -> np.ndarray:
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        rem_ratio = max(0.0, (self.max_fes - self.sum_fes) / max(1, self.max_fes))
        step_ratio = self.step_count / max(1, self.episode_steps)
        gbest_norm = float(np.tanh(np.log10(abs(self.gbest_f) + 1.0) / 10.0))

        for i, dims in enumerate(self.grouping_result):
            dim_ratio = len(dims) / max(1, self.D)
            action_ratio = self.last_action_idx[i] / max(1, len(self.profile_candidates) - 1)
            opt_ratio = self.last_optimizer_idx[i] / max(1, len(self.optimizer_candidates) - 1)
            res_ratio = self.last_resource_idx[i] / max(1, len(self.resource_factors) - 1)
            local_last = self.last_local_improve[i]
            local_mean = float(np.mean(self.local_improve_hist[i][-5:])) if self.local_improve_hist[i] else 0.0

            obs[i, 0] = dim_ratio
            obs[i, 1] = action_ratio
            obs[i, 2] = local_last
            obs[i, 3] = self.subfes_per_agent / max(1, self.max_fes)
            obs[i, 4] = rem_ratio
            obs[i, 5] = step_ratio
            obs[i, 6] = local_mean
            obs[i, 7] = self.last_team_improve
            obs[i, 8] = gbest_norm
            obs[i, 9] = opt_ratio
            obs[i, 10] = res_ratio
            # new features from contact.md
            obs[i, 11] = self.last_commit_success
            obs[i, 12] = self.last_joint_vs_mean_local_gap
            obs[i, 13] = self.last_local_vs_joint_gap[i]
            obs[i, 14] = self.last_local_centered[i]
            obs[i, 15] = float(i) / float(max(1, self.n_agents - 1))

        return obs

    def reset(self):
        self.current_x = np.zeros((self.D,), dtype=np.float64)
        self.current_f = float(self.fun(self.current_x))
        self.gbest_x = self.current_x.copy()
        self.gbest_f = self.current_f

        self.sum_fes = 0
        self.step_count = 0
        self.last_action_idx.fill(0)
        self.last_optimizer_idx.fill(0)
        self.last_resource_idx.fill(0)
        self.last_local_improve.fill(0.0)
        self.last_team_improve = 0.0
        self.local_improve_hist = [[] for _ in range(self.n_agents)]
        self.last_commit_success = 0.0
        self.last_joint_vs_mean_local_gap = 0.0
        self.last_local_vs_joint_gap.fill(0.0)
        self.last_local_centered.fill(0.0)
        del self.current_eval_fitness_record[:]
        del self.current_eval_individual_record[:]
        self.param_state_cache = [dict() for _ in range(self.n_agents)]

        return self._build_obs()

    def export_state(self) -> Dict:
        """
        Export lightweight environment state for two-stage continuation.
        """
        return {
            "current_x": self.current_x.astype(np.float64, copy=True).tolist(),
            "current_f": float(self.current_f),
            "gbest_x": self.gbest_x.astype(np.float64, copy=True).tolist(),
            "gbest_f": float(self.gbest_f),
            "sum_fes": int(self.sum_fes),
            "step_count": int(self.step_count),
            "last_action_idx": self.last_action_idx.astype(np.int64, copy=True).tolist(),
            "last_optimizer_idx": self.last_optimizer_idx.astype(np.int64, copy=True).tolist(),
            "last_resource_idx": self.last_resource_idx.astype(np.int64, copy=True).tolist(),
            "last_local_improve": self.last_local_improve.astype(np.float64, copy=True).tolist(),
            "last_team_improve": float(self.last_team_improve),
            "local_improve_hist": [[float(v) for v in h] for h in self.local_improve_hist],
            "last_commit_success": float(self.last_commit_success),
            "last_joint_vs_mean_local_gap": float(self.last_joint_vs_mean_local_gap),
            "last_local_vs_joint_gap": self.last_local_vs_joint_gap.astype(np.float64, copy=True).tolist(),
            "last_local_centered": self.last_local_centered.astype(np.float64, copy=True).tolist(),
            "param_state_cache": self.param_state_cache,
        }

    def import_state(self, state: Dict):
        """
        Import state previously exported by export_state().
        """
        self.current_x = np.asarray(state.get("current_x", self.current_x), dtype=np.float64).reshape(self.D)
        self.current_f = float(state.get("current_f", self.current_f))
        self.gbest_x = np.asarray(state.get("gbest_x", self.gbest_x), dtype=np.float64).reshape(self.D)
        self.gbest_f = float(state.get("gbest_f", self.gbest_f))
        self.sum_fes = int(state.get("sum_fes", self.sum_fes))
        self.step_count = int(state.get("step_count", self.step_count))

        self.last_action_idx = np.asarray(
            state.get("last_action_idx", self.last_action_idx), dtype=np.int64
        ).reshape(self.n_agents)
        self.last_optimizer_idx = np.asarray(
            state.get("last_optimizer_idx", self.last_optimizer_idx), dtype=np.int64
        ).reshape(self.n_agents)
        self.last_resource_idx = np.asarray(
            state.get("last_resource_idx", self.last_resource_idx), dtype=np.int64
        ).reshape(self.n_agents)
        self.last_local_improve = np.asarray(
            state.get("last_local_improve", self.last_local_improve), dtype=np.float64
        ).reshape(self.n_agents)
        self.last_team_improve = float(state.get("last_team_improve", self.last_team_improve))
        hist = state.get("local_improve_hist", None)
        if isinstance(hist, list) and len(hist) == self.n_agents:
            self.local_improve_hist = [[float(v) for v in h] for h in hist]
        self.last_commit_success = float(state.get("last_commit_success", self.last_commit_success))
        self.last_joint_vs_mean_local_gap = float(
            state.get("last_joint_vs_mean_local_gap", self.last_joint_vs_mean_local_gap)
        )
        self.last_local_vs_joint_gap = np.asarray(
            state.get("last_local_vs_joint_gap", self.last_local_vs_joint_gap), dtype=np.float64
        ).reshape(self.n_agents)
        self.last_local_centered = np.asarray(
            state.get("last_local_centered", self.last_local_centered), dtype=np.float64
        ).reshape(self.n_agents)
        psc = state.get("param_state_cache", None)
        if isinstance(psc, list) and len(psc) == self.n_agents:
            self.param_state_cache = psc

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        actions = np.asarray(actions)
        if actions.ndim == 2 and actions.shape == (self.n_agents, 3):
            opt_actions = np.clip(actions[:, 0].astype(np.int64), 0, len(self.optimizer_candidates) - 1)
            cfg_actions = np.clip(actions[:, 1].astype(np.int64), 0, len(self.profile_candidates) - 1)
            res_actions = np.clip(actions[:, 2].astype(np.int64), 0, len(self.resource_factors) - 1)
            cfg_actions_block = np.repeat(cfg_actions[:, None], self.cfg_param_num, axis=1)
        elif actions.ndim == 2 and actions.shape == (self.n_agents, 2 + self.cfg_param_num):
            opt_actions = np.clip(actions[:, 0].astype(np.int64), 0, len(self.optimizer_candidates) - 1)
            cfg_actions_block = np.clip(
                actions[:, 1 : 1 + self.cfg_param_num].astype(np.int64),
                0,
                len(self.profile_candidates) - 1,
            )
            cfg_actions = cfg_actions_block[:, 0].copy()
            res_actions = np.clip(actions[:, 1 + self.cfg_param_num].astype(np.int64), 0, len(self.resource_factors) - 1)
        else:
            raise ValueError(
                f"Expected actions shape [A,3] or [A,{2 + self.cfg_param_num}] with A={self.n_agents}, got {actions.shape}."
            )

        self.last_optimizer_idx = opt_actions.copy()
        self.last_action_idx = cfg_actions.copy()
        self.last_resource_idx = res_actions.copy()

        # Parallel-sync semantics: all subproblems evaluate against same base snapshot.
        x_base = self.current_x.copy()
        f_prev = self.current_f
        step_fitness_chunks: List[np.ndarray] = []
        step_individual_chunks: List[np.ndarray] = []

        proposals = []
        for i, dims in enumerate(self.grouping_result):
            if len(dims) <= 0:
                raise ValueError(
                    f"Invalid empty group at index {i} for problem {self.question} "
                    f"(family={self.problem_family}). Please check grouping settings."
                )
            optimizer_name = self.optimizer_candidates[int(opt_actions[i])]
            subfes_i = max(1, int(round(self.subfes_per_agent * float(self.resource_factors[int(res_actions[i])]))))
            seed = int(self.opts.seed + self.step_count * 1000 + i)

            # Build fitness function / problem / options at step layer.
            def fitness_sub(z_batch: np.ndarray, _dims=dims, _x_base=x_base):
                z_batch = np.asarray(z_batch, dtype=np.float64)
                dims_arr = np.asarray(_dims, dtype=np.int64)
                # Reuse workspace to reduce repeated allocations in hot path.
                if not hasattr(fitness_sub, "_workspace"):
                    fitness_sub._workspace = None
                    fitness_sub._workspace_n = 0

                if z_batch.ndim == 1:
                    z_batch = z_batch[None, :]
                elif z_batch.ndim == 3 and z_batch.shape[1] == 1:
                    z_batch = np.squeeze(z_batch, axis=1)

                if z_batch.ndim != 2 or z_batch.shape[1] != len(_dims):
                    raise ValueError(f"fitness_sub expected [N,{len(_dims)}], got {z_batch.shape}")
                n_batch = int(z_batch.shape[0])
                d_total = int(_x_base.shape[0])
                if (fitness_sub._workspace is None) or (fitness_sub._workspace_n < n_batch):
                    fitness_sub._workspace = np.empty((n_batch, d_total), dtype=np.float64)
                    fitness_sub._workspace_n = n_batch
                x_eval_batch = fitness_sub._workspace[:n_batch]
                x_eval_batch[:] = _x_base[None, :]
                x_eval_batch[:, dims_arr] = z_batch

                vals = np.asarray(self.fun(x_eval_batch), dtype=np.float64).reshape(-1)

                # record each evaluation (batched)
                self.current_eval_fitness_record.extend(vals.tolist())
                if self.record_eval_individual:
                    self.current_eval_individual_record.extend(
                        x_eval_batch.astype(np.float32, copy=False).ravel()
                    )

                return vals

            problem = {
                "fitness_function": fitness_sub,
                "ndim_problem": len(dims),
                "lower_boundary": self.lb * np.ones((len(dims),), dtype=np.float64),
                "upper_boundary": self.ub * np.ones((len(dims),), dtype=np.float64),
            }
            options = self._build_optimizer_options(
                agent_id=i,
                optimizer_name=optimizer_name,
                cfg_levels=cfg_actions_block[i].tolist(),
                dims=dims,
                x_base=x_base,
                subfes_i=subfes_i,
                seed=seed,
            )
            p = self._run_subproblem(optimizer_name, problem, options, x_base, f_prev, dims)
            # keep lightweight parameter state for inherit profile in future steps
            int_keys = {"n_individuals", "m", "ms", "k_init", "kmax", "lam", "distance"}
            float_keys = {"sigma", "c_s", "a_z", "c_a", "gamma", "cs", "ds", "k_inc_cond", "k_dec_cond"}
            if not self._uses_pre_caf4a62_arch():
                float_keys = set(float_keys) | {"cov_lr_scale", "c_s_scale"}
            rec = {}
            for k, v in options.items():
                if k in int_keys:
                    rec[k] = int(v)
                elif k in float_keys:
                    rec[k] = float(v)
            self.param_state_cache[int(i)][str(optimizer_name).lower()] = rec
            proposals.append(p)
            step_fitness_chunks.append(p["fitness_record"])
            # optional heavy record; keep for compatibility with opt.py style
            if self.record_eval_individual and p["individual_record"].size > 0:
                step_individual_chunks.append(p["individual_record"])

        # Commit once after all agents finish the round.
        x_commit = x_base.copy()
        for i, dims in enumerate(self.grouping_result):
            x_commit[np.asarray(dims, dtype=np.int64)] = proposals[i]["best_sub_x"]

        f_commit = float(self.fun(x_commit))
        # Commit evaluation is also a real function evaluation:
        # include it in records and FE accounting.
        step_fitness_chunks.append(np.asarray([f_commit], dtype=np.float64))
        if self.record_eval_individual:
            step_individual_chunks.append(x_commit.astype(np.float32, copy=False)[None, :])
        if f_commit <= self.current_f:
            self.current_x = x_commit
            self.current_f = f_commit

        if self.current_f <= self.gbest_f:
            self.gbest_f = self.current_f
            self.gbest_x = self.current_x.copy()

        self.last_team_improve = _safe_log_improvement(f_prev, self.current_f)
        self.last_commit_success = 1.0 if f_commit <= f_prev else 0.0

        local_improvements = []
        total_evals = 0
        for i, p in enumerate(proposals):
            li = float(p["local_improve"])
            self.last_local_improve[i] = li
            self.local_improve_hist[i].append(li)
            local_improvements.append(li)
            total_evals += int(p["n_eval"])
        total_evals += 1  # x_commit evaluation

        # derived features from local/global improvements
        if len(local_improvements) > 0:
            local_arr = np.asarray(local_improvements, dtype=np.float64)
            mean_local_improve = float(np.mean(local_arr))
        else:
            local_arr = np.zeros((self.n_agents,), dtype=np.float64)
            mean_local_improve = 0.0

        team_improve = float(self.last_team_improve)
        self.last_joint_vs_mean_local_gap = float(team_improve - mean_local_improve)
        self.last_local_vs_joint_gap = (local_arr - team_improve).astype(np.float64, copy=False)
        self.last_local_centered = (local_arr - mean_local_improve).astype(np.float64, copy=False)

        self.sum_fes += total_evals
        self.step_count += 1

        # Terminate only by true evaluation budget.
        done = (self.sum_fes >= self.max_fes)
        obs_next = self._build_obs()

        if step_fitness_chunks:
            step_fitness_record = np.concatenate(step_fitness_chunks, axis=0)
        else:
            step_fitness_record = np.empty((0,), dtype=np.float64)

        if self.record_eval_individual and step_individual_chunks:
            step_individual_record = np.concatenate(step_individual_chunks, axis=0).astype(np.float32, copy=False)
        else:
            step_individual_record = np.empty((0, self.D), dtype=np.float32)

        # Stage-A reward shaping:
        #   r_i = alpha * team + (1-alpha) * local_centered_i
        team_reward = float(self.last_team_improve)
        local_centered = self.last_local_centered.astype(np.float32, copy=False)
        mixed_reward = (
            float(self.reward_team_weight) * team_reward
            + float(self.reward_local_weight) * local_centered
        ).astype(np.float32, copy=False)
        info = {
            "team_improve": self.last_team_improve,
            "local_improve": np.asarray(local_improvements, dtype=np.float32),
            "current_f": self.current_f,
            "gbest_fitness": self.gbest_f,
            "sumFEs": self.sum_fes,
            "step": self.step_count,
            "fitness_record": step_fitness_record,
            "individual_record": step_individual_record,
            "commit_success_flag": float(self.last_commit_success),
            "joint_vs_mean_local_gap": float(self.last_joint_vs_mean_local_gap),
            "local_vs_joint_gap": self.last_local_vs_joint_gap.astype(np.float32, copy=False),
            "local_centered": self.last_local_centered.astype(np.float32, copy=False),
            "team_reward": float(team_reward),
            "mixed_reward": mixed_reward,
        }
        return obs_next, mixed_reward, bool(done), info
