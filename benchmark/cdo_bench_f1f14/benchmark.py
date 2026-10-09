from pathlib import Path
from typing import Dict, List

import numpy as np

from benchmark._cdo_shared.math_functions import (
    FUNC_MAP_VEC,
    _get_elliptic_w,
    _griewank_vec,
    _read_matrix,
    _rosenbrock_vec,
    _schwefel_vec,
    _t_asy_vec,
    _t_osz_vec,
)


def _default_data_root() -> Path:
    return Path(__file__).resolve().parent / "data"


CDO_BENCH_F1F14_CONFIG: Dict[int, Dict] = {
    1: {"base": "elliptic", "heterogeneous": False, "node_num": 20, "weight": 100.0, "shift": True, "rotate": True},
    2: {"base": "schwefel", "heterogeneous": False, "node_num": 20, "weight": 100.0, "shift": True, "rotate": True},
    3: {"base": "rosenbrock", "heterogeneous": False, "node_num": 20, "weight": 100.0, "shift": True, "rotate": True},
    4: {"base": "griewank", "heterogeneous": False, "node_num": 20, "weight": 100.0, "shift": True, "rotate": True},
    5: {"base": ["elliptic", "schwefel"], "heterogeneous": True, "node_num": 20, "weight": 100.0, "shift": True, "rotate": True},
    6: {"base": ["elliptic", "rosenbrock"], "heterogeneous": True, "node_num": 20, "weight": 100.0, "shift": True, "rotate": True},
    7: {"base": ["schwefel", "rosenbrock"], "heterogeneous": True, "node_num": 20, "weight": 100.0, "shift": True, "rotate": True},
    8: {"base": ["elliptic", "griewank"], "heterogeneous": True, "node_num": 20, "weight": 100.0, "shift": True, "rotate": True},
    9: {"base": ["schwefel", "griewank"], "heterogeneous": True, "node_num": 20, "weight": 100.0, "shift": True, "rotate": True},
    10: {"base": ["rosenbrock", "griewank"], "heterogeneous": True, "node_num": 20, "weight": 100.0, "shift": True, "rotate": True},
    11: {"base": "schwefel", "heterogeneous": False, "node_num": 20, "weight": 100.0, "shift": False, "rotate": True},
    12: {"base": "rosenbrock", "heterogeneous": False, "node_num": 20, "weight": 100.0, "shift": False, "rotate": True},
    13: {"base": "schwefel", "heterogeneous": False, "node_num": 20, "weight": 100.0, "shift": True, "rotate": False},
    14: {"base": "rosenbrock", "heterogeneous": False, "node_num": 20, "weight": 100.0, "shift": True, "rotate": False},
}


class CDOBenchF1F14Function:
    def __init__(self, func_id: int, data_root: Path):
        self.func_id = int(func_id)
        if self.func_id not in CDO_BENCH_F1F14_CONFIG:
            raise ValueError(f"CDOBenchF1F14 function id {self.func_id} is out of range. Available: 1..14.")
        self.cfg = dict(CDO_BENCH_F1F14_CONFIG[self.func_id])
        self.data_root = Path(data_root)
        self.node_num = int(self.cfg["node_num"])
        self.dimension = 100
        self.lower = -100.0
        self.upper = 100.0
        self.weight = float(self.cfg["weight"])
        self.do_shift = bool(self.cfg["shift"])
        self.do_rotate = bool(self.cfg["rotate"])
        self.best = 0.0
        self.family = "CDOBenchF1F14"

        a_path = self.data_root / f"A_{self.node_num}n{self.dimension}D"
        w_path = self.data_root / f"W_{self.node_num}n"
        r_path = self.data_root / f"R_{self.dimension}D"
        xopt_path = self.data_root / f"xopt_{self.dimension}D"

        self.A = _read_matrix(a_path)
        self.W = _read_matrix(w_path)
        self.R = _read_matrix(r_path) if self.do_rotate else np.eye(self.dimension, dtype=np.float64)
        self.xopt = _read_matrix(xopt_path) if self.do_shift else np.zeros((self.dimension,), dtype=np.float64)

        missing = []
        if self.A is None:
            missing.append(a_path.name)
        if self.W is None:
            missing.append(w_path.name)
        if self.do_rotate and self.R is None:
            missing.append(r_path.name)
        if self.do_shift and self.xopt is None:
            missing.append(xopt_path.name)
        if missing:
            raise FileNotFoundError(
                f"Missing CDOBenchF1F14 F{self.func_id} data under {self.data_root}: {', '.join(missing)}"
            )

        self.A = np.asarray(self.A, dtype=np.float64)
        self.W = np.asarray(self.W, dtype=np.float64)
        self.R = np.asarray(self.R, dtype=np.float64)
        self.xopt = np.asarray(self.xopt, dtype=np.float64).reshape(-1)
        if self.A.shape[0] < self.node_num:
            raise ValueError(f"CDO A data has insufficient node rows: need {self.node_num}, got {self.A.shape[0]}")
        if self.A.shape[1] != self.dimension:
            raise ValueError(f"CDO A data dim mismatch: expected {self.dimension}, got {self.A.shape[1]}")
        if self.R.shape != (self.dimension, self.dimension):
            raise ValueError(f"CDO R data must be [{self.dimension},{self.dimension}], got {self.R.shape}")
        if self.xopt.shape[0] != self.dimension:
            raise ValueError(f"CDO xopt data dim mismatch: expected {self.dimension}, got {self.xopt.shape[0]}")

        base = self.cfg["base"]
        if isinstance(base, str):
            self.base_keys = [base] * self.node_num
        else:
            self.base_keys = [base[i % 2] for i in range(self.node_num)]
        self.base_funcs = [FUNC_MAP_VEC[k] for k in self.base_keys]
        self.base_key_counts: Dict[str, int] = {}
        for key in self.base_keys:
            self.base_key_counts[key] = int(self.base_key_counts.get(key, 0) + 1)
        self.A_used = self.A[: self.node_num, :]
        self.A_row_sum = np.sum(self.A_used, axis=0, dtype=np.float64)

    def _to_z(self, x_batch: np.ndarray) -> np.ndarray:
        x = np.asarray(x_batch, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError(f"CDOBenchF1F14 F{self.func_id} expected [N,{self.dimension}], got {x.shape}")
        z = x - self.xopt[None, :]
        if self.do_rotate:
            z = z @ self.R.T
        return z

    def _local_eval_from_z(self, agent_id: int, z_batch: np.ndarray) -> np.ndarray:
        aid = int(agent_id)
        if aid < 0 or aid >= self.node_num:
            raise ValueError(f"agent_id out of range: {aid}, node_num={self.node_num}")
        z = np.asarray(z_batch, dtype=np.float64).reshape(-1, self.dimension)
        f_elem = self.base_funcs[aid](z.copy())
        linear = z @ self.A[aid, :] * self.weight
        return np.nan_to_num(f_elem + linear, nan=1e300, posinf=1e300, neginf=-1e300)

    def local_eval_batch(self, agent_id: int, x_batch: np.ndarray) -> np.ndarray:
        return self._local_eval_from_z(int(agent_id), self._to_z(x_batch))

    def local_eval_all_batch(self, x_batch: np.ndarray) -> np.ndarray:
        z = self._to_z(x_batch)
        vals = np.empty((z.shape[0], self.node_num), dtype=np.float64)
        for i in range(self.node_num):
            vals[:, i] = self._local_eval_from_z(i, z)
        return np.nan_to_num(vals, nan=1e300, posinf=1e300, neginf=-1e300)

    def __call__(self, x):
        z = self._to_z(x)
        key_set = set(self.base_key_counts.keys())
        z_osz = None
        z_osz_asy = None
        if ("elliptic" in key_set) or ("schwefel" in key_set):
            z_osz = _t_osz_vec(z)
        if "schwefel" in key_set:
            z_osz_asy = _t_asy_vec(z_osz, beta=0.2)

        vals = np.zeros((z.shape[0],), dtype=np.float64)
        for key, count in self.base_key_counts.items():
            if key == "elliptic":
                w = _get_elliptic_w(z.shape[1])
                f_val = (z_osz * z_osz) @ w
            elif key == "schwefel":
                f_val = np.sum(np.cumsum(z_osz_asy, axis=1) ** 2, axis=1)
            elif key == "rosenbrock":
                f_val = _rosenbrock_vec(z)
            elif key == "griewank":
                f_val = _griewank_vec(z)
            else:
                f_val = FUNC_MAP_VEC[key](z)
            vals += float(count) * f_val
        vals += self.weight * (z @ self.A_row_sum)
        vals /= float(self.node_num)
        return np.nan_to_num(vals, nan=1e300, posinf=1e300, neginf=-1e300)

    def info(self) -> Dict:
        return {
            "best": self.best,
            "dimension": self.dimension,
            "lower": self.lower,
            "upper": self.upper,
            "node_num": self.node_num,
            "family": self.family,
            "data_root": str(self.data_root),
            "base": self.cfg["base"],
            "heterogeneous": bool(self.cfg["heterogeneous"]),
            "weight": self.weight,
            "if_shift": self.do_shift,
            "if_rotate": self.do_rotate,
        }


class Benchmark:
    def __init__(self, opts=None):
        self.opts = opts
        if opts is not None and str(getattr(opts, "cdo_bench_data_root", "")).strip():
            self.data_root = Path(str(getattr(opts, "cdo_bench_data_root"))).expanduser().resolve()
        else:
            self.data_root = _default_data_root()
        if not self.data_root.exists():
            raise FileNotFoundError(f"CDOBenchF1F14 data root not found: {self.data_root}")
        self.problem_ids = list(range(1, 15))
        self._func_cache: Dict[int, CDOBenchF1F14Function] = {}

    def get_function(self, func_id: int) -> CDOBenchF1F14Function:
        fid = int(func_id)
        if fid not in self._func_cache:
            self._func_cache[fid] = CDOBenchF1F14Function(fid, self.data_root)
        return self._func_cache[fid]

    def get_info(self, func_id: int) -> Dict:
        return self.get_function(func_id).info()

    def get_num_functions(self) -> int:
        return len(self.problem_ids)

    def get_function_names(self) -> List[str]:
        return [f"CDO_Bench_F{i}" for i in self.problem_ids]
