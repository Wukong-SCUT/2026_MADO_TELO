import os
from pathlib import Path
from typing import Dict, List

import numpy as np


def _read_matrix(path: Path) -> np.ndarray:
    data = np.loadtxt(str(path), dtype=np.float64)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"WSN W matrix must be square, got shape={arr.shape} from {path}")
    return arr


def _measurement_rss_log(distance: np.ndarray) -> np.ndarray:
    d = np.maximum(np.asarray(distance, dtype=np.float64), 1e-12)
    return 100.0 - 20.0 * np.log10(d)


class WSNFunction:
    """
    Wireless Sensor Network localization objective.
    Decision vector x encodes all target coordinates:
      x = [t0_x, t0_y, t0_z, t1_x, ...]
    Objective:
      mean squared error between predicted sensor measurements and observed measurements.
    """

    def __init__(
        self,
        source_xyz: np.ndarray,
        target_xyz: np.ndarray,
        node_num: int,
        target_num: int,
        coordinate_dim: int,
        lower: float,
        upper: float,
        measurement: str = "RSS",
        metric_mode: str = "ccsa_readme",
        noisy: bool = True,
        noisy_degree: float = 0.1,
        seed: int = 42,
        W: np.ndarray = None,
        w_source: str = "",
    ):
        self.node_num = int(node_num)
        self.target_num = int(target_num)
        self.coordinate_dim = int(coordinate_dim)
        self.dimension = int(self.target_num * self.coordinate_dim)
        self.lower = float(lower)
        self.upper = float(upper)
        self.best = 0.0

        self.measurement = str(measurement).upper()
        if self.measurement != "RSS":
            raise ValueError(f"Unsupported WSN measurement type: {measurement}. Only RSS is currently supported.")
        self.metric_mode = str(metric_mode).lower()
        if self.metric_mode not in {"ccsa_readme", "normalized"}:
            raise ValueError(f"Unsupported WSN metric_mode: {metric_mode}")

        self.source = np.asarray(source_xyz, dtype=np.float64)[: self.node_num, : self.coordinate_dim]
        self.target = np.asarray(target_xyz, dtype=np.float64)[: self.target_num, : self.coordinate_dim]
        if self.source.shape[0] < self.node_num:
            raise ValueError(f"source locations are insufficient: need {self.node_num}, got {self.source.shape[0]}")
        if self.target.shape[0] < self.target_num:
            raise ValueError(f"target locations are insufficient: need {self.target_num}, got {self.target.shape[0]}")

        self.rng = np.random.default_rng(int(seed))
        self.observed = self._build_observed(noisy=bool(noisy), noisy_degree=float(noisy_degree))
        if W is not None:
            weight = np.asarray(W, dtype=np.float64)
            if weight.shape != (self.node_num, self.node_num):
                raise ValueError(
                    f"WSN W shape mismatch: expected {(self.node_num, self.node_num)}, got {weight.shape}."
                )
            self.W = weight
            self.w_source = str(w_source or "W_16_WSN_location")
        else:
            self.w_source = ""

    def _predict(self, x_batch: np.ndarray) -> np.ndarray:
        # x_batch: [N, D] where D = target_num * coordinate_dim
        x_batch = np.asarray(x_batch, dtype=np.float64).reshape(-1, self.target_num, self.coordinate_dim)
        # diff: [N, node_num, target_num, coord]
        diff = x_batch[:, None, :, :] - self.source[None, :, None, :]
        dist = np.linalg.norm(diff, axis=-1)
        return _measurement_rss_log(dist)  # [N, node_num, target_num]

    def _build_observed(self, noisy: bool, noisy_degree: float) -> np.ndarray:
        true_pred = self._predict(self.target.reshape(1, -1))[0]  # [node_num, target_num]
        if noisy:
            true_pred = true_pred + self.rng.normal(0.0, noisy_degree, size=true_pred.shape)
        return true_pred.astype(np.float64, copy=False)

    def __call__(self, x):
        local_vals = self.local_eval_all_batch(x)
        return np.mean(local_vals, axis=1).astype(np.float64, copy=False)

    def local_eval_all_batch(self, x_batch: np.ndarray) -> np.ndarray:
        """
        Return all sensor-local objectives f_i(x) for each candidate x.

        Shape: [N, node_num]. The global WSN objective used by __call__ is the
        mean over this second axis, so F(x) = mean_i f_i(x).
        """
        x = np.asarray(x_batch, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2:
            raise ValueError(f"local_eval_all_batch expected [N,D] or [D], got {x.shape}")
        if x.shape[1] != self.dimension:
            raise ValueError(f"local_eval_all_batch dimension mismatch: expected {self.dimension}, got {x.shape[1]}")

        pred = self._predict(x)  # [N, node_num, target_num]
        err2 = (pred - self.observed[None, :, :]) ** 2
        if self.metric_mode == "ccsa_readme":
            local_vals = np.sum(err2, axis=2)
        else:
            local_vals = np.mean(err2, axis=2)
        return local_vals.astype(np.float64, copy=False)

    def local_eval_batch(self, agent_id: int, x_batch: np.ndarray) -> np.ndarray:
        """
        Sensor-local objective f_i(x).

        This is the objective-decomposition view used by distributed WSN
        optimizers: each sensor agent evaluates the full target-location vector
        with only its own measurements.
        """
        aid = int(agent_id)
        if aid < 0 or aid >= self.node_num:
            raise ValueError(f"agent_id out of range: {aid}, node_num={self.node_num}")
        x = np.asarray(x_batch, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2:
            raise ValueError(f"local_eval_batch expected [N,D] or [D], got {x.shape}")
        if x.shape[1] != self.dimension:
            raise ValueError(f"local_eval_batch dimension mismatch: expected {self.dimension}, got {x.shape[1]}")

        target_x = x.reshape(-1, self.target_num, self.coordinate_dim)
        diff = target_x - self.source[aid][None, None, :]
        dist = np.linalg.norm(diff, axis=-1)
        pred = _measurement_rss_log(dist)
        err2 = (pred - self.observed[aid][None, :]) ** 2
        if self.metric_mode == "ccsa_readme":
            local_vals = np.sum(err2, axis=1)
        else:
            local_vals = np.mean(err2, axis=1)
        return local_vals.astype(np.float64, copy=False)

    def global_eval(self, x):
        return self.__call__(x)

    def info(self) -> Dict:
        return {
            "best": self.best,
            "dimension": self.dimension,
            "lower": self.lower,
            "upper": self.upper,
            "node_num": self.node_num,
            "target_num": self.target_num,
            "coordinate_dim": self.coordinate_dim,
            "measurement": self.measurement,
            "metric_mode": self.metric_mode,
            "has_W": bool(hasattr(self, "W")),
            "w_source": self.w_source,
        }


class Benchmark:
    """
    WSN benchmark manager, aligned with CEC interface:
      - get_function(func_id)
      - get_info(func_id)
      - get_num_functions()
    """

    def __init__(self, opts=None):
        self.opts = opts
        self._func_cache: Dict[int, WSNFunction] = {}

        target_num_list = [10, 20, 30, 40, 50]
        if opts is not None and getattr(opts, "wsn_target_num_list", None):
            target_num_list = [int(x) for x in getattr(opts, "wsn_target_num_list")]

        self.problem_specs: Dict[int, Dict] = {}
        for i, t in enumerate(target_num_list, start=1):
            self.problem_specs[i] = {
                "target_num": int(t),
                "node_num": int(getattr(opts, "wsn_node_num", 16)) if opts is not None else 16,
                "coordinate_dim": int(getattr(opts, "wsn_coordinate_dim", 3)) if opts is not None else 3,
                "measurement": str(getattr(opts, "wsn_measurement", "RSS")) if opts is not None else "RSS",
                "metric_mode": str(getattr(opts, "wsn_metric_mode", "ccsa_readme")) if opts is not None else "ccsa_readme",
                "noisy": bool(int(getattr(opts, "wsn_noisy", 1))) if opts is not None else True,
                "noisy_degree": float(getattr(opts, "wsn_noisy_degree", 0.1)) if opts is not None else 0.1,
                "lower": float(getattr(opts, "wsn_lower_bound", -100.0)) if opts is not None else -100.0,
                "upper": float(getattr(opts, "wsn_upper_bound", 100.0)) if opts is not None else 100.0,
            }

        base_data_root = None
        if opts is not None and getattr(opts, "wsn_data_root", None):
            base_data_root = Path(getattr(opts, "wsn_data_root")).expanduser().resolve()
        if base_data_root is None:
            base_data_root = Path(__file__).resolve().parent / "data"
        self.data_root = base_data_root
        self.source_path = str(self.data_root / "source_WSN_location")
        self.target_path = str(self.data_root / "target_WSN_location")

        if not os.path.exists(self.source_path) or not os.path.exists(self.target_path):
            raise FileNotFoundError(
                f"WSN data files not found under {self.data_root}. "
                "Expected source_WSN_location and target_WSN_location."
            )
        self.source_xyz = np.loadtxt(self.source_path, dtype=np.float64)
        self.target_xyz = np.loadtxt(self.target_path, dtype=np.float64)
        self.w_path = self.data_root / "W_16_WSN_location"
        self.W = _read_matrix(self.w_path)

    def _build_seed(self, func_id: int) -> int:
        base = int(getattr(self.opts, "seed", 42)) if self.opts is not None else 42
        return int(base + 10007 * int(func_id))

    def _build_function(self, func_id: int) -> WSNFunction:
        if int(func_id) not in self.problem_specs:
            raise ValueError(
                f"WSN function id {func_id} is out of range. "
                f"Available ids: {sorted(self.problem_specs.keys())}"
            )
        spec = self.problem_specs[int(func_id)]
        return WSNFunction(
            source_xyz=self.source_xyz,
            target_xyz=self.target_xyz,
            node_num=spec["node_num"],
            target_num=spec["target_num"],
            coordinate_dim=spec["coordinate_dim"],
            lower=spec["lower"],
            upper=spec["upper"],
            measurement=spec["measurement"],
            metric_mode=spec["metric_mode"],
            noisy=spec["noisy"],
            noisy_degree=spec["noisy_degree"],
            seed=self._build_seed(int(func_id)),
            W=self.W,
            w_source=str(self.w_path),
        )

    def get_function(self, func_id: int) -> WSNFunction:
        fid = int(func_id)
        if fid not in self._func_cache:
            self._func_cache[fid] = self._build_function(fid)
        return self._func_cache[fid]

    def get_info(self, func_id: int) -> Dict:
        return self.get_function(func_id).info()

    def get_num_functions(self) -> int:
        return int(len(self.problem_specs))

    def get_function_names(self) -> List[str]:
        names = []
        for fid in sorted(self.problem_specs.keys()):
            t = self.problem_specs[fid]["target_num"]
            n = self.problem_specs[fid]["node_num"]
            c = self.problem_specs[fid]["coordinate_dim"]
            m = self.problem_specs[fid]["measurement"]
            names.append(f"WSN_location_{m}_{c}d_{n}s_{t}t")
        return names
