from typing import Dict, List

import numpy as np


def _build_ring_weight(node_num: int) -> np.ndarray:
    n = int(node_num)
    if n < 3:
        raise ValueError("MASOIE WSN ring W requires at least 3 nodes.")
    W = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        W[i, i] = 0.5
        W[i, (i - 1) % n] = 0.25
        W[i, (i + 1) % n] = 0.25
    return W


class MASOIEWSNFunction:
    """
    MASOIE F15-style WSN localization objective.

    Decision vector:
      x = [target_0_xyz, target_1_xyz, ..., target_(Nt-1)_xyz], dim = Nt * 3

    Local objective at sensor i:
      f_i(x) = sum_t (phi_it - ||x_t - sensor_i||^2)^2
      where phi_it = ||true_target_t - sensor_i||^2 + noise

    Global objective:
      F(x) = mean_i f_i(x)
    """

    def __init__(
        self,
        node_num: int,
        target_num: int,
        space_size: float,
        noise_std: float,
        seed: int,
    ):
        self.node_num = int(node_num)
        self.target_num = int(target_num)
        self.coordinate_dim = 3
        self.dimension = int(self.target_num * self.coordinate_dim)
        self.lower = 0.0
        self.upper = float(space_size)
        self.best = 0.0
        self.noise_std = float(noise_std)
        self.W = _build_ring_weight(self.node_num)
        self.w_source = "masoie_ring_20n" if self.node_num == 20 else f"masoie_ring_{self.node_num}n"

        rng = np.random.default_rng(int(seed))
        self.sensor_pos = rng.uniform(0.0, self.upper, size=(self.node_num, 3))
        self.true_targets = rng.uniform(20.0, max(20.0, self.upper - 20.0), size=(self.target_num, 3))

        # Follow MASOIE implementation semantics: normal(0, noise_std^2).
        self.measurements = np.zeros((self.node_num, self.target_num), dtype=np.float64)
        noise_scale = float(self.noise_std ** 2)
        for i in range(self.node_num):
            for t in range(self.target_num):
                dist2 = float(np.sum((self.true_targets[t] - self.sensor_pos[i]) ** 2))
                noise = float(rng.normal(0.0, noise_scale))
                self.measurements[i, t] = dist2 + noise

    def _local_eval_batch(self, agent_id: int, x_batch: np.ndarray) -> np.ndarray:
        x_batch = np.asarray(x_batch, dtype=np.float64).reshape(-1, self.target_num, 3)
        sensor = self.sensor_pos[int(agent_id)]  # [3]
        phi = self.measurements[int(agent_id)]  # [Nt]

        # est_dist2: [N, Nt]
        est_dist2 = np.sum((x_batch - sensor[None, None, :]) ** 2, axis=2)
        err = phi[None, :] - est_dist2
        return np.sum(err ** 2, axis=1).astype(np.float64, copy=False)

    def local_eval_batch(self, agent_id: int, x_batch: np.ndarray) -> np.ndarray:
        """
        Sensor-local objective f_i(x).

        Shape:
          x_batch: [N, D] or [D]
          return:  [N]
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
        return self._local_eval_batch(aid, x)

    def local_eval_all_batch(self, x_batch: np.ndarray) -> np.ndarray:
        """
        Return all sensor-local objectives f_i(x) for each candidate x.

        Shape: [N, node_num]. The global MASOIE-WSN objective used by __call__
        is the mean over this second axis, so F(x) = mean_i f_i(x).
        """
        x = np.asarray(x_batch, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2:
            raise ValueError(f"local_eval_all_batch expected [N,D] or [D], got {x.shape}")
        if x.shape[1] != self.dimension:
            raise ValueError(f"local_eval_all_batch dimension mismatch: expected {self.dimension}, got {x.shape[1]}")

        vals = np.empty((x.shape[0], self.node_num), dtype=np.float64)
        for i in range(self.node_num):
            vals[:, i] = self._local_eval_batch(i, x)
        return vals.astype(np.float64, copy=False)

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2:
            raise ValueError(f"MASOIEWSNFunction expected [N,D] or [D], got {x.shape}")
        if x.shape[1] != self.dimension:
            raise ValueError(f"dimension mismatch: expected {self.dimension}, got {x.shape[1]}")

        vals = np.zeros((x.shape[0],), dtype=np.float64)
        for i in range(self.node_num):
            vals += self._local_eval_batch(i, x)
        vals /= float(self.node_num)
        return vals

    def info(self) -> Dict:
        return {
            "best": self.best,
            "dimension": self.dimension,
            "lower": self.lower,
            "upper": self.upper,
            "node_num": self.node_num,
            "target_num": self.target_num,
            "coordinate_dim": self.coordinate_dim,
            "noise_std": self.noise_std,
            "problem_style": "MASOIE_F15_WSN",
            "has_W": True,
            "w_source": self.w_source,
        }


class Benchmark:
    """
    MASOIE-style WSN benchmark manager, aligned with CEC interface.
    """

    def __init__(self, opts=None):
        self.opts = opts
        self._func_cache: Dict[int, MASOIEWSNFunction] = {}

        target_num_list = [5]
        if opts is not None and getattr(opts, "masoie_wsn_target_num_list", None):
            target_num_list = [int(x) for x in getattr(opts, "masoie_wsn_target_num_list")]

        self.problem_specs: Dict[int, Dict] = {}
        for i, t in enumerate(target_num_list, start=1):
            self.problem_specs[i] = {
                "target_num": int(t),
                "node_num": int(getattr(opts, "masoie_wsn_node_num", 20)) if opts is not None else 20,
                "space_size": float(getattr(opts, "masoie_wsn_space_size", 100.0)) if opts is not None else 100.0,
                "noise_std": float(getattr(opts, "masoie_wsn_noise_std", 2.0)) if opts is not None else 2.0,
            }

    def _build_seed(self, func_id: int) -> int:
        base = int(getattr(self.opts, "seed", 42)) if self.opts is not None else 42
        return int(base + 13013 * int(func_id))

    def _build_function(self, func_id: int) -> MASOIEWSNFunction:
        fid = int(func_id)
        if fid not in self.problem_specs:
            raise ValueError(
                f"MASOIE WSN function id {func_id} is out of range. "
                f"Available ids: {sorted(self.problem_specs.keys())}"
            )
        spec = self.problem_specs[fid]
        return MASOIEWSNFunction(
            node_num=spec["node_num"],
            target_num=spec["target_num"],
            space_size=spec["space_size"],
            noise_std=spec["noise_std"],
            seed=self._build_seed(fid),
        )

    def get_function(self, func_id: int) -> MASOIEWSNFunction:
        fid = int(func_id)
        if fid not in self._func_cache:
            self._func_cache[fid] = self._build_function(fid)
        return self._func_cache[fid]

    def get_info(self, func_id: int) -> Dict:
        return self.get_function(func_id).info()

    def get_num_functions(self) -> int:
        return int(len(self.problem_specs))

    def get_function_names(self) -> List[str]:
        out = []
        for fid in sorted(self.problem_specs.keys()):
            t = self.problem_specs[fid]["target_num"]
            n = self.problem_specs[fid]["node_num"]
            out.append(f"MASOIE_WSN_3d_{n}s_{t}t")
        return out
