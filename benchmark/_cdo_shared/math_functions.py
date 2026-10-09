from typing import Dict
import numpy as np


def _read_matrix(path):
    try:
        return np.loadtxt(str(path), dtype=np.float64)
    except Exception:
        return None

_ELLIPTIC_W_CACHE: Dict[int, np.ndarray] = {}
_GRIEWANK_SQRT_IDX_CACHE: Dict[int, np.ndarray] = {}
_ASY_EXPV_CACHE: Dict[int, np.ndarray] = {}


def _get_elliptic_w(d: int) -> np.ndarray:
    d = int(d)
    w = _ELLIPTIC_W_CACHE.get(d)
    if w is None:
        w = 10 ** (6.0 * np.arange(d, dtype=np.float64) / max(d - 1, 1))
        _ELLIPTIC_W_CACHE[d] = w
    return w


def _get_griewank_sqrt_idx(d: int) -> np.ndarray:
    d = int(d)
    idx = _GRIEWANK_SQRT_IDX_CACHE.get(d)
    if idx is None:
        idx = np.sqrt(np.arange(1, d + 1, dtype=np.float64))
        _GRIEWANK_SQRT_IDX_CACHE[d] = idx
    return idx


def _get_asy_expv(d: int, beta: float) -> np.ndarray:
    d = int(d)
    expv = _ASY_EXPV_CACHE.get(d)
    if expv is None:
        idx = np.arange(d, dtype=np.float64)
        expv = beta * idx / max(d - 1, 1)
        _ASY_EXPV_CACHE[d] = expv
    return expv


def _t_osz_vec(z: np.ndarray) -> np.ndarray:
    hat = np.where(z != 0, np.log(np.abs(z) + 1e-300), 0.0)
    sgn = np.sign(z)
    c1 = np.where(z > 0, 10.0, 5.5)
    c2 = np.where(z > 0, 7.9, 3.1)
    return sgn * np.exp(hat + 0.049 * (np.sin(c1 * hat) + np.sin(c2 * hat)))


def _t_asy_vec(z: np.ndarray, beta: float = 0.2) -> np.ndarray:
    _, d = z.shape
    out = z.copy()
    expv = _get_asy_expv(d, beta)
    rows, cols = np.where(z > 0)
    with np.errstate(over="ignore", invalid="ignore"):
        out[rows, cols] = z[rows, cols] ** (1.0 + expv[cols] * np.sqrt(z[rows, cols]))
    return out


def _elliptic_vec(z: np.ndarray) -> np.ndarray:
    z2 = _t_osz_vec(z)
    d = z2.shape[1]
    w = _get_elliptic_w(d)
    return z2 ** 2 @ w


def _schwefel_vec(z: np.ndarray) -> np.ndarray:
    z2 = _t_osz_vec(z)
    z2 = _t_asy_vec(z2, beta=0.2)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.sum(np.cumsum(z2, axis=1) ** 2, axis=1)


def _rosenbrock_vec(z: np.ndarray) -> np.ndarray:
    z0 = z[:, :-1]
    z1 = z[:, 1:]
    t1 = z0 * z0 - z1
    t2 = z0 - 1.0
    return np.sum(100.0 * t1 * t1 + t2 * t2, axis=1)


def _griewank_vec(z: np.ndarray) -> np.ndarray:
    d = z.shape[1]
    sqrt_idx = _get_griewank_sqrt_idx(d)
    return np.sum(z ** 2, axis=1) / 4000.0 - np.prod(np.cos(z / sqrt_idx), axis=1) + 1.0


FUNC_MAP_VEC = {
    "elliptic": _elliptic_vec,
    "schwefel": _schwefel_vec,
    "rosenbrock": _rosenbrock_vec,
    "griewank": _griewank_vec,
}


