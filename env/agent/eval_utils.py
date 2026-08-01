"""Small evaluation helpers used by the standalone C8c release."""

from concurrent.futures import ProcessPoolExecutor
import inspect
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
import numpy as np


def run_parallel_task(target_func, parallel_num, *args, **kwargs):
    """Run repeated evaluations and inject each worker index into the last argument."""
    worker_count = int(parallel_num)
    if worker_count <= 0:
        raise ValueError("parallel_num must be positive")

    parameters = list(inspect.signature(target_func).parameters)
    if not parameters:
        raise ValueError("target_func must accept a worker-index argument")
    index_name = parameters[-1]

    futures = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for worker_index in range(worker_count):
            worker_kwargs = dict(kwargs)
            worker_kwargs[index_name] = worker_index
            futures.append(executor.submit(target_func, *args, **worker_kwargs))

        results = []
        total_time = 0.0
        for future in futures:
            value = future.result()
            if isinstance(value, (tuple, list)) and len(value) >= 2:
                results.append(value[0])
                total_time += float(value[1])
            else:
                results.append(value)

    return results, total_time / worker_count


def running_data_record(output_data, output_dir):
    """Store aligned best-so-far curves in ``running_data.h5``."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path / "running_data.h5", "w") as handle:
        for algorithm, runs in output_data.items():
            if "_time" in algorithm or not runs:
                continue
            min_len = min(len(run) for run in runs)
            if min_len <= 0:
                continue
            values = np.asarray([run[:min_len] for run in runs], dtype=np.float64)
            handle.create_dataset(algorithm, data=values, compression="gzip")


def _iter_aligned_runs(data):
    for algorithm, runs in data.items():
        if "_time" in algorithm or not runs:
            continue
        min_len = min(len(run) for run in runs)
        if min_len <= 0:
            continue
        yield algorithm, np.asarray([run[:min_len] for run in runs], dtype=np.float64)


def plot_evaluation_curve(data, output_path, font_size, log_scale=False, show_variance=False):
    """Plot the mean raw objective curve and an optional standard-deviation band."""
    figure, axis = plt.subplots(figsize=(9, 6))
    for algorithm, values in _iter_aligned_runs(data):
        mean = np.mean(values, axis=0)
        x_axis = np.arange(mean.size)
        line, = axis.plot(x_axis, mean, label=algorithm)
        if show_variance and values.shape[0] > 1:
            std = np.std(values, axis=0)
            axis.fill_between(x_axis, mean - std, mean + std, color=line.get_color(), alpha=0.2)

    if log_scale:
        axis.set_yscale("log")
    axis.set_xlabel("FEs", fontsize=font_size)
    axis.set_ylabel("Objective Value", fontsize=font_size)
    axis.set_title("Evaluation Curves", fontsize=font_size)
    axis.grid(True, alpha=0.3)
    if axis.lines:
        axis.legend(fontsize=font_size)
    figure.tight_layout()
    figure.savefig(Path(output_path) / "evaluation_curves.png", bbox_inches="tight")
    plt.close(figure)


def plot_evaluation_curve_best_so_far(
    data,
    output_path,
    maxfes,
    figsize=(3.5, 2.16),
    font_size=8,
    log_scale=True,
    show_variance=True,
    eps=1e-12,
):
    """Plot aligned best-so-far curves against the configured FE budget."""
    figure, axis = plt.subplots(figsize=figsize)
    for algorithm, values in _iter_aligned_runs(data):
        values = np.minimum.accumulate(values, axis=1)
        values = np.clip(values, eps, None)
        mean = np.mean(values, axis=0)
        x_axis = np.linspace(0, float(maxfes), mean.size)
        line, = axis.plot(x_axis, mean, label=algorithm, alpha=0.9)
        if show_variance and values.shape[0] > 1:
            factor = 10.0 ** np.std(np.log10(values), axis=0)
            axis.fill_between(
                x_axis,
                mean / factor,
                mean * factor,
                color=line.get_color(),
                alpha=0.2,
                linewidth=0,
            )

    if log_scale:
        axis.set_yscale("log")
        axis.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8)))
        axis.yaxis.set_minor_formatter(NullFormatter())
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    axis.xaxis.set_major_formatter(formatter)
    axis.set_xlabel("FEs", fontsize=font_size)
    axis.set_ylabel("Objective Value", fontsize=font_size)
    axis.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if axis.lines:
        axis.legend(frameon=False, fontsize=max(font_size - 1, 1))
    figure.tight_layout()

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "evaluation_curves_best_so_far.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "evaluation_curves_best_so_far.png", bbox_inches="tight", dpi=600)
    plt.close(figure)
