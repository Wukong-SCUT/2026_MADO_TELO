import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from options import get_options


MEAN_RE = re.compile(r"^final_fitness_mean:\s*(\S+)", re.MULTILINE)
STD_RE = re.compile(r"^final_fitness_std:\s*(\S+)", re.MULTILINE)
TIME_RE = re.compile(r"^avg_time:\s*(\S+)", re.MULTILINE)
FUN_RE = re.compile(r"^f(\d+)$")

RELEASE_RUNS = {
    "CDOBenchF1F15": {
        "note": "C8c-release-epoch20-f1f15-3e6",
        "model": "C8c-epoch20-F1F15",
    },
    "WSNLocation": {
        "note": "C8c-release-epoch24-wsn-3e6",
        "model": "C8c-epoch24-WSN",
    },
}


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _find_latest_run(test_root: Path, benchmark: str, note: str) -> Path:
    candidates = []
    for path in test_root.iterdir():
        if not path.is_dir():
            continue
        options_path = path / "options_test.json"
        if not options_path.is_file():
            continue
        try:
            options = _load_json(options_path)
        except (OSError, ValueError, TypeError):
            continue
        if (
            str(options.get("benchmark_name", "")) == benchmark
            and str(options.get("note", "")) == note
        ):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No release evaluation for benchmark={benchmark!r}, note={note!r} "
            f"under {test_root}. Run eval_mappo first."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_run_values(strategy_dir: Path) -> Tuple[float, float, float]:
    text = (strategy_dir / "per_run_values.txt").read_text(encoding="utf-8")
    mean_match = MEAN_RE.search(text)
    std_match = STD_RE.search(text)
    time_match = TIME_RE.search(text)
    if mean_match is None or std_match is None or time_match is None:
        raise ValueError(f"Incomplete per_run_values.txt: {strategy_dir}")
    return (
        float(mean_match.group(1)),
        float(std_match.group(1)),
        float(time_match.group(1)),
    )


def _read_comm_count(strategy_dir: Path) -> Tuple[float, float]:
    path = strategy_dir / "graph_metrics_per_run.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    values = [
        float(item["graph_comm_rounds"])
        for item in raw
        if isinstance(item, dict) and item.get("graph_comm_rounds") is not None
    ]
    if not values:
        raise ValueError(f"No graph_comm_rounds in {path}")
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(variance)


def _max_fes(options: Dict, benchmark: str, fun_id: int) -> float:
    key = "cdo_bench_max_fes_list" if benchmark == "CDOBenchF1F15" else "wsn_max_fes_list"
    values = options.get(key, [])
    if isinstance(values, str):
        values = [x.strip() for x in values.split(",") if x.strip()]
    index = int(fun_id) - 1
    if isinstance(values, list) and index < len(values):
        value = float(values[index])
        if value > 0:
            return value
    return float(options.get("max_fes", 0))


def _collect_run(run_dir: Path, benchmark: str) -> Dict[int, Dict[str, float]]:
    options = _load_json(run_dir / "options_test.json")
    records: Dict[int, Dict[str, float]] = {}
    for fun_dir in sorted(run_dir.iterdir(), key=lambda p: p.name):
        match = FUN_RE.match(fun_dir.name)
        if not (fun_dir.is_dir() and match):
            continue
        fun_id = int(match.group(1))
        strategy_dir = fun_dir / "mappo_deterministic"
        if not strategy_dir.is_dir():
            continue
        mean, std, elapsed = _read_run_values(strategy_dir)
        comm_mean, comm_std = _read_comm_count(strategy_dir)
        max_fes = _max_fes(options, benchmark, fun_id)
        records[fun_id] = {
            "objective_mean": mean,
            "objective_std": std,
            "time_per_1e5": elapsed / max_fes * 1e5,
            "comm_mean": comm_mean,
            "comm_std": comm_std,
        }
    return records


def _reference_table(raw: Dict, section: str) -> Dict[str, Dict[str, float]]:
    value = raw.get("default_main_preset", {}).get(section, {})
    if not isinstance(value, dict):
        return {}
    return {
        str(name): {str(k): float(v) for k, v in values.items()}
        for name, values in value.items()
        if isinstance(values, dict)
    }


def _columns() -> List[str]:
    return [f"CDOBenchF1F15:f{i:02d}" for i in range(1, 16)] + [
        f"WSNLocation:f{i:02d}" for i in range(1, 6)
    ]


def _write_table(
    output_dir: Path,
    filename: str,
    columns: List[str],
    rows: Dict[str, Dict[str, str]],
) -> Tuple[Path, Path]:
    csv_path = output_dir / f"{filename}.csv"
    md_path = output_dir / f"{filename}.md"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_or_strategy", *columns])
        for label, values in rows.items():
            writer.writerow([label, *[values.get(col, "") for col in columns]])
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| model_or_strategy | " + " | ".join(columns) + " |\n")
        f.write("|---|" + "---|" * len(columns) + "\n")
        for label, values in rows.items():
            f.write(f"| {label} | " + " | ".join(values.get(col, "") for col in columns) + " |\n")
    return csv_path, md_path


def _write_long_csv(output_dir: Path, filename: str, rows: List[Dict]) -> Path:
    path = output_dir / f"{filename}.long.csv"
    fieldnames = list(rows[0].keys()) if rows else ["benchmark", "fun_id", "model", "value"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _plot_metric(
    output_dir: Path,
    benchmark: str,
    metric: str,
    series: Dict[str, Dict[int, float]],
    ylabel: str,
) -> List[Path]:
    paths = []
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    markers = ["o", "s", "^", "D"]
    for index, (label, values) in enumerate(series.items()):
        xs = sorted(values)
        ys = [values[x] for x in xs]
        if not xs:
            continue
        ax.plot(xs, ys, marker=markers[index % len(markers)], linewidth=1.8, label=label)
    ax.set_xlabel("Function")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{benchmark}: {metric}")
    ax.set_xticks(range(1, 16 if benchmark == "CDOBenchF1F15" else 6))
    if all(v > 0 for values in series.values() for v in values.values()):
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = plot_dir / f"{metric}_{benchmark}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def summarize(release_args: Optional[List[str]] = None) -> Path:
    opts = get_options(list(release_args or []))
    test_root = Path(opts.release_summary_test_root).expanduser().resolve()
    if not test_root.is_dir():
        raise FileNotFoundError(f"Evaluation root not found: {test_root}. Run eval_mappo first.")

    runs = {
        benchmark: _find_latest_run(test_root, benchmark, spec["note"])
        for benchmark, spec in RELEASE_RUNS.items()
    }
    records = {benchmark: _collect_run(path, benchmark) for benchmark, path in runs.items()}
    expected = {"CDOBenchF1F15": 15, "WSNLocation": 5}
    for benchmark, count in expected.items():
        if len(records[benchmark]) != count:
            raise ValueError(
                f"Incomplete {benchmark} evaluation: expected {count} functions, "
                f"found {len(records[benchmark])} in {runs[benchmark]}"
            )

    preset = _load_json(Path(opts.release_summary_preset).expanduser().resolve())
    objective_refs = _reference_table(preset, "reference_values")
    time_refs = _reference_table(preset, "time_reference_values")
    comm_refs = _reference_table(preset, "comm_reference_values")

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir = Path(opts.release_summary_output_dir).expanduser().resolve() / (
        f"{opts.release_summary_name_prefix}_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    prefix = str(opts.release_summary_output_prefix)
    columns = _columns()

    objective_rows: Dict[str, Dict[str, str]] = {}
    objective_long = []
    for benchmark, values in records.items():
        label = RELEASE_RUNS[benchmark]["model"]
        objective_rows[label] = {}
        for fun_id, record in values.items():
            col = f"{benchmark}:f{fun_id:02d}"
            objective_rows[label][col] = (
                f"{record['objective_mean']:.3e} ({record['objective_std']:.3e})"
            )
            objective_long.append({
                "benchmark": benchmark,
                "fun_id": fun_id,
                "model": label,
                "mean": record["objective_mean"],
                "std": record["objective_std"],
            })
    for label, values in objective_refs.items():
        objective_rows[label] = {key: f"{value:.3e}" for key, value in values.items()}
    objective_csv, objective_md = _write_table(output_dir, prefix, columns, objective_rows)
    objective_long_path = _write_long_csv(output_dir, prefix, objective_long)

    time_rows: Dict[str, Dict[str, str]] = {}
    comm_rows: Dict[str, Dict[str, str]] = {}
    time_long = []
    comm_long = []
    for benchmark, values in records.items():
        label = RELEASE_RUNS[benchmark]["model"]
        time_rows[label] = {}
        comm_rows[label] = {}
        for fun_id, record in values.items():
            col = f"{benchmark}:f{fun_id:02d}"
            time_rows[label][col] = f"{record['time_per_1e5']:.3e}"
            comm_rows[label][col] = f"{record['comm_mean']:.3e} ({record['comm_std']:.3e})"
            time_long.append({"benchmark": benchmark, "fun_id": fun_id, "model": label, "value": record["time_per_1e5"]})
            comm_long.append({"benchmark": benchmark, "fun_id": fun_id, "model": label, "mean": record["comm_mean"], "std": record["comm_std"]})
    for label, values in time_refs.items():
        time_rows[label] = {key: f"{value:.3e}" for key, value in values.items()}
    for label, values in comm_refs.items():
        comm_rows[label] = {key: f"{value:.3e}" for key, value in values.items()}
    time_csv, time_md = _write_table(output_dir, f"{prefix}.time_per_1e5", columns, time_rows)
    comm_csv, comm_md = _write_table(output_dir, f"{prefix}.communication_count", columns, comm_rows)
    time_long_path = _write_long_csv(output_dir, f"{prefix}.time_per_1e5", time_long)
    comm_long_path = _write_long_csv(output_dir, f"{prefix}.communication_count", comm_long)

    comparisons = [
        ("CDOBenchF1F15", "MASOIE-paper"),
        ("WSNLocation", "CCSA-paper"),
    ]
    log_rows = []
    for benchmark, reference_name in comparisons:
        diff = 0.0
        for fun_id, record in records[benchmark].items():
            ref = objective_refs[reference_name][f"{benchmark}:f{fun_id:02d}"]
            diff += math.log10(max(record["objective_mean"], 1e-300)) - math.log10(max(ref, 1e-300))
        log_rows.append({
            "model_or_strategy": RELEASE_RUNS[benchmark]["model"],
            "reference": reference_name,
            "problem_count": len(records[benchmark]),
            "log10_difference_sum": diff,
        })
    log_csv = output_dir / f"{prefix}.log10_difference_sum.csv"
    with log_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    log_md = output_dir / f"{prefix}.log10_difference_sum.md"
    with log_md.open("w", encoding="utf-8") as f:
        f.write("| model_or_strategy | reference | problem_count | log10_difference_sum |\n")
        f.write("|---|---|---:|---:|\n")
        for row in log_rows:
            f.write(
                f"| {row['model_or_strategy']} | {row['reference']} | "
                f"{row['problem_count']} | {row['log10_difference_sum']:+.6f} |\n"
            )
        f.write("\nSmaller is better; negative means better than the reference overall.\n")

    excel_path = output_dir / f"{prefix}.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.read_csv(objective_csv).to_excel(writer, sheet_name="objective", index=False)
        pd.read_csv(time_csv).to_excel(writer, sheet_name="time_per_1e5", index=False)
        pd.read_csv(comm_csv).to_excel(writer, sheet_name="communication", index=False)
        pd.read_csv(log_csv).to_excel(writer, sheet_name="log10_difference", index=False)

    for benchmark, reference_name in comparisons:
        model_label = RELEASE_RUNS[benchmark]["model"]
        _plot_metric(output_dir, benchmark, "objective", {
            model_label: {i: x["objective_mean"] for i, x in records[benchmark].items()},
            reference_name: {i: objective_refs[reference_name][f"{benchmark}:f{i:02d}"] for i in records[benchmark]},
        }, "Objective value")
        _plot_metric(output_dir, benchmark, "time_per_1e5", {
            model_label: {i: x["time_per_1e5"] for i, x in records[benchmark].items()},
            **{
                label: {i: values[f"{benchmark}:f{i:02d}"] for i in records[benchmark]}
                for label, values in time_refs.items()
            },
        }, "Seconds per 1e5 FEs")
        _plot_metric(output_dir, benchmark, "communication_count", {
            model_label: {i: x["comm_mean"] for i, x in records[benchmark].items()},
            **{
                label: {i: values[f"{benchmark}:f{i:02d}"] for i in records[benchmark]}
                for label, values in comm_refs.items()
            },
        }, "Communication count")

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": {key: value.name for key, value in runs.items()},
        "files": [
            str(path.name)
            for path in (
                objective_csv, objective_md, objective_long_path,
                time_csv, time_md, time_long_path,
                comm_csv, comm_md, comm_long_path,
                log_csv, log_md, excel_path,
            )
        ],
    }
    (output_dir / "summary_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"[C8c Summary] CDOBench run: {runs['CDOBenchF1F15'].name}")
    print(f"[C8c Summary] WSN run      : {runs['WSNLocation'].name}")
    for row in log_rows:
        print(
            f"[C8c Summary] {row['model_or_strategy']} vs {row['reference']}: "
            f"{row['log10_difference_sum']:+.6f}"
        )
    print(f"[C8c Summary] Output: {output_dir}")
    return output_dir


def main() -> None:
    summarize(sys.argv[1:])


if __name__ == "__main__":
    main()
