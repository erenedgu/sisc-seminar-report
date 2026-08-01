#!/usr/bin/env python3
"""Analyze whether simulation duration is a reliable proxy for energy consumption."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


TIME_COL_CANDIDATES = ("timestamp", "time", "Time", "Timestamp", "t")
ENERGY_COL_CANDIDATES = (
    "energy",
    "Energy",
    "cum_energy",
    "cumulative_energy",
    "energy_j",
    "energy_J",
    "energy_joules",
)


@dataclass(frozen=True)
class RunSeries:
    group: str
    iteration: str
    time_sec: np.ndarray
    cum_energy: np.ndarray


def _resolve_path(path: str, base_dir: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (base_dir / candidate)


def _extract_pid_from_execution_log(log_path: Path) -> int | None:
    if not log_path.exists():
        return None

    pattern = re.compile(r"spawned with pid (\d+)")
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                return int(match.group(1))
    return None


def _format_group_label(dir_name: str) -> str:
    lowered = dir_name.lower()
    if lowered.startswith("original"):
        return "Original"
    match = re.search(r"lowres(\d+)", lowered)
    if match:
        return f"Low-res {int(match.group(1))}"
    return dir_name.replace("_", " ")


def _group_sort_key(dir_name: str) -> tuple[int, int, str]:
    lowered = dir_name.lower()
    if lowered.startswith("original"):
        return (0, 0, dir_name)
    match = re.search(r"lowres(\d+)", lowered)
    if match:
        return (1, int(match.group(1)), dir_name)
    return (2, 0, dir_name)


def _build_total_energy_timeline(
    cpu_pid: pd.DataFrame,
    gpu_pid: pd.DataFrame,
) -> pd.DatetimeIndex:
    cpu_index = pd.DatetimeIndex(pd.Index(cpu_pid["timestamp"]).unique()).sort_values()
    gpu_index = pd.DatetimeIndex(pd.Index(gpu_pid["timestamp"]).unique()).sort_values()

    if cpu_index.empty:
        return gpu_index
    return cpu_index


def _align_cumulative_energy_to_timeline(
    df_metric: pd.DataFrame,
    timeline: pd.DatetimeIndex,
    value_name: str,
) -> pd.Series:
    if df_metric.empty:
        return pd.Series(0.0, index=timeline, name=value_name)

    source = (
        df_metric.groupby("timestamp", as_index=True)["value"]
        .sum()
        .sort_index()
        .astype(float)
    )
    interpolation_index = pd.DatetimeIndex(source.index.union(timeline).sort_values())
    aligned = source.reindex(interpolation_index)

    first_valid = aligned.first_valid_index()
    if first_valid is None:
        return pd.Series(0.0, index=timeline, name=value_name)

    aligned.loc[aligned.index < first_valid] = 0.0
    aligned = aligned.interpolate(method="time", limit_area="inside")
    aligned = aligned.ffill().fillna(0.0)
    aligned = aligned.reindex(timeline)
    aligned.name = value_name
    return aligned


def _load_energy_timeseries_from_telemetry(telemetry_path: Path, pid: int) -> pd.DataFrame | None:
    usecols = ["metric", "timestamp", "value", "consumer_kind", "consumer_id"]
    df = pd.read_csv(telemetry_path, sep=";", usecols=usecols)

    if df.empty:
        return None

    df["consumer_id"] = pd.to_numeric(df["consumer_id"], errors="coerce")
    df_energy = df.loc[
        (df["consumer_kind"] == "process")
        & (df["consumer_id"] == pid)
        & (df["metric"].str.contains("attributed_energy", case=False, na=False))
    ]

    if df_energy.empty:
        return None

    df_cpu = df_energy.loc[
        df_energy["metric"].str.contains("attributed_energy_cpu", case=False, na=False),
        ["timestamp", "value"],
    ].copy()
    df_gpu = df_energy.loc[
        df_energy["metric"].str.contains("attributed_energy_gpu", case=False, na=False),
        ["timestamp", "value"],
    ].copy()

    if df_cpu.empty and df_gpu.empty:
        return None

    for frame in (df_cpu, df_gpu):
        if frame.empty:
            continue
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce").dt.floor("100ms")
        frame.dropna(subset=["timestamp"], inplace=True)
        frame.sort_values("timestamp", inplace=True)
        frame["value"] = frame.groupby("timestamp")["value"].transform("sum")
        frame.drop_duplicates(subset=["timestamp"], inplace=True)
        frame["value"] = frame["value"].cumsum()

    timeline = _build_total_energy_timeline(df_cpu, df_gpu)
    if timeline.empty:
        return None

    cpu_aligned = _align_cumulative_energy_to_timeline(df_cpu, timeline, "cpu_cum")
    gpu_aligned = _align_cumulative_energy_to_timeline(df_gpu, timeline, "gpu_cum")

    df_merged = pd.DataFrame({
        "timestamp": timeline,
        "cum_energy": cpu_aligned.to_numpy() + gpu_aligned.to_numpy(),
    })
    df_merged["time_sec"] = (
        df_merged["timestamp"] - df_merged["timestamp"].iloc[0]
    ).dt.total_seconds()

    df_merged.dropna(subset=["time_sec", "cum_energy"], inplace=True)
    df_merged = df_merged.loc[df_merged["time_sec"] >= 0]

    if df_merged.empty:
        return None

    return df_merged[["time_sec", "cum_energy"]]


def _load_energy_timeseries_from_simple_csv(csv_path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    time_col = next((col for col in TIME_COL_CANDIDATES if col in df.columns), None)
    energy_col = next((col for col in ENERGY_COL_CANDIDATES if col in df.columns), None)

    if time_col is None or energy_col is None:
        return None

    time_raw = df[time_col]
    energy_raw = pd.to_numeric(df[energy_col], errors="coerce")

    if np.issubdtype(time_raw.dtype, np.number):
        time_sec = pd.to_numeric(time_raw, errors="coerce")
        time_sec = time_sec - time_sec.min()
    else:
        time_dt = pd.to_datetime(time_raw, errors="coerce")
        time_sec = (time_dt - time_dt.iloc[0]).dt.total_seconds()

    mask = time_sec.notna() & energy_raw.notna()
    time_sec = time_sec.loc[mask].astype(float)
    energy_raw = energy_raw.loc[mask].astype(float)

    if time_sec.empty:
        return None

    order = np.argsort(time_sec.to_numpy())
    time_sorted = time_sec.to_numpy()[order]
    energy_sorted = energy_raw.to_numpy()[order]

    return pd.DataFrame({"time_sec": time_sorted, "cum_energy": energy_sorted})


def _compute_ols_slope(time_sec: np.ndarray, cum_energy: np.ndarray) -> tuple[float, float]:
    if time_sec.size < 2:
        return float("nan"), float("nan")

    design = np.vstack([time_sec, np.ones_like(time_sec)]).T
    slope, intercept = np.linalg.lstsq(design, cum_energy, rcond=None)[0]
    return float(slope), float(intercept)


def _find_run_entries(root_dir: Path) -> list[dict]:
    run_entries: list[dict] = []

    iter_dirs = sorted(root_dir.glob("iter_*"))
    for iter_dir in iter_dirs:
        telemetry_path = iter_dir / "telemetry.csv"
        if telemetry_path.exists():
            run_entries.append({
                "kind": "telemetry",
                "iteration": iter_dir.name,
                "telemetry": telemetry_path,
                "log": iter_dir / "execution.log",
            })

    if run_entries:
        return run_entries

    for csv_path in sorted(root_dir.glob("*.csv")):
        if csv_path.name == "uq_water_depth_samples.csv":
            continue
        run_entries.append({
            "kind": "simple",
            "iteration": csv_path.stem,
            "csv": csv_path,
        })

    return run_entries


def _collect_runs(root_dir: Path, label: str) -> tuple[list[dict], list[RunSeries]]:
    records: list[dict] = []
    series: list[RunSeries] = []

    for entry in _find_run_entries(root_dir):
        iteration = entry["iteration"]
        if entry["kind"] == "telemetry":
            pid = _extract_pid_from_execution_log(entry["log"])
            if pid is None:
                print(f"[{label}] {iteration}: PID not found; skipping.")
                continue
            df_timeseries = _load_energy_timeseries_from_telemetry(entry["telemetry"], pid)
        else:
            df_timeseries = _load_energy_timeseries_from_simple_csv(entry["csv"])

        if df_timeseries is None or df_timeseries.empty:
            print(f"[{label}] {iteration}: No energy data; skipping.")
            continue

        time_sec = df_timeseries["time_sec"].to_numpy(dtype=float)
        cum_energy = df_timeseries["cum_energy"].to_numpy(dtype=float)

        if time_sec.size < 2 or cum_energy.size < 2:
            print(f"[{label}] {iteration}: Insufficient samples; skipping.")
            continue

        duration = float(time_sec[-1])
        total_energy = float(cum_energy[-1])
        slope, intercept = _compute_ols_slope(time_sec, cum_energy)
        energy_per_second = total_energy / duration if duration > 0 else float("nan")

        records.append({
            "group": label,
            "iteration": iteration,
            "total_energy_j": total_energy,
            "total_duration_s": duration,
            "avg_power_w": slope,
            "energy_per_second": energy_per_second,
            "intercept": intercept,
        })

        series.append(RunSeries(
            group=label,
            iteration=iteration,
            time_sec=time_sec,
            cum_energy=cum_energy,
        ))

    return records, series


def _zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    mean = values.mean(skipna=True)
    std = values.std(skipna=True)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    z = (values - mean) / std
    return z.fillna(0.0)


def _average_trajectory(series_list: list[RunSeries], points: int) -> tuple[np.ndarray, np.ndarray] | None:
    if not series_list:
        return None

    max_durations = [run.time_sec[-1] for run in series_list if run.time_sec.size > 1]
    if not max_durations:
        return None

    common_max = float(min(max_durations))
    if common_max <= 0:
        return None

    grid = np.linspace(0.0, common_max, points)
    samples: list[np.ndarray] = []

    for run in series_list:
        if run.time_sec.size < 2:
            continue
        energy_interp = np.interp(grid, run.time_sec, run.cum_energy)
        samples.append(energy_interp)

    if not samples:
        return None

    return grid, np.mean(samples, axis=0)


def _plot_power_signature(
    output_dir: Path,
    series_list: list[RunSeries],
    groups: list[str],
    points: int,
    power_summary: dict[str, dict[str, float | int]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for group in groups:
        group_series = [run for run in series_list if run.group == group]
        result = _average_trajectory(group_series, points)
        if result is None:
            continue
        grid, avg_energy = result
        label = f"{group} (n={len(group_series)})"
        if power_summary and group in power_summary:
            mean_power = power_summary[group].get("mean_power")
            if isinstance(mean_power, (int, float)) and np.isfinite(mean_power):
                label = f"{group} (n={len(group_series)}, P={mean_power:.2f} W)"
        ax.plot(grid, avg_energy, linewidth=2, label=label)

    ax.set_title("Average Energy vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Energy (J)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

    output_path = output_dir / "power_signature.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_proxy_fallacy(
    output_dir: Path,
    df_all: pd.DataFrame,
    groups: list[str],
    r2_all: float | None = None,
    r2_clean: float | None = None,
    slope: float | None = None,
    intercept: float | None = None,
    output_name: str = "proxy_fallacy.png",
    title: str = "Energy vs Duration",
    x_pad_ratio: float = 0.4,
    x_pad_min: float = 4.0,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    colors = {group: palette[idx % len(palette)] for idx, group in enumerate(groups)}

    for group in groups:
        df_group = df_all.loc[df_all["group"] == group]
        df_clean = df_group.loc[~df_group["is_outlier"]]
        df_out = df_group.loc[df_group["is_outlier"]]

        ax.scatter(
            df_clean["total_duration_s"],
            df_clean["total_energy_j"],
            color=colors[group],
            alpha=0.7,
            label=f"{group} clean",
        )

        if not df_out.empty:
            ax.scatter(
                df_out["total_duration_s"],
                df_out["total_energy_j"],
                color=colors[group],
                marker="x",
                s=60,
                label=f"{group} outlier",
            )

    max_duration = float(df_all["total_duration_s"].max())
    x_max: float | None = None
    if np.isfinite(max_duration) and max_duration > 0:
        x_pad = max(x_pad_min, max_duration * x_pad_ratio)
        x_max = max_duration + x_pad
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(bottom=0.0)

    if (
        slope is not None
        and intercept is not None
        and np.isfinite(slope)
        and np.isfinite(intercept)
    ):
        line_max = x_max if x_max is not None else max_duration
        if line_max > 0:
            t_vals = np.linspace(0.0, line_max, 200)
            e_vals = slope * t_vals + intercept
            sign = "+" if intercept >= 0 else "-"
            ax.plot(
                t_vals,
                e_vals,
                "k--",
                linewidth=1.5,
                label=rf"Linear Fit ($E = {slope:.2f}t {sign} {abs(intercept):.2f}$)",
            )

    ax.set_title(title)
    ax.set_xlabel("Total Duration (s)")
    ax.set_ylabel("Total Energy (J)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

    r2_lines: list[str] = []
    if r2_all is not None and np.isfinite(r2_all):
        r2_lines.append(rf"$R^2$ all: {r2_all:.3f}")
    if r2_clean is not None and np.isfinite(r2_clean):
        r2_lines.append(rf"$R^2$ clean: {r2_clean:.3f}")
    if r2_lines:
        ax.text(
            0.98,
            0.98,
            "\n".join(r2_lines),
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=10,
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.8,
            },
        )

    output_path = output_dir / output_name
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze whether duration is a reliable proxy for energy consumption."
    )
    parser.add_argument(
        "--runs-root",
        default="data_archive/time_proxy_test",
        help="Folder containing run subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/energy_proxy",
        help="Output directory for plots and CSVs.",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=2.0,
        help="Z-score threshold for outlier detection.",
    )
    parser.add_argument(
        "--avg-grid-points",
        type=int,
        default=200,
        help="Number of points for the average trajectory plot.",
    )

    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parents[1]

    runs_root = _resolve_path(args.runs_root, base_dir)
    output_dir = _resolve_path(args.output_dir, base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not runs_root.exists():
        print(f"Runs root does not exist: {runs_root}")
        return

    subdirs = [path for path in runs_root.iterdir() if path.is_dir()]
    if not subdirs:
        print(f"No run folders found in: {runs_root}")
        return

    subdirs = sorted(subdirs, key=lambda path: _group_sort_key(path.name))
    group_dirs: dict[str, Path] = {}
    for path in subdirs:
        label = _format_group_label(path.name)
        if label in group_dirs:
            label = f"{label} ({path.name})"
        group_dirs[label] = path

    groups = list(group_dirs.keys())

    records: list[dict] = []
    series: list[RunSeries] = []

    for group in groups:
        group_records, group_series = _collect_runs(group_dirs[group], group)
        records.extend(group_records)
        series.extend(group_series)

    if not records:
        print("No runs found; nothing to analyze.")
        return

    df_all = pd.DataFrame(records)

    df_all["duration_z"] = _zscore(df_all["total_duration_s"])
    df_all["eps_z"] = _zscore(df_all["energy_per_second"])

    df_all["is_outlier"] = (
        (df_all["duration_z"] > args.z_threshold)
        | (df_all["eps_z"] < -args.z_threshold)
    )

    df_clean = df_all.loc[~df_all["is_outlier"]].copy()
    df_outliers = df_all.loc[df_all["is_outlier"]].copy()

    df_all.to_csv(output_dir / "energy_proxy_all.csv", index=False)
    df_clean.to_csv(output_dir / "energy_proxy_clean.csv", index=False)
    df_outliers.to_csv(output_dir / "energy_proxy_outliers.csv", index=False)

    clean_keys = set(zip(df_clean["group"], df_clean["iteration"]))
    series_clean = [run for run in series if (run.group, run.iteration) in clean_keys]

    power_summary: dict[str, dict[str, float | int]] = {}
    for group in groups:
        group_power = df_clean.loc[df_clean["group"] == group, "avg_power_w"].dropna()
        if group_power.empty:
            continue
        power_summary[group] = {
            "mean_power": float(group_power.mean()),
            "std_power": float(group_power.std(ddof=1)) if len(group_power) > 1 else 0.0,
            "n": int(group_power.size),
        }

    _plot_power_signature(
        output_dir,
        series_clean,
        groups,
        args.avg_grid_points,
        power_summary=power_summary,
    )

    r2_all = None
    r2_clean = None
    slope_all = None
    intercept_all = None
    slope_clean = None
    intercept_clean = None
    mask_all = df_all["total_duration_s"].notna() & df_all["total_energy_j"].notna()
    if mask_all.sum() >= 2:
        result_all = stats.linregress(
            df_all.loc[mask_all, "total_duration_s"],
            df_all.loc[mask_all, "total_energy_j"],
        )
        r2_all = result_all.rvalue ** 2
        slope_all = result_all.slope
        intercept_all = result_all.intercept

    mask_clean = df_clean["total_duration_s"].notna() & df_clean["total_energy_j"].notna()
    if mask_clean.sum() >= 2:
        result_clean = stats.linregress(
            df_clean.loc[mask_clean, "total_duration_s"],
            df_clean.loc[mask_clean, "total_energy_j"],
        )
        r2_clean = result_clean.rvalue ** 2
        slope_clean = result_clean.slope
        intercept_clean = result_clean.intercept

    _plot_proxy_fallacy(
        output_dir,
        df_all,
        groups,
        r2_all=r2_all,
        r2_clean=None,
        slope=slope_all,
        intercept=intercept_all,
        output_name="proxy_fallacy.png",
        title="Energy vs Duration",
    )

    _plot_proxy_fallacy(
        output_dir,
        df_clean,
        groups,
        r2_clean=r2_clean,
        slope=slope_clean,
        intercept=intercept_clean,
        output_name="proxy_fallacy_clean.png",
        title="Energy vs Duration (Clean Only)",
    )

    print("\n--- Power Draw Summary (Clean Data) ---")
    for group in groups:
        group_power = df_clean.loc[df_clean["group"] == group, "avg_power_w"].dropna()
        if group_power.empty:
            print(f"{group}: no clean samples.")
            continue
        mean_power = float(group_power.mean())
        std_power = float(group_power.std(ddof=1)) if len(group_power) > 1 else 0.0
        print(f"{group}: mean={mean_power:.2f} W, std={std_power:.2f} W, n={len(group_power)}")

    print("\n--- Welch t-test (Clean Data) ---")
    ttest_pairs: list[tuple[str, str]] = []
    for i, left in enumerate(groups):
        for right in groups[i + 1:]:
            ttest_pairs.append((left, right))

    any_test = False
    for left, right in ttest_pairs:
        left_vals = df_clean.loc[df_clean["group"] == left, "avg_power_w"].dropna()
        right_vals = df_clean.loc[df_clean["group"] == right, "avg_power_w"].dropna()
        if left_vals.size < 2 or right_vals.size < 2:
            continue
        t_stat, p_val = stats.ttest_ind(
            left_vals,
            right_vals,
            equal_var=False,
            nan_policy="omit",
        )
        print(f"{left} vs {right}: t={t_stat:.3f}, p={p_val:.4f}")
        any_test = True

    if not any_test:
        print("Not enough samples for t-test.")

    mask = df_all["total_duration_s"].notna() & df_all["total_energy_j"].notna()
    if mask.sum() >= 2:
        result = stats.linregress(
            df_all.loc[mask, "total_duration_s"],
            df_all.loc[mask, "total_energy_j"],
        )
        r2 = result.rvalue ** 2
        print("\n--- Energy-to-Time Correlation ---")
        print(f"R^2: {r2:.3f}")
        if r2 < 0.8:
            print(
                "Low R^2 suggests power draw varies across runs (e.g., throttling, "
                "background load, or GPU utilization shifts), making time-to-solution "
                "a weak proxy for energy."
            )
    else:
        print("\n--- Energy-to-Time Correlation ---")
        print("Not enough samples for R^2.")

    print("\n--- Outlier Summary ---")
    print(f"Total runs: {len(df_all)}")
    print(f"Clean runs: {len(df_clean)}")
    print(f"Outliers: {len(df_outliers)}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
