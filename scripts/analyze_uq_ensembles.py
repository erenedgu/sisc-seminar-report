import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _extract_pid_from_execution_log(iteration: int, log_file: str) -> int:

    if not os.path.exists(log_file):
        return None

    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(r'spawned with pid (\d+)', line)
            if match:
                return int(match.group(1))

    return None


def _align_cumulative_energy_to_timeline(df_metric: pd.DataFrame, timeline: pd.DatetimeIndex, value_name: str) -> pd.Series:

    if df_metric.empty:
        return pd.Series(0.0, index=timeline, name=value_name)

    aligned = (
        df_metric.groupby("timestamp", as_index=True)["value"]
        .sum()
        .sort_index()
        .reindex(timeline)
        .astype(float)
    )

    first_valid = aligned.first_valid_index()
    if first_valid is None:
        return pd.Series(0.0, index=timeline, name=value_name)

    aligned.loc[aligned.index < first_valid] = 0.0
    aligned = aligned.interpolate(method="time", limit_area="inside")
    aligned = aligned.fillna(0.0)

    last_valid = df_metric["timestamp"].max()
    aligned.loc[aligned.index > last_valid] = np.nan
    aligned.name = value_name
    return aligned


def _safe_zscore(series: pd.Series) -> pd.Series:
    std_val = series.std(ddof=0)
    if pd.isna(std_val) or std_val == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std_val


def _build_total_energy_timeline(
    cpu_pid: pd.DataFrame,
    gpu_pid: pd.DataFrame,
) -> pd.DatetimeIndex:
    cpu_index = pd.DatetimeIndex(pd.Index(cpu_pid["timestamp"]).unique()).sort_values()
    gpu_index = pd.DatetimeIndex(pd.Index(gpu_pid["timestamp"]).unique()).sort_values()

    if cpu_index.empty:
        return gpu_index
    return cpu_index


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
base_dir = os.path.join(project_root, "data_archive", "uq_energy_combined")
results = []

print("--- Multi-Ensemble UQ Energy Analysis (CPU + GPU) ---")

sigma_dirs = [
    d for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("sigma_")
]

for sigma_dir in sorted(sigma_dirs):
    match = re.match(r"^sigma_(\d+(?:\.\d+)?)$", sigma_dir)
    if not match:
        continue
    sigma_val = float(match.group(1))

    sigma_path = os.path.join(base_dir, sigma_dir)
    iter_dirs = [
        d for d in os.listdir(sigma_path)
        if os.path.isdir(os.path.join(sigma_path, d)) and d.startswith("iter_")
    ]

    for iter_dir in sorted(iter_dirs, key=lambda x: int(x.split("_")[1])):
        iteration = int(iter_dir.split("_")[1])
        filename = os.path.join(sigma_path, iter_dir, "telemetry.csv")
        log_file = os.path.join(sigma_path, iter_dir, "execution.log")

        if not os.path.exists(filename):
            continue

        target_pid = _extract_pid_from_execution_log(iteration, log_file)
        if target_pid is None:
            raise FileNotFoundError(
                f"Iteration {iteration} in {sigma_dir}: Could not extract PID from execution.log."
            )

        df = pd.read_csv(filename, sep=';', dtype={'resource_id': 'str'})

        # 1. Extract Joules & Isolate the Process by PID
        df_gpu_raw = df[(df['metric'].str.contains('attributed_energy_gpu', na=False))
                        & (df['consumer_kind'] == 'process')
                        & (df['consumer_id'] == target_pid)]
        df_cpu_raw = df[(df['metric'].str.contains('attributed_energy_cpu', na=False))
                        & (df['consumer_kind'] == 'process')
                        & (df['consumer_id'] == target_pid)]

        if df_gpu_raw.empty and df_cpu_raw.empty:
            print(f"Iteration {iteration} (sigma={sigma_val}): Missing CPU or GPU data for PID {target_pid}.")
            continue

        df_gpu_raw = df_gpu_raw[['timestamp', 'value']].copy()
        df_cpu_raw = df_cpu_raw[['timestamp', 'value']].copy()

        # 2. Format Time
        df_gpu_raw['timestamp'] = pd.to_datetime(df_gpu_raw['timestamp']).dt.floor('100ms')
        df_cpu_raw['timestamp'] = pd.to_datetime(df_cpu_raw['timestamp']).dt.floor('100ms')

        # 3. Squash Duplicates
        df_gpu = df_gpu_raw.groupby('timestamp', as_index=False).sum()
        df_cpu = df_cpu_raw.groupby('timestamp', as_index=False).sum()

        df_gpu = df_gpu.sort_values('timestamp')
        df_cpu = df_cpu.sort_values('timestamp')

        # 4. Cumulative Energy Calculation
        df_gpu['value'] = df_gpu['value'].cumsum()
        df_cpu['value'] = df_cpu['value'].cumsum()

        # 5. Timeline alignment
        timeline = _build_total_energy_timeline(df_cpu, df_gpu)

        if timeline.empty:
            continue

        cpu_aligned = _align_cumulative_energy_to_timeline(df_cpu, timeline, "cpu_cum")
        gpu_aligned = _align_cumulative_energy_to_timeline(df_gpu, timeline, "gpu_cum")

        df_merged = pd.DataFrame({
            "timestamp": timeline,
            "cpu_cum": cpu_aligned.to_numpy(),
            "gpu_cum": gpu_aligned.to_numpy(),
        })

        df_merged.dropna(subset=["cpu_cum", "gpu_cum"], inplace=True)

        if df_merged.empty:
            continue

        # 6. Final Energy Calculation
        df_merged['cum_energy'] = df_merged['cpu_cum'] + df_merged['gpu_cum']

        run_total = df_merged['cum_energy'].iloc[-1]
        df_merged['time_sec'] = (df_merged['timestamp'] - df_merged['timestamp'].iloc[0]).dt.total_seconds()
        duration_s = df_merged['time_sec'].iloc[-1]

        results.append({
            "sigma_val": sigma_val,
            "iteration": iteration,
            "total_energy_j": float(run_total),
            "duration_s": float(duration_s),
        })


df_results = pd.DataFrame(results)

if df_results.empty:
    raise RuntimeError(f"No valid telemetry runs were processed from {base_dir}.")


# 3. Statistical Analysis Table
stats = (
    df_results.groupby("sigma_val", sort=True)["total_energy_j"]
    .agg(
        mean_j="mean",
        median_j="median",
        std_j="std",
        p05=lambda s: np.percentile(s, 5),
        p95=lambda s: np.percentile(s, 95),
    )
    .reset_index()
)

print("\n--- Statistical Summary (All Runs) ---")
header = f"{'Topographic Noise (m)':>22} | {'Mean (J)':>12} | {'Median (J)':>12} | {'Std Dev (J)':>12} | {'90% Confidence Interval (J)':>30}"
print(header)
print("-" * len(header))
stats_lines = ["Energy Statistics:"]
for _, row in stats.iterrows():
    mean_e = row['mean_j']
    median_e = row['median_j']
    std_e = 0.0 if pd.isna(row['std_j']) else row['std_j']
    sigma = row['sigma_val']
    ci_str = f"[{row['p05']:.2f}, {row['p95']:.2f}]"
    stats_lines.append(f"σ={sigma}m: Mean={mean_e:.0f} J, Median={median_e:.0f} J, Std={std_e:.1f} J")
    print(
        f"{sigma:>22.2f} | "
        f"{mean_e:>12.2f} | "
        f"{median_e:>12.2f} | "
        f"{std_e:>12.2f} | "
        f"{ci_str:>30}"
    )


# 4. Boxplot Visualization
fig1, ax1 = plt.subplots(figsize=(10, 6))

sns.boxplot(
    ax=ax1,
    data=df_results,
    x="sigma_val",
    y="total_energy_j",
    hue="sigma_val",
    palette="deep",
    legend=False,
    showfliers=False,
)
ax1.set_title("Computational Energy Variance relative to Topographic Noise (σ)")
ax1.set_xlabel("Topographic Error / σ (meters)")
ax1.set_ylabel("Total GPU/CPU Energy Consumed (Joules)")
ax1.grid(axis="y", linestyle="--", alpha=0.6)
ax1.text(0.02, 0.98, "\n".join(stats_lines), transform=ax1.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", alpha=0.85))

fig2, ax2 = plt.subplots(figsize=(10, 6))

sns.stripplot(
    ax=ax2,
    data=df_results,
    x="sigma_val",
    y="total_energy_j",
    hue="sigma_val",
    palette="deep",
    jitter=True,
    alpha=0.6,
    size=4,
    legend=False,
)
ax2.set_title("Raw Distribution of Total Energy Consumption across UQ Ensembles")
ax2.set_xlabel("Topographic Error / σ (meters)")
ax2.set_ylabel("Total GPU/CPU Energy Consumed (Joules)")
ax2.grid(axis="y", linestyle="--", alpha=0.6)

output_dir = os.path.join(project_root, "plots", "uq_final")
os.makedirs(output_dir, exist_ok=True)
boxplot_path = os.path.join(output_dir, "energy_uq_boxplot.png")
stripplot_path = os.path.join(output_dir, "energy_uq_stripplot.png")
fig1.tight_layout()
fig1.savefig(boxplot_path, dpi=300)
fig2.tight_layout()
fig2.savefig(stripplot_path, dpi=300)
print(f"\nBoxplot saved to: {boxplot_path}")
print(f"Stripplot saved to: {stripplot_path}")
# --- Combined 1x2 Comparison (overwrite previous combined image) ---
figc, (cax1, cax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Left: raw telemetry (dots)
sns.stripplot(
    ax=cax1,
    data=df_results,
    x="sigma_val",
    y="total_energy_j",
    hue="sigma_val",
    palette="deep",
    jitter=True,
    alpha=0.6,
    size=4,
    legend=False,
)
cax1.set_title("Raw Distribution of Total Energy Consumption across UQ Ensembles")
cax1.set_xlabel("Topographic Error / σ (meters)")
cax1.set_ylabel("Total GPU/CPU Energy Consumed (Joules)")
cax1.grid(axis="y", linestyle="--", alpha=0.6)

sns.boxplot(
    ax=cax2,
    data=df_results,
    x="sigma_val",
    y="total_energy_j",
    hue="sigma_val",
    palette="viridis",
    legend=False,
    showfliers=False,
)
cax2.set_title("Computational Energy Variance relative to Topographic Noise (σ)")
cax2.set_xlabel("Topographic Error / σ (meters)")
cax2.set_ylabel("")
cax2.grid(axis="y", linestyle="--", alpha=0.6)
cax2.text(0.02, 0.98, "\n".join(stats_lines), transform=cax2.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", alpha=0.85))

combined_path = os.path.join(output_dir, "energy_uq_comparison.png")
figc.tight_layout()
figc.savefig(combined_path, dpi=300)
print(f"Combined comparison saved to: {combined_path}")

plt.show(block=False)
plt.pause(0.001)