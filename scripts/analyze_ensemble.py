import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import yaml
from pathlib import Path
from PIL import Image
from scipy import stats
import pickle


def _extract_pid_from_execution_log(iteration: int) -> int:
    """
    Extract the SynxFlow process PID from execution.log.
    
    Searches for the line: "Child process '...' spawned with pid XXXXX."
    Returns the PID as an integer, or None if not found.
    """
    log_file = f'ensemble_results/iter_{iteration}/execution.log'
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(r'spawned with pid (\d+)', line)
            if match:
                return int(match.group(1))
    
    return None


def _align_cumulative_energy_to_timeline(df_metric: pd.DataFrame, timeline: pd.DatetimeIndex, value_name: str) -> pd.Series:
    """
    Align a cumulative metric to a shared timeline.

    Missing values between the earlier start/end of this stream and later start/end of the other stream are returned as 0.
    Between observed samples, values are linearly interpolated on timestamp.
    After the last observed sample, the cumulative value is carried forward so the total does not drop back to zero.
    """
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


def _build_total_energy_timeline(
    cpu_pid: pd.DataFrame,
    gpu_pid: pd.DataFrame,
) -> pd.DatetimeIndex:
    """
    Build timestamps for attributed total energy using the union of both sensors.
    """
    cpu_index = pd.DatetimeIndex(pd.Index(cpu_pid["timestamp"]).unique()).sort_values()
    gpu_index = pd.DatetimeIndex(pd.Index(gpu_pid["timestamp"]).unique()).sort_values()

    return pd.DatetimeIndex(cpu_index.union(gpu_index).sort_values())


def _create_gif_from_pngs(
    image_dir: Path,
    output_path: Path,
    duration_ms: int = 800,
    include_regex: str | None = None,
) -> None:
    """Build a GIF slideshow from PNG images in a directory."""
    image_paths = list(image_dir.glob("*.png"))
    if include_regex:
        pattern = re.compile(include_regex)
        image_paths = [path for path in image_paths if pattern.search(path.stem)]

    image_paths = sorted(
        image_paths,
        key=lambda path: (
            int(match.group(1)) if (match := re.search(r"_iter_(\d+)", path.stem)) else float("inf"),
            path.name,
        ),
    )

    if not image_paths:
        print(f"No PNG files found in {image_dir}; skipping GIF creation.")
        return

    frames = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            frame = img.convert("RGBA")
            frames.append(frame.copy())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    print(f"GIF saved to '{output_path}' from {len(frames)} frame(s).")

# ================================================


def load_config(config_file: str = 'config.yml') -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file {config_file} not found!")
    
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


def _get_uq_config(cfg: dict) -> tuple[bool, dict]:
    uq_cfg = cfg.get('uq', {})
    if isinstance(uq_cfg, dict):
        return bool(uq_cfg.get('enabled', False)), uq_cfg
    return bool(uq_cfg), {}


def load_uq_samples(uq_cfg: dict) -> pd.DataFrame:
    # Use a single hardcoded, non-configurable location for UQ samples
    output_csv = os.path.join('ensemble_results', 'uq_water_depth_samples.csv')

    if not os.path.exists(output_csv):
        print(f"UQ samples not found at {output_csv}.")
        return pd.DataFrame()

    df = pd.read_csv(output_csv)
    if 'water_depth_m' not in df.columns:
        print(f"UQ samples missing 'water_depth_m' column in {output_csv}.")
        return pd.DataFrame()

    return df


def plot_uq_distribution(uq_samples: np.ndarray, config: dict, uq_cfg: dict) -> None:
    """Generate and save QoI water depth distribution plot."""
    if uq_samples.size == 0:
        return

    analysis_visualization_cfg = config.get('analysis', {}).get('visualization', {})

    dpi = analysis_visualization_cfg.get('dpi', 300)
    figsize = tuple(analysis_visualization_cfg.get('figsize', [10, 6]))
    grid_cfg = analysis_visualization_cfg.get('grid', {})
    grid_enabled = grid_cfg.get('enabled', True)
    grid_linestyle = grid_cfg.get('linestyle', '--')
    grid_alpha = grid_cfg.get('alpha', 0.6)

    uq_plot_cfg = analysis_visualization_cfg.get(
        'uq_distribution_plot', analysis_visualization_cfg.get('distribution_plot', {})
    )
    hist_color = uq_plot_cfg.get('hist_color', 'seagreen')
    hist_edgecolor = uq_plot_cfg.get('hist_edgecolor', 'black')
    hist_alpha = uq_plot_cfg.get('hist_alpha', 0.8)
    kde_color = uq_plot_cfg.get('kde_color', 'darkgreen')
    kde_linewidth = uq_plot_cfg.get('kde_linewidth', 2)
    kde_bw_method = uq_plot_cfg.get('kde_bw_method', 0.3)
    dist_mean_color = uq_plot_cfg.get('mean_color', 'red')
    dist_mean_linestyle = uq_plot_cfg.get('mean_linestyle', ':')

    fig, ax = plt.subplots(figsize=figsize)

    min_val = float(np.min(uq_samples))
    max_val = float(np.max(uq_samples))

    if np.isclose(min_val, max_val):
        bins = 3
    else:
        q75, q25 = np.percentile(uq_samples, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr / (len(uq_samples) ** (1/3)) if iqr > 0 else (max_val - min_val) / 10
        bins = max(3, int(np.ceil((max_val - min_val) / bin_width)))

    n, bins_edges, _ = ax.hist(
        uq_samples,
        bins=bins,
        color=hist_color,
        edgecolor=hist_edgecolor,
        alpha=hist_alpha,
    )

    ax_twin = None
    if len(uq_samples) > 1 and not np.isclose(min_val, max_val):
        kde = stats.gaussian_kde(uq_samples, bw_method=kde_bw_method)
        x_range = np.linspace(min_val, max_val, 200)
        kde_values = kde(x_range)
        kde_values = kde_values * n.sum() * (bins_edges[1] - bins_edges[0])
        ax_twin = ax.twinx()
        ax_twin.plot(x_range, kde_values, color=kde_color, linewidth=kde_linewidth,
                     label='Kernel Density Estimate')
        ax_twin.set_yticklabels([])
        ax_twin.set_yticks([])

    mean_val = float(np.mean(uq_samples))
    ax.axvline(mean_val, color=dist_mean_color, linestyle=dist_mean_linestyle,
               label=f'Mean: {mean_val:.3f} m')

    point_xy = uq_cfg.get('point_xy')
    if isinstance(point_xy, (list, tuple)) and len(point_xy) == 2:
        ax.set_title(f"Distribution of Maximum Water Depth for each Iteration at ({point_xy[0]}, {point_xy[1]})")
    else:
        ax.set_title("Distribution of Maximum Water Depth (QoI)")

    ax.set_xlabel("Water Depth (m)")
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.grid(False)

    lines1, labels1 = ax.get_legend_handles_labels()
    if ax_twin is not None:
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax.legend()

    output_plot = uq_cfg.get('output_plot', os.path.join('plots', 'uq_water_depth_distribution.png'))
    os.makedirs(os.path.dirname(output_plot) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_plot, dpi=dpi)
    plt.close(fig)
    print(f"Plot saved as '{output_plot}'")


def print_uq_statistics(uq_samples: np.ndarray, uq_cfg: dict) -> None:
    if uq_samples.size == 0:
        return

    mean_val = float(np.mean(uq_samples))
    std_val = float(np.std(uq_samples))
    cov = (std_val / mean_val) * 100 if mean_val != 0 else float('inf')
    point_xy = uq_cfg.get('point_xy')
    point_label = f" at ({point_xy[0]}, {point_xy[1]})" if isinstance(point_xy, (list, tuple)) and len(point_xy) == 2 else ""

    print("\n--- UQ Water Depth Statistics ---")
    print(f"Mean Water Depth{point_label}: {mean_val:.3f} m")
    print(f"Standard Deviation: ±{std_val:.3f} m")
    print(f"Coefficient of Variation: {cov:.2f}%")
    print("----------------------------------\n")


def collect_energy_data(iterations: int) -> tuple[list[float], list[dict]]:
    """
    Collect energy data for the target PID across all iterations.
    """
    energy_results = []
    iteration_data = []
    
    print("--- Ensemble Total Energy Analysis (CPU + GPU) ---")
    
    for i in range(iterations):
        filename = f'ensemble_results/iter_{i}/telemetry.csv'
        
        if not os.path.exists(filename):
            continue
        
        target_pid = _extract_pid_from_execution_log(i)
        if target_pid is None:
            print(f"Iteration {i}: Could not extract PID from execution.log.")
            continue
            
        # Ensure resource_id and consumer_id are read as strings to prevent matching bugs
        df = pd.read_csv(filename, sep=';', dtype={'resource_id': str, 'consumer_id': str})
        target_pid_str = str(target_pid)
        
        # 1. Isolate the Process by PID safely
        df_gpu_raw = df[(df['metric'].str.contains('attributed_energy_gpu', na=False)) 
                        & (df['consumer_kind'] == 'process')
                        & (df['consumer_id'] == target_pid_str)].copy()
                        
        df_cpu_raw = df[(df['metric'].str.contains('attributed_energy_cpu', na=False)) 
                        & (df['consumer_kind'] == 'process')
                        & (df['consumer_id'] == target_pid_str)].copy()
        
        if df_gpu_raw.empty and df_cpu_raw.empty:
            print(f"Iteration {i}: Missing CPU or GPU data for PID {target_pid}.")
            continue

        # 2. Format Time (Use exact time, do NOT use dt.floor('100ms'))
        df_gpu_raw['timestamp'] = pd.to_datetime(df_gpu_raw['timestamp'])
        df_cpu_raw['timestamp'] = pd.to_datetime(df_cpu_raw['timestamp'])
        
        # 3. Squash Duplicates
        df_gpu = df_gpu_raw.groupby('timestamp', as_index=False)['value'].sum().sort_values('timestamp')
        df_cpu = df_cpu_raw.groupby('timestamp', as_index=False)['value'].sum().sort_values('timestamp')

        # 4. Cumulative Energy Calculation (Convert interval deltas to running total)
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
            print(f"Iteration {i}: No overlapping CPU/GPU data. Skipping.")
            continue

        # 6. Final Energy Calculation
        df_merged['cum_energy'] = df_merged['cpu_cum'] + df_merged['gpu_cum']
        df_merged['time_sec'] = (df_merged['timestamp'] - df_merged['timestamp'].iloc[0]).dt.total_seconds()
        
        run_total = df_merged['cum_energy'].iloc[-1]
        cpu_total = df_merged['cpu_cum'].iloc[-1]
        gpu_total = df_merged['gpu_cum'].iloc[-1]
        
        energy_results.append(run_total)
        iteration_data.append({
            'iteration': i,
            'df': df_merged,
            'cpu_total': cpu_total,
            'gpu_total': gpu_total,
            'run_total': run_total,
        })
        
        print(f"Iteration {i}: {run_total:.2f} Total J [CPU: {cpu_total:.2f} J | GPU: {gpu_total:.2f} J] (Duration: {df_merged['time_sec'].iloc[-1]:.2f}s)")
    
    return energy_results, iteration_data


def plot_cumulative_energy(iteration_data: list[dict], config: dict) -> None:
    """Generate and save cumulative energy vs time plot."""
    analysis_visualization_cfg = config.get('analysis', {}).get('visualization', {})
    
    dpi = analysis_visualization_cfg.get('dpi', 300)
    figsize = tuple(analysis_visualization_cfg.get('figsize', [10, 6]))
    grid_cfg = analysis_visualization_cfg.get('grid', {})
    grid_enabled = grid_cfg.get('enabled', True)
    grid_linestyle = grid_cfg.get('linestyle', '--')
    grid_alpha = grid_cfg.get('alpha', 0.6)
    
    cumulative_plot_cfg = analysis_visualization_cfg.get('cumulative_plot', {})
    cumulative_linewidth = cumulative_plot_cfg.get('linewidth', 1.5)
    cumulative_alpha = cumulative_plot_cfg.get('alpha', 0.7)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for idx, data in enumerate(iteration_data):
        df_merged = data['df']
        line_label = 'Monte Carlo Iterations' if idx == 0 else None
        ax.plot(df_merged['time_sec'], df_merged['cum_energy'], 
                alpha=cumulative_alpha, linewidth=cumulative_linewidth, label=line_label)
    
    ax.set_title("Cumulative Energy Consumption (CPU + GPU)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Total Energy Consumed (Joules)")
    if grid_enabled:
        ax.grid(True, linestyle=grid_linestyle, alpha=grid_alpha)
    ax.legend()
    
    fig.tight_layout()
    os.makedirs('plots', exist_ok=True)
    fig.savefig('plots/cumulative_energy_consumption.png', dpi=dpi)
    plt.close(fig)
    print("Plot saved as 'cumulative_energy_consumption.png'")


def plot_energy_per_iteration(energy_results: list[float], config: dict) -> None:
    """Generate and save energy cost per iteration plot."""
    if not energy_results:
        return
    
    analysis_visualization_cfg = config.get('analysis', {}).get('visualization', {})
    
    dpi = analysis_visualization_cfg.get('dpi', 300)
    figsize = tuple(analysis_visualization_cfg.get('figsize', [10, 6]))
    grid_cfg = analysis_visualization_cfg.get('grid', {})
    grid_enabled = grid_cfg.get('enabled', True)
    grid_linestyle = grid_cfg.get('linestyle', '--')
    grid_alpha = grid_cfg.get('alpha', 0.6)
    
    iteration_plot_cfg = analysis_visualization_cfg.get('iteration_plot', {})
    scatter_color = iteration_plot_cfg.get('scatter_color', 'orange')
    scatter_edgecolor = iteration_plot_cfg.get('scatter_edgecolor', 'black')
    scatter_size = iteration_plot_cfg.get('scatter_size', 80)
    line_color = iteration_plot_cfg.get('line_color', 'orange')
    line_alpha = iteration_plot_cfg.get('line_alpha', 0.4)
    line_style = iteration_plot_cfg.get('line_style', '--')
    iter_mean_color = iteration_plot_cfg.get('mean_color', 'red')
    iter_mean_linestyle = iteration_plot_cfg.get('mean_linestyle', ':')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    iters = range(len(energy_results))
    ax.scatter(iters, energy_results, color=scatter_color, edgecolors=scatter_edgecolor, 
               s=scatter_size, zorder=3)
    ax.plot(iters, energy_results, color=line_color, alpha=line_alpha, linestyle=line_style)
    
    mean_val = np.mean(energy_results)
    ax.axhline(mean_val, color=iter_mean_color, linestyle=iter_mean_linestyle, 
               label=f'Mean: {mean_val:.2f} J')
    
    ax.set_title("Total Energy Cost per Iteration")
    ax.set_xlabel("Iteration Number")
    ax.set_ylabel("Final Joules")
    if grid_enabled:
        ax.grid(True, linestyle=grid_linestyle, alpha=grid_alpha)
    ax.legend()
    
    fig.tight_layout()
    fig.savefig('plots/energy_cost_per_iteration.png', dpi=dpi)
    plt.close(fig)
    print("Plot saved as 'energy_cost_per_iteration.png'")


def plot_energy_distribution(energy_results: list[float], config: dict) -> None:
    """Generate and save energy distribution plot."""
    if not energy_results:
        return
    
    analysis_visualization_cfg = config.get('analysis', {}).get('visualization', {})
    
    dpi = analysis_visualization_cfg.get('dpi', 300)
    figsize = tuple(analysis_visualization_cfg.get('figsize', [10, 6]))
    grid_cfg = analysis_visualization_cfg.get('grid', {})
    grid_enabled = grid_cfg.get('enabled', True)
    grid_linestyle = grid_cfg.get('linestyle', '--')
    grid_alpha = grid_cfg.get('alpha', 0.6)
    
    distribution_plot_cfg = analysis_visualization_cfg.get('distribution_plot', {})
    hist_color = distribution_plot_cfg.get('hist_color', 'steelblue')
    hist_edgecolor = distribution_plot_cfg.get('hist_edgecolor', 'black')
    hist_alpha = distribution_plot_cfg.get('hist_alpha', 0.8)
    kde_color = distribution_plot_cfg.get('kde_color', 'darkblue')
    kde_linewidth = distribution_plot_cfg.get('kde_linewidth', 2)
    kde_bw_method = distribution_plot_cfg.get('kde_bw_method', 0.3)
    dist_mean_color = distribution_plot_cfg.get('mean_color', 'red')
    dist_mean_linestyle = distribution_plot_cfg.get('mean_linestyle', ':')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Freedman-Diaconis rule for bin width
    q75, q25 = np.percentile(energy_results, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(energy_results) ** (1/3)) if iqr > 0 else (max(energy_results) - min(energy_results)) / 10
    bins = max(3, int(np.ceil((max(energy_results) - min(energy_results)) / bin_width)))
    
    n, bins_edges, patches = ax.hist(
        energy_results,
        bins=bins,
        color=hist_color,
        edgecolor=hist_edgecolor,
        alpha=hist_alpha,
    )

    # Add Kernel Density Estimate smooth line overlay
    ax_twin = None
    if len(energy_results) > 1:
        kde = stats.gaussian_kde(energy_results, bw_method=kde_bw_method)
        x_range = np.linspace(min(energy_results), max(energy_results), 200)
        kde_values = kde(x_range)
        # Scale KDE to match histogram height
        kde_values = kde_values * n.sum() * (bins_edges[1] - bins_edges[0])
        ax_twin = ax.twinx()
        ax_twin.plot(x_range, kde_values, color=kde_color, linewidth=kde_linewidth, 
                     label='Kernel Density Estimate')
        ax_twin.set_yticklabels([])
        ax_twin.set_yticks([])

    mean_val = np.mean(energy_results)
    ax.axvline(mean_val, color=dist_mean_color, linestyle=dist_mean_linestyle, 
               label=f'Mean: {mean_val:.2f} J')

    ax.set_title("Distribution of Total Energy Cost")
    ax.set_xlabel("Final Joules")
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.grid(False)
    
    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    if ax_twin is not None:
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax.legend()
    
    fig.tight_layout()
    fig.savefig('plots/energy_cost_distribution.png', dpi=dpi)
    plt.close(fig)
    print("Plot saved as 'energy_cost_distribution.png'")


def print_statistics(energy_results: list[float]) -> None:
    """Print statistical summary of energy results."""
    if not energy_results:
        return
    
    energy_array = np.array(energy_results)
    mean_energy = np.mean(energy_array)
    std_dev_energy = np.std(energy_array)
    cov = (std_dev_energy / mean_energy) * 100

    print("\n--- Final Statistics ---")
    print(f"Mean Energy Cost: {mean_energy:.2f} Joules")
    print(f"Standard Deviation: ±{std_dev_energy:.2f} Joules")
    print(f"Coefficient of Variation: {cov:.2f}%")
    print("------------------------\n")



def _regenerate_water_height_with_fixed_colorbar(water_height_dir: Path) -> None:
    """Regenerate water height PNGs with fixed colorbar range across all frames.
    
    This ensures that colors are consistent across animation frames, making changes meaningful.
    Loads previously saved raster data from flood_model.py executions.
    Uses only matplotlib (no synxflow dependency).
    
    Args:
        water_height_dir: Path to directory containing water_height_*.png files
    """
    raster_data_dir = Path('plots/.raster_data')
    if not raster_data_dir.exists():
        print(f"Raster data directory not found at {raster_data_dir}. Skipping colorbar fix.")
        return
    
    # Find all saved raster data files
    pkl_files = sorted(raster_data_dir.glob('water_height_iter_*.pkl'))
    
    if not pkl_files:
        print("No saved raster data found for colorbar fix.")
        return
    
    # Find global min/max across all iterations
    global_min = float('inf')
    global_max = float('-inf')
    raster_data_list = []
    
    print("Scanning water depth range across all iterations...")
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                array, header = pickle.load(f)
            # Exclude NODATA values
            valid_data = array[array != header['NODATA_value']]
            if len(valid_data) > 0:
                global_min = min(global_min, float(np.nanmin(valid_data)))
                global_max = max(global_max, float(np.nanmax(valid_data)))
                raster_data_list.append((pkl_file, array, header))
        except Exception as e:
            print(f"  Warning: Could not load {pkl_file}: {e}")
    
    if global_min == float('inf') or global_max == float('-inf'):
        print("Could not determine valid water depth range.")
        return
    
    print(f"Water depth range: {global_min:.4f} - {global_max:.4f} m")
    print(f"Regenerating {len(raster_data_list)} water height visualizations with fixed colorbar...")
    
    # Regenerate PNG files with fixed vmin/vmax using matplotlib
    for pkl_file, array, header in raster_data_list:
        # Extract iteration number
        iteration_match = re.search(r'water_height_iter_(\d+)', pkl_file.stem)
        if not iteration_match:
            continue
        
        iteration_num = iteration_match.group(1)
        png_filename = water_height_dir / f'water_height_iter_{iteration_num}.png'
        
        try:
            # Set NODATA values to NaN for proper visualization
            array_display = array.astype(float)
            array_display[array == header['NODATA_value']] = np.nan
            
            # Create figure with matplotlib
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Get extent from header (assuming standard grid structure)
            ncols = header['ncols']
            nrows = header['nrows']
            xllcorner = header['xllcorner']
            yllcorner = header['yllcorner']
            cellsize = header['cellsize']
            
            extent = [xllcorner, xllcorner + ncols * cellsize,
                      yllcorner, yllcorner + nrows * cellsize]
            
            # Plot with fixed colorbar range
            im = ax.imshow(array_display, extent=extent, origin='upper', 
                          cmap='viridis', vmin=global_min, vmax=global_max,
                          aspect='equal', interpolation='nearest')
            
            ax.set_title('Final Water Depth for each Iteration', fontsize=14)
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, pad=0.05)
            cbar.set_label('Depth (m)', fontsize=12, labelpad=20)
            cbar.ax.tick_params(labelsize=10)
            
            fig.subplots_adjust(right=0.88)
            fig.savefig(str(png_filename), dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"  Warning: Could not regenerate {png_filename}: {e}")
            import traceback
            traceback.print_exc()


def _regenerate_dem_with_fixed_colorbar(dem_output_dir: Path) -> None:
    """Regenerate DEM PNGs with fixed colorbar range across all frames.
    
    This ensures that colors are consistent across animation frames, making changes meaningful.
    Loads previously saved raster data from flood_model.py executions.
    Uses only matplotlib (no synxflow dependency).
    
    Args:
        dem_output_dir: Path to directory containing dem_3d_*.png files
    """
    raster_data_dir = Path('plots/.raster_data')
    if not raster_data_dir.exists():
        print(f"Raster data directory not found at {raster_data_dir}. Skipping DEM colorbar fix.")
        return
    
    # Find all saved DEM raster data files
    pkl_files = sorted(raster_data_dir.glob('dem_3d_iter_*.pkl'))
    
    if not pkl_files:
        print("No saved DEM raster data found for colorbar fix.")
        return
    
    # Find global min/max across all iterations
    global_min = float('inf')
    global_max = float('-inf')
    raster_data_list = []
    
    print("Scanning DEM elevation range across all iterations...")
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                array, header = pickle.load(f)
            # For DEM, include all values (typically no NODATA)
            valid_data = array[~np.isnan(array)]
            if len(valid_data) > 0:
                global_min = min(global_min, float(np.nanmin(valid_data)))
                global_max = max(global_max, float(np.nanmax(valid_data)))
                raster_data_list.append((pkl_file, array, header))
        except Exception as e:
            print(f"  Warning: Could not load {pkl_file}: {e}")
    
    if global_min == float('inf') or global_max == float('-inf'):
        print("Could not determine valid DEM elevation range.")
        return
    
    print(f"DEM elevation range: {global_min:.2f} - {global_max:.2f} m")
    print(f"Regenerating {len(raster_data_list)} DEM visualizations with fixed colorbar...")
    
    # Regenerate PNG files with fixed vmin/vmax using matplotlib
    for pkl_file, array, header in raster_data_list:
        # Extract iteration number
        iteration_match = re.search(r'dem_3d_iter_(\d+)', pkl_file.stem)
        if not iteration_match:
            continue
        
        iteration_num = iteration_match.group(1)
        png_filename = dem_output_dir / f'dem_3d_iter_{iteration_num}.png'
        
        try:
            # Create figure with matplotlib
            fig, ax = plt.subplots(figsize=(12, 11))
            ncols = header['ncols']
            nrows = header['nrows']
            xllcorner = header['xllcorner']
            yllcorner = header['yllcorner']
            cellsize = header['cellsize']

            extent = [xllcorner, xllcorner + ncols * cellsize,
                      yllcorner, yllcorner + nrows * cellsize]
            
            # Plot with fixed colorbar range
            im = ax.imshow(array, cmap='terrain', extent=extent, origin='upper',
                          vmin=global_min, vmax=global_max,
                          aspect='equal', interpolation='nearest')
            
            ax.set_title('Digital Elevation Model for each Iteration', fontsize=16)
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, pad=0.05)
            cbar.set_label('Elevation (m)', fontsize=12, labelpad=20)
            cbar.ax.tick_params(labelsize=10)
            
            fig.subplots_adjust(right=0.88)
            fig.savefig(str(png_filename), dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"  Warning: Could not regenerate {png_filename}: {e}")
            import traceback
            traceback.print_exc()


def generate_gifs(config: dict) -> None:
    """Generate GIF animations from PNG sequences if configured."""
    analysis_visualization_cfg = config.get('analysis', {}).get('visualization', {})
    generate_gif = analysis_visualization_cfg.get('generate_gif', False)
    gif_duration_ms = analysis_visualization_cfg.get('duration_ms', 800)
    
    if generate_gif:
        # Fix colorbar range before creating DEM GIF
        print("Fixing DEM colorbar to consistent range...")
        _regenerate_dem_with_fixed_colorbar(Path("plots/dem_3d"))
        
        _create_gif_from_pngs(
            Path("plots/dem_3d"),
            Path("plots/dem_3d.gif"),
            duration_ms=gif_duration_ms,
            include_regex=r"_iter_\d+$",
        )
        
        # Fix colorbar range before creating water height GIF
        print("Fixing water height colorbar to consistent range...")
        _regenerate_water_height_with_fixed_colorbar(Path("plots/water_height"))
        
        _create_gif_from_pngs(
            Path("plots/water_height"),
            Path("plots/water_height.gif"),
            duration_ms=gif_duration_ms,
            include_regex=r"_iter_\d+$",
        )
    else:
        print("GIF creation skipped because analysis.visualization.generate_gif is false.")


def main() -> None:
    """Main analysis orchestration function."""
    cfg = load_config('config.yml')
    iterations = cfg['monte_carlo']['iterations']
    
    # Collect energy data from all iterations
    energy_results, iteration_data = collect_energy_data(iterations)
    
    # Generate plots if data was collected
    if energy_results:
        plot_cumulative_energy(iteration_data, cfg)
        plot_energy_per_iteration(energy_results, cfg)
        plot_energy_distribution(energy_results, cfg)
        print_statistics(energy_results)

    uq_enabled, uq_cfg = _get_uq_config(cfg)
    if uq_enabled:
        uq_df = load_uq_samples(uq_cfg)
        if not uq_df.empty:
            uq_samples = uq_df['water_depth_m'].to_numpy(dtype=float)
            plot_uq_distribution(uq_samples, cfg, uq_cfg)
            print_uq_statistics(uq_samples, uq_cfg)
        else:
            print("UQ enabled but no samples found; skipping UQ distribution plot.")
    
    # Generate GIFs
    generate_gifs(cfg)


if __name__ == "__main__":
    main()