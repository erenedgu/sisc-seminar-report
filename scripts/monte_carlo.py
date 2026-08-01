import numpy as np
import os
import subprocess
import tomllib
import yaml
import rasterio
from rasterio.windows import Window
import textwrap


def _normalize_point_type(point_type: str) -> str:
    kind = str(point_type).strip().lower()
    if kind in {"index", "grid", "pixel", "array"}:
        return "index"
    return "map"


def _resolve_uq_point(cfg: dict, point_xy, point_type: str):
    if point_xy is not None:
        return point_xy, point_type
    gauges = cfg.get("model", {}).get("observation", {}).get("gauges_position", [])
    if gauges:
        if point_type == "index":
            print(
                "WARNING: uq.point_xy is missing while uq.point_type is 'index'. "
                "Falling back to model.observation.gauges_position uses map coordinates."
            )
            point_type = "map"
        return gauges[0], point_type
    return None, point_type


def _is_valid_point(point_xy) -> bool:
    return isinstance(point_xy, (list, tuple)) and len(point_xy) == 2


def _select_water_depth_output(output_dir: str, end_time: int) -> str | None:
    h_max_path = os.path.join(output_dir, f"h_max_{end_time}.asc")
    h_final_path = os.path.join(output_dir, f"h_{end_time}.asc")
    if os.path.exists(h_max_path):
        return h_max_path
    if os.path.exists(h_final_path):
        return h_final_path
    return None


def _sample_water_depth_at_point(output_path: str, point_xy, point_type: str) -> float | None:
    with rasterio.open(output_path) as src:
        if point_type == "index":
            col = int(round(float(point_xy[0])))
            row = int(round(float(point_xy[1])))
        else:
            row, col = src.index(float(point_xy[0]), float(point_xy[1]))
            row = int(row)
            col = int(col)

        if row < 0 or col < 0 or row >= src.height or col >= src.width:
            print(
                f"WARNING: UQ point ({point_xy[0]}, {point_xy[1]}) is out of bounds for {output_path}."
            )
            return None

        value = src.read(1, window=Window(col, row, 1, 1))[0, 0]
        nodata = src.nodata
        if nodata is not None and value == nodata:
            print(f"WARNING: UQ sample landed on NODATA at row={row}, col={col}.")
            return None
        if np.isnan(value):
            print(f"WARNING: UQ sample is NaN at row={row}, col={col}.")
            return None

        return float(value)


def _render_baseline_dem_with_point(
    python_exe: str,
    dem_path: str,
    point_xy,
    point_type: str,
    output_path: str,
) -> None:
    if not _is_valid_point(point_xy):
        print("WARNING: Cannot render baseline DEM; uq.point_xy is missing or invalid.")
        return

    script = textwrap.dedent(
        """
        import os
        import rasterio
        import matplotlib.pyplot as plt
        import numpy as np

        dem_path = os.environ.get("DEMO_DEM_PATH", "")
        point_type = os.environ.get("DEMO_POINT_TYPE", "map").strip().lower()
        point_xy = [float(os.environ.get("DEMO_POINT_X", "0")), float(os.environ.get("DEMO_POINT_Y", "0"))]
        output_path = os.environ.get("DEMO_OUT_PATH", "dem_3d_baseline_marked.png")

        abs_dem = os.path.abspath(dem_path)
        vfs_path = f"/vsigzip/{abs_dem}" if dem_path.endswith(".gz") else abs_dem

        with rasterio.open(vfs_path) as src:
            dem = src.read(1)
            nodata = src.nodata
            bounds = src.bounds
            if nodata is not None:
                dem = dem.astype(float)
                dem[dem == nodata] = np.nan

            if point_type == "index":
                col = int(round(point_xy[0]))
                row = int(round(point_xy[1]))
                map_x, map_y = src.xy(row, col)
            else:
                map_x = float(point_xy[0])
                map_y = float(point_xy[1])
                row, col = src.index(map_x, map_y)

        fig, ax = plt.subplots(figsize=(10, 8))
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        im = ax.imshow(
            dem,
            cmap="terrain",
            extent=extent,
            origin="upper",
            interpolation="nearest",
            aspect="equal",
        )
        ax.scatter([map_x], [map_y], s=120, c="red", edgecolors="white", linewidths=1.5, zorder=5)
        ax.set_title("Digital Elevation Model")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        cbar = plt.colorbar(im, ax=ax, pad=0.03)
        cbar.set_label("Elevation (m)")

        fig.tight_layout()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        """
    ).strip()

    env = os.environ.copy()
    env["DEMO_DEM_PATH"] = dem_path
    env["DEMO_POINT_TYPE"] = point_type
    env["DEMO_POINT_X"] = str(point_xy[0])
    env["DEMO_POINT_Y"] = str(point_xy[1])
    env["DEMO_OUT_PATH"] = output_path

    try:
        subprocess.run(
            [python_exe, "-c", script],
            check=True,
            env=env,
            text=True,
            capture_output=True,
        )
        print(f"Saved baseline DEM with UQ point: {output_path}")
    except subprocess.CalledProcessError as exc:
        print("WARNING: Failed to render baseline DEM with UQ point.")
        if exc.stdout:
            print(exc.stdout)
        if exc.stderr:
            print(exc.stderr)

# --- Load Configuration ---
config_file = 'config.yml'
if not os.path.exists(config_file):
    raise FileNotFoundError(f"Configuration file {config_file} not found!")

with open(config_file, 'r') as file:
    cfg = yaml.safe_load(file)

project_root = os.path.dirname(os.path.abspath(config_file))

# --- Research Parameters (from YAML) ---
iterations = cfg['monte_carlo']['iterations']
std_dev = cfg['monte_carlo']['std_dev']

uq_cfg = cfg.get('uq', {})
uq_enabled = False
uq_point_xy = None
uq_point_type = "map"
uq_output_csv = os.path.join('ensemble_results', 'uq_water_depth_samples.csv')

if isinstance(uq_cfg, dict):
    uq_enabled = bool(uq_cfg.get('enabled', False))
    uq_point_xy = uq_cfg.get('point_xy')
    uq_point_type = uq_cfg.get('point_type', uq_point_type)
else:
    uq_enabled = bool(uq_cfg)

uq_point_type = _normalize_point_type(uq_point_type)
uq_point_xy, uq_point_type = _resolve_uq_point(cfg, uq_point_xy, uq_point_type)

uq_output_dir = os.path.dirname(uq_output_csv)
uq_end_time = None
uq_output_grid_dir = None

if uq_enabled:
    if not _is_valid_point(uq_point_xy):
        print("WARNING: UQ enabled but uq.point_xy is missing or invalid. Skipping UQ sampling.")
        uq_enabled = False
    else:
        solver_cfg = cfg.get('model', {}).get('solver', {})
        if 'end_time' not in solver_cfg:
            print("WARNING: UQ enabled but model.solver.end_time is missing. Skipping UQ sampling.")
            uq_enabled = False
        else:
            uq_end_time = int(solver_cfg['end_time'])
            uq_output_grid_dir = os.path.join(project_root, 'gaia_flood_case', 'output')

            if uq_output_dir:
                os.makedirs(uq_output_dir, exist_ok=True)

            with open(uq_output_csv, 'w') as file:
                file.write("iteration,water_depth_m\n")

            print(
                "UQ enabled: sampling water depth at "
                f"({uq_point_xy[0]}, {uq_point_xy[1]}) using {uq_point_type} coordinates."
            )

# 0. Setup Master Output Directory
BASE_OUT_DIR = "ensemble_results"
os.makedirs(BASE_OUT_DIR, exist_ok=True)

# 1. Locate the pristine baseline map
base_dem_path = cfg['dem']

if not os.path.exists(base_dem_path):
    raise FileNotFoundError(f"Baseline DEM not found at: {base_dem_path}")

print(f"Starting Monte Carlo Ensemble: {iterations} runs, sigma={std_dev}m")

abs_dem_path = os.path.abspath(base_dem_path)
vfs_path = f'/vsigzip/{abs_dem_path}' if base_dem_path.endswith('.gz') else base_dem_path

# Read the map using Rasterio
with rasterio.open(vfs_path) as src:
    original_elevation = src.read(1)  # Read the first band (elevation data)
    dem_meta = src.meta.copy()  # Save the coordinates/projection data for later

# Force the output to be a standard TIFF to avoid compression write errors
dem_meta.update(driver='GTiff')

# Path to the Alumet binary (from config)
alumet_bin = cfg['monte_carlo']['alumet_bin']

if not os.path.exists(alumet_bin):
    raise FileNotFoundError(f"Alumet binary not found at: {alumet_bin}")

# Read the CSV output name from Alumet's own config so the filename stays single-sourced.
with open('alumet-config.toml', 'rb') as file:
    alumet_cfg = tomllib.load(file)

default_alumet_output = alumet_cfg['plugins']['csv']['output_path']

python_exe = subprocess.check_output(
    "micromamba run -n env-model which python", shell=True, text=True
).strip()

if uq_enabled:
    baseline_output = os.path.join("plots", "dem_3d", "dem_3d_baseline_marked.png")
    _render_baseline_dem_with_point(
        python_exe,
        base_dem_path,
        uq_point_xy,
        uq_point_type,
        baseline_output,
    )

for i in range(iterations):
    print(f"\n==========================================")
    print(f"      Running Iteration {i+1} / {iterations}      ")
    print(f"==========================================")

    iter_dir = os.path.join(BASE_OUT_DIR, f"iter_{i}")
    os.makedirs(iter_dir, exist_ok=True)

    # 2. Inject Gaussian Noise
    noise = np.random.normal(0, std_dev, original_elevation.shape)
    noisy_elevation = original_elevation + noise

    nodata_val = dem_meta.get('nodata')
    if nodata_val is not None:
        # Revert any no-data pixels back to their exact original value
        noisy_elevation[original_elevation == nodata_val] = nodata_val

    # 3. Save the noisy map using Rasterio (as a safe .tif)
    noisy_filename = f'DEM_noisy_{i}.tif'
    with rasterio.open(noisy_filename, 'w', **dem_meta) as dst:
        dst.write(noisy_elevation.astype(dem_meta['dtype']), 1)

    # 4. Execute the Simulation & Measurement Pipeline
    iter_log = os.path.join(iter_dir, "execution.log")
    archive_csv = os.path.join(iter_dir, "telemetry.csv")

    # Remove any leftover telemetry from a previous run before starting a new iteration.
    if os.path.exists(default_alumet_output):
        os.remove(default_alumet_output)
    
    cmd = (
        f"{alumet_bin} --config alumet-config.toml "
        f"exec {python_exe} scripts/flood_model.py --dem {noisy_filename} --config {config_file} "
        f"2>&1 | tee {iter_log}"
    )
    
    print(f"Executing: {cmd}")
    # Use bash -o pipefail to catch failures in the pipeline, and check=True to raise on failure
    subprocess.run(f"bash -o pipefail -c \"{cmd}\"", shell=True, check=True)

    # 5. Archive Telemetry
    if os.path.exists(default_alumet_output):
        os.rename(default_alumet_output, archive_csv)
        print(f"Saved energy telemetry to {archive_csv}")
    else:
        print(f"WARNING: Telemetry missing ({default_alumet_output} not found) for iteration {i}!")

    if uq_enabled:
        output_h_path = _select_water_depth_output(uq_output_grid_dir, uq_end_time)
        if output_h_path is None:
            print("WARNING: UQ output water depth file not found; skipping sample for this iteration.")
        else:
            water_depth = _sample_water_depth_at_point(output_h_path, uq_point_xy, uq_point_type)
            if water_depth is not None:
                with open(uq_output_csv, 'a') as file:
                    file.write(f"{i},{water_depth:.6f}\n")
                print(f"Saved UQ sample: iteration {i}, water depth {water_depth:.6f} m")
    
    # 6. Cleanup
    if os.path.exists(noisy_filename):
        os.remove(noisy_filename)

print("\nEnsemble complete!")