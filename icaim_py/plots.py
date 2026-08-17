from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import rasterio
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds as window_from_bounds
from scipy.io import netcdf_file

from .common import find_repo_root, infer_gps_layout, load_results_file


def _normalize_decomposition_name(value: str) -> str:
    normalized = str(value).strip().upper()
    if normalized not in {"ICA", "PCA"}:
        raise ValueError(f"Unsupported decomposition {value!r}. Expected 'ICA' or 'PCA'.")
    return normalized


def _component_prefix(decomposition: str) -> str:
    return "IC" if decomposition == "ICA" else "PC"


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _stringify(value.item())
        if value.size == 1:
            return _stringify(value.reshape(-1)[0])
        return " ".join(_stringify(item) for item in value.reshape(-1))
    return str(value)


def _first_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _first_string(value.item())
        for item in value.reshape(-1):
            text = _first_string(item)
            if text:
                return text
        return ""
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _first_string(item)
            if text:
                return text
        return ""
    return _stringify(value)


def _as_component_matrix(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array[:, np.newaxis]
    return array


def _as_diagonal(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.array([float(array)])
    if array.ndim == 1:
        return array
    return np.diag(array)


def _safe_component_indices(components: Sequence[int] | None, n_components: int) -> list[int]:
    if components is None:
        return list(range(n_components))
    selected: list[int] = []
    for component in components:
        if component < 1 or component > n_components:
            raise ValueError(f"Component index {component} is outside the valid range 1..{n_components}")
        selected.append(component - 1)
    return selected


def _nice_step(span: float) -> float:
    if span <= 0:
        return 0.5
    raw = span / 4.0
    exponent = math.floor(math.log10(raw))
    base = raw / (10**exponent)
    if base <= 1:
        mantissa = 1
    elif base <= 2:
        mantissa = 2
    elif base <= 5:
        mantissa = 5
    else:
        mantissa = 10
    return mantissa * 10**exponent


def _rounded_extent(lon: np.ndarray, lat: np.ndarray) -> tuple[float, float, float, float]:
    lon_min = float(np.nanmin(lon))
    lon_max = float(np.nanmax(lon))
    lat_min = float(np.nanmin(lat))
    lat_max = float(np.nanmax(lat))
    lon_span = max(lon_max - lon_min, 0.3)
    lat_span = max(lat_max - lat_min, 0.3)
    lon_pad = max(0.12 * lon_span, 0.15)
    lat_pad = max(0.12 * lat_span, 0.15)
    lon_step = _nice_step(lon_span + 2.0 * lon_pad)
    lat_step = _nice_step(lat_span + 2.0 * lat_pad)
    return (
        math.floor((lon_min - lon_pad) / lon_step) * lon_step,
        math.ceil((lon_max + lon_pad) / lon_step) * lon_step,
        math.floor((lat_min - lat_pad) / lat_step) * lat_step,
        math.ceil((lat_max + lat_pad) / lat_step) * lat_step,
    )


def _padded_station_extent(lon: np.ndarray, lat: np.ndarray) -> tuple[float, float, float, float]:
    lon_min = float(np.nanmin(lon))
    lon_max = float(np.nanmax(lon))
    lat_min = float(np.nanmin(lat))
    lat_max = float(np.nanmax(lat))
    lon_span = max(lon_max - lon_min, 0.3)
    lat_span = max(lat_max - lat_min, 0.3)
    lon_pad = max(0.08 * lon_span, 0.06)
    lat_pad = max(0.08 * lat_span, 0.06)
    return (lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad)


def _plot_extent(
    lon: np.ndarray,
    lat: np.ndarray,
    background: dict[str, Any] | None,
) -> tuple[float, float, float, float]:
    rounded = _rounded_extent(lon, lat)
    if background is None:
        return rounded

    bg_left, bg_right, bg_bottom, bg_top = [float(value) for value in background["extent"]]
    if (
        float(np.nanmin(lon)) >= bg_left
        and float(np.nanmax(lon)) <= bg_right
        and float(np.nanmin(lat)) >= bg_bottom
        and float(np.nanmax(lat)) <= bg_top
    ):
        return (bg_left, bg_right, bg_bottom, bg_top)
    return rounded


def _extent_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    left = max(float(first[0]), float(second[0]))
    right = min(float(first[1]), float(second[1]))
    bottom = max(float(first[2]), float(second[2]))
    top = min(float(first[3]), float(second[3]))
    if left >= right or bottom >= top:
        return None
    return (left, right, bottom, top)


def _extent_covers(
    container: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
    atol: float = 1e-6,
) -> bool:
    return (
        float(container[0]) <= float(target[0]) + atol
        and float(container[1]) >= float(target[1]) - atol
        and float(container[2]) <= float(target[2]) + atol
        and float(container[3]) >= float(target[3]) - atol
    )


def _extent_area(extent: tuple[float, float, float, float]) -> float:
    return max(0.0, float(extent[1]) - float(extent[0])) * max(0.0, float(extent[3]) - float(extent[2]))


def _nice_reference_value(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    scaled = value / (10**exponent)
    if scaled < 1.5:
        mantissa = 1.0
    elif scaled < 3.5:
        mantissa = 2.0
    elif scaled < 7.5:
        mantissa = 5.0
    else:
        mantissa = 10.0
    return mantissa * 10**exponent


def _normalization_factor(series: np.ndarray, normalization: str) -> float:
    if normalization == "none":
        return 1.0
    if normalization == "peak-to-peak":
        amplitude = float(np.nanmax(series) - np.nanmin(series))
    elif normalization == "unit-max":
        amplitude = float(np.nanmax(np.abs(series)))
    else:
        raise ValueError(f"Unsupported normalization mode: {normalization}")
    if not np.isfinite(amplitude) or amplitude == 0.0:
        return 1.0
    return 1.0 / amplitude


def _flip_sign(series: np.ndarray) -> float:
    max_value = float(np.nanmax(series))
    min_value = float(np.nanmin(series))
    if abs(min_value) > abs(max_value):
        return -1.0
    return 1.0


def _load_results(results_or_file: dict[str, Any] | str | Path) -> tuple[dict[str, Any], str]:
    if isinstance(results_or_file, dict):
        return results_or_file, "results"
    path = Path(results_or_file).resolve()
    return load_results_file(path), path.stem


def _infer_repo_root(results: dict[str, Any], repo_root: str | Path | None) -> Path:
    cfg = results.get("cfg") if isinstance(results, dict) else None
    cfg_repo = _field(cfg, "repo_root")
    if repo_root is not None:
        resolved = Path(repo_root).resolve()
        if cfg_repo:
            cfg_resolved = Path(_stringify(cfg_repo)).resolve()
            has_background_assets = (
                (resolved / "Scenarios" / "casestudy" / "case1" / "gmt" / "maps" / "dem_gradient.grd").exists()
                or (resolved / "Data" / "DEM").exists()
            )
            if has_background_assets:
                return resolved
            return cfg_resolved
        return resolved
    if cfg_repo:
        return Path(_stringify(cfg_repo)).resolve()
    return find_repo_root(Path(__file__).resolve())


def _extract_station_names(results: dict[str, Any], decomp: Any, n_stations: int, component_size: int) -> list[str]:
    names = _field(results, "STATIONS_name")
    if names is not None:
        array = np.atleast_1d(names)
        return [_stringify(item)[:4] for item in array[:n_stations]]
    decomp_names = _field(decomp, "name")
    if decomp_names is None:
        return [f"S{i + 1:03d}" for i in range(n_stations)]
    array = np.atleast_1d(decomp_names)
    return [_stringify(array[index * component_size])[:4] for index in range(n_stations)]


def _extract_plot_data(results: dict[str, Any], decomposition: str) -> dict[str, Any]:
    decomposition = _normalize_decomposition_name(decomposition)
    prefix = _component_prefix(decomposition)
    if f"{decomposition}_U" in results:
        u = _as_component_matrix(results[f"{decomposition}_U"])
        v = _as_component_matrix(results[f"{decomposition}_V"])
        diag_s = _as_diagonal(results[f"{decomposition}_S"])
        llh = np.asarray(results.get(f"{decomposition}_llh", results["Xd_llh"]), dtype=float)
        timeline = np.asarray(results.get(f"{decomposition}_timeline", results["Xd_timeline"]), dtype=float).reshape(-1)
        _, components = infer_gps_layout(results.get(f"{decomposition}_type", results.get("Xd_type")), u.shape[0])
        if llh.ndim != 2 or llh.shape[1] < 2:
            raise ValueError(f"{decomposition}_llh/Xd_llh is not in the expected Mx3 format.")

        sort_index = np.argsort(diag_s)[::-1]
        u = u[:, sort_index]
        v = v[:, sort_index]
        diag_s = diag_s[sort_index]

        component_size = len(components)
        n_stations = u.shape[0] // component_size
        if "STATIONS_name" in results:
            station_names = [_stringify(item)[:4] for item in np.atleast_1d(results["STATIONS_name"])[:n_stations]]
        else:
            station_names = _extract_station_names(results, None, n_stations, component_size)
        station_lon = llh[0::component_size, 0]
        station_lat = llh[0::component_size, 1]

        return {
            "decomposition": decomposition,
            "component_prefix": prefix,
            "components": components,
            "component_size": component_size,
            "U": u,
            "V": v,
            "S": diag_s,
            "timeline": timeline,
            "station_lon": station_lon,
            "station_lat": station_lat,
            "station_names": station_names,
            "n_stations": n_stations,
            "n_components": u.shape[1],
        }

    decomp = _field(results, decomposition)
    if decomp is None:
        raise ValueError(f"The provided results do not contain a {decomposition} decomposition.")
    xd = _field(results, "Xd")

    u = _as_component_matrix(_field(decomp, "U"))
    v = _as_component_matrix(_field(decomp, "V"))
    diag_s = _as_diagonal(_field(decomp, "S"))
    llh = np.asarray(_field(decomp, "llh", _field(xd, "llh")), dtype=float)
    timeline = np.asarray(_field(decomp, "timeline", _field(xd, "timeline")), dtype=float).reshape(-1)
    _, components = infer_gps_layout(_field(decomp, "type", _field(xd, "type")), u.shape[0])
    if llh.ndim != 2 or llh.shape[1] < 2:
        raise ValueError(f"{decomposition}.llh is not in the expected Mx3 format.")

    sort_index = np.argsort(diag_s)[::-1]
    u = u[:, sort_index]
    v = v[:, sort_index]
    diag_s = diag_s[sort_index]

    component_size = len(components)
    n_stations = u.shape[0] // component_size
    station_names = _extract_station_names(results, decomp, n_stations, component_size)
    station_lon = llh[0::component_size, 0]
    station_lat = llh[0::component_size, 1]

    return {
        "decomposition": decomposition,
        "component_prefix": prefix,
        "components": components,
        "component_size": component_size,
        "U": u,
        "V": v,
        "S": diag_s,
        "timeline": timeline,
        "station_lon": station_lon,
        "station_lat": station_lat,
        "station_names": station_names,
        "n_stations": n_stations,
        "n_components": u.shape[1],
    }


def _load_background(
    repo_root: Path,
    background_grid: str | Path | None,
    extent: tuple[float, float, float, float],
) -> dict[str, np.ndarray | tuple[float, float, float, float]] | None:
    if background_grid in (None, "auto"):
        candidates: list[Path] = []
        case1_candidate = repo_root / "Scenarios" / "casestudy" / "case1" / "gmt" / "maps" / "dem_gradient.grd"
        if case1_candidate.exists():
            candidates.append(case1_candidate)
        dem_root = repo_root / "Data" / "DEM"
        if dem_root.exists():
            candidates.extend(sorted(dem_root.rglob("*hillshade*.tif")))
            candidates.extend(sorted(dem_root.rglob("*hillshade*.tiff")))
            candidates.extend(sorted(dem_root.rglob("*dem*.tif")))
            candidates.extend(sorted(dem_root.rglob("*dem*.tiff")))
    elif _stringify(background_grid).lower() == "none":
        candidates = []
    else:
        candidates = [Path(background_grid).resolve()]

    best_background: dict[str, Any] | None = None
    best_score: tuple[int, float] | None = None
    for path in candidates:
        if not path.exists():
            continue
        suffix = path.suffix.lower()
        if suffix in {".tif", ".tiff"}:
            background = _load_geotiff_background(path, extent)
        else:
            background = _load_netcdf_background(path, extent)
        if background is not None:
            score = (
                1 if bool(background.get("covers_full_extent", False)) else 0,
                float(background.get("coverage_area", 0.0)),
            )
            if best_score is None or score > best_score:
                best_background = background
                best_score = score
    return best_background


def _load_geotiff_background(path: Path, extent: tuple[float, float, float, float]) -> dict[str, Any] | None:
    lon_min, lon_max, lat_min, lat_max = extent
    requested_extent = (float(lon_min), float(lon_max), float(lat_min), float(lat_max))
    with rasterio.open(path) as src:
        if src.crs is None:
            return None
        try:
            source_bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        except Exception:
            return None
        source_extent = (
            float(source_bounds[0]),
            float(source_bounds[2]),
            float(source_bounds[1]),
            float(source_bounds[3]),
        )
        overlap = _extent_overlap(requested_extent, source_extent)
        if overlap is None:
            return None
        try:
            source_overlap = transform_bounds(
                "EPSG:4326",
                src.crs,
                overlap[0],
                overlap[2],
                overlap[1],
                overlap[3],
                densify_pts=21,
            )
        except Exception:
            return None
        window = window_from_bounds(*source_overlap, transform=src.transform).intersection(Window(0, 0, src.width, src.height))
        data = src.read(1, window=window, boundless=False, masked=True)
        source_transform = src.window_transform(window)

    if data.size == 0:
        return None
    source_grid = np.asarray(data.filled(np.nan), dtype=np.float32)
    if not np.isfinite(source_grid).any():
        return None

    target_height, target_width = source_grid.shape
    step = max(1, int(max(target_height, target_width) / 1200))
    if step > 1:
        target_height = max(1, int(math.ceil(target_height / step)))
        target_width = max(1, int(math.ceil(target_width / step)))
    target_transform = transform_from_bounds(overlap[0], overlap[2], overlap[1], overlap[3], target_width, target_height)
    target_grid = np.full((target_height, target_width), np.nan, dtype=np.float32)
    reproject(
        source=source_grid,
        destination=target_grid,
        src_transform=source_transform,
        src_crs=src.crs,
        src_nodata=np.nan,
        dst_transform=target_transform,
        dst_crs="EPSG:4326",
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    if not np.isfinite(target_grid).any():
        return None
    return {
        "grid": np.asarray(target_grid, dtype=float),
        "extent": overlap,
        "origin": "upper",
        "covers_full_extent": _extent_covers(source_extent, requested_extent),
        "coverage_area": _extent_area(overlap),
    }


def _load_netcdf_background(path: Path, extent: tuple[float, float, float, float]) -> dict[str, Any] | None:
    with netcdf_file(path, "r", mmap=False) as dataset:
        lon_key = "lon" if "lon" in dataset.variables else "x"
        lat_key = "lat" if "lat" in dataset.variables else "y"
        lon = np.array(dataset.variables[lon_key].data, dtype=float)
        lat = np.array(dataset.variables[lat_key].data, dtype=float)
        grid = np.array(dataset.variables["z"].data, dtype=float)

    if lon[0] > lon[-1]:
        lon = lon[::-1]
        grid = grid[:, ::-1]
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        grid = grid[::-1, :]

    requested_extent = tuple(float(value) for value in extent)
    source_extent = (float(lon[0]), float(lon[-1]), float(lat[0]), float(lat[-1]))
    overlap = _extent_overlap(requested_extent, source_extent)
    if overlap is None:
        return None

    lon_step = abs(float(lon[1] - lon[0])) if lon.size > 1 else 0.0
    lat_step = abs(float(lat[1] - lat[0])) if lat.size > 1 else 0.0
    lon_mask = (lon >= overlap[0] - lon_step) & (lon <= overlap[1] + lon_step)
    lat_mask = (lat >= overlap[2] - lat_step) & (lat <= overlap[3] + lat_step)
    if lon_mask.any():
        lon = lon[lon_mask]
        grid = grid[:, lon_mask]
    if lat_mask.any():
        lat = lat[lat_mask]
        grid = grid[lat_mask, :]

    if grid.size == 0 or not np.isfinite(grid).any():
        return None
    step = max(1, int(max(grid.shape) / 1200))
    lon = lon[::step]
    lat = lat[::step]
    grid = grid[::step, ::step]
    return {
        "grid": grid,
        "extent": (float(lon[0]), float(lon[-1]), float(lat[0]), float(lat[-1])),
        "origin": "lower",
        "covers_full_extent": _extent_covers(source_extent, requested_extent),
        "coverage_area": _extent_area(overlap),
    }


def _draw_background(ax: plt.Axes, background: dict[str, Any] | None) -> None:
    if background is None:
        return
    grid = np.asarray(background["grid"], dtype=float)
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return
    vmin = float(np.nanpercentile(finite, 5))
    vmax = float(np.nanpercentile(finite, 95))
    ax.imshow(
        grid,
        extent=background["extent"],
        origin=background.get("origin", "lower"),
        cmap="Greys",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        alpha=0.95,
        zorder=0,
    )


def _component_station_vectors(spatial_mm: np.ndarray, components: Sequence[str], n_stations: int) -> dict[str, np.ndarray]:
    vectors = {code: np.zeros(n_stations, dtype=float) for code in ("e", "n", "u")}
    component_size = len(components)
    for offset, component in enumerate(components):
        vectors[component] = spatial_mm[offset::component_size]
    return vectors


def _component_figure(
    plot_data: dict[str, Any],
    component_index: int,
    normalization: str,
    background: dict[str, Any] | None,
    label_stations: bool,
) -> plt.Figure:
    u = plot_data["U"][:, component_index].copy()
    v = plot_data["V"][:, component_index].copy()
    singular_value = float(plot_data["S"][component_index])
    sign = _flip_sign(v)
    u *= sign
    v *= sign

    factor = _normalization_factor(v, normalization)
    plotted_v = v * factor
    spatial_mm = u * singular_value / factor
    components = tuple(plot_data["components"])
    vectors = _component_station_vectors(spatial_mm, components, int(plot_data["n_stations"]))
    east_mm = vectors["e"]
    north_mm = vectors["n"]
    up_mm = vectors["u"]

    station_lon = plot_data["station_lon"]
    station_lat = plot_data["station_lat"]
    station_names = plot_data["station_names"]
    n_stations = int(plot_data["n_stations"])
    timeline = plot_data["timeline"]
    component_prefix = plot_data["component_prefix"]

    extent = _plot_extent(station_lon, station_lat, background)
    lon_min, lon_max, lat_min, lat_max = extent
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    mean_lat = float(np.nanmean(station_lat))

    horizontal_mm = np.hypot(east_mm, north_mm)
    max_horizontal_mm = float(np.nanmax(horizontal_mm)) if horizontal_mm.size else 0.0
    map_span_deg = max(lon_span, lat_span)
    deg_per_mm = 0.0 if max_horizontal_mm == 0 else 0.16 * map_span_deg / max_horizontal_mm
    reference_mm = _nice_reference_value(max_horizontal_mm / 3.0 if max_horizontal_mm > 0 else 1.0)
    has_horizontal = "e" in components or "n" in components
    has_vertical = "u" in components

    figure = plt.figure(figsize=(8.27, 11.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=[1.0, 1.6])
    ax_time = figure.add_subplot(grid[0, 0])
    ax_map = figure.add_subplot(grid[1, 0])
    ax_map.set_facecolor("#ececec")

    ax_time.plot(timeline, plotted_v, color="black", linewidth=1.0)
    ax_time.axhline(0.0, color="0.6", linewidth=0.8, linestyle="--")
    ax_time.set_xlim(float(np.nanmin(timeline)), float(np.nanmax(timeline)))
    ax_time.set_xlabel("Time (yr)")
    if normalization == "peak-to-peak":
        ax_time.set_ylabel("V (peak-to-peak = 1)")
    elif normalization == "unit-max":
        ax_time.set_ylabel("V (max|.| = 1)")
    else:
        ax_time.set_ylabel("V")
    ax_time.set_title(f"{component_prefix}{component_index + 1} temporal evolution", loc="left", fontsize=12, pad=8)
    if normalization != "none":
        norm_label = "peak-to-peak" if normalization == "peak-to-peak" else "maxabs"
        ax_time.text(
            0.01,
            0.97,
            f"norm = {norm_label}, V_plot = {factor:.3g} * V, map = U*S / {factor:.3g}",
            transform=ax_time.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8", "pad": 3},
        )

    _draw_background(ax_map, background)
    if has_vertical:
        color_values = up_mm
        color_label = "Vertical displacement (mm)"
        cmap = "seismic"
        color_limit = float(np.nanmax(np.abs(color_values))) if color_values.size else 1.0
        if not np.isfinite(color_limit) or color_limit == 0.0:
            color_limit = 1.0
        scatter_kwargs = {"vmin": -color_limit, "vmax": color_limit}
    else:
        color_values = horizontal_mm
        color_label = "Horizontal amplitude (mm)"
        cmap = "viridis"
        color_limit = float(np.nanmax(color_values)) if color_values.size else 1.0
        if not np.isfinite(color_limit) or color_limit == 0.0:
            color_limit = 1.0
        scatter_kwargs = {"vmin": 0.0, "vmax": color_limit}
    scatter = ax_map.scatter(
        station_lon,
        station_lat,
        c=color_values,
        cmap=cmap,
        s=52,
        edgecolors="black",
        linewidths=0.4,
        zorder=3,
        **scatter_kwargs,
    )

    if has_horizontal and deg_per_mm > 0.0:
        ax_map.quiver(
            station_lon,
            station_lat,
            east_mm * deg_per_mm,
            north_mm * deg_per_mm,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#22dd22",
            width=0.0045,
            headwidth=4.0,
            headlength=5.0,
            headaxislength=4.5,
            zorder=4,
        )
        ref_x = lon_min + 0.08 * lon_span
        ref_y = lat_min + 0.08 * lat_span
        ax_map.quiver(
            ref_x,
            ref_y,
            reference_mm * deg_per_mm,
            0,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#22dd22",
            width=0.005,
            headwidth=4.0,
            headlength=5.0,
            headaxislength=4.5,
            zorder=5,
        )
        ax_map.text(
            ref_x,
            ref_y - 0.04 * lat_span,
            f"{reference_mm:g} mm",
            ha="left",
            va="top",
            fontsize=9,
            color="#117711",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1},
            zorder=6,
        )

    if label_stations:
        for lon_value, lat_value, name in zip(station_lon, station_lat, station_names):
            ax_map.text(lon_value, lat_value, f" {name}", fontsize=7, ha="left", va="center", zorder=6)

    ax_map.set_xlim(lon_min, lon_max)
    ax_map.set_ylim(lat_min, lat_max)
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    title_suffix = f"{n_stations} stations"
    if background is None:
        title_suffix += ", no DEM"
    elif not bool(background.get("covers_full_extent", True)):
        title_suffix += ", partial DEM"
    ax_map.set_title(f"{component_prefix}{component_index + 1} spatial pattern ({title_suffix})", loc="left", fontsize=12, pad=8)
    ax_map.grid(color="0.8", linewidth=0.6, linestyle=":")
    ax_map.set_aspect(1.0 / math.cos(math.radians(mean_lat)))

    colorbar = figure.colorbar(scatter, ax=ax_map, orientation="horizontal", fraction=0.055, pad=0.08)
    colorbar.set_label(color_label)

    figure.suptitle(f"{component_prefix}{component_index + 1}  |  {n_stations} stations used", fontsize=16, y=0.995)
    return figure


def create_ica_component_plots(
    results_or_file: dict[str, Any] | str | Path,
    output_dir: str | Path,
    repo_root: str | Path | None = None,
    components: Sequence[int] | None = None,
    normalization: str = "peak-to-peak",
    background_grid: str | Path | None = "auto",
    label_stations: bool = False,
    dpi: int = 200,
    prefix: str | None = None,
    decomposition: str = "ICA",
) -> list[Path]:
    if normalization not in {"peak-to-peak", "unit-max", "none"}:
        raise ValueError("normalization must be one of 'peak-to-peak', 'unit-max', or 'none'")

    decomposition = _normalize_decomposition_name(decomposition)
    results, _ = _load_results(results_or_file)
    resolved_repo_root = _infer_repo_root(results, repo_root)
    plot_data = _extract_plot_data(results, decomposition=decomposition)
    selected = _safe_component_indices(components, plot_data["n_components"])
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    output_prefix = prefix or _component_prefix(decomposition)

    background_request_extent = _padded_station_extent(plot_data["station_lon"], plot_data["station_lat"])
    background = _load_background(resolved_repo_root, background_grid, background_request_extent)

    generated: list[Path] = []
    multipage_pdf = output_path / f"{output_prefix}_components.pdf"
    with PdfPages(multipage_pdf) as pdf:
        for component_index in selected:
            figure = _component_figure(plot_data, component_index, normalization, background, label_stations)
            pdf.savefig(figure)
            pdf_file = output_path / f"{output_prefix}{component_index + 1}.pdf"
            png_file = output_path / f"{output_prefix}{component_index + 1}.png"
            figure.savefig(pdf_file)
            figure.savefig(png_file, dpi=dpi)
            generated.extend([pdf_file, png_file])
            plt.close(figure)
    generated.insert(0, multipage_pdf)
    return generated
