from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import csv

from datetime import datetime

import math
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .common import (
    Config,
    parse_data_input,
    parse_station_list,
    load_tseri_series,
    _flatten_text_values,
    select_stations,
    apply_filters
)

from .plots import (
    _normalize_decomposition_name,
    _component_prefix,
    _field,
    _stringify,
    _load_results,
    _safe_component_indices,
    _as_component_matrix,
    _as_diagonal,
    _flip_sign,
    _normalization_factor,
    _plot_extent,
    _infer_repo_root,
    _extract_station_names,
    _draw_background,
    _load_background
)

SUPPORTED_DATASET_COMPONENTS: dict[str, tuple[str, ...]] = {
    "GPS1": ("u",),
    "GPS2": ("e", "n"),
    "GPS3": ("e", "n", "u"),
    "INSARLOS": ("los",),
}

POINTSIZE = 1
VARIANCES = 1

def dataset_components_for_type(value: Any) -> tuple[str, ...]:
    return SUPPORTED_DATASET_COMPONENTS[normalize_dataset_type(value)]

def normalize_dataset_type(value: Any) -> str:
    normalized = _flatten_text_values(value)
    if not normalized:
        raise ValueError("Dataset type is empty.")
    dataset_type = normalized[0].strip().upper()
    if dataset_type not in SUPPORTED_DATASET_COMPONENTS:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    return dataset_type

def load_case_dataset(cfg: Config) -> tuple[list[dict[str, Any]], dict[str, Any]]:

    datasets = [entry for entry in parse_data_input(cfg.data_input_file, cfg.repo_root) if entry["instruction"] == "decomp"]

    if not datasets:
        raise RuntimeError(f"No dataset marked as 'decomp' in {cfg.data_input_file}")

    stations: list[dict[str, Any]] = []

    for dataset in datasets:

        dataset_type = normalize_dataset_type(dataset['type'])
        print(f"Loading dataset type: {dataset_type}")

        if dataset_type.startswith('GPS'):
            for listed in parse_station_list(dataset["list_path"], cfg.repo_root):
                series = load_tseri_series(listed["file"], dataset["unit_input"], cfg.unit_output)
                station = dict(listed)
                station["type"] = dataset_type
                station["timeline"] = series["timeline"]
                station["pos"] = series["pos"]
                stations.append(station)

        elif dataset_type == 'INSARLOS':
            insar_points = read_insar_matrix(
                    dataset["list_path"],
                    time_prefix = "D",
                    x_col  = "lon",
                    y_col = "lat",
                    z_col = "height",
                    sep = ','
                    )
            stations.extend(insar_points)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")


    stations = select_stations(
        stations,
        origin_lon=cfg.select_origin_lon,
        origin_lat=cfg.select_origin_lat,
        radius_km=cfg.select_radius_km,
    )


    xd = build_xcalc(
        stations,
        first_epoch=cfg.first_epoch,
        last_epoch=cfg.last_epoch,
        skip_epochs=cfg.skip_epochs,
        threshold_ts_missingdata=cfg.threshold_ts_missingdata,
        threshold_epochs_missingdata=cfg.threshold_epochs_missingdata,
    )

    return stations, xd

def date_to_decimal_year(date: datetime) -> float:
    year_start = datetime(date.year, 1, 1)
    next_year_start = datetime(date.year + 1, 1, 1)
    return date.year + (
        (date - year_start).total_seconds()
        / (next_year_start - year_start).total_seconds()
    )

def parse_time_column(
    column: str,
    time_prefix: str = "D",
    time_format: str = "decimal",
) -> float:
    value = column.removeprefix(time_prefix).strip()

    if time_format == "decimal":
        return float(value)

    date = datetime.strptime(value, time_format)
    return date_to_decimal_year(date)

def read_insar_matrix(input_file,indexcol =None,sep =',',x_col ='lon', y_col ='lat', z_col =None, time_prefix ='D',time_format="decimal"):
    stations: list[dict[str, Any]] = []
    with open(input_file, "r") as f:
        data = csv.reader(f, delimiter=sep)
        for k,row in enumerate(data):
            if not k:
                headers=row.copy()
                if indexcol is None:
                    indexcol=headers[0]
                name_index = headers.index(indexcol)
                x_index = headers.index(x_col)
                y_index = headers.index(y_col)
                z_index = ''
                for candidate in (z_col, "height"):
                    if candidate in headers:
                        z_index = headers.index(candidate)
                        break
                time_cols = [col for col in headers if col.startswith(time_prefix)]
                times_index = [headers.index(ct) for ct in time_cols]

                timeline = np.asarray([parse_time_column(col, time_prefix=time_prefix, time_format=time_format) for col in time_cols],dtype=float)
            else:
                series=[]
                for t in times_index:
                    if not len(row[t]):
                        row[t] = 'nan'
                    series.append(row[t])

                values = np.asarray(series, dtype=float)
                var_values = np.full(values.shape, VARIANCES, dtype=float)
                var_values[~np.isfinite(values)] = np.inf

                pos = {
                    "los": values,
                    "var_los": var_values,
                }


                station = {
                    "name": str(row[name_index]),
                    "file": str(input_file),
                    "lon": float(row[x_index]),
                    "lat": float(row[y_index]),
                    "height": float(row[z_index]) if z_index else 0.0,
                    "type": "INSARLOS",
                    "timeline": timeline,
                    "pos": pos,
                }

                stations.append(station)
    return(stations)

def build_xcalc(
    stations: list[dict[str, Any]],
    first_epoch: float,
    last_epoch: float,
    skip_epochs: tuple[float, ...],
    threshold_ts_missingdata: float,
    threshold_epochs_missingdata: float,
) -> dict[str, Any]:
    if not stations:
        return empty_xcalc()

    all_timeline = np.concatenate([station["timeline"] for station in stations])
    timeline = np.unique(np.sort(all_timeline))
    timeline = timeline[(timeline >= first_epoch) & (timeline <= last_epoch)]

    n_series = sum(len(dataset_components_for_type(station["type"])) for station in stations)
    nt = timeline.size
    ts = np.full((n_series, nt), np.nan, dtype=float)
    var_ts = np.full((n_series, nt), np.inf, dtype=float)
    llh = np.full((n_series, 3), np.nan, dtype=float)
    names: list[str] = []
    types: list[str] = []

    row = 0
    k,n = 0,len(stations)
    for station in stations:
        # print('Filtering station - %d / %d'%(k,n))
        k=k+1
        index_map = {float(epoch): idx for idx, epoch in enumerate(timeline)}
        llh_row = np.array([station["lon"], station["lat"], station["height"]], dtype=float)
        station_type = normalize_dataset_type(station["type"])
        for component in dataset_components_for_type(station_type):               #modified function name (dataset_components_for_type)
            names.append(f"{station['name']}{component}")
            types.append(f"{station_type}{component}")
            llh[row, :] = llh_row
            for src_idx, epoch in enumerate(station["timeline"]):
                dst_idx = index_map.get(float(epoch))
                if dst_idx is None:
                    continue
                ts[row, dst_idx] = station["pos"][component][src_idx]
                var_ts[row, dst_idx] = station["pos"][f"var_{component}"][src_idx]
            row += 1

    xd = {
        "name": names,
        "type": types,
        "llh": llh,
        "timeline": timeline.copy(),
        "ts": ts,
        "var_ts": var_ts,
    }
    prefilter_xd = {
        "name": list(names),
        "type": list(types),
        "llh": llh.copy(),
        "timeline": timeline.copy(),
        "ts": ts.copy(),
        "var_ts": var_ts.copy(),
    }
    filtered_xd = apply_filters(xd, skip_epochs, threshold_ts_missingdata, threshold_epochs_missingdata)
    filtered_xd["filter_debug"] = {"prefilter": prefilter_xd}
    return filtered_xd

def empty_xcalc() -> dict[str, Any]:
    return {
        "name": [],
        "type": [],
        "llh": np.empty((0, 3), dtype=float),
        "timeline": np.empty((0,), dtype=float),
        "ts": np.empty((0, 0), dtype=float),
        "var_ts": np.empty((0, 0), dtype=float),
    }


def _extract_plot_data_insar(results: dict[str, Any], decomposition: str) -> dict[str, Any]:
    decomposition = _normalize_decomposition_name(decomposition)
    prefix = _component_prefix(decomposition)
    if f"{decomposition}_U" in results:
        u = _as_component_matrix(results[f"{decomposition}_U"])
        v = _as_component_matrix(results[f"{decomposition}_V"])
        diag_s = _as_diagonal(results[f"{decomposition}_S"])
        llh = np.asarray(results.get(f"{decomposition}_llh", results["Xd_llh"]), dtype=float)
        timeline = np.asarray(results.get(f"{decomposition}_timeline", results["Xd_timeline"]), dtype=float).reshape(-1)
        _, components = infer_dataset_layout(results.get(f"{decomposition}_type", results.get("Xd_type")), u.shape[0])
        if llh.ndim != 2 or llh.shape[1] < 2:
            raise ValueError(f"{decomposition}_llh/Xd_llh is not in the expected Mx3 format.")

        sort_index = np.argsort(diag_s)[::-1]
        u = u[:, sort_index]
        v = v[:, sort_index]
        diag_s = diag_s[sort_index]

        component_size = len(components)
        n_stations = u.shape[0] // component_size
        if "STATIONS_name" in results:
            station_names = [_stringify(item) for item in np.atleast_1d(results["STATIONS_name"])[:n_stations]]
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
    _, components = infer_dataset_layout(_field(decomp, "type", _field(xd, "type")), u.shape[0])

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

def create_ica_component_plots_insar(
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
    plot_data = _extract_plot_data_insar(results, decomposition=decomposition)
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
            figure = _component_figure_insarlos(plot_data, component_index, normalization, background, label_stations)
            pdf.savefig(figure)
            pdf_file = output_path / f"{output_prefix}{component_index + 1}.pdf"
            png_file = output_path / f"{output_prefix}{component_index + 1}.png"
            figure.savefig(pdf_file)
            figure.savefig(png_file, dpi=dpi)
            generated.extend([pdf_file, png_file])
            plt.close(figure)
    generated.insert(0, multipage_pdf)
    return generated


def _component_figure_insarlos(
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

    color_values = spatial_mm
    color_label = "LOS displacement (mm)"
    cmap = "seismic"
    color_limit = float(np.nanmax(np.abs(color_values))) if color_values.size else 1.0
    if not np.isfinite(color_limit) or color_limit == 0.0:
        color_limit = 1.0
    scatter_kwargs = {"vmin": -color_limit, "vmax": color_limit}

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
    map_span_deg = max(lon_span, lat_span)
    point_size = POINTSIZE

    scatter = ax_map.scatter(
        station_lon,
        station_lat,
        c=color_values,
        cmap=cmap,
        s=point_size,
        edgecolors="none",
        linewidths=0.0,
        alpha=0.95,
        zorder=3,
        **scatter_kwargs,
    )

    if label_stations:
        for lon_value, lat_value, name in zip(station_lon, station_lat, station_names):
            ax_map.text(lon_value, lat_value, f" {name}", fontsize=5.5, ha="left", va="center", zorder=6)

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

def _padded_station_extent(lon: np.ndarray, lat: np.ndarray) -> tuple[float, float, float, float]:
    lon_min = float(np.nanmin(lon))
    lon_max = float(np.nanmax(lon))
    lat_min = float(np.nanmin(lat))
    lat_max = float(np.nanmax(lat))
    lon_span = max(lon_max - lon_min, 0.3)
    lat_span = max(lat_max - lat_min, 0.3)
    lon_pad = max(0.03 * lon_span, 0.02)                                                                    #modified padding
    lat_pad = max(0.03 * lat_span, 0.02)                                                                    #modified padding
    return (lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad)

def infer_dataset_layout(type_value: Any, n_rows: int | None = None) -> tuple[str, tuple[str, ...]]:
    for text in _flatten_text_values(type_value):
        candidate = text

        if candidate.startswith("INSARLOS") and len(candidate) >= 8 and candidate[:8] in SUPPORTED_DATASET_COMPONENTS:
            dataset_type = candidate[:8]
            components = SUPPORTED_DATASET_COMPONENTS[dataset_type]
            if n_rows is not None and n_rows < 1:
                raise ValueError(f"Series count {n_rows} is not compatible with dataset type INSARLOS.")
            return dataset_type, components


        if candidate.startswith("GPS") and len(candidate) >= 4 and candidate[:4] in SUPPORTED_DATASET_COMPONENTS:
            dataset_type = candidate[:4]
            components = SUPPORTED_DATASET_COMPONENTS[dataset_type]
            if n_rows is not None and n_rows % len(components) != 0:
                raise ValueError(
                    f"Series count {n_rows} is not compatible with dataset type {dataset_type} "
                    f"({len(components)} components per station)."
                )
            return dataset_type, components

    raise ValueError(f"Unsupported dataset type information: {type_value!r}")

def station_names_from_series(series_names: Any, type_value: Any) -> list[str]:
    names = _flatten_text_values(series_names)
    if not names:
        return []

    dataset_type, components = infer_dataset_layout(type_value, len(names))
    component_size = len(components)

    if dataset_type == "INSARLOS":
        return [name[:-3] for name in names[::component_size]]

    return [name[:4] for name in names[::component_size]]
