from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .common import GPS_COMPONENT_LABELS, infer_gps_layout, load_results_file


COMPONENT_SUFFIXES = ("E", "N", "U")
COMPONENT_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2")


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


def _load_results(results_or_file: dict[str, Any] | str | Path) -> tuple[dict[str, Any], str]:
    if isinstance(results_or_file, dict):
        return results_or_file, "results"
    path = Path(results_or_file).resolve()
    return load_results_file(path), path.stem


def _station_names(results: dict[str, Any], n_stations: int, component_size: int) -> list[str]:
    if "STATIONS_name" in results:
        names = [_stringify(item)[:4] for item in np.atleast_1d(results["STATIONS_name"])[:n_stations]]
        if len(names) == n_stations:
            return names
    xd_names = _field(results.get("Xd"), "name")
    if xd_names is None:
        return [f"S{i + 1:03d}" for i in range(n_stations)]
    return [_stringify(np.atleast_1d(xd_names)[idx * component_size])[:4] for idx in range(n_stations)]


def _diag_values(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.array([float(array)])
    if array.ndim == 1:
        return array
    return np.diag(array)


def _extract_station_fit_data(results: dict[str, Any], decomposition: str) -> dict[str, Any]:
    decomposition = _normalize_decomposition_name(decomposition)
    xd = _field(results, "Xd")
    decomp = _field(results, decomposition)
    if xd is None or decomp is None:
        raise ValueError(f"The provided results do not contain both Xd and {decomposition} structures.")

    observed_centered = np.asarray(_field(xd, "ts"), dtype=float)
    observed_var = np.asarray(_field(xd, "var_ts"), dtype=float)
    centering_offsets = np.asarray(_field(xd, "centering_offsets"), dtype=float).reshape(-1)
    timeline = np.asarray(_field(xd, "timeline"), dtype=float).reshape(-1)
    llh = np.asarray(_field(xd, "llh"), dtype=float)
    model_centered = np.asarray(_field(decomp, "ts"), dtype=float)
    u = np.asarray(_field(decomp, "U"), dtype=float)
    s_diag = _diag_values(_field(decomp, "S"))
    v = np.asarray(_field(decomp, "V"), dtype=float)

    if observed_centered.ndim != 2:
        raise ValueError("Xd.ts is not a 2-D matrix.")
    if observed_centered.shape != observed_var.shape or observed_centered.shape != model_centered.shape:
        raise ValueError(f"Xd and {decomposition} matrices do not have compatible shapes.")
    _, components = infer_gps_layout(_field(decomp, "type", _field(xd, "type")), observed_centered.shape[0])
    component_size = len(components)
    if u.shape[0] != observed_centered.shape[0] or v.shape[0] != observed_centered.shape[1]:
        raise ValueError(f"{decomposition} U/V dimensions are not consistent with Xd.ts.")
    if u.shape[1] != s_diag.size or v.shape[1] != s_diag.size:
        raise ValueError(f"{decomposition} U/S/V dimensions are not consistent.")

    observed_raw = observed_centered + centering_offsets[:, None]
    observed_raw[~np.isfinite(observed_var)] = np.nan

    component_centered = np.empty((s_diag.size, observed_centered.shape[0], observed_centered.shape[1]), dtype=float)
    for idx, singular_value in enumerate(s_diag):
        component_centered[idx] = np.outer(u[:, idx] * singular_value, v[:, idx])

    model_mean = model_centered - np.sum(component_centered, axis=0)
    model_raw = model_centered + centering_offsets[:, None]
    baseline_raw = model_mean + centering_offsets[:, None]

    n_stations = observed_centered.shape[0] // component_size
    station_lon = llh[0::component_size, 0] if llh.ndim == 2 and llh.shape[1] >= 2 else np.full(n_stations, np.nan)
    station_lat = llh[0::component_size, 1] if llh.ndim == 2 and llh.shape[1] >= 2 else np.full(n_stations, np.nan)

    return {
        "decomposition": decomposition,
        "component_prefix": _component_prefix(decomposition),
        "components": components,
        "timeline": timeline,
        "observed_raw": observed_raw,
        "model_raw": model_raw,
        "component_centered": component_centered,
        "baseline_raw": baseline_raw,
        "n_stations": n_stations,
        "n_components": s_diag.size,
        "station_names": _station_names(results, n_stations, component_size),
        "station_lon": station_lon,
        "station_lat": station_lat,
    }


def _select_station_indices(
    station_names: Sequence[str],
    stations: Sequence[str] | None,
) -> list[int]:
    if not stations:
        return list(range(len(station_names)))

    normalized = {name.upper(): idx for idx, name in enumerate(station_names)}
    selected: list[int] = []
    for item in stations:
        key = item.upper()
        if key not in normalized:
            raise ValueError(f"Unknown station code {item!r}. Available stations: {', '.join(station_names)}")
        selected.append(normalized[key])
    return selected


def _station_figure(
    data: dict[str, Any],
    station_index: int,
    show_components: bool,
) -> plt.Figure:
    station_name = data["station_names"][station_index]
    timeline = data["timeline"]
    component_codes = tuple(data["components"])
    component_size = len(component_codes)
    rows = slice(component_size * station_index, component_size * station_index + component_size)
    observed = data["observed_raw"][rows, :]
    modeled = data["model_raw"][rows, :]
    baseline = data["baseline_raw"][rows, :]
    component_series = data["component_centered"][:, rows, :]
    lon = float(data["station_lon"][station_index])
    lat = float(data["station_lat"][station_index])
    decomposition = data["decomposition"]
    component_prefix = data["component_prefix"]

    figure, axes = plt.subplots(
        component_size,
        1,
        figsize=(10.5, 2.6 * component_size + 0.8),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    if np.isfinite(lon) and np.isfinite(lat):
        title = f"{station_name}  |  {decomposition}  |  lon {lon:.4f}, lat {lat:.4f}"
    else:
        title = f"{station_name}  |  {decomposition}"

    for component_index, ax in enumerate(axes):
        component_code = component_codes[component_index]
        observed_series = observed[component_index]
        modeled_series = modeled[component_index]
        baseline_series = baseline[component_index]

        ax.plot(timeline, observed_series, color="black", linewidth=1.0, label="Observed")
        ax.plot(timeline, modeled_series, color="#c62828", linewidth=1.4, label="Modeled sum")

        if show_components:
            for ic_index in range(component_series.shape[0]):
                color = COMPONENT_COLORS[ic_index % len(COMPONENT_COLORS)]
                ax.plot(
                    timeline,
                    component_series[ic_index, component_index, :],
                    color=color,
                    linewidth=0.9,
                    linestyle="--",
                    alpha=0.9,
                    label=f"{component_prefix}{ic_index + 1}",
                )

        baseline_valid = baseline_series[np.isfinite(baseline_series)]
        if baseline_valid.size and np.nanmax(np.abs(baseline_valid - baseline_valid[0])) < 1e-9:
            ax.axhline(
                float(np.nanmean(baseline_valid)),
                color="0.7",
                linewidth=0.8,
                linestyle=":",
                label="Offset + mean" if component_index == 0 else None,
            )

        residual = observed_series - modeled_series
        valid = np.isfinite(residual)
        if np.any(valid):
            rms = float(np.sqrt(np.mean(residual[valid] ** 2)))
            ax.text(
                0.99,
                0.96,
                f"RMS misfit = {rms:.3f} mm",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8", "pad": 2},
            )

        ax.set_ylabel(f"{GPS_COMPONENT_LABELS[component_code]} (mm)")
        ax.grid(color="0.85", linewidth=0.6, linestyle=":")
        ax.axhline(0.0, color="0.85", linewidth=0.8)

    axes[-1].set_xlabel("Time (yr)")
    handles, labels = axes[0].get_legend_handles_labels()
    unique_labels: list[str] = []
    unique_handles: list[Any] = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels and label:
            unique_labels.append(label)
            unique_handles.append(handle)
    axes[0].set_title(title, fontsize=14, pad=10)
    axes[0].legend(unique_handles, unique_labels, loc="upper left", ncol=min(4, len(unique_labels)), fontsize=8)
    return figure


def create_station_fit_plots(
    results_or_file: dict[str, Any] | str | Path,
    output_dir: str | Path,
    stations: Sequence[str] | None = None,
    show_components: bool = True,
    dpi: int = 180,
    prefix: str = "station_fit",
    decomposition: str = "ICA",
) -> list[Path]:
    results, _ = _load_results(results_or_file)
    decomposition = _normalize_decomposition_name(decomposition)
    data = _extract_station_fit_data(results, decomposition=decomposition)
    selected = _select_station_indices(data["station_names"], stations)

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    multipage_prefix = prefix if decomposition == "ICA" else f"{prefix}_{decomposition.lower()}"
    multipage_pdf = output_path / f"{multipage_prefix}_all.pdf"
    with PdfPages(multipage_pdf) as pdf:
        for station_index in selected:
            figure = _station_figure(data, station_index, show_components)
            station_name = data["station_names"][station_index]
            stem = station_name if decomposition == "ICA" else f"{station_name}_{decomposition.lower()}"
            pdf.savefig(figure)
            pdf_file = output_path / f"{stem}.pdf"
            png_file = output_path / f"{stem}.png"
            figure.savefig(pdf_file)
            figure.savefig(png_file, dpi=dpi)
            generated.extend([pdf_file, png_file])
            plt.close(figure)

    generated.insert(0, multipage_pdf)
    return generated
