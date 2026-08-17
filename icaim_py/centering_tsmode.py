from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from .common import Config, default_case1_output_dir
from .plots import (
    _draw_background,
    _infer_repo_root,
    _load_background,
    _load_results,
    _padded_station_extent,
)



def center_basic(xd: dict[str, Any], cfg: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    from .legacy_algorithms import build_imposed_v, center_basic_legacy

    imposed_v, _ = build_imposed_v(xd["timeline"], cfg)
    return center_basic_legacy(xd, cfg.n_components, imposed_v)


def center_data(xd: dict[str, Any], cfg: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    from .legacy_algorithms import build_imposed_v, center_advanced_legacy, center_basic_legacy

    imposed_v, _ = build_imposed_v(xd["timeline"], cfg)
    if cfg.centering.type == "basic":
        return center_basic_legacy(xd, cfg.n_components, imposed_v)
    if cfg.centering.type == "advanced":
        return center_advanced_legacy(xd, cfg, imposed_v)
    if cfg.centering.type == "tsmode":
        return center_tsmode(xd, cfg)
    raise ValueError(f"Unsupported centering.type='{cfg.centering.type}'")




def weighted_column_means(data: np.ndarray, weight: np.ndarray) -> np.ndarray:
    means = np.zeros(data.shape[1], dtype=float)

    for idx in range(data.shape[1]):
        valid = (weight[:,idx] > 0.0) & np.isfinite(data[:,idx])
        if np.any(valid):
            means[idx] = float(np.sum(weight[valid,idx] * data[valid,idx]) / np.sum(weight[valid,idx]))
    return means


def compute_tsmode_centering_offsets(
    xd: dict[str, Any],
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    from .common import normalize_decomposition_mode
    from .legacy_algorithms import weighted_row_means
    from .insar import infer_dataset_layout

    mode = normalize_decomposition_mode(cfg.decomposition_mode)
    
    weights = (1.0 / xd["var_ts"]) ** 2
    weights[~np.isfinite(weights)] = 0.0

    _, components = infer_dataset_layout(xd["type"], xd["ts"].shape[0])
    component_size = len(components)

    offsets = np.zeros_like(xd["ts"])

    if mode == "t":
        for component_index in range(component_size):
            rows = slice(component_index, xd["ts"].shape[0], component_size)
            
            component_ts = xd["ts"][rows, :]
            component_weights = weights[rows, :]
            row_offsets = weighted_row_means(component_ts, component_weights)
            offsets[rows, :] = row_offsets[:, None]

        return offsets

    if mode == "s":
        for component_index in range(component_size):
            rows = slice(component_index, xd["ts"].shape[0], component_size)
            
            component_ts = xd["ts"][rows, :]
            component_weights = weights[rows, :]
            offsets[rows,:] = weighted_column_means(component_ts,component_weights)

        return offsets

    raise ValueError(f"Unsupported decomposition_mode='{cfg.decomposition_mode}'")


def center_tsmode(xd: dict[str, Any], cfg: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    from .common import normalize_decomposition_mode
    from .legacy_algorithms import fill_missing_rows

    mode = normalize_decomposition_mode(cfg.decomposition_mode)
    
    print(f"Centering with tsmode ({mode}-mode)")
    
    filled_ts = fill_missing_rows(xd["ts"], xd["var_ts"])
    
    offsets = compute_tsmode_centering_offsets(xd, cfg)

    if mode == "t":
        centered = dict(xd)
        centered["ts"] = filled_ts - offsets
        centered["centering_offsets"] = offsets
        return centered, None
    elif mode == "s":
        centered = dict(xd)
        centered["ts"] = filled_ts - offsets
        centered["centering_offsets"] = offsets
        return centered, None
    else:
        raise ValueError(f"Unsupported decomposition_mode='{cfg.decomposition_mode}'")


def plot_centering_grid(
    xd_precen,
    xd,
    component_size=3,
    station_indices=None,
    component_labels=None,
    figsize=None,
    show=True,
):
    ts_precen = np.asarray(xd_precen["ts"])
    ts_centered = np.asarray(xd["ts"])
    timeline = np.asarray(xd_precen["timeline"])

    n_series = ts_precen.shape[0]
    n_stations = n_series // component_size

    if station_indices is None:
        station_indices = range(n_stations)

    station_indices = list(station_indices)

    if component_labels is None:
        component_labels = [f"Comp {i + 1}" for i in range(component_size)]

    if figsize is None:
        figsize = (4.5 * component_size, 2.4 * len(station_indices))

    fig, ax = plt.subplots(
        len(station_indices),
        component_size,
        figsize=figsize,
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )

    offsets = np.asarray(xd["centering_offsets"])

    for i, station_index in enumerate(station_indices):
        for component_index in range(component_size):
            r = station_index * component_size + component_index
            axis = ax[i, component_index]

            axis.plot(
                timeline,
                ts_precen[r, :],
                color="black",
                linewidth=1.0,
                label="Orig",
            )

            if offsets.ndim == 1:
                axis.axhline(
                    offsets[r],
                    color="red",
                    linewidth=1.0,
                    label="Offset",
                )
            else:
                axis.plot(
                    timeline,
                    offsets[r, :],
                    color="red",
                    linewidth=1.0,
                    label="Offset",
                )

            axis.plot(
                timeline,
                ts_centered[r, :],
                color="tab:blue",
                linewidth=1.0,
                label="Centered",
            )

            if i == 0:
                axis.set_title(component_labels[component_index])

            if component_index == 0:
                name = str(np.asarray(xd_precen["name"])[r])
                axis.set_ylabel(name[:4])

            axis.grid(True, alpha=0.3)

    ax[-1, 0].set_xlabel("Time")
    for component_index in range(1, component_size):
        ax[-1, component_index].set_xlabel("Time")

    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    if show:
        plt.show()

    return fig, ax






def plot_stations(
    lon,
    lat,
    gps_data,
    labels=None,
    show_labels=True,
    ax=None,
    cmap="viridis",
    marker_size=70,
    label_size=8,
    title=None,
    scale=None,
    arrow_color='red',
    colorbar_label=None,
    quiver_label=None,
    background=None
):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if len(gps_data.keys()) == 3:
        e = np.asarray(gps_data['e'], dtype=float)
        n = np.asarray(gps_data['n'], dtype=float)
        u = np.asarray(gps_data['u'], dtype=float)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
        else:
            fig = ax.figure

        _draw_background(ax, background)

        sc = ax.scatter(
            lon,
            lat,
            c=u,
            s=marker_size,
            cmap=cmap,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

        q = ax.quiver(
            lon,
            lat,
            e,
            n,
            angles="xy",
            scale_units="xy",
            scale=scale,
            color=arrow_color,
            width=0.004,
            zorder=4,
        )


        if show_labels and labels is not None:
            for x, y, label in zip(lon, lat, labels):
                ax.text(
                    x,
                    y,
                    str(label),
                    fontsize=label_size,
                    ha="left",
                    va="bottom",
                    zorder=4,
                )

        if quiver_label is not None:
            ax.quiverkey(
                q,
                X=0.88,
                Y=1.03,
                U=quiver_label[0],
                label=quiver_label[1],
                labelpos="E",
                coordinates="axes",
            )


        cb = fig.colorbar(sc, ax=ax)
        if colorbar_label is not None:
            cb.set_label(colorbar_label)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        if title is not None:
            ax.set_title(title)

        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        return fig, ax, sc
    elif len(gps_data.keys())==2:
        e = np.asarray(gps_data['e'], dtype=float)
        n = np.asarray(gps_data['n'], dtype=float)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
        else:
            fig = ax.figure

        _draw_background(ax, background)
        
        sc = ax.scatter(
            lon,
            lat,
            c='black',
            s=marker_size,
            edgecolor="black",
            linewidth=0.5,
        )

        q = ax.quiver(
            lon,
            lat,
            e,
            n,
            angles="xy",
            scale_units="xy",
            scale=scale,
            color=arrow_color,
            width=0.004,
            zorder=4,
        )

        if show_labels and labels is not None:
            for x, y, label in zip(lon, lat, labels):
                ax.text(
                    x,
                    y,
                    str(label),
                    fontsize=label_size,
                    ha="left",
                    va="bottom",
                    zorder=4,
                )

        if quiver_label is not None:
            ax.quiverkey(
                q,
                X=0.88,
                Y=1.03,
                U=quiver_label[0],
                label=quiver_label[1],
                labelpos="E",
                coordinates="axes",
            )


        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        if title is not None:
            ax.set_title(title)

        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        return fig, ax, sc
    elif len(gps_data.keys())==1:
        u = np.asarray(gps_data['u'], dtype=float)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
        else:
            fig = ax.figure

        _draw_background(ax, background)
        
        sc = ax.scatter(
            lon,
            lat,
            c=u,
            s=marker_size,
            cmap=cmap,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

        if show_labels and labels is not None:
            for x, y, label in zip(lon, lat, labels):
                ax.text(
                    x,
                    y,
                    str(label),
                    fontsize=label_size,
                    ha="left",
                    va="bottom",
                    zorder=4,
                )

        cb = fig.colorbar(sc, ax=ax)
        if colorbar_label is not None:
            cb.set_label(colorbar_label)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        if title is not None:
            ax.set_title(title)

        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        return fig, ax, sc
    else:
        raise ValueError("gps_data must contain 1, 2, or 3 components")

def plot_centers_stmode(
    xd: dict[str, Any],
    xd_precen: dict[str, Any],
    cfg: Config,
    background=None,
    label_stations: bool = False,
    station_indices=None,
) -> tuple[plt.Figure, plt.Figure | None]:
    from .common import normalize_decomposition_mode
    from .insar import infer_dataset_layout

    mode = normalize_decomposition_mode(cfg.decomposition_mode)
    _, components = infer_dataset_layout(xd["type"], xd["ts"].shape[0])
    component_size = len(components)

    fig_t, _ = plot_centering_grid(
        xd_precen,
        xd,
        component_size=component_size,
        station_indices=station_indices,
        component_labels=components,
        show=False,
    )

    lon = xd_precen["llh"][0::component_size, 0]
    lat = xd_precen["llh"][0::component_size, 1]
    labels = [str(name)[:4] for name in xd_precen["name"][0::component_size]]
    gps_data = {cname : xd['centering_offsets'][i::component_size,0] for i,cname in enumerate(components)}

    if mode == 't':
        fig_s, _, _ = plot_stations(
            lon,
            lat,
            gps_data,
            labels=labels,
            show_labels=label_stations,
            scale=None,
            title="Centering offsets vectors",
            quiver_label=(1.0, "1 mm"),
            background=background,
        )
        return fig_t,fig_s
    return fig_t,None


def create_centers_stmode_plots(
    results_or_file: dict[str, Any] | str | Path,
    cfg: Config | None,
    output_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    background_grid: str | Path | None = "auto",
    label_stations: bool = False,
    station_indices=None,
    dpi: int = 200,
) -> list[Path]:
    from .common import normalize_decomposition_mode
    from .insar import infer_dataset_layout

    if cfg is None:
        raise ValueError("cfg is required to create centering plots.")

    results, _ = _load_results(results_or_file)

    if "Xd" not in results or "Xd_precen" not in results:
        raise ValueError("results must contain both 'Xd' and 'Xd_precen'.")

    if output_dir is None:
        output_path = default_case1_output_dir(cfg) / "plots"
    else:
        output_path = Path(output_dir).resolve()

    output_path.mkdir(parents=True, exist_ok=True)
    mode = normalize_decomposition_mode(cfg.decomposition_mode)
    _, components = infer_dataset_layout(results["Xd"]["type"], results["Xd"]["ts"].shape[0])
    component_size = len(components)
    lon = results["Xd_precen"]["llh"][0::component_size, 0]
    lat = results["Xd_precen"]["llh"][0::component_size, 1]
    
    background=None
    if mode == "t":
        background_request_extent = _padded_station_extent(lon, lat)
        background = _load_background(_infer_repo_root(results, repo_root), background_grid, background_request_extent)

    fig_t, fig_s = plot_centers_stmode(
        xd=results["Xd"],
        xd_precen=results["Xd_precen"],
        cfg=cfg,
        background=background,
        label_stations=label_stations,
        station_indices=station_indices,
    )

    generated: list[Path] = []

    pdf_t = output_path / "centers_t.pdf"
    png_t = output_path / "centers_t.png"
    fig_t.savefig(pdf_t)
    fig_t.savefig(png_t, dpi=dpi)
    generated.extend([pdf_t, png_t])
    plt.close(fig_t)

    if fig_s is not None:
        pdf_s = output_path / "centers_s.pdf"
        png_s = output_path / "centers_s.png"
        fig_s.savefig(pdf_s)
        fig_s.savefig(png_s, dpi=dpi)
        generated.extend([pdf_s, png_s])
        plt.close(fig_s)

    return generated




























def centering_offsets_matrix(
    offsets: Any,
    data_shape: tuple[int, int],
    cfg: Config | None = None,
    decomposition_mode: str | None = None,
    centering_type: str | None = None,
) -> np.ndarray:
    from .common import normalize_decomposition_mode

    n_series, n_epochs = data_shape
    values = np.asarray(offsets, dtype=float)

    if values.ndim == 2:
        if values.shape != data_shape:
            raise ValueError(f"centering_offsets has shape {values.shape}, expected {data_shape}.")
        return values

    values = values.reshape(-1)
    mode = normalize_decomposition_mode(
        decomposition_mode if decomposition_mode is not None else getattr(cfg, "decomposition_mode", "t")
    )
    ctype = centering_type if centering_type is not None else getattr(getattr(cfg, "centering", None), "type", None)

    if ctype == "tsmode" and mode == "s":
        expected = n_epochs
        if values.size != expected:
            raise ValueError(
                "centering.type='tsmode' with decomposition_mode='s' expects one centering offset per epoch: "
                f"got {values.size}, expected n_epochs={expected}."
            )
        return values[None, :]

    expected = n_series
    if values.size != expected:
        raise ValueError(
            f"centering.type={ctype!r} with decomposition_mode={mode!r} expects one centering offset per series: "
            f"got {values.size}, expected n_series={expected}."
        )
    return values[:, None]


def add_centering_offsets(
    centered_ts: np.ndarray,
    offsets: Any,
    cfg: Config | None = None,
    decomposition_mode: str | None = None,
    centering_type: str | None = None,
) -> np.ndarray:
    centered = np.asarray(centered_ts, dtype=float)
    if centered.ndim != 2:
        raise ValueError("centered_ts must be a 2-D matrix.")
    offset_matrix = centering_offsets_matrix(
        offsets,
        centered.shape,
        cfg=cfg,
        decomposition_mode=decomposition_mode,
        centering_type=centering_type,
    )
    return centered + offset_matrix


def subtract_centering_offsets(
    raw_ts: np.ndarray,
    offsets: Any,
    cfg: Config | None = None,
    decomposition_mode: str | None = None,
    centering_type: str | None = None,
) -> np.ndarray:
    raw = np.asarray(raw_ts, dtype=float)
    if raw.ndim != 2:
        raise ValueError("raw_ts must be a 2-D matrix.")
    offset_matrix = centering_offsets_matrix(
        offsets,
        raw.shape,
        cfg=cfg,
        decomposition_mode=decomposition_mode,
        centering_type=centering_type,
    )
    return raw - offset_matrix


def station_fit_raw_series_from_centering(
    xd: dict[str, Any],
    decomp: dict[str, Any],
    cfg: Config | None = None,
) -> dict[str, np.ndarray]:
    observed_centered = np.asarray(xd["ts"], dtype=float)
    model_centered = np.asarray(decomp["ts"], dtype=float)
    offsets = xd["centering_offsets"]
    decomposition_mode = decomp.get("decomposition_mode", getattr(cfg, "decomposition_mode", "t") if cfg is not None else "t")
    centering_type = getattr(getattr(cfg, "centering", None), "type", None)

    if observed_centered.shape != model_centered.shape:
        raise ValueError("Xd.ts and decomposition ts do not have compatible shapes.")

    return {
        "observed_raw": add_centering_offsets(
            observed_centered,
            offsets,
            cfg=cfg,
            decomposition_mode=decomposition_mode,
            centering_type=centering_type,
        ),
        "model_raw": add_centering_offsets(
            model_centered,
            offsets,
            cfg=cfg,
            decomposition_mode=decomposition_mode,
            centering_type=centering_type,
        ),
    }
