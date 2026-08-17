from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    Config,
    align_components,
    build_effective_config,
    build_summary,
    build_ica_init_parameters,
    build_pca,
    compute_quality_metrics,
    default_case1_output_dir,
    flat_quality_metrics,
    json_ready,
    reorder_decomp,
    save_json,
    save_results_mat,
    save_results_npz,
    validate_and_describe_config,
)

from .vbica import decompose_ica


from .centering_tsmode import center_data
from .insar import load_case_dataset, station_names_from_series

def run_decomposition(
    repo_root: str | Path,
    output_dir: str | Path | None = None,
    save_mat: bool = True,
    data_input_file: str | Path | None = None,
    config_file: str | Path | None = None,
    config_overrides: dict[str, Any] | None = None,
    cfg: Config | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if cfg is None:
        cfg = build_effective_config(
            repo_root,
            data_input_file=data_input_file,
            config_file=config_file,
            config_overrides=config_overrides,
        )
    elif any(value is not None for value in (data_input_file, config_file, config_overrides)):
        raise ValueError("Pass either cfg or config_file/config_overrides/data_input_file, not both.")

    config_notes = validate_and_describe_config(cfg)
    output_path = Path(output_dir) if output_dir is not None else default_case1_output_dir(cfg)
    output_path.mkdir(parents=True, exist_ok=True)

    xd_stations, xd_precen = load_case_dataset(cfg)

    xd, pca_4cen = center_data(xd_precen, cfg)

    pca = build_pca(xd, cfg, pca_4cen=pca_4cen)
    
    init_parameters = build_ica_init_parameters(cfg, pca)
    ica, ica_aux = decompose_ica(xd, pca, init_parameters)

    station_names = station_names_from_series(pca["name"], pca["type"])
    results = {
        "cfg": cfg,
        "config_notes": config_notes,
        "flags": json_ready(cfg.flags),
        "XD": xd_stations,
        "Xd_precen": xd_precen,
        "Xd": xd,
        "PCA_4cen": pca_4cen,
        "PCA": pca,
        "ICA": ica,
        "A_recon": ica_aux["A_recon"],
        "S_recon": ica_aux["S_recon"],
        "var_A_recon": ica_aux["var_A_recon"],
        "var_S_recon": ica_aux["var_S_recon"],
        "data_mask": ica_aux["data_mask"],
        "ind_missing_data": ica_aux["ind_missing_data"],
        "STATIONS_name": station_names,
        "M": [len(pca["name"])],
        "Nt": pca["V"].shape[0],
        "init_parameters": init_parameters,
        "run_metadata": json_ready(run_metadata) if run_metadata is not None else None,
        "metrics": {
            "chi2_PCA": ica_aux["chi2_PCA"],
            "chi2_ICA": ica_aux["chi2_ICA"],
            "variance_explained_PCA": ica_aux["variance_explained_PCA"],
            "variance_explained_ICA": ica_aux["variance_explained_ICA"],
            "ard": ica_aux["ard"],
        },
    }
    quality_metrics = compute_quality_metrics(results)
    results["quality_metrics"] = quality_metrics
    results["metrics"].update(flat_quality_metrics(quality_metrics))
    generated_files = {
        "mat": str(output_path / "all_python.mat"),
        "npz": str(output_path / "all_python.npz"),
        "summary_json": str(output_path / "summary.json"),
    }
    if save_mat:
        save_results_mat(output_path / "all_python.mat", results)
        save_results_npz(output_path / "all_python.npz", results)
        save_json(output_path / "summary.json", build_summary(results, generated_files))
    return results


def compare_with_reference(results: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    comparisons: dict[str, float] = {}

    mask = np.isfinite(results["Xd"]["ts"]) & np.isfinite(reference["Xd"]["ts"])
    comparisons["Xd_ts_diff"] = float(np.linalg.norm(results["Xd"]["ts"][mask] - reference["Xd"]["ts"][mask]))
    mask = np.isfinite(results["Xd"]["var_ts"]) & np.isfinite(reference["Xd"]["var_ts"])
    comparisons["Xd_var_diff"] = float(np.linalg.norm(results["Xd"]["var_ts"][mask] - reference["Xd"]["var_ts"][mask]))

    perm, signs = align_components(results["PCA"]["U"], results["PCA"]["V"], reference["PCA"]["U"], reference["PCA"]["V"])
    pca = reorder_decomp(results["PCA"], perm, signs, recompute_ts=True)
    comparisons["PCA_U_diff"] = float(np.linalg.norm(pca["U"] - reference["PCA"]["U"]))
    comparisons["PCA_V_diff"] = float(np.linalg.norm(pca["V"] - reference["PCA"]["V"]))
    comparisons["PCA_ts_diff"] = float(np.linalg.norm(pca["ts"] - reference["PCA"]["ts"]))

    perm, signs = align_components(results["ICA"]["U"], results["ICA"]["V"], reference["ICA"]["U"], reference["ICA"]["V"])
    ica = reorder_decomp(results["ICA"], perm, signs, recompute_ts=False)
    comparisons["ICA_U_diff"] = float(np.linalg.norm(ica["U"] - reference["ICA"]["U"]))
    comparisons["ICA_V_diff"] = float(np.linalg.norm(ica["V"] - reference["ICA"]["V"]))
    comparisons["ICA_ts_diff"] = float(np.linalg.norm(results["ICA"]["ts"] - reference["ICA"]["ts"]))

    comparisons["chi2_PCA_diff"] = abs(results["metrics"]["chi2_PCA"] - reference["metrics"]["chi2_PCA"])
    comparisons["chi2_ICA_diff"] = abs(results["metrics"]["chi2_ICA"] - reference["metrics"]["chi2_ICA"])
    comparisons["var_explained_PCA_diff"] = abs(
        results["metrics"]["variance_explained_PCA"] - reference["metrics"]["variance_explained_PCA"]
    )
    comparisons["var_explained_ICA_diff"] = abs(
        results["metrics"]["variance_explained_ICA"] - reference["metrics"]["variance_explained_ICA"]
    )
    return comparisons
