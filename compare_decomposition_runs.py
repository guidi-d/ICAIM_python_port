from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import f as f_distribution

os.environ.setdefault("MPLCONFIGDIR", "/tmp/icaim_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/icaim_cache")

from icaim_py.common import (
    align_components,
    compute_quality_metrics,
    find_repo_root,
    json_ready,
    load_results_file,
    reorder_decomp,
    save_json,
)


def resolve_result_files(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for raw_path in inputs:
        path = raw_path.resolve()
        if path.is_file():
            files.append(path)
            continue
        if path.is_dir():
            files.extend(sorted(path.rglob("all_python.npz")))
            continue
        raise FileNotFoundError(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _quality_entry(path: Path, results: dict[str, Any]) -> dict[str, Any]:
    quality = results.get("quality_metrics")
    if not isinstance(quality, dict):
        quality = compute_quality_metrics(results)

    resolved_config = results.get("resolved_config")
    if not isinstance(resolved_config, dict):
        resolved_config = results.get("config")
    label = path.parent.name if path.name == "all_python.npz" else path.stem

    entry = {
        "label": label,
        "path": str(path),
        "n_series": quality["data"]["n_series"],
        "n_epochs": quality["data"]["n_epochs"],
        "n_components": quality["data"]["n_components"],
        "decomposition_mode": quality["data"].get("decomposition_mode", "t"),
        "n_observations": quality["data"]["n_observations"],
        "missing_data_fraction": quality["data"]["missing_data_fraction"],
        "chi2_PCA": quality["PCA"]["chi2"],
        "reduced_chi2_PCA": quality["PCA"]["reduced_chi2"],
        "variance_explained_PCA": quality["PCA"]["variance_explained"],
        "chi2_ICA": quality["ICA"]["chi2"],
        "reduced_chi2_ICA": quality["ICA"]["reduced_chi2"],
        "variance_explained_ICA": quality["ICA"]["variance_explained"],
        "chi2_gain_ICA_vs_PCA_pct": quality["ICA"]["chi2_gain_vs_PCA_pct"],
        "ICA_energy": quality["ICA"]["energy"],
        "ICA_iterations": quality["ICA"]["iterations"],
        "ARD_ratio": quality["ICA"]["ard_ratio"],
        "ARD_too_many_components": quality["ICA"]["ard_suggests_too_many_components"],
        "PCA_n_parameters": quality["PCA"]["n_parameters"],
        "ICA_n_parameters": quality["ICA"]["n_parameters"],
        "PCA_fit_method": quality["PCA"]["fit_method"],
        "ICA_net_init": None,
        "centering_type": None,
        "centering_function": None,
        "config_signature": None,
    }
    if isinstance(resolved_config, dict):
        entry["centering_type"] = resolved_config.get("centering", {}).get("type")
        entry["centering_function"] = resolved_config.get("centering", {}).get("function_resolved") or resolved_config.get(
            "centering", {}
        ).get("function")
        entry["PCA_fit_method"] = entry["PCA_fit_method"] or resolved_config.get("decompositionPCA", {}).get("decomp_fcn_resolved")
        entry["ICA_net_init"] = resolved_config.get("decompositionICA", {}).get("net_init")
        entry["config_signature"] = config_signature(resolved_config)
    return entry


def config_signature(resolved_config: dict[str, Any]) -> str:
    cfg = deepcopy(json_ready(resolved_config))
    cfg.pop("n_components", None)
    if isinstance(cfg.get("decompositionICA"), dict):
        cfg["decompositionICA"].pop("states_resolved", None)
        cfg["decompositionICA"].pop("source_resolved", None)
        cfg["decompositionICA"].pop("mix_resolved", None)
        cfg["decompositionICA"].pop("noise_resolved", None)
    return json.dumps(cfg, sort_keys=True)


def compare_pairwise(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    results: dict[str, float] = {}

    mask = np.isfinite(a["Xd"]["ts"]) & np.isfinite(b["Xd"]["ts"])
    results["Xd_ts_diff"] = float(np.linalg.norm(a["Xd"]["ts"][mask] - b["Xd"]["ts"][mask]))

    mask = np.isfinite(a["Xd"]["var_ts"]) & np.isfinite(b["Xd"]["var_ts"])
    results["Xd_var_diff"] = float(np.linalg.norm(a["Xd"]["var_ts"][mask] - b["Xd"]["var_ts"][mask]))

    if a["PCA"]["U"].shape[1] == b["PCA"]["U"].shape[1]:
        perm, signs = align_components(a["PCA"]["U"], a["PCA"]["V"], b["PCA"]["U"], b["PCA"]["V"])
        pca_a = reorder_decomp(a["PCA"], perm, signs, recompute_ts=True)
        results["PCA_U_diff"] = float(np.linalg.norm(pca_a["U"] - b["PCA"]["U"]))
        results["PCA_V_diff"] = float(np.linalg.norm(pca_a["V"] - b["PCA"]["V"]))
        results["PCA_ts_diff"] = float(np.linalg.norm(pca_a["ts"] - b["PCA"]["ts"]))

    if a["ICA"]["U"].shape[1] == b["ICA"]["U"].shape[1]:
        perm, signs = align_components(a["ICA"]["U"], a["ICA"]["V"], b["ICA"]["U"], b["ICA"]["V"])
        ica_a = reorder_decomp(a["ICA"], perm, signs, recompute_ts=False)
        results["ICA_U_diff"] = float(np.linalg.norm(ica_a["U"] - b["ICA"]["U"]))
        results["ICA_V_diff"] = float(np.linalg.norm(ica_a["V"] - b["ICA"]["V"]))
        results["ICA_ts_diff"] = float(np.linalg.norm(a["ICA"]["ts"] - b["ICA"]["ts"]))

    return results


def model_order_checks(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str | None], list[dict[str, Any]]] = {}
    for entry in entries:
        key = (entry["n_series"], entry["n_epochs"], entry.get("config_signature"))
        grouped.setdefault(key, []).append(entry)

    checks: list[dict[str, Any]] = []
    for (_, _, signature), group_entries in grouped.items():
        ordered = sorted(group_entries, key=lambda item: (item["n_components"], item["path"]))
        for current, nxt in zip(ordered, ordered[1:]):
            if current["n_components"] >= nxt["n_components"]:
                continue
            v1_pca = nxt["PCA_n_parameters"] - current["PCA_n_parameters"]
            v2_pca = nxt["n_observations"] - nxt["PCA_n_parameters"]
            v1_ica = nxt["ICA_n_parameters"] - current["ICA_n_parameters"]
            v2_ica = nxt["n_observations"] - nxt["ICA_n_parameters"]

            f_comp_pca = float("nan")
            if v1_pca > 0 and nxt["chi2_PCA"] != 0.0 and v2_pca > 0:
                f_comp_pca = float(
                    ((current["chi2_PCA"] - nxt["chi2_PCA"]) / nxt["chi2_PCA"]) * (v2_pca / v1_pca)
                )
            f_comp_ica = float("nan")
            if v1_ica > 0 and nxt["chi2_ICA"] != 0.0 and v2_ica > 0:
                f_comp_ica = float(
                    ((current["chi2_ICA"] - nxt["chi2_ICA"]) / nxt["chi2_ICA"]) * (v2_ica / v1_ica)
                )

            checks.append(
                {
                    "from_label": current["label"],
                    "to_label": nxt["label"],
                    "from_path": current["path"],
                    "to_path": nxt["path"],
                    "from_n_components": current["n_components"],
                    "to_n_components": nxt["n_components"],
                    "config_signature": signature,
                    "F_comp_PCA": f_comp_pca,
                    "F_crit_PCA_95": float(f_distribution.ppf(0.95, v1_pca, v2_pca)) if v1_pca > 0 and v2_pca > 0 else float("nan"),
                    "F_comp_ICA": f_comp_ica,
                    "F_crit_ICA_95": float(f_distribution.ppf(0.95, v1_ica, v2_ica)) if v1_ica > 0 and v2_ica > 0 else float("nan"),
                    "ARD_ratio_from": current["ARD_ratio"],
                    "ARD_ratio_to": nxt["ARD_ratio"],
                }
            )
    return checks


def print_summary(entries: list[dict[str, Any]], checks: list[dict[str, Any]], pairwise: dict[str, float] | None) -> None:
    print("Runs")
    for entry in entries:
        print(
            " | ".join(
                [
                    f"label={entry['label']}",
                    f"n={entry['n_components']}",
                    f"mode={entry['decomposition_mode']}",
                    f"chi2_ICA={entry['chi2_ICA']:.6g}",
                    f"redchi2_ICA={entry['reduced_chi2_ICA']:.6g}",
                    f"varExp_ICA={entry['variance_explained_ICA']:.6g}",
                    f"ICA_energy={entry['ICA_energy']:.6g}",
                    f"ARD_ratio={entry['ARD_ratio']:.6g}",
                    f"path={entry['path']}",
                ]
            )
        )

    if checks:
        print()
        print("Model-order checks")
        for check in checks:
            print(
                " | ".join(
                    [
                        f"{check['from_n_components']}->{check['to_n_components']}",
                        f"F_comp_PCA={check['F_comp_PCA']:.6g}",
                        f"F_crit_PCA_95={check['F_crit_PCA_95']:.6g}",
                        f"F_comp_ICA={check['F_comp_ICA']:.6g}",
                        f"F_crit_ICA_95={check['F_crit_ICA_95']:.6g}",
                        f"ARD_from={check['ARD_ratio_from']:.6g}",
                        f"ARD_to={check['ARD_ratio_to']:.6g}",
                    ]
                )
            )

    if pairwise:
        print()
        print("Pairwise differences")
        for key, value in pairwise.items():
            print(f"{key}={value:.12g}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple ICAIM decomposition outputs (.npz or .mat), summarizing quality metrics and model-order checks."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Result files or directories. Directories are scanned recursively for all_python.npz.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["n_components", "variance_explained_ICA", "reduced_chi2_ICA", "ICA_energy", "path"],
        default="n_components",
        help="Primary sort key for the run summary.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        type=Path,
        help="Optional JSON report file.",
    )
    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve())
    files = resolve_result_files(args.inputs)
    if not files:
        raise ValueError("No result files found.")

    loaded = [(path, load_results_file(path)) for path in files]
    entries = [_quality_entry(path, results) for path, results in loaded]
    entries.sort(key=lambda item: (item[args.sort_by], item["path"]))
    checks = model_order_checks(entries)

    pairwise: dict[str, float] | None = None
    if len(loaded) == 2:
        pairwise = compare_pairwise(loaded[0][1], loaded[1][1])

    print_summary(entries, checks, pairwise)

    if args.output_json is not None:
        report = {
            "repo_root": str(repo_root),
            "files": entries,
            "model_order_checks": checks,
            "pairwise_differences": pairwise,
        }
        save_json(args.output_json, report)


if __name__ == "__main__":
    main()
