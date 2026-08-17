from __future__ import annotations

import argparse
import itertools
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any

from icaim_py.common import (
    build_effective_config,
    default_case1_batch_output_dir,
    effective_config_payload,
    find_repo_root,
    json_ready,
    legacy_combination_is_compatible,
    resolve_case1_batch_file,
    resolve_case1_config_file,
    resolve_case1_data_input_file,
    save_json,
)
from icaim_py.pipeline import run_decomposition

from icaim_py.plots import create_ica_component_plots as create_ica_component_plots_gps
from icaim_py.insar import create_ica_component_plots_insar

from icaim_py.station_plots import create_station_fit_plots


def load_batch_spec(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in batch file {path}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Batch file must contain a JSON object: {path}")
    return payload


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label} {path}: line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def nested_override_from_path(path: str, value: Any) -> dict[str, Any]:
    parts = path.split(".")
    payload: dict[str, Any] = value
    for part in reversed(parts):
        payload = {part: payload}
    return payload


def slugify(value: Any) -> str:
    text = str(value)
    text = text.replace("&", "and")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "value"


def run_name_from_grid_item(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in sorted(item):
        label = slugify(key.replace(".", "_"))
        value = slugify(item[key])
        parts.append(f"{label}-{value}")
    return "__".join(parts)


def enumerate_grid_runs(grid: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(grid, dict) or not grid:
        raise ValueError("The batch 'grid' section must be a non-empty JSON object.")

    ordered_keys = list(grid)
    ordered_values: list[list[Any]] = []
    for key in ordered_keys:
        values = grid[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Grid key '{key}' must contain a non-empty JSON array.")
        ordered_values.append(values)

    runs: list[dict[str, Any]] = []
    for combo in itertools.product(*ordered_values):
        item = {key: value for key, value in zip(ordered_keys, combo)}
        overrides: dict[str, Any] = {}
        for key, value in item.items():
            overrides = deep_merge(overrides, nested_override_from_path(key, value))
        runs.append({"name": run_name_from_grid_item(item), "overrides": overrides, "grid_item": item})
    return runs


def explicit_runs(spec_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for idx, entry in enumerate(spec_runs, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Batch run #{idx} must be a JSON object.")
        overrides = entry.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"Batch run #{idx} has invalid 'overrides'. Expected a JSON object.")
        name = entry.get("name") or f"run-{idx:03d}"
        runs.append({"name": slugify(name), "overrides": overrides, "grid_item": None})
    return runs


def resolve_batch_reference(path: str | Path | None, batch_file: Path, repo_root: Path) -> Path | None:
    if path is None:
        return None

    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path.resolve()

    candidate_from_batch = (batch_file.resolve().parent / raw_path).resolve()
    if candidate_from_batch.exists():
        return candidate_from_batch

    candidate_from_repo = (repo_root.resolve() / raw_path).resolve()
    if candidate_from_repo.exists():
        return candidate_from_repo

    return resolve_case1_config_file(raw_path, repo_root)


def resolve_batch_run_data_input_file(
    cli_data_input_file: Path | None,
    merged_config_payload: dict[str, Any],
    repo_root: Path,
) -> Path:
    if cli_data_input_file is not None:
        return cli_data_input_file
    if "data_input_file" not in merged_config_payload:
        raise ValueError(
            "--data-input-file is required unless each effective batch run config defines top-level 'data_input_file'. "
            "No implicit default dataset is allowed."
        )
    return resolve_case1_data_input_file(merged_config_payload["data_input_file"], repo_root)


def per_run_aux_output_dir(
    requested_root: Path | None,
    run_output_dir: Path,
    run_name: str,
    default_leaf: str,
) -> Path:
    if requested_root is None:
        return run_output_dir / default_leaf
    return requested_root.resolve() / run_name


def _shell_join(parts: list[str]) -> str:
    return shlex.join(parts)


def _write_reproduce_script(path: Path, command: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n")
    path.chmod(0o755)


def _single_run_command(
    repro_config_file: Path,
    run_output_dir: Path,
    args: argparse.Namespace,
) -> str:
    command = [
        "icaim-run",
        "--config-file",
        str(repro_config_file),
        "--output-dir",
        str(run_output_dir),
    ]
    if args.make_plots:
        command.append("--make-plots")
        if args.plot_output_dir is not None:
            command.extend(["--plot-output-dir", str(per_run_aux_output_dir(args.plot_output_dir, run_output_dir, run_output_dir.name, "plots"))])
        command.extend(["--plot-normalization", args.plot_normalization])
        command.extend(["--decomposition", args.decomposition])
        command.extend(["--background-grid", str(args.background_grid)])
        if args.components:
            command.append("--components")
            command.extend(str(component) for component in args.components)
        if args.label_stations:
            command.append("--label-stations")
        command.extend(["--dpi", str(args.dpi)])
        if args.prefix is not None:
            command.extend(["--prefix", args.prefix])
    if args.make_station_fit_plots:
        command.append("--make-station-fit-plots")
        if args.station_fit_output_dir is not None:
            command.extend(
                [
                    "--station-fit-output-dir",
                    str(per_run_aux_output_dir(args.station_fit_output_dir, run_output_dir, run_output_dir.name, "station_fits")),
                ]
            )
        if args.hide_station_components:
            command.append("--hide-station-components")
        if args.stations:
            command.append("--stations")
            command.extend(args.stations)
        if not args.make_plots:
            command.extend(["--decomposition", args.decomposition])
            command.extend(["--dpi", str(args.dpi)])
    if args.decomposition_mode is not None:
        command.extend(["--decomposition-mode", args.decomposition_mode])
    return _shell_join(command)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple ICAIM decomposition configurations from a batch JSON file.")
    parser.add_argument(
        "--repo-root",
        default=find_repo_root(Path(__file__).resolve()),
        type=Path,
        help="Path to the ICAIM repository root.",
    )
    parser.add_argument(
        "--batch-file",
        required=True,
        type=Path,
        help="JSON file describing the run grid and shared config overrides. You can pass an absolute path, a repo-relative path, or just the batch name/file name stored under Scenarios/casestudy/case1/python_port/config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Directory where per-run outputs and the batch summary will be written. Default: Scenarios/casestudy/case1/python_port/output_batch/<batch-name>",
    )
    parser.add_argument(
        "--data-input-file",
        default=None,
        type=Path,
        help="Optional dataset/data_input file for every run. You can pass an absolute path, a repo-relative path, or just the dataset name/file name, for example: resi_ATF2026. If omitted, each effective batch run config can inherit top-level data_input_file from default.config.json, the batch base_config_file, or per-run overrides.",
    )
    parser.add_argument(
        "--decomposition-mode",
        choices=["t", "s", "t-mode", "s-mode"],
        default=None,
        help="Override the decomposition orientation for every run. Without this flag, use each run's JSON configuration.",
    )
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Also generate ICA or PCA component PDF/PNG figures for every completed run.",
    )
    parser.add_argument(
        "--plot-output-dir",
        default=None,
        type=Path,
        help="Optional root directory for generated component plots. In batch mode a per-run subdirectory is created inside it.",
    )
    parser.add_argument(
        "--plot-normalization",
        choices=["peak-to-peak", "unit-max", "none"],
        default="peak-to-peak",
        help="Temporal normalization used in component figures. Default: peak-to-peak.",
    )
    parser.add_argument(
        "--decomposition",
        choices=["ICA", "PCA"],
        default="ICA",
        help="Which decomposition to analyze when generating optional plots. Default: ICA.",
    )
    parser.add_argument(
        "--background-grid",
        default="auto",
        help="Optional GMT/NetCDF grid for the plot background. Use 'auto' or 'none'.",
    )
    parser.add_argument(
        "--components",
        nargs="*",
        type=int,
        default=None,
        help="Optional 1-based component indices to plot when --make-plots is enabled. Default: all components.",
    )
    parser.add_argument(
        "--label-stations",
        action="store_true",
        help="Annotate station names on component maps when --make-plots is enabled.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG resolution in dpi for generated plots and station-fit figures. Default: 200.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional output filename prefix for component plots. Default: IC for ICA, PC for PCA.",
    )
    parser.add_argument(
        "--make-station-fit-plots",
        action="store_true",
        help="Also generate per-station observed vs modeled East/North/Up figures for every completed run.",
    )
    parser.add_argument(
        "--station-fit-output-dir",
        default=None,
        type=Path,
        help="Optional root directory for generated station-fit plots. In batch mode a per-run subdirectory is created inside it.",
    )
    parser.add_argument(
        "--hide-station-components",
        action="store_true",
        help="Hide individual component contribution curves in station-fit figures.",
    )
    parser.add_argument(
        "--stations",
        nargs="*",
        default=None,
        help="Optional station codes to include in station-fit plots when --make-station-fit-plots is enabled.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    batch_file = resolve_case1_batch_file(args.batch_file, repo_root)
    cli_data_input_file = (
        resolve_case1_data_input_file(args.data_input_file, repo_root) if args.data_input_file is not None else None
    )
    output_dir = args.output_dir.resolve() if args.output_dir is not None else default_case1_batch_output_dir(repo_root, batch_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = load_batch_spec(batch_file)
    shared_overrides = spec.get("shared_overrides", {})
    if not isinstance(shared_overrides, dict):
        raise ValueError("The batch 'shared_overrides' section must be a JSON object.")

    base_config_file = resolve_batch_reference(spec.get("base_config_file"), batch_file, repo_root)
    skip_incompatible = bool(spec.get("skip_incompatible", True))
    skip_not_implemented = bool(spec.get("skip_not_implemented", True))
    print(f"batch_file={batch_file}")
    print(f"base_config_file={base_config_file}")
    if cli_data_input_file is not None:
        print(f"data_input_file={cli_data_input_file}")
    else:
        print("data_input_file=<from batch/config overrides>")
    if args.decomposition_mode is not None:
        print(f"decomposition_mode={args.decomposition_mode}")
    print(f"output_dir={output_dir}")

    batch_command = _shell_join(["icaim-run-batch", *sys.argv[1:]])
    _write_reproduce_script(output_dir / "reproduce_batch.sh", batch_command)

    runs: list[dict[str, Any]] = []
    if "grid" in spec:
        runs.extend(enumerate_grid_runs(spec["grid"]))
    if "runs" in spec:
        if not isinstance(spec["runs"], list):
            raise ValueError("The batch 'runs' section must be a JSON array.")
        runs.extend(explicit_runs(spec["runs"]))
    if not runs:
        raise ValueError("Batch file must contain at least one run, via 'grid' and/or 'runs'.")

    summary: dict[str, Any] = {
        "repo_root": str(repo_root),
        "batch_file": str(batch_file),
        "base_config_file": str(base_config_file) if base_config_file is not None else None,
        "output_dir": str(output_dir),
        "data_input_file": str(cli_data_input_file) if cli_data_input_file is not None else None,
        "data_input_file_source": "cli" if cli_data_input_file is not None else "per-run-config",
        "decomposition_mode": args.decomposition_mode,
        "skip_incompatible": skip_incompatible,
        "skip_not_implemented": skip_not_implemented,
        "reproduce_batch_command": batch_command,
        "reproduce_batch_script": str(output_dir / "reproduce_batch.sh"),
        "runs": [],
    }

    for run_spec in runs:
        run_name = run_spec["name"]
        run_output_dir = output_dir / run_name
        merged_overrides = deep_merge(shared_overrides, run_spec["overrides"])
        record: dict[str, Any] = {
            "name": run_name,
            "output_dir": str(run_output_dir),
            "overrides": json_ready(merged_overrides),
        }
        if run_spec["grid_item"] is not None:
            record["grid_item"] = json_ready(run_spec["grid_item"])
        if args.decomposition_mode is not None:
            merged_overrides = deep_merge(merged_overrides, {"decomposition_mode": args.decomposition_mode})
            record["overrides"] = json_ready(merged_overrides)

        try:
            cfg = build_effective_config(
                repo_root=repo_root,
                data_input_file=cli_data_input_file,
                config_file=base_config_file,
                config_overrides=merged_overrides,
            )
            record["data_input_file"] = str(cfg.data_input_file)
            record["decomposition_mode"] = cfg.decomposition_mode
            compatible, reason = legacy_combination_is_compatible(cfg)
            if skip_incompatible and not compatible:
                record["status"] = "skipped_incompatible"
                record["reason"] = reason
                summary["runs"].append(record)
                print(f"[skip incompatible] {run_name}: {reason}")
                continue

            run_output_dir.mkdir(parents=True, exist_ok=True)
            save_json(run_output_dir / "batch_run.json", {"name": run_name, "config": cfg, "overrides": merged_overrides})
            repro_config_file = run_output_dir / "repro_config.json"
            save_json(repro_config_file, effective_config_payload(cfg))
            record["repro_config_file"] = str(repro_config_file)
            reproduce_run_command = _single_run_command(repro_config_file, run_output_dir, args)
            record["reproduce_run_command"] = reproduce_run_command
            record["reproduce_run_script"] = str(run_output_dir / "reproduce_run.sh")
            _write_reproduce_script(run_output_dir / "reproduce_run.sh", reproduce_run_command)
            results = run_decomposition(
                repo_root=repo_root,
                output_dir=run_output_dir,
                save_mat=True,
                cfg=cfg,
                run_metadata={
                    "entrypoint": "run_decomposition_batch.py",
                    "wrapper_command": "icaim-run-batch",
                    "batch_file": str(batch_file),
                    "base_config_file": str(base_config_file) if base_config_file is not None else None,
                    "data_input_file": str(cfg.data_input_file),
                    "decomposition_mode": cfg.decomposition_mode,
                    "run_name": run_name,
                    "output_dir": str(run_output_dir),
                },
            )

            if args.make_plots:

                plot_output_dir = per_run_aux_output_dir(args.plot_output_dir, run_output_dir, run_name, "plots")
                dataset_types = {str(value).upper() for value in results["Xd"]["type"]}
                create_ica_component_plots = (
                    create_ica_component_plots_insar
                    if any(value.startswith("INSARLOS") for value in dataset_types)
                    else create_ica_component_plots_gps
                )
                plot_paths = create_ica_component_plots(
                    results_or_file=results,
                    output_dir=plot_output_dir,
                    repo_root=repo_root,
                    components=args.components,
                    normalization=args.plot_normalization,
                    background_grid=args.background_grid,
                    label_stations=args.label_stations,
                    dpi=args.dpi,
                    prefix=args.prefix,
                    decomposition=args.decomposition,
                )
                record["plot_output_dir"] = str(plot_output_dir)
                record["plot_file_count"] = len(plot_paths)

            if args.make_station_fit_plots:

                station_fit_output_dir = per_run_aux_output_dir(
                    args.station_fit_output_dir,
                    run_output_dir,
                    run_name,
                    "station_fits",
                )
                station_plot_paths = create_station_fit_plots(
                    results_or_file=results,
                    output_dir=station_fit_output_dir,
                    stations=args.stations,
                    show_components=not args.hide_station_components,
                    dpi=args.dpi,
                    decomposition=args.decomposition,
                )
                record["station_fit_output_dir"] = str(station_fit_output_dir)
                record["station_fit_file_count"] = len(station_plot_paths)

            record["status"] = "completed"
            record["config_notes"] = list(results.get("config_notes", []))
            summary["runs"].append(record)
            extra_labels: list[str] = []
            if args.make_plots:
                extra_labels.append("plots")
            if args.make_station_fit_plots:
                extra_labels.append("station_fits")
            suffix = f" ({', '.join(extra_labels)})" if extra_labels else ""
            print(f"[ok] {run_name}{suffix}")
        except (NotImplementedError, ValueError) as exc:
            record["status"] = "skipped_not_implemented" if skip_not_implemented else "failed"
            record["reason"] = str(exc)
            summary["runs"].append(record)
            prefix = "skip" if skip_not_implemented else "fail"
            print(f"[{prefix}] {run_name}: {exc}")
            if not skip_not_implemented:
                raise
        except Exception as exc:
            record["status"] = "failed"
            record["reason"] = f"{type(exc).__name__}: {exc}"
            summary["runs"].append(record)
            print(f"[fail] {run_name}: {type(exc).__name__}: {exc}")
            raise

    save_json(output_dir / "batch_summary.json", summary)


if __name__ == "__main__":
    main()
