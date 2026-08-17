from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from scipy.io import loadmat

from icaim_py.common import (
    build_effective_config,
    case1_python_port_compare_batch_output_root,
    effective_config_payload,
    find_repo_root,
    legacy_combination_is_compatible,
    resolve_case1_batch_file,
    resolve_case1_config_file,
    resolve_case1_data_input_file,
    save_json,
    save_text,
)
from icaim_py.pipeline import compare_with_reference, run_decomposition
from run_decomposition_batch import deep_merge, enumerate_grid_runs, explicit_runs, load_batch_spec, resolve_batch_reference


def matlab_quote(path: str | Path) -> str:
    return str(path).replace("'", "''")


def build_run_specs(batch_spec: dict[str, Any] | None) -> tuple[str | None, list[dict[str, Any]], dict[str, Any], bool]:
    if batch_spec is None:
        return None, [{"name": "run-001", "overrides": {}, "grid_item": None}], {}, True

    shared_overrides = batch_spec.get("shared_overrides", {})
    if not isinstance(shared_overrides, dict):
        raise ValueError("The batch 'shared_overrides' section must be a JSON object.")
    skip_incompatible = bool(batch_spec.get("skip_incompatible", True))
    runs: list[dict[str, Any]] = []
    if "grid" in batch_spec:
        runs.extend(enumerate_grid_runs(batch_spec["grid"]))
    if "runs" in batch_spec:
        if not isinstance(batch_spec["runs"], list):
            raise ValueError("The batch 'runs' section must be a JSON array.")
        runs.extend(explicit_runs(batch_spec["runs"]))
    if not runs:
        raise ValueError("Batch file must contain at least one run, via 'grid' and/or 'runs'.")
    return batch_spec.get("base_config_file"), runs, shared_overrides, skip_incompatible


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Python and legacy MATLAB ICAIM pipelines on one or more configurations and compare outputs."
    )
    repo_root = find_repo_root(Path(__file__).resolve())
    parser.add_argument("--repo-root", default=repo_root, type=Path, help="Path to the ICAIM repository root.")
    parser.add_argument(
        "--config-file",
        default=None,
        type=Path,
        help="JSON config file for a single run. You can pass an absolute path, a repo-relative path, or the exact config file name stored under Scenarios/casestudy/case1/python_port/config. The .json suffix is optional.",
    )
    parser.add_argument(
        "--batch-file",
        default=None,
        type=Path,
        help="Batch JSON file for multiple runs. You can pass an absolute path, a repo-relative path, or just the batch name/file name stored under Scenarios/casestudy/case1/python_port/config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Directory where per-run Python/MATLAB outputs and comparison reports will be written. Default: Scenarios/casestudy/case1/python_port/output_compare_batch/<config-or-batch-name>",
    )
    parser.add_argument(
        "--matlab-binary",
        default=Path("/usr/local/MATLAB/R2024a/bin/matlab"),
        type=Path,
        help="Path to the MATLAB executable.",
    )
    parser.add_argument(
        "--data-input-file",
        default=None,
        type=Path,
        help="Optional dataset/data_input file to use in both Python and MATLAB. You can pass an absolute path, a repo-relative path, or just the dataset name/file name. If omitted, each effective run config can inherit top-level data_input_file from default.config.json, the selected config file, or per-run overrides.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()

    if args.batch_file is not None and args.config_file is not None:
        raise ValueError("Pass either --config-file or --batch-file, not both.")
    if args.batch_file is None and args.config_file is None:
        raise ValueError("Pass one of --config-file or --batch-file. No implicit default configuration is allowed.")

    batch_file = resolve_case1_batch_file(args.batch_file, repo_root) if args.batch_file is not None else None
    cli_data_input_file = (
        resolve_case1_data_input_file(args.data_input_file, repo_root) if args.data_input_file is not None else None
    )
    batch_spec = load_batch_spec(batch_file) if batch_file is not None else None
    base_config_file, run_specs, shared_overrides, skip_incompatible = build_run_specs(batch_spec)
    if base_config_file is not None and batch_file is not None:
        base_config_file = resolve_batch_reference(base_config_file, batch_file, repo_root)
    if args.config_file is not None:
        base_config_file = resolve_case1_config_file(args.config_file, repo_root)

    default_output_name = batch_file.stem if batch_file is not None else (base_config_file.stem if base_config_file is not None else "single_run")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else case1_python_port_compare_batch_output_root(repo_root) / default_output_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"config_file={base_config_file}")
    print(f"batch_file={batch_file}")
    if cli_data_input_file is not None:
        print(f"data_input_file={cli_data_input_file}")
    else:
        print("data_input_file=<from config/batch overrides>")
    print(f"output_dir={output_dir}")

    summary: dict[str, Any] = {
        "repo_root": str(repo_root),
        "config_file": str(base_config_file) if args.config_file is not None and base_config_file is not None else None,
        "batch_file": str(batch_file) if batch_file is not None else None,
        "data_input_file": str(cli_data_input_file) if cli_data_input_file is not None else None,
        "data_input_file_source": "cli" if cli_data_input_file is not None else "per-run-config",
        "output_dir": str(output_dir),
        "runs": [],
    }

    matlab_workdir = repo_root / "rewrite" / "matlab_clean"
    for run_spec in run_specs:
        run_name = run_spec["name"]
        run_output_dir = output_dir / run_name
        python_output_dir = run_output_dir / "python"
        matlab_output_dir = run_output_dir / "matlab"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        merged_overrides = deep_merge(shared_overrides, run_spec["overrides"])

        record: dict[str, Any] = {
            "name": run_name,
            "output_dir": str(run_output_dir),
            "overrides": merged_overrides,
        }
        if run_spec["grid_item"] is not None:
            record["grid_item"] = run_spec["grid_item"]

        cfg = build_effective_config(
            repo_root=repo_root,
            data_input_file=cli_data_input_file,
            config_file=base_config_file,
            config_overrides=merged_overrides,
        )
        record["data_input_file"] = str(cfg.data_input_file)
        compatible, reason = legacy_combination_is_compatible(cfg)
        if not compatible and skip_incompatible:
            record["status"] = "skipped_incompatible"
            record["reason"] = reason
            summary["runs"].append(record)
            print(f"[skip incompatible] {run_name}: {reason}")
            continue

        merged_config_file = run_output_dir / "merged_config.json"
        save_json(merged_config_file, effective_config_payload(cfg))

        python_results = run_decomposition(
            repo_root=repo_root,
            output_dir=python_output_dir,
            save_mat=True,
            cfg=cfg,
            run_metadata={
                "entrypoint": "compare_matlab_batch.py",
                "wrapper_command": "icaim-compare-matlab-batch",
                "config_file": str(base_config_file) if base_config_file is not None else None,
                "batch_file": str(batch_file) if batch_file is not None else None,
                "data_input_file": str(cfg.data_input_file),
                "run_name": run_name,
                "output_dir": str(python_output_dir),
            },
        )

        data_input_expr = f"'{matlab_quote(cfg.data_input_file)}'"
        matlab_command = (
            f"cd('{matlab_quote(matlab_workdir)}'); "
            f"run_case1_legacy_configured('{matlab_quote(matlab_output_dir)}', "
            f"'{matlab_quote(merged_config_file)}', {data_input_expr});"
        )
        subprocess.run(
            [str(args.matlab_binary), "-batch", matlab_command],
            check=True,
            cwd=repo_root,
        )

        matlab_result_file = matlab_output_dir / "all_legacy_configured.mat"
        matlab_results = loadmat(matlab_result_file, simplify_cells=True)
        comparison = compare_with_reference(python_results, matlab_results)

        report = {
            "name": run_name,
            "python_output_dir": str(python_output_dir),
            "matlab_output_dir": str(matlab_output_dir),
            "matlab_result_file": str(matlab_result_file),
            "comparison": comparison,
        }
        save_json(run_output_dir / "comparison_with_matlab.json", report)
        lines = [
            f"name={run_name}",
            f"python_output_dir={python_output_dir}",
            f"matlab_output_dir={matlab_output_dir}",
            f"matlab_result_file={matlab_result_file}",
            "",
        ]
        lines.extend(f"{key}={value:.12g}" for key, value in comparison.items())
        save_text(run_output_dir / "comparison_with_matlab.txt", "\n".join(lines) + "\n")

        record["status"] = "completed"
        record["comparison"] = comparison
        summary["runs"].append(record)
        print(f"[ok] {run_name}")

    save_json(output_dir / "comparison_summary.json", summary)


if __name__ == "__main__":
    main()
