from __future__ import annotations

import argparse
from pathlib import Path

from scipy.io import loadmat

from icaim_py.common import (
    build_effective_config,
    default_case1_output_dir,
    find_repo_root,
    infer_case1_dataset_label,
    resolve_case1_config_file,
    resolve_case1_data_input_file,
    save_json,
    save_text,
)
from icaim_py.pipeline import compare_with_reference, run_decomposition


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the Python ICAIM decomposition port against MATLAB reference outputs using an explicit config file and an explicit or config-provided dataset input."
    )
    repo_root = find_repo_root(Path(__file__).resolve())
    parser.add_argument(
        "--repo-root",
        default=repo_root,
        type=Path,
        help="Path to the ICAIM repository root.",
    )
    parser.add_argument(
        "--reference",
        default=repo_root / "Scenarios" / "casestudy" / "case1" / "matfiles" / "all.mat",
        type=Path,
        help="MATLAB reference .mat file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Directory where the Python output and comparison report will be written. Default: Scenarios/casestudy/case1/python_port/output/<dataset>",
    )
    parser.add_argument(
        "--data-input-file",
        default=None,
        type=Path,
        help="Optional dataset/data_input file. You can pass an absolute path, a repo-relative path, or just the dataset name/file name. If omitted, the effective config chain must define top-level data_input_file, for example in default.config.json or the selected override JSON.",
    )
    parser.add_argument(
        "--config-file",
        required=True,
        type=Path,
        help="Required JSON config file with overrides applied on top of Scenarios/casestudy/case1/python_port/config/default.config.json. You can pass an absolute path, a repo-relative path, or the exact config file name stored under Scenarios/casestudy/case1/python_port/config. The .json suffix is optional, for example: config.atf2026 or config.case1.verify.quick.basic.",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    resolved_config_file = resolve_case1_config_file(args.config_file, repo_root)
    cli_data_input_file = resolve_case1_data_input_file(args.data_input_file, repo_root) if args.data_input_file is not None else None
    cfg = build_effective_config(
        repo_root=repo_root,
        data_input_file=cli_data_input_file,
        config_file=resolved_config_file,
    )
    resolved_data_input_file = cfg.data_input_file
    dataset_label = infer_case1_dataset_label(resolved_data_input_file, repo_root)
    output_dir = args.output_dir.resolve() if args.output_dir is not None else default_case1_output_dir(cfg)
    print(f"reference_file={args.reference.resolve()}")
    print(f"config_file={resolved_config_file}")
    print(f"data_input_file={resolved_data_input_file}")
    print(f"dataset_label={dataset_label}")
    print(f"output_dir={output_dir}")

    results = run_decomposition(
        repo_root,
        output_dir,
        save_mat=True,
        cfg=cfg,
        run_metadata={
            "entrypoint": "compare_with_matlab.py",
            "wrapper_command": "icaim-compare-matlab",
            "config_file": str(resolved_config_file),
            "data_input_file": str(resolved_data_input_file),
            "dataset_label": dataset_label,
            "output_dir": str(output_dir),
            "reference_file": str(args.reference.resolve()),
        },
    )
    reference = loadmat(args.reference, simplify_cells=True)
    if "metrics" not in reference:
        reference["metrics"] = {
            "chi2_PCA": 8070.346263,
            "chi2_ICA": 8988.288843,
            "variance_explained_PCA": 85.213464,
            "variance_explained_ICA": 84.717402,
        }

    comparison = compare_with_reference(results, reference)
    for key, value in comparison.items():
        print(f"{key}={value:.12g}")

    report = {
        "reference_file": str(args.reference),
        "python_output_dir": str(output_dir),
        "comparison": comparison,
    }
    save_json(output_dir / "comparison_with_matlab.json", report)
    lines = [f"reference_file={args.reference}", f"python_output_dir={output_dir}", ""]
    lines.extend(f"{key}={value:.12g}" for key, value in comparison.items())
    save_text(output_dir / "comparison_with_matlab.txt", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
