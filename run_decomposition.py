from __future__ import annotations

import argparse
import json
from pathlib import Path

from icaim_py.pipeline import run_decomposition
from icaim_py.common import (
    build_effective_config,
    default_case1_output_dir,
    effective_config_payload,
    find_repo_root,
    infer_case1_dataset_label,
    resolve_case1_config_file,
    resolve_case1_data_input_file,
    save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ICAIM decomposition pipeline in Python starting from Scenarios/casestudy/case1/python_port/config/default.config.json, then applying the selected JSON overrides, and finally any CLI dataset override."
    )
    parser.add_argument(
        "--repo-root",
        default=find_repo_root(Path(__file__).resolve()),
        type=Path,
        help="Path to the ICAIM repository root.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Directory where all_python.npz, all_python.mat, and summary files will be written. Default: Scenarios/casestudy/case1/python_port/output/<dataset>",
    )
    parser.add_argument(
        "--data-input-file",
        default=None,
        type=Path,
        help="Optional dataset/data_input file. You can pass an absolute path, a repo-relative path, or just the dataset name/file name, for example: resi_ATF, data_input_resi_ATF.txt. If omitted, the effective config chain must define top-level data_input_file, for example in default.config.json or the selected override JSON.",
    )
    parser.add_argument(
        "--config-file",
        required=True,
        type=Path,
        help="Required JSON config file with overrides applied on top of Scenarios/casestudy/case1/python_port/config/default.config.json. You can pass an absolute path, a repo-relative path, or the exact config file name stored under Scenarios/casestudy/case1/python_port/config. The .json suffix is optional, for example: config.atf2026 or config.case1.verify.quick.basic.",
    )
    parser.add_argument(
        "--decomposition-mode",
        choices=["t", "s", "t-mode", "s-mode"],
        default=None,
        help="Override the decomposition orientation. 't' keeps the historical series x epochs mode; 's' runs PCA/ICA on the transposed epochs x series matrix.",
    )
    parser.add_argument(
        "--print-effective-config",
        action="store_true",
        help="Print the fully resolved effective configuration JSON before the decomposition starts.",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Resolve the effective configuration, write effective_config.json, optionally print it, and exit without running the decomposition.",
    )
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Also generate ICA or PCA component PDF/PNG figures after the run.",
    )
    parser.add_argument(
        "--plot-output-dir",
        default=None,
        type=Path,
        help="Optional directory for the generated component plots. Default: <output-dir>/plots",
    )
    parser.add_argument(
        "--plot-normalization",
        choices=["peak-to-peak", "unit-max", "none"],
        default="peak-to-peak",
        help="Temporal normalization used in the component figures. Default: peak-to-peak.",
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
        help="PNG resolution in dpi for generated plots. Default: 200.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional output filename prefix for component plots. Default: IC for ICA, PC for PCA.",
    )
    parser.add_argument(
        "--make-station-fit-plots",
        action="store_true",
        help="Also generate per-station observed vs modeled East/North/Up figures after the run.",
    )
    parser.add_argument(
        "--station-fit-output-dir",
        default=None,
        type=Path,
        help="Optional directory for the generated station-fit plots. Default: <output-dir>/station_fits",
    )
    parser.add_argument(
        "--hide-station-components",
        action="store_true",
        help="Hide individual component contribution curves in the station-fit figures.",
    )
    parser.add_argument(
        "--stations",
        nargs="*",
        default=None,
        help="Optional station codes to include in station-fit plots when --make-station-fit-plots is enabled.",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    resolved_config_file = resolve_case1_config_file(args.config_file, repo_root)
    cli_data_input_file = resolve_case1_data_input_file(args.data_input_file, repo_root) if args.data_input_file is not None else None
    cfg = build_effective_config(
        repo_root=repo_root,
        data_input_file=cli_data_input_file,
        config_file=resolved_config_file,
        config_overrides={"decomposition_mode": args.decomposition_mode} if args.decomposition_mode is not None else None,
    )
    resolved_data_input_file = cfg.data_input_file
    dataset_label = infer_case1_dataset_label(resolved_data_input_file, repo_root)
    output_dir = args.output_dir.resolve() if args.output_dir is not None else default_case1_output_dir(cfg)
    effective_config_file = output_dir / "effective_config.json"
    effective_config = effective_config_payload(cfg)
    save_json(effective_config_file, effective_config)
    print(f"config_file={resolved_config_file}")
    print(f"data_input_file={resolved_data_input_file}")
    print(f"dataset_label={dataset_label}")
    print(f"decomposition_mode={cfg.decomposition_mode}")
    print(f"output_dir={output_dir}")
    print(f"effective_config_file={effective_config_file}")
    if args.print_effective_config:
        print(json.dumps(effective_config, indent=2, sort_keys=True))
    if args.config_only:
        return

    results = run_decomposition(
        repo_root,
        output_dir,
        save_mat=True,
        cfg=cfg,
        run_metadata={
            "entrypoint": "run_decomposition.py",
            "wrapper_command": "icaim-run",
            "config_file": str(resolved_config_file),
            "data_input_file": str(resolved_data_input_file),
            "dataset_label": dataset_label,
            "decomposition_mode": cfg.decomposition_mode,
            "output_dir": str(output_dir),
            "effective_config_file": str(effective_config_file),
        },
    )
    metrics = results["metrics"]
    print(f"chi2_PCA={metrics['chi2_PCA']:.6f}")
    print(f"chi2_ICA={metrics['chi2_ICA']:.6f}")
    print(f"variance_explained_PCA={metrics['variance_explained_PCA']:.6f}")
    print(f"variance_explained_ICA={metrics['variance_explained_ICA']:.6f}")
    print(f"reduced_chi2_PCA={metrics['reduced_chi2_PCA']:.6f}")
    print(f"reduced_chi2_ICA={metrics['reduced_chi2_ICA']:.6f}")
    print(f"chi2_gain_ICA_vs_PCA_pct={metrics['chi2_gain_ICA_vs_PCA_pct']:.6f}")
    print(f"ICA_energy={metrics['ICA_energy']:.6f}")
    print(f"ARD_ratio={metrics['ARD_ratio']:.6f}")
    for note in results.get("config_notes", []):
        print(f"config_note={note}")
    if args.make_plots:
        from icaim_py.plots import create_ica_component_plots

        plot_output_dir = args.plot_output_dir.resolve() if args.plot_output_dir is not None else output_dir / "plots"
        generated = create_ica_component_plots(
            results,
            plot_output_dir,
            repo_root=repo_root,
            components=args.components,
            normalization=args.plot_normalization,
            background_grid=args.background_grid,
            label_stations=args.label_stations,
            dpi=args.dpi,
            prefix=args.prefix,
            decomposition=args.decomposition,
        )
        for path in generated:
            print(path)
    if args.make_station_fit_plots:
        from icaim_py.station_plots import create_station_fit_plots

        station_fit_output_dir = (
            args.station_fit_output_dir.resolve() if args.station_fit_output_dir is not None else output_dir / "station_fits"
        )
        generated = create_station_fit_plots(
            results,
            station_fit_output_dir,
            stations=args.stations,
            show_components=not args.hide_station_components,
            dpi=args.dpi,
            decomposition=args.decomposition,
        )
        for path in generated:
            print(path)


if __name__ == "__main__":
    main()
