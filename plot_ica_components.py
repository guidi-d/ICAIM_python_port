from __future__ import annotations

import argparse
from pathlib import Path

from icaim_py.common import find_repo_root, load_results_file, resolve_case1_results_file

from icaim_py.plots import create_ica_component_plots as create_ica_component_plots_gps
from icaim_py.insar import create_ica_component_plots_insar

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ICA or PCA component figures from a Python or clean-MATLAB ICAIM .npz/.mat output."
    )
    repo_root = find_repo_root(Path(__file__).resolve())
    parser.add_argument(
        "--repo-root",
        default=repo_root,
        type=Path,
        help="Path to the ICAIM repository root.",
    )
    parser.add_argument(
        "--results-mat",
        dest="results_file",
        default=None,
        type=Path,
        help="Path to the .npz or .mat file containing saved ICAIM results. You can also pass an output directory or just the dataset name, for example: resi_ATF.",
    )
    parser.add_argument(
        "--results-file",
        dest="results_file",
        type=Path,
        help="Alias for --results-mat. Accepts either .npz or .mat.",
    )
    parser.add_argument(
        "--decomposition",
        choices=["ICA", "PCA"],
        default="ICA",
        help="Which decomposition to analyze. Default: ICA.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Directory where the figures will be written. Default: <results-dir>/plots. PCA outputs use PC-prefixed filenames.",
    )
    parser.add_argument(
        "--components",
        nargs="*",
        type=int,
        default=None,
        help="Optional 1-based component indices to plot. Default: all components.",
    )
    parser.add_argument(
        "--normalization",
        choices=["peak-to-peak", "unit-max", "none"],
        default="peak-to-peak",
        help="Temporal-series normalization used for the top panel. Default: peak-to-peak.",
    )
    parser.add_argument(
        "--background-grid",
        default="auto",
        help="Optional GMT/NetCDF grid for the background map. Use 'auto' or 'none'.",
    )
    parser.add_argument(
        "--label-stations",
        action="store_true",
        help="Annotate station names on the map.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG resolution in dpi. Default: 200.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional output filename prefix. Default: IC for ICA, PC for PCA.",
    )
    args = parser.parse_args()
    if args.results_file is None:
        parser.error("--results-file is required. No implicit default results path is allowed.")
    repo_root = args.repo_root.resolve()
    results_file = resolve_case1_results_file(args.results_file, repo_root)
    output_dir = args.output_dir.resolve() if args.output_dir is not None else results_file.parent / "plots"
    results = load_results_file(results_file)
    dataset_types = {str(value).upper() for value in results["Xd"]["type"]}
    create_ica_component_plots = (
        create_ica_component_plots_insar
        if any(value.startswith("INSARLOS") for value in dataset_types)
        else create_ica_component_plots_gps
    )

    generated = create_ica_component_plots(
        results_or_file=results,
        output_dir=output_dir,
        repo_root=repo_root,
        components=args.components,
        normalization=args.normalization,
        background_grid=args.background_grid,
        label_stations=args.label_stations,
        dpi=args.dpi,
        prefix=args.prefix,
        decomposition=args.decomposition,
    )

    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
