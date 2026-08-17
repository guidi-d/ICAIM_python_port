from __future__ import annotations

import argparse
from pathlib import Path

from icaim_py.common import find_repo_root, resolve_case1_results_file
from icaim_py.station_plots import create_station_fit_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create per-station observed vs modeled East/North/Up plots from ICA or PCA ICAIM results."
    )
    repo_root = find_repo_root(Path(__file__).resolve())
    parser.add_argument(
        "--repo-root",
        default=repo_root,
        type=Path,
        help="Path to the ICAIM repository root.",
    )
    parser.add_argument(
        "--results-file",
        default=None,
        type=Path,
        help="Path to the results file (.npz recommended, .mat also supported). You can also pass an output directory or just the dataset name, for example: resi_ATF.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Directory where the station fit PDF/PNG files will be written. Default: <results-dir>/station_fits",
    )
    parser.add_argument(
        "--decomposition",
        choices=["ICA", "PCA"],
        default="ICA",
        help="Which decomposition to analyze. Default: ICA.",
    )
    parser.add_argument(
        "--stations",
        nargs="*",
        default=None,
        help="Optional station codes to plot, for example: --stations ACOM AFAL",
    )
    parser.add_argument(
        "--hide-components",
        action="store_true",
        help="Hide the individual component contribution curves and only show observed vs modeled sum.",
    )
    args = parser.parse_args()
    if args.results_file is None:
        parser.error("--results-file is required. No implicit default results path is allowed.")
    repo_root = args.repo_root.resolve()
    results_file = resolve_case1_results_file(args.results_file, repo_root)
    output_dir = args.output_dir.resolve() if args.output_dir is not None else results_file.parent / "station_fits"

    generated = create_station_fit_plots(
        results_or_file=results_file,
        output_dir=output_dir,
        stations=args.stations,
        show_components=not args.hide_components,
        decomposition=args.decomposition,
    )
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
