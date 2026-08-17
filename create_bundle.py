from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from icaim_py.common import case1_python_port_bundle_dir, find_repo_root, parse_data_input, parse_station_list


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "output", "bundle_case1"))


def relative_to_repo(repo_root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path {path} is outside repo root {repo_root}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a self-contained Python bundle for the active ICAIM case.")
    parser.add_argument("--repo-root", default=find_repo_root(Path(__file__).resolve()), type=Path, help="Path to the ICAIM repository root.")
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Directory where the bundle will be created. Default: Scenarios/casestudy/case1/python_port/bundle_case1",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else case1_python_port_bundle_dir(repo_root)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copy_tree(repo_root / "rewrite" / "python_port", output_dir / "python_port")
    copy_file(repo_root / "rewrite" / "build_dataset_files.py", output_dir / "build_dataset_files.py")
    copy_file(repo_root / "rewrite" / "compare_outputs.py", output_dir / "compare_outputs.py")
    copy_file(repo_root / "rewrite" / "MANUALE_REWRITE.md", output_dir / "MANUALE_REWRITE.md")
    copy_file(repo_root / "rewrite" / "MANUAL_REWRITE_EN.md", output_dir / "MANUAL_REWRITE_EN.md")

    case_dir = repo_root / "Scenarios" / "casestudy" / "case1"
    config_dir = case_dir / "python_port" / "config"
    copy_tree(case_dir / "dataset", output_dir / "Scenarios" / "casestudy" / "case1" / "dataset")
    if config_dir.exists():
        copy_tree(config_dir, output_dir / "Scenarios" / "casestudy" / "case1" / "python_port" / "config")
    copy_file(case_dir / "matfiles" / "all.mat", output_dir / "Scenarios" / "casestudy" / "case1" / "matfiles" / "all.mat")

    datasets = parse_data_input(case_dir / "dataset" / "data_input_file.txt", repo_root)
    copied_lists: set[Path] = set()
    for dataset in datasets:
        if dataset["instruction"] != "decomp":
            continue
        list_path = Path(dataset["list_path"]).resolve()
        if list_path not in copied_lists:
            copy_file(list_path, output_dir / relative_to_repo(repo_root, list_path))
            copied_lists.add(list_path)
        for station in parse_station_list(list_path, repo_root):
            station_file = Path(station["file"]).resolve()
            copy_file(station_file, output_dir / relative_to_repo(repo_root, station_file))

    print(f"Bundle created in: {output_dir}")
    print("Recipient quick start:")
    print(f"  cd {output_dir / 'python_port'}")
    print("  python -m venv .venv")
    print("  source .venv/bin/activate")
    print("  pip install -r requirements.txt")
    print("  python run_decomposition.py --config-file config.atf2026")
    print("  python compare_with_matlab.py --config-file config.case1.verify.quick.basic --data-input-file resi_ATF")
    print("  python compare_matlab_batch.py --config-file config.case1.verify.quick.basic --data-input-file resi_ATF")
    print("  python plot_ica_components.py --results-file resi_ATF")
    print("  python plot_station_fits.py --results-file resi_ATF")


if __name__ == "__main__":
    main()
