# ICAIM Python Port

Versione italiana: `README.md`

This directory contains the Python port of the active `Scenarios/casestudy/case1` workflow plus the tools used to run it, compare it with MATLAB, and compare different decomposition outputs against each other.

Path note:
- in the full repository this directory is `rewrite/python_port`
- in the distributed bundle the same directory is simply `python_port`

Working layout:
- code, Python package, `requirements.txt`, and documentation stay in `rewrite/python_port`
- `case1` JSON configurations live in `Scenarios/casestudy/case1/python_port/config`
- single-run outputs live in `Scenarios/casestudy/case1/python_port/output/<dataset>`
- batch outputs live in `Scenarios/casestudy/case1/python_port/output_batch/<batch-name>`
- Python/MATLAB batch comparisons live in `Scenarios/casestudy/case1/python_port/output_compare_batch/<config-or-batch>`
- the distributable bundle lives in `Scenarios/casestudy/case1/python_port/bundle_case1`

Main scripts inside `python_port`:
- `run_decomposition.py`
- `run_decomposition_batch.py`
- `compare_with_matlab.py`
- `compare_matlab_batch.py`
- `plot_ica_components.py`
- `plot_station_fits.py`
- `compare_decomposition_runs.py`
- `select_best_decomposition_runs.py`
- `create_bundle.py`

Repository-level helpers also exposed through wrappers:
- `rewrite/build_dataset_files.py`
- `rewrite/compare_outputs.py`

Commands exposed through `PATH`:
- `icaim-run`
- `icaim-run-batch`
- `icaim-compare-matlab`
- `icaim-compare-matlab-batch`
- `icaim-plot-components`
- `icaim-plot-station-fits`
- `icaim-compare-runs`
- `icaim-select-runs`
- `icaim-create-bundle`
- `icaim-build-dataset-files`
- `icaim-compare-results`

Wrapper-to-script mapping:
- `icaim-run` -> `run_decomposition.py`
- `icaim-run-batch` -> `run_decomposition_batch.py`
- `icaim-compare-matlab` -> `compare_with_matlab.py`
- `icaim-compare-matlab-batch` -> `compare_matlab_batch.py`
- `icaim-plot-components` -> `plot_ica_components.py`
- `icaim-plot-station-fits` -> `plot_station_fits.py`
- `icaim-compare-runs` -> `compare_decomposition_runs.py`
- `icaim-select-runs` -> `select_best_decomposition_runs.py`
- `icaim-create-bundle` -> `create_bundle.py`
- `icaim-build-dataset-files` -> `rewrite/build_dataset_files.py` in the full repository, `build_dataset_files.py` in the bundle
- `icaim-compare-results` -> `rewrite/compare_outputs.py` in the full repository, `compare_outputs.py` in the bundle

Available documentation:
- Italian quick guide: `README.md`
- Italian operational how-to: `HOWTO.md`
- English quick guide: `README_EN.md`
- English operational how-to: `HOWTO_EN.md`
- Italian configuration guide: `CASE1_CONFIG_GUIDE.md`
- English configuration guide: `CASE1_CONFIG_GUIDE_EN.md`
- general rewrite manual in Italian: `../MANUALE_REWRITE.md`
- general rewrite manual in English: `../MANUAL_REWRITE_EN.md`

Which document to read:
- if you only need orientation inside `python_port`, start from `README_EN.md`
- if you want to actually run decompositions starting from a data folder, start from `HOWTO_EN.md`
- if you need the meaning of the JSON fields, use `CASE1_CONFIG_GUIDE_EN.md`
- if you also need architecture, MATLAB workflow, and general rewrite limits, use `../MANUAL_REWRITE_EN.md`

Current status:
- raw `tseri` loader, preprocessing, and station selection for `case1`
- `basic` centering
- `advanced` centering for the legacy `decomp_CG_means` branch
- `centering.Vimposed.type`: `None`, `Heaviside`, `Linear`, `V`
- PCA through `empca` and `decomp_srebro_CG_simultaneous`
- Gaussian `vbICA` with `net_init='SVD'` or `net_init='SVD_S&J'`
- legacy presets `legacy_o1..legacy_o4` and `legacy_r1..legacy_r4`
- JSON-driven batch search and Python vs MATLAB comparisons on the same configuration
- quality metrics saved for every run plus decomposition-to-decomposition comparison through `compare_decomposition_runs.py`

Current limits:
- `decompositionICA.source_type='g'`
- `decompositionICA.ICA_num=1`
- `decompositionICA.source_init='kmeans'`
- `decompositionPCA.rand_init=0`
- `centering.type='advanced'` requires `centering.function='decomp_CG_means'`

Quick start:

```bash
cd /path/to/ICAIM/rewrite/python_port
# in the bundle: cd /path/to/bundle_case1/python_port
python run_decomposition.py --config-file config.case1.verify.quick.basic --data-input-file resi_ATF
python run_decomposition.py --config-file config.atf2026
python run_decomposition_batch.py --batch-file batch.case1.search.example.json --data-input-file resi_ATF
python compare_with_matlab.py --config-file config.case1.verify.quick.basic --data-input-file resi_ATF
python compare_decomposition_runs.py \
  /path/to/ICAIM/Scenarios/casestudy/case1/python_port/output/resi_ATF/all_python.npz \
  /path/to/ICAIM/Scenarios/casestudy/case1/python_port/output_batch/batch.case1.search.example/*/all_python.npz
python create_bundle.py
icaim-run --config-file config.atf2026
```

For the full end-to-end procedure from data ingestion to comparisons and bundle creation, use `HOWTO_EN.md`.

Configuration:
- all JSON files live in `Scenarios/casestudy/case1/python_port/config`
- complete base config: `default.config.json`
- single-run override example: `config.case1.example.json`
- single-run example with embedded dataset: `config.atf2026.json`
- single-run example with embedded dataset, explicit season alias: `config.atf2026_season.json`
- batch example: `batch.case1.search.example.json`
- quick checks: `config.case1.verify.quick.basic.json`, `config.case1.verify.quick.advanced.json`, `config.case1.verify.quick.srebro.json`
- `default.config.json` is the complete base case1 configuration; the other `config.*.json` files are scenario-specific overrides applied on top of it
- `--config-file` accepts full paths, repo-relative paths, or the exact config file name with optional `.json`; for example `--config-file config.atf2026` or `--config-file config.case1.verify.quick.basic`. Short aliases such as `basic` are not accepted
- `--batch-file` and `--data-input-file` still accept full paths or short names; for example `--batch-file example`, `--data-input-file resi_ATF2026`
- `run_decomposition.py` and `compare_with_matlab.py` always require `--config-file`; `--data-input-file` is optional when the effective configuration already defines `data_input_file`, for example in `default.config.json` or in the selected override JSON
- `run_decomposition_batch.py` and `compare_matlab_batch.py` let you omit `--data-input-file` when every effective batch run config inherits or defines `data_input_file`
- `default.config.json` points to `Scenarios/casestudy/case1/dataset/data_input_file.txt` by default, but any selected `config.*.json` file or CLI `--data-input-file` can override it
- the practical parameter guide is `CASE1_CONFIG_GUIDE_EN.md`
- to inspect the full effective parameter set before computation, use `icaim-run --config-file ... [--data-input-file ...] --config-only`
- during a normal run `icaim-run` writes `effective_config.json` immediately inside the output directory, before the decomposition starts
- `default.config.json` exposes `flags`, `outliers`, `velocity`, and the legacy centering fields `offsets_epoch_imposed` plus `Ustart/Sstart/Vstart`
- `flags` and `outliers` remain compatibility-oriented configuration; `velocity` is kept mainly for the clean MATLAB/legacy path when detrending is enabled, but it does not alter the main `run_decomposition` computational flow

Portability on another PC:
- keep the repository layout with the `Data/` and `Scenarios/` directories
- the example JSON files now use explicit file names or repo-relative paths, so they do not need to be rewritten when the local repo root changes
- `data_input` and `stn_list` files may still contain legacy absolute paths with `/Data/` or `/Scenarios/` segments: the Python port remaps them automatically to the current repository root
- only absolute paths that point outside the repository need to be updated manually
- if you want to use the `icaim-...` wrappers from any directory, add `export PATH="/path/to/ICAIM/rewrite/python_port/bin:$PATH"` to `~/.bashrc` or `~/.zshrc`, then reopen the shell

Metrics and decomposition comparison:
- `effective_config.json` stores the full effective configuration as soon as the run starts
- `summary.json` stores `config`, `resolved_config`, `config_notes`, `metrics`, and `quality_metrics`
- `all_python.npz` also stores flat metrics, `ICA.net.energy`, `ICA.net.alphas`, `quality_metrics_json`, and the JSON-serialized configuration
- `decomposition_mode` controls decomposition orientation: `t` is the historical default, `s` internally uses the transposed matrix and can also be selected with `--decomposition-mode s`
- detailed variable guide and `GPS1`/`GPS2`/`GPS3` layout notes: `ALL_PYTHON_NPZ_GUIDE_EN.md`
- `compare_decomposition_runs.py` compares multiple `all_python.npz` or `.mat` files, prints a compact summary table, and computes F-tests across compatible model orders
- `select_best_decomposition_runs.py` uses fit metrics, F-tests, and ARD diagnostics to recommend the best run for each `n_components` and an overall final choice
- the most useful metrics currently include `variance_explained_*`, `reduced_chi2_*`, `weighted_rms_*`, `ICA_energy`, and `ARD_ratio`

Main tools:
- `run_decomposition.py`: single run, with optional IC plots and observed-vs-modeled station plots
- `run_decomposition_batch.py`: JSON-driven multi-configuration campaigns
- `plot_ica_components.py`: ICA or PCA component plots from an existing output
- `plot_station_fits.py`: station-wise observed-vs-reconstructed comparison
- `compare_with_matlab.py`: comparison against `all.mat`
- `compare_decomposition_runs.py`: comparison across multiple Python decompositions
- `select_best_decomposition_runs.py`: automatic selection of the best runs from a set of outputs
- `compare_matlab_batch.py`: Python vs MATLAB comparison on the same JSON configuration
- `build_dataset_files.py`: generation of `stn_list` and `data_input`
- `create_bundle.py`: distributable bundle generator

For the full CLI options:

```bash
python run_decomposition.py --help
python run_decomposition_batch.py --help
python plot_ica_components.py --help
python plot_station_fits.py --help
python compare_with_matlab.py --help
python compare_decomposition_runs.py --help
python select_best_decomposition_runs.py --help
python compare_matlab_batch.py --help
```

Current verification state:
- quick `basic`: almost machine-precision agreement, `ICA_ts_diff=3.10e-07`
- quick `advanced + decomp_CG_means`: still almost identical, `ICA_ts_diff=1.65e-07`
- quick `decomp_srebro_CG_simultaneous + SVD_S&J`: not bit-identical, but `var_explained_*` stays within about `1e-5`

Main outputs:
- `Scenarios/casestudy/case1/python_port/output/<dataset>/all_python.npz`
- `Scenarios/casestudy/case1/python_port/output/<dataset>/all_python.mat`
- `Scenarios/casestudy/case1/python_port/output/<dataset>/summary.json`
- `Scenarios/casestudy/case1/python_port/output/<dataset>/comparison_with_matlab.json`
- `Scenarios/casestudy/case1/python_port/output/<dataset>/comparison_with_matlab.txt`
- `Scenarios/casestudy/case1/python_port/output_batch/<batch-name>/*/summary.json`
- `Scenarios/casestudy/case1/python_port/bundle_case1`
