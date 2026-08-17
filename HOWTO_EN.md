# ICAIM Python Port Operational How-To

Italian version: `HOWTO.md`

This file is the end-to-end practical guide for the Python port, starting from a folder that contains `tseri` data files.

If you need to understand which document to read:

- `README_EN.md`
  Quick map of the `python_port` directory: what is there, where config, input, output, and wrappers live.
- `HOWTO_EN.md`
  Operational procedure from start to finish: dataset preparation, runs, plots, batch search, comparisons, and bundle creation.
- `CASE1_CONFIG_GUIDE_EN.md`
  Meaning of the JSON fields and mapping from the MATLAB `parameter_files`.
- `../MANUAL_REWRITE_EN.md`
  General rewrite manual: architecture, MATLAB and Python workflows, limits, and design rationale.

## 1. Prerequisites

Full repository:

- root: `/path/to/ICAIM`
- Python code: `/path/to/ICAIM/rewrite/python_port`
- JSON config: `/path/to/ICAIM/Scenarios/casestudy/case1/python_port/config`
- dataset descriptors: `/path/to/ICAIM/Scenarios/casestudy/case1/dataset`

Typical setup:

```bash
cd /path/to/ICAIM/rewrite/python_port
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want to use wrappers from any directory:

```bash
export PATH="/path/to/ICAIM/rewrite/python_port/bin:$PATH"
```

Practical wrapper note:

- `icaim-run`, `icaim-run-batch`, `icaim-plot-components`, `icaim-plot-station-fits`, `icaim-compare-matlab`, `icaim-compare-matlab-batch`, `icaim-compare-runs`, `icaim-select-runs`, `icaim-create-bundle` point to scripts inside `rewrite/python_port`
- `icaim-build-dataset-files` points to `rewrite/build_dataset_files.py` in the full repository
- `icaim-compare-results` points to `rewrite/compare_outputs.py` in the full repository
- in the distributed bundle, `build_dataset_files.py` and `compare_outputs.py` are copied into the bundle root, so the wrapper is the most stable form to document

## 2. Start from a data folder

Assume you have a folder with `tseri` files, for example:

```bash
/path/to/ICAIM/Data/resi_ATF2026
```

To generate the `stn_list` and `data_input` files:

```bash
icaim-build-dataset-files --series-dir /path/to/ICAIM/Data/resi_ATF2026 --data-type GPS2
```

By default this creates:

- `Scenarios/casestudy/case1/dataset/stn_list/resi_ATF2026.txt`
- `Scenarios/casestudy/case1/dataset/data_input_resi_ATF2026.txt`

Available options:

- `--series-dir`
  Required. Folder containing `tseri` files.
- `--list-name`
  Station-list filename. Default: `<folder_name>.txt`.
- `--data-input-name`
  Data-input filename. Default: `data_input_<folder_name>.txt`.
- `--unit-input`
  Deformation unit written into the generated `data_input`. Default: `mm`.
- `--data-type`
  GPS type written into the generated `data_input`: `GPS1`, `GPS2`, `GPS3`. Default: `GPS3`.
- `--operation`
  Operation field written into the generated `data_input`. Default: `decomp`.
- `--activate`
  Also updates `dataset/data_input_file.txt`.
- `--backup-active`
  When used with `--activate`, keeps a `.bak` copy of the previous `data_input_file.txt`.

Important note:

- `icaim-run` no longer uses `data_input_file.txt` as an implicit default
- `--activate` is mostly for legacy workflow compatibility or to keep track of the historical active dataset
- for Python runs you must pass `--data-input-file` only when the effective config does not already define `data_input_file`
- in the Python port, `data_input` can use `GPS1` for `Up`, `GPS2` for `East/North`, and `GPS3` for `East/North/Up`
- the clean MATLAB rewrite remains `GPS3`-only, so Python/MATLAB comparisons are still limited for now to `GPS3` datasets

## 3. Choose the configuration file

JSON configurations live in:

```text
Scenarios/casestudy/case1/python_port/config
```

Useful examples:

- `default.config.json`
- `config.case1.verify.quick.basic.json`
- `config.case1.verify.quick.advanced.json`
- `config.case1.verify.quick.srebro.json`
- `config.case1.example.json`
- `batch.case1.search.example.json`

Commands accept three forms:

- absolute path
- repository-relative path
- exact config file name, with optional `.json`

Equivalent examples:

```bash
--config-file /path/to/ICAIM/Scenarios/casestudy/case1/python_port/config/config.atf2026.json
--config-file Scenarios/casestudy/case1/python_port/config/config.atf2026.json
--config-file config.atf2026
```

For `--config-file`, do not use opaque aliases such as `basic`: the CLI only accepts the full JSON file name.

Important: `Scenarios/casestudy/case1/python_port/config/default.config.json` is the complete base configuration. The JSON passed through `--config-file` is applied as an override on top of that base.

To understand what the JSON actually changes, use `CASE1_CONFIG_GUIDE_EN.md`.

## 4. Run a single decomposition

Recommended command when the JSON already contains `data_input_file`:

```bash
icaim-run --config-file config.atf2026
```

If the JSON is only an override and does not contain `data_input_file`, pass the dataset explicitly:

```bash
icaim-run --config-file config.case1.verify.quick.basic --data-input-file resi_ATF2026
```

To inspect the full effective parameter set before computation:

```bash
icaim-run --config-file config.case1.verify.quick.basic --data-input-file resi_ATF2026 --config-only
```

If you also want the full JSON dump on stdout, add `--print-effective-config`.

Equivalent script form:

```bash
python /path/to/ICAIM/rewrite/python_port/run_decomposition.py \
  --config-file config.atf2026
```

At startup the CLI prints:

- `config_file=...`
- `data_input_file=...`
- `dataset_label=...`
- `output_dir=...`
- `effective_config_file=...`

At startup `icaim-run` also writes `effective_config.json` inside the output directory, before the decomposition begins.

The same information is also stored in `summary.json` under `run_metadata`.

If both the JSON and the CLI specify the dataset, `--data-input-file` wins.

Default output:

```text
Scenarios/casestudy/case1/python_port/output/<dataset>
```

Main generated files:

- `effective_config.json`
- `all_python.npz`
- `all_python.mat`
- `summary.json`

Dedicated guide for the contents of `all_python.npz`:

- `ALL_PYTHON_NPZ_GUIDE_EN.md`

Available options for `icaim-run` / `run_decomposition.py`:

- `--repo-root`
  ICAIM repository root.
- `--output-dir`
  Run output directory. Default: `Scenarios/casestudy/case1/python_port/output/<dataset>`.
- `--data-input-file`
  Optional if the JSON defines `data_input_file`. Otherwise it is required.
- `--config-file`
  Required. JSON override file on top of the code-defined base config; accepts the full file name, with optional `.json`.
- `--print-effective-config`
  Print the full effective configuration JSON to stdout before the computation starts.
- `--config-only`
  Resolve the effective configuration, write `effective_config.json`, then exit without running the decomposition.
- `--make-plots`
  Also generate component plots after the run.
- `--plot-output-dir`
  Component-plot directory. Default: `<output-dir>/plots`.
- `--plot-normalization`
  One of `peak-to-peak`, `unit-max`, `none`. Default: `peak-to-peak`.
- `--decomposition`
  One of `ICA`, `PCA`. Default: `ICA`.
- `--background-grid`
  Optional GMT/NetCDF background grid. Use `auto` or `none`.
- `--components`
  Optional 1-based component indices to plot when using `--make-plots`. Default: all components.
- `--label-stations`
  Annotate station names on component maps when using `--make-plots`.
- `--dpi`
  PNG resolution for generated plots. Default: `200`.
- `--prefix`
  Optional component-plot filename prefix. Default: `IC` for ICA and `PC` for PCA.
- `--make-station-fit-plots`
  Also generate observed-vs-modeled station plots.
- `--station-fit-output-dir`
  Station-plot directory. Default: `<output-dir>/station_fits`.
- `--hide-station-components`
  Hide individual component contribution curves in station-fit plots.
- `--stations`
  Optional station-code list for station-fit plots generated inside `icaim-run`, for example `--stations ANCG AT01`.

Example with plots included:

```bash
icaim-run \
  --config-file config.atf2026 \
  --make-plots \
  --label-stations \
  --make-station-fit-plots
```

## 5. Generate plots from an existing output

Component plots:

```bash
icaim-plot-components --results-file resi_ATF2026
```

Available options for `icaim-plot-components` / `plot_ica_components.py`:

- `--repo-root`
  Repository root.
- `--results-mat`
  Historical alias for the `.npz` or `.mat` result file.
- `--results-file`
  Result file, output directory, or dataset name. Example: `resi_ATF2026`.
- `--decomposition`
  One of `ICA`, `PCA`. Default: `ICA`.
- `--output-dir`
  Plot output directory. Default: `<results-dir>/plots`.
- `--components`
  Optional 1-based component indices. Default: all.
- `--normalization`
  One of `peak-to-peak`, `unit-max`, `none`. Default: `peak-to-peak`.
- `--background-grid`
  Optional background grid. Use `auto` or `none`.
- `--label-stations`
  Annotate station names on the map.
- `--dpi`
  PNG resolution. Default: `200`.
- `--prefix`
  Output filename prefix. Default: `IC` for ICA and `PC` for PCA.

Station-fit plots:

```bash
icaim-plot-station-fits --results-file resi_ATF2026
```

Available options for `icaim-plot-station-fits` / `plot_station_fits.py`:

- `--repo-root`
  Repository root.
- `--results-file`
  Result file, output directory, or dataset name.
- `--output-dir`
  Plot output directory. Default: `<results-dir>/station_fits`.
- `--decomposition`
  One of `ICA`, `PCA`. Default: `ICA`.
- `--stations`
  Optional list of station codes, for example `--stations ANCG AT01`.
- `--hide-components`
  Show only observed vs total modeled signal.

## 6. Run a batch campaign

Recommended command:

```bash
icaim-run-batch --batch-file example --data-input-file resi_ATF2026
```

If the batch `base_config_file` or the effective overrides already define `data_input_file`, you can omit it:

```bash
icaim-run-batch --batch-file batch.case1.search.atf2026_season.json
```

Equivalent script form:

```bash
python /path/to/ICAIM/rewrite/python_port/run_decomposition_batch.py \
  --batch-file batch.case1.search.example.json \
  --data-input-file resi_ATF2026
```

Default output:

```text
Scenarios/casestudy/case1/python_port/output_batch/<batch-name>
```

The batch workflow:

- expands the configuration grid
- filters incompatible combinations
- creates one output directory per run
- writes a `batch_summary.json`

Available options for `icaim-run-batch` / `run_decomposition_batch.py`:

- `--repo-root`
  Repository root.
- `--batch-file`
  Required. Batch JSON with run grid and shared overrides.
- `--output-dir`
  Batch output directory. Default: `Scenarios/casestudy/case1/python_port/output_batch/<batch-name>`.
- `--data-input-file`
  Optional if every effective batch run config defines `data_input_file`. Otherwise it is required.
- `--make-plots`
  Generate component plots for every completed run.
- `--plot-output-dir`
  Root directory for component plots.
- `--plot-normalization`
  One of `peak-to-peak`, `unit-max`, `none`. Default: `peak-to-peak`.
- `--decomposition`
  One of `ICA`, `PCA`. Default: `ICA`.
- `--background-grid`
  Optional background grid. Use `auto` or `none`.
- `--make-station-fit-plots`
  Generate station-fit plots for every completed run.
- `--station-fit-output-dir`
  Root directory for station-fit plots.
- `--hide-station-components`
  Hide component curves in station-fit plots.

## 7. Compare Python and MATLAB

Comparison against the historical `all.mat` baseline:

```bash
icaim-compare-matlab --config-file config.case1.verify.quick.basic --data-input-file resi_ATF2026
```

Available options for `icaim-compare-matlab` / `compare_with_matlab.py`:

- `--repo-root`
  Repository root.
- `--reference`
  MATLAB `.mat` reference file.
- `--output-dir`
  Output directory. Default: `Scenarios/casestudy/case1/python_port/output/<dataset>`.
- `--data-input-file`
  Optional if the JSON defines `data_input_file`. Otherwise it is required.
- `--config-file`
  Required. JSON configuration; accepts the full file name, with optional `.json`.

Important note:

- this workflow also goes through the clean MATLAB rewrite
- so the dataset must still be `GPS3` here

Python vs MATLAB on the same config or batch:

```bash
icaim-compare-matlab-batch --config-file config.case1.verify.quick.basic --data-input-file resi_ATF2026
```

Or:

```bash
icaim-compare-matlab-batch --batch-file example --data-input-file resi_ATF2026
```

If the single config or the batch already define `data_input_file`, you can omit it:

```bash
icaim-compare-matlab-batch --batch-file batch.case1.search.atf2026_season.json
```

Available options for `icaim-compare-matlab-batch` / `compare_matlab_batch.py`:

- `--repo-root`
  Repository root.
- `--config-file`
  JSON config for a single run.
- `--batch-file`
  Batch JSON for multiple runs.
- `--output-dir`
  Output directory. Default: `Scenarios/casestudy/case1/python_port/output_compare_batch/<config-or-batch-name>`.
- `--matlab-binary`
  Path to the MATLAB executable.
- `--data-input-file`
  Optional if every effective config defines `data_input_file`. Otherwise it is required.

Note:

- for `icaim-compare-matlab-batch` you must pass either `--config-file` or `--batch-file`
- `--data-input-file` is optional only if every effective config defines `data_input_file`
- this workflow is also still limited, for now, to `GPS3`, because the clean MATLAB side does not yet support `GPS1/GPS2`

## 8. Compare different runs

Compact comparison across multiple outputs:

```bash
icaim-compare-runs \
  /path/to/ICAIM/Scenarios/casestudy/case1/python_port/output/resi_ATF2026/all_python.npz \
  /path/to/ICAIM/Scenarios/casestudy/case1/python_port/output_batch/batch.case1.search.example
```

Available options for `icaim-compare-runs` / `compare_decomposition_runs.py`:

- positional `inputs`
  Result files or directories. Directories are scanned recursively for `all_python.npz`.
- `--sort-by`
  One of `n_components`, `variance_explained_ICA`, `reduced_chi2_ICA`, `ICA_energy`, `path`.
- `--output-json`
  Optional JSON report file.

Automatic best-run selection:

```bash
icaim-select-runs \
  /path/to/ICAIM/Scenarios/casestudy/case1/python_port/output_batch/batch.case1.search.example \
  --output-json /tmp/icaim_select_report.json \
  --output-markdown /tmp/icaim_select_report.md
```

Practical difference relative to `icaim-compare-runs`:

- `icaim-compare-runs` summarizes and compares all runs, but does not pick a winner
- `icaim-select-runs` starts from the same outputs and produces a recommendation: best run for each `n_components`, preferred choice for each configuration family, and one final recommended run

Available options for `icaim-select-runs` / `select_best_decomposition_runs.py`:

- positional `inputs`
  Result files or directories. Directories are scanned recursively for `all_python.npz`.
- `--output-json`
  Optional JSON file with the full selection report.
- `--output-markdown`
  Optional Markdown file with a readable report.

Direct pairwise comparison between two files:

```bash
icaim-compare-results \
  --a /path/to/run_a/all_python.npz \
  --b /path/to/run_b/all_python.npz \
  --label-a run_a \
  --label-b run_b
```

Available options for `icaim-compare-results` / `compare_outputs.py`:

- `--a`
  First `.npz` or `.mat` file.
- `--b`
  Second `.npz` or `.mat` file.
- `--label-a`
  Short label for the first file.
- `--label-b`
  Short label for the second file.

## 9. Understand where outputs go

Single run:

- `Scenarios/casestudy/case1/python_port/output/<dataset>/effective_config.json`
- `Scenarios/casestudy/case1/python_port/output/<dataset>/all_python.npz`
- `Scenarios/casestudy/case1/python_port/output/<dataset>/all_python.mat`
- `Scenarios/casestudy/case1/python_port/output/<dataset>/summary.json`

Batch:

- `Scenarios/casestudy/case1/python_port/output_batch/<batch-name>/`
- `Scenarios/casestudy/case1/python_port/output_batch/<batch-name>/batch_summary.json`

Python/MATLAB comparison:

- `Scenarios/casestudy/case1/python_port/output_compare_batch/<config-or-batch>/`

Inside `summary.json` you always find:

- the effective configuration
- the effective dataset used
- fit metrics
- `run_metadata` with the resolved paths also printed by the CLI

Inside the output directory you also always find `effective_config.json`, written at the beginning of the run and reusable as a fully explicit config file.

## 10. Move the code to another PC

In the full repository:

1. copy the whole repository while keeping the `Data/`, `Scenarios/`, and `rewrite/` layout
2. recreate the virtual environment
3. reinstall `requirements.txt`
4. add the wrapper `PATH` again if you want to run them from any directory

In the bundle:

1. enter `bundle_case1/python_port`
2. create the virtual environment
3. install `requirements.txt`
4. run either the Python scripts or the bundled wrappers

Legacy `data_input` and `stn_list` files that contain absolute paths with `/Data/` or `/Scenarios/` segments are remapped automatically against the local repository or bundle root.

## 11. Create a distributable bundle

Command:

```bash
icaim-create-bundle
```

Equivalent:

```bash
python /path/to/ICAIM/rewrite/python_port/create_bundle.py
```

Available options for `icaim-create-bundle` / `create_bundle.py`:

- `--repo-root`
  Repository root.
- `--output-dir`
  Bundle destination. Default: `Scenarios/casestudy/case1/python_port/bundle_case1`.

## 12. Minimal commands to remember

If you have a new data folder:

```bash
icaim-build-dataset-files --series-dir /path/to/ICAIM/Data/my_dataset
icaim-run --config-file config.case1.verify.quick.basic --data-input-file my_dataset
icaim-plot-components --results-file my_dataset
icaim-plot-station-fits --results-file my_dataset
```

If you want to search across multiple configurations:

```bash
icaim-run-batch --batch-file example --data-input-file my_dataset
icaim-compare-runs Scenarios/casestudy/case1/python_port/output_batch/batch.case1.search.example
icaim-select-runs Scenarios/casestudy/case1/python_port/output_batch/batch.case1.search.example
```

If you want to compare with MATLAB:

```bash
icaim-compare-matlab --config-file config.case1.verify.quick.basic --data-input-file my_dataset
icaim-compare-matlab-batch --config-file config.case1.verify.quick.basic --data-input-file my_dataset
```
