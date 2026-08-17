# Case1 Config Guide

Italian version: [CASE1_CONFIG_GUIDE.md](CASE1_CONFIG_GUIDE.md)

This file accompanies the `case1` JSON configuration, with the complete base in [default.config.json](../../Scenarios/casestudy/case1/python_port/config/default.config.json) and an example override in [config.case1.example.json](../../Scenarios/casestudy/case1/python_port/config/config.case1.example.json), and maps the MATLAB `parameter_files` of `case1` to the Python port.

## Before You Start

- [default.config.json](../../Scenarios/casestudy/case1/python_port/config/default.config.json) is the complete base case1 configuration.
- The other `config.*.json` files in the same folder are scenario-specific overrides applied on top of `default.config.json`.
- [config.case1.example.json](../../Scenarios/casestudy/case1/python_port/config/config.case1.example.json) is a small example override, not a full dump of all defaults.
- To inspect the full effective parameter set before the run, use `icaim-run --config-file ... [--data-input-file ...] --config-only`.
- A normal run writes `effective_config.json` immediately inside the output directory and also stores `resolved_config` in `summary.json`.

## If You Are Starting From Scratch

If you do not know which options to choose, start here:

- `centering.type='basic'`
- `centering.function='empca'`
- `decompositionPCA.decomp_fcn='empca'`
- `decompositionICA.net_init='SVD'`
- `decompositionICA.n_mixed_pdfs=4`
- `decomposition_mode='t'`

Use these variants only for specific cases:

- `decomposition_mode='s'` when you want to decompose `Xd_ts.T`, useful when `n_series` is much larger than `n_epochs`.
- `centering.type='advanced'` only if you want to reproduce the legacy joint mean/decomposition branch.
- `centering.function='decomp_CG_means'` only together with `centering.type='advanced'`.
- `decompositionPCA.decomp_fcn='decomp_srebro_CG_simultaneous'` only if you want to compare against the simultaneous legacy branch and accept longer runtimes.
- `decompositionICA.net_init='SVD_S&J'` as a search alternative, not as the first default to try.

Recommended minimal override:

```json
{
  "data_input_file": "data_input_resi_ATF.txt",
  "n_components": 3,
  "decomposition_mode": "t",
  "centering": {
    "type": "basic",
    "function": "empca"
  },
  "decompositionPCA": {
    "decomp_fcn": "empca"
  },
  "decompositionICA": {
    "net_init": "SVD"
  }
}
```

## How Configuration Resolution Works

Application order:

1. `default.config.json` builds the base.
2. The JSON passed through `--config-file` applies overrides.
3. If you also pass `--data-input-file`, the CLI value wins over the JSON value.
4. Validation normalizes legacy aliases and checks compatibility constraints.

Practical consequences:

- `decomp_empca` is accepted as a legacy alias, but it is normalized to `empca`.
- `decomposition_mode` accepts `t`, `t-mode`, `temporal`, `time`, `s`, `s-mode`, `spatial`, `space`; outputs also store `decomposition_mode_resolved`.
- Unknown JSON keys make configuration loading fail.
- `repo_root` and `case_dir` are derived fields and cannot be overridden.
- Tuple-like fields must be passed as JSON arrays.
- For some ICA fields you can pass either a single value or an array of length `1` or `n_components`.
- In `icaim-run` and `compare_with_matlab.py`, `--data-input-file` is an optional override: if you omit it, the value may come from `default.config.json` or from the JSON selected with `--config-file`.

## Path Conventions

- JSON files live in `Scenarios/casestudy/case1/python_port/config`.
- `data_input_file` can be a full path, a repo-relative path, or just the file name, for example `data_input_resi_ATF.txt`.
- For `--config-file`, pass the exact JSON file name, with optional `.json`, for example `config.atf2026` or `config.case1.verify.quick.basic`.
- Short aliases such as `basic` are not accepted for `--config-file`.
- `--batch-file` and `--data-input-file` still accept short names.
- In `icaim-run-batch` and `icaim-compare-matlab-batch`, `--data-input-file` is optional when every effective batch config inherits or defines `data_input_file`; if both JSON and CLI specify it, the CLI wins.
- If the repository is moved to another PC, repo-relative paths and legacy absolute paths that contain `/Data/`, `/Scenarios/`, or `/rewrite/` are resolved against the new local repository root.

## Mapping From MATLAB parameter_files

- `scen_parameters.m`
  JSON keys: `first_epoch`, `last_epoch`, `n_components`, `threshold_ts_missingdata`, `threshold_epochs_missingdata`, `skip_epochs`, `unit_output`, `select_origin_lon`, `select_origin_lat`, `select_radius_km`, `velocity.file`, `velocity.format`.
- `centering_parameters.m`
  JSON keys: `centering.type`, `centering.function`, `centering.iter_max`, `centering.tol`, `centering.func`, `centering.dfunc`, `centering.Vimposed.*`, `centering.offsets_epoch_imposed`, `centering.Ustart`, `centering.Sstart`, `centering.Vstart`.
- `decompositionPCA_parameters.m`
  JSON keys: `decompositionPCA.decomp_fcn`, `decompositionPCA.iter_max_decomp`, `decompositionPCA.tol_decomp`, `decompositionPCA.rand_missingdata`, `decompositionPCA.rand_init`.
- `decompositionICA_parameters.m`
  JSON keys: `decompositionICA.*`, including `mix.*`, `noise.*`, `source.*`, plus the optional legacy presets.
- `flags_parameters.m`
  JSON keys: `flags.*`.
- `outliers_parameters.m`, `center_parameters_outliers.m`, `decomposition_parameters_outliers.m`
  JSON keys: `outliers.*`, `outliers.centering.*`, `outliers.decompositionPCA.*`.
- `seismicity_parameters.m`
  No active mapping in the current Python port: the `seismicity` block has been removed because it is unused in the active workflow.
- `plot_parameters.m`
  No active mapping in the current Python port: the `plot` block has been removed because it is unused in the active workflow.

## Quick Reference For Top-level Sections

Effective code defaults:

- `data_input_file`
  The CLI requires it to be defined either in the JSON or through `--data-input-file`. At internal API level there is a historical fallback to `Scenarios/casestudy/case1/dataset/data_input_file.txt`, but `icaim-run` and `compare_with_matlab.py` explicitly disable it.
- `first_epoch`
  Default: `2010.0`.
- `last_epoch`
  Default: `2019.26164336`.
- `n_components`
  Default: `2`.
- `unit_output`
  Default: `mm`.
- `skip_epochs`
  Default: `[]`.
- `threshold_ts_missingdata`
  Default: `80.0`.
- `threshold_epochs_missingdata`
  Default: `100.0`.
- `select_origin_lon`
  Default: `12.0151`.
- `select_origin_lat`
  Default: `45.9753`.
- `select_radius_km`
  Default: `95000.0`.

Practical use:

- Edit `centering`, `decompositionPCA`, and `decompositionICA` first.
- Edit `first_epoch`, `last_epoch`, `n_components`, `skip_epochs`, missing-data thresholds, and geographic selection when you are changing the dataset or the analysis window.
- `flags` and `outliers` remain compatibility-oriented sections, but they do not yet drive the main numerical flow.
- `velocity` does not alter the main `run_decomposition` flow, but it is still available mainly for the clean MATLAB/legacy path when detrending is enabled.

## Reference For centering

Effective defaults:

- `type='basic'`
- `function='empca'`
- `iter_max=1000000`
- `tol=1e-7`
- `func='func_mean_zero_sum_V_transform_corrected'`
- `dfunc='dfunc_mean_zero_sum_V_transform_corrected'`
- `offsets_epoch_imposed=[]`
- `Vimposed.type='None'`
- `Vimposed.param=[]`
- `Ustart=[]`
- `Sstart=[]`
- `Vstart=[]`

Allowed values and constraints:

- `type`
  Supported values: `basic`, `advanced`.
- `function`
  Accepted values: `empca`, `decomp_empca`, `decomp_CG_means`.
- `function`
  `decomp_empca` is a legacy alias and resolves to `empca`.
- `type='advanced'`
  Requires `function='decomp_CG_means'`.
- `Vimposed.type`
  Supported values: `None`, `Heaviside`, `Linear`, `V`.
- `offsets_epoch_imposed`
  This is a compatibility alias for defining `Heaviside` through epochs.
- `offsets_epoch_imposed`
  It is compatible only with `Vimposed.type='None'` or `Vimposed.type='Heaviside'`.
- `offsets_epoch_imposed`
  Do not use it together with `Vimposed.param` when `Vimposed.type='Heaviside'`.
- `Ustart`, `Sstart`, `Vstart`
  If you pass one of them, you must pass the other two as well.

Recommended choice:

- `basic + empca` is the robust default and the best starting point.
- `advanced + decomp_CG_means` makes sense when you want to reproduce the legacy joint mean/decomposition branch.
- `iter_max`
  Start from `1e6` and reduce it only for large parameter scans.
- `tol`
  `1e-7` is a practical compromise; smaller values may improve accuracy but can slow the run substantially.
- `Vimposed`
  `Heaviside` makes sense for coseismic steps at known epochs, `Linear` for an imposed trend, and `V` when you want to pass a full temporal matrix explicitly.

## Reference For decompositionPCA

Effective defaults:

- `decomp_fcn='empca'`
- `iter_max_decomp=500000`
- `tol_decomp=1e-7`
- `rand_missingdata=0`
- `rand_init=0`

Allowed values and constraints:

- `decomp_fcn`
  Accepted values: `empca`, `decomp_empca`, `decomp_srebro_CG_simultaneous`.
- `decomp_fcn`
  `decomp_empca` is a legacy alias and resolves to `empca`.
- `decomp_fcn`
  Actual Python implementations: `empca`, `decomp_srebro_CG_simultaneous`.
- `rand_init`
  It must remain `0`; other values are not implemented in the Python port yet.

Recommended choice:

- `empca` is the recommended default.
- `decomp_srebro_CG_simultaneous` makes sense when you want to align with the simultaneous legacy branch and accept longer runtimes.
- `tol_decomp`
  Start from `1e-7`; go smaller only if you have a clear accuracy reason.

## Reference For decompositionICA

Effective defaults:

- `source_type='g'`
- `learning_percent=100`
- `ICA_num=1`
- `n_mixed_pdfs=4`
- `states=null`
- `mix_prior_preset=null`
- `source_prior_preset=null`
- `net_init='SVD'`
- `source_init='kmeans'`
- `max_steps=500`
- `isonoise=1`
- `ARD=1`
- `tol=1e-8`
- `eta=1.0`
- `mix.b_alpha_0=1e3`
- `mix.c_alpha_0=1e-3`
- `noise.b_Lam_0=1e1`
- `noise.c_Lam_0=1e-1`
- `noise.mb0=1.0`
- `noise.mn0=0.0`
- `source.m_0=0.0`
- `source.tau_0=1.0`
- `source.b_0=1e1`
- `source.c_0=1e-1`
- `source.lambda_0=null`
- `source.setSource=1`

Allowed values and constraints:

- `source_type`
  Today the Python port implements only `g`.
- `ICA_num`
  Today the Python port implements only `1`.
- `source_init`
  Today the Python port implements only `kmeans`.
- `net_init`
  Implemented values: `SVD`, `SVD_S&J`.
- `mix_prior_preset`
  Supported values: `legacy_r1`, `legacy_r2`, `legacy_r3`, `legacy_r4`.
- `source_prior_preset`
  Supported values: `legacy_o1`, `legacy_o2`, `legacy_o3`, `legacy_o4`.
- `states`
  If left as `null`, the port uses `n_mixed_pdfs` for each component.
- `n_mixed_pdfs`, `states`, `mix.*`, `source.m_0`, `source.tau_0`, `source.b_0`, `source.c_0`, `source.lambda_0`
  You can pass either a single value or an array of length `1` or `n_components`.
- `source.lambda_0`
  If left as `null`, the port uses a legacy rule that depends on the vbICA sample count: `n_epochs` in `t-mode`, `n_series` in `s-mode`.

Available legacy presets:

- `source_prior_preset='legacy_o1'`
  Resolves to `b_0=1e3`, `c_0=1e-3`.
- `source_prior_preset='legacy_o2'`
  Resolves to `b_0=1e1`, `c_0=1e-1`.
- `source_prior_preset='legacy_o3'`
  Resolves to `b_0=1e-1`, `c_0=1e1`.
- `source_prior_preset='legacy_o4'`
  Resolves to `b_0=1e-3`, `c_0=1e3`.
- `mix_prior_preset='legacy_r1'`
  Resolves to `b_alpha_0=1e5`, `c_alpha_0=1e-1`.
- `mix_prior_preset='legacy_r2'`
  Resolves to `b_alpha_0=1e1`, `c_alpha_0=1e-1`.
- `mix_prior_preset='legacy_r3'`
  Resolves to `b_alpha_0=1e-1`, `c_alpha_0=1e1`.
- `mix_prior_preset='legacy_r4'`
  Resolves to `b_alpha_0=1e-3`, `c_alpha_0=1e3`.

Recommended choice:

- `net_init='SVD'` is the most stable default.
- `net_init='SVD_S&J'` is useful as a search alternative, but always compare `summary.json`.
- `n_mixed_pdfs=4` is the first value to keep for legacy comparisons.
- For `mix.*` and `source.*`, it is usually better to explore orders of magnitude such as `1e5`, `1e3`, `1e1`, `1e-1`, `1e-3` rather than very fine local perturbations.

## Compact Reference For Support Sections

These sections exist in the shared configuration model and are visible in the complete base file [default.config.json](../../Scenarios/casestudy/case1/python_port/config/default.config.json). Today they do not drive the main numerical flow of `run_decomposition`, but they are still valid configuration keys.

- `velocity`
  Keys: `file`, `format`.
- `flags`
  Keys: `flag_detrend`, `flag_disp`, `flag_decomp_err`, `flag_fault_model`, `flag_load_rand_guess`, `flag_load_rand`, `mixnet_flag`, `flag_visible`, `flag_whitening`, `flag_plot_ts`, `flag_plot_ts_offsets`, `flag_rm_offsets`, `flag_ICA_decomp`, `flag_invert_ICs`, `flag_invert_offsets`.
- `outliers`
  Keys: `blunder_threshold_unit`, `blunder_threshold_hor`, `blunder_threshold_ver`, `outlier_threshold`, `centering.*`, `decompositionPCA.*`.
- `outliers.centering`
  Keys: `type`, `function`, `n_components`, `n_comp_mean`, `iter_max`, `tol`, `func`, `dfunc`, `Vimposed.*`.
- `outliers.decompositionPCA`
  Keys: `iter_max_decomp`, `tol_decomp`, `decomp_fcn`, `rand_missingdata`, `rand_init`.

## Sections Documented But Not Yet Operational In The Main Flow

- `flags`
  They are accepted and stored in the output, but they do not yet alter the Python computational flow the way they do in legacy MATLAB.
- `outliers`
  Exposed to cover the `parameter_files` surface, but the dedicated outlier preprocessing is not yet implemented in the Python port.
- `velocity`
  It does not enter the main `run_decomposition` flow, but it is still kept in the configuration for compatibility with the clean MATLAB/legacy path when detrending is enabled.

## Recommended Cross-checks

- When comparing decompositions built from the same dataset and preprocessing, inspect `summary.json` and `quality_metrics` first.
- For multi-run comparisons, use `compare_decomposition_runs.py` on the generated `all_python.npz` files.
- `resolved_config` is the best reference when you want to see normalized names, resolved presets, and the per-component parameters actually used.
- With `decomposition_mode='s'`, `PCA_U/ICA_U` and `PCA_V/ICA_V` are mapped back to the historical `(n_series,k)` and `(n_epochs,k)` shapes; ICA independence is nevertheless estimated in the transposed problem.
- `ICA_energy` and `ARD_ratio` are meaningful only across comparable runs; `variance_explained_*` and `reduced_chi2_*` remain the most direct fit metrics.
