# Guide To The Variables In `all_python.npz`

`all_python.npz` is the compressed NumPy archive produced by the case 1 Python pipeline.
It stores:

- observed and reconstructed matrices
- PCA and ICA factorizations
- flat metrics for batch comparisons
- serialized JSON payloads with configuration and structured metrics

This guide explains what each variable means, how to read its shape and indexing, and how the layout changes in the `GPS1`, `GPS2`, and `GPS3` cases.

## Notation

In this guide:

- `n_stations` = number of GPS stations
- `component_size` = number of observed components per station
- `n_series = n_stations * component_size`
- `n_epochs` = number of time epochs
- `k` = number of PCA/ICA components requested through `n_components`

## Data Layout For `GPS1`, `GPS2`, `GPS3`

The dataset type determines how many observed series each station contributes and in which order they are stored in the rows of every matrix with shape `(n_series, ...)`.

| Dataset type | Components per station | `component_size` | Row order per station |
| --- | --- | ---: | --- |
| `GPS1` | vertical only | `1` | `u` |
| `GPS2` | horizontal plane | `2` | `e`, `n` |
| `GPS3` | full 3D | `3` | `e`, `n`, `u` |

So:

- in `GPS1`, row `s` is the `u` component of station `s`
- in `GPS2`, rows `2*s` and `2*s+1` are `e` and `n`
- in `GPS3`, rows `3*s`, `3*s+1`, `3*s+2` are `e`, `n`, `u`

Examples:

- `Xd_name`: `ANCGe`, `ANCGn`, `ANCGu`, `ANCNe`, `ANCNn`, `ANCNu`, ...
- `Xd_type`: `GPS3e`, `GPS3n`, `GPS3u`, `GPS3e`, `GPS3n`, `GPS3u`, ...
- `STATIONS_name`: `ANCG`, `ANCN`, `ARCE`, ...

The same row convention applies to:

- `Xd_ts`, `Xd_var_ts`, `Xd_llh`
- `PCA_U`, `PCA_ts`
- `ICA_U`, `ICA_ts`
- `A_recon`, `var_A_recon`
- `data_mask`

## How To Read `ICA_U`, `ICA_S`, `ICA_V`

This is the most important point to avoid confusion.

The columns do not represent `East`, `North`, `Up`.
The columns represent ICA components.

For each component `j` with `0 <= j < k`:

- `ICA_U[:, j]` is the spatial pattern of component `j` over all observed series
- `ICA_S[j, j]` is the scaling factor of component `j`
- `ICA_V[:, j]` is the time series of component `j`

The reconstruction of component `j` alone is:

```text
ICA_component_j = ICA_U[:, j] * ICA_S[j, j] * ICA_V[:, j]^T
```

The full reconstruction is:

```text
ICA_ts = ICA_U @ ICA_S @ ICA_V.T
```

Correct interpretation:

- column `j=0` is not "East"
- column `j=1` is not "North"
- column `j=2` is not "Up"
- `e`, `n`, `u` live in the rows of `ICA_U`, not in the columns

So, to isolate the spatial contribution of ICA component `j`:

- `GPS1`: `up = ICA_U[:, j]`
- `GPS2`: `east = ICA_U[0::2, j]`, `north = ICA_U[1::2, j]`
- `GPS3`: `east = ICA_U[0::3, j]`, `north = ICA_U[1::3, j]`, `up = ICA_U[2::3, j]`

## Component Ordering And Sign

ICA and PCA components are sorted in descending order of the diagonal values of `S`.
So:

- the first column is the component with the largest scale in `S`
- it is not a physical direction
- the sign of `U` and `V` can be flipped together without changing the reconstruction

For ICA, the file stores a normalized version:

- `ICA_U` comes from `A_recon` after column-wise normalization
- `ICA_V` comes from `S_recon` after row-wise normalization
- `ICA_S` collects the norms moved out of `A_recon` and `S_recon`

So:

- `A_recon`, `S_recon` = raw VBICA factors
- `ICA_U`, `ICA_S`, `ICA_V` = normalized and sorted factorization used for analysis and plotting

## Main Variables

### Observed data `Xd_*`

| Variable | Shape | Meaning | Notes |
| --- | --- | --- | --- |
| `Xd_ts` | `(n_series, n_epochs)` | Observed series after filtering and centering | Each row is one observed component of one station |
| `Xd_var_ts` | `(n_series, n_epochs)` | Variance associated with `Xd_ts` | Missing values are encoded as `inf` |
| `Xd_timeline` | `(n_epochs,)` | Common time axis | Same epochs used by PCA and ICA |
| `Xd_llh` | `(n_series, 3)` | Longitude, latitude, height for each series row | Coordinates are repeated across `e/n/u` rows of the same station |
| `Xd_centering_offsets` | `(n_series,)` | Offsets removed during centering | Add them back to recover the original level |
| `Xd_name` | `(n_series,)` | Full series name | Example `ANCGe`, `ANCNn`, `ARCGu` |
| `Xd_type` | `(n_series,)` | Full series type | Example `GPS3e`, `GPS2n`, `GPS1u` |

### PCA

| Variable | Shape | Meaning | Notes |
| --- | --- | --- | --- |
| `PCA_U` | `(n_series, k)` | PCA spatial patterns | Rows follow the `GPS1/2/3` layout |
| `PCA_S` | `(k, k)` | Diagonal matrix of PCA scales | Diagonal sorted in descending order |
| `PCA_V` | `(n_epochs, k)` | PCA temporal patterns | Column `j` is the time series of component `j` |
| `PCA_ts` | `(n_series, n_epochs)` | PCA reconstruction in data space | `PCA_U @ PCA_S @ PCA_V.T` |
| `PCA_decomposition_mode` | scalar string | PCA decomposition mode | `t` or `s`; in `s`, factors are mapped back to the historical shapes |

### Normalized ICA

| Variable | Shape | Meaning | Notes |
| --- | --- | --- | --- |
| `ICA_U` | `(n_series, k)` | ICA spatial patterns | Columns are ICA components, not `e/n/u` directions |
| `ICA_S` | `(k, k)` | Diagonal matrix of ICA scales | Components are sorted by descending diagonal value |
| `ICA_V` | `(n_epochs, k)` | ICA temporal patterns | Column `j` is the time history of component `j` |
| `ICA_ts` | `(n_series, n_epochs)` | ICA reconstruction in data space | `ICA_U @ ICA_S @ ICA_V.T` |
| `ICA_llh` | `(n_series, 3)` | Coordinates associated with the ICA solution | Normally the same as `Xd_llh` |
| `ICA_timeline` | `(n_epochs,)` | Time axis associated with the ICA solution | Normally the same as `Xd_timeline` |
| `ICA_name` | `(n_series,)` | Series names associated with ICA | Normally the same as `Xd_name` |
| `ICA_type` | `(n_series,)` | Series types associated with ICA | Normally the same as `Xd_type` |
| `ICA_decomposition_mode` | scalar string | ICA decomposition mode | `t` or `s`; in `s`, vbICA is estimated on the transposed matrix |

### Raw VBICA factors

| Variable | Shape | Meaning | Notes |
| --- | --- | --- | --- |
| `A_recon` | `(n_series, k)` | Raw mixing matrix from the ICA network | Before the normalization that produces `ICA_U` |
| `S_recon` | `(k, n_epochs)` | Raw reconstructed sources from the ICA network | Before the normalization that produces `ICA_V` |
| `var_A_recon` | `(n_series, k)` | Posterior variance of `A_recon` | Derived from posterior precisions |
| `var_S_recon` | `(k, n_epochs)` | Posterior variance of `S_recon` | Derived from posterior precisions |

Note: in `decomposition_mode='s'`, these raw factors are the coherent transpose of the internal vbICA problem; they keep the historical shapes for compatibility with analysis, metrics, and plots.

### Missing data and station metadata

| Variable | Shape | Meaning | Notes |
| --- | --- | --- | --- |
| `data_mask` | `(n_series, n_epochs)` | Valid-data mask | `1` = present datum, `0` = missing datum |
| `ind_missing_data` | `(n_missing,)` | Flat indexes of missing entries | 1-based indexes, kept this way for MATLAB compatibility |
| `STATIONS_name` | `(n_stations,)` | Station codes without component suffix | One entry per station, not per series |

## Flat Metrics

These keys are meant for quick run comparison and are also reused by the batch scripts.

| Variable | Shape | Meaning | Notes |
| --- | --- | --- | --- |
| `chi2_PCA` | scalar | Total chi-square of the PCA reconstruction | |
| `chi2_ICA` | scalar | Total chi-square of the ICA reconstruction | |
| `variance_explained_PCA` | scalar | Variance explained by PCA | In percent |
| `variance_explained_ICA` | scalar | Variance explained by ICA | In percent |
| `ard` | `(k,)` | Normalized ARD weights | Derived from ICA `alpha` values |
| `n_observations` | scalar integer | Number of valid observations | Missing data excluded |
| `missing_data_fraction` | scalar | Fraction of missing data | Computed from `Xd_var_ts` |
| `reduced_chi2_PCA` | scalar | PCA reduced chi-square | |
| `reduced_chi2_ICA` | scalar | ICA reduced chi-square | |
| `weighted_rms_PCA` | scalar | PCA weighted RMS | |
| `weighted_rms_ICA` | scalar | ICA weighted RMS | |
| `chi2_gain_ICA_vs_PCA_pct` | scalar | Percent chi-square improvement of ICA over PCA | Larger is better |
| `var_explained_gain_ICA_vs_PCA` | scalar | Variance-explained improvement of ICA over PCA | |
| `ICA_energy` | scalar | Final ICA network energy | Usually matches `ICA_net_energy` |
| `ICA_iterations` | scalar integer | Number of ICA iterations | Usually equals `len(ICA_net_energy_path)` |
| `ARD_ratio` | scalar | `max(alpha) / min(alpha)` | Large values suggest too many components |

## PCA Fit And ICA Network Diagnostics

| Variable | Shape | Meaning | Notes |
| --- | --- | --- | --- |
| `PCA_fit_method` | string scalar | PCA method used | Example `empca`, `srebro` |
| `PCA_fit_iterations` | integer scalar | Number of PCA fit iterations | |
| `PCA_fit_objective_name` | string scalar | Name of the optimized objective | Example `chi2`, `residual` |
| `PCA_fit_objective_final` | scalar | Final objective value | |
| `PCA_fit_objective_path` | `(n_iter+1,)` or absent | Full PCA objective trace | Present for `empca` |
| `ICA_net_energy` | scalar | Final ICA network energy | Useful direct-access duplicate |
| `ICA_net_energy_path` | `(n_iter,)` | ICA energy history during training | Useful to inspect convergence |
| `ICA_net_alphas` | `(k,)` | Raw ARD hyperparameters from the ICA network | `ard` is derived from these |

## Serialized JSON Payloads

These keys contain JSON strings, not numeric arrays.

| Variable | Type | Meaning | Notes |
| --- | --- | --- | --- |
| `config_json` | JSON string | Initial run configuration | Contains the main parameters requested at startup |
| `resolved_config_json` | JSON string | Effective resolved configuration | Also includes code-resolved fields such as `seismicity_resolved` |
| `quality_metrics_json` | JSON string | Full structured quality metrics | Contains the `data`, `PCA`, and `ICA` blocks |

## What Changes Between `GPS1`, `GPS2`, `GPS3`

The keys in the file stay the same.
What changes is:

- `component_size`
- `n_series`
- the row ordering inside spatially indexed matrices
- how you must read one column of `U`

### `GPS1`

- `component_size = 1`
- only the `u` component exists
- `n_series = n_stations`
- one column `ICA_U[:, j]` already gives one vertical value per station

### `GPS2`

- `component_size = 2`
- per-station row order is `e`, `n`
- `n_series = 2 * n_stations`
- one column `ICA_U[:, j]` must be split as:

```text
east  = ICA_U[0::2, j]
north = ICA_U[1::2, j]
```

### `GPS3`

- `component_size = 3`
- per-station row order is `e`, `n`, `u`
- `n_series = 3 * n_stations`
- one column `ICA_U[:, j]` must be split as:

```text
east  = ICA_U[0::3, j]
north = ICA_U[1::3, j]
up    = ICA_U[2::3, j]
```

The same rule applies to `PCA_U`, `A_recon`, `var_A_recon`, `Xd_ts`, `ICA_ts`, `data_mask`, and every other matrix indexed by observed series.

## Always-Present And Conditional Keys

Normally always present:

- all `Xd_*` fields
- all `PCA_*` and `ICA_*` fields listed above
- `A_recon`, `S_recon`, `var_A_recon`, `var_S_recon`
- `data_mask`, `ind_missing_data`, `STATIONS_name`
- the main flat metrics
- `config_json`, `resolved_config_json`, `quality_metrics_json`

Potentially conditional:

- `PCA_fit_objective_path`: present for some PCA methods, especially `empca`
- `ICA_net_energy`, `ICA_net_energy_path`, `ICA_net_alphas`: present when the ICA solution stores the full network diagnostics

## Quick Mental Model

- columns of `U` = components
- columns of `V` = components
- rows of `U` = observed series
- rows of `Xd_ts` and `ICA_ts` = observed series
- `e/n/u` depend on dataset type and live in the rows
- `GPS1 -> u`
- `GPS2 -> e, n`
- `GPS3 -> e, n, u`
- `A_recon` and `S_recon` are the raw factors
- `ICA_U`, `ICA_S`, `ICA_V` are the normalized and sorted form
