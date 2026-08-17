from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass, replace
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat

from .legacy_algorithms import (
    build_imposed_v,
    center_advanced_legacy,
    center_basic_legacy,
    decomp_srebro_cg_simultaneous,
)


EPS = np.finfo(float).eps
CASE1_RELATIVE_DIR = Path("Scenarios") / "casestudy" / "case1"
CASE1_PYTHON_PORT_DIRNAME = "python_port"
CASE1_PYTHON_CONFIG_DIRNAME = "config"
DEFAULT_CASE1_CONFIG_FILENAME = "default.config.json"
DEFAULT_CONFIG_BOOTSTRAP_DATA_INPUT_NAME = "__default_config_bootstrap__.txt"
SUPPORTED_GPS_COMPONENTS: dict[str, tuple[str, ...]] = {
    "GPS1": ("u",),
    "GPS2": ("e", "n"),
    "GPS3": ("e", "n", "u"),
}
GPS_COMPONENT_LABELS: dict[str, str] = {"e": "East", "n": "North", "u": "Up"}


@dataclass(frozen=True)
class CenteringVImposedConfig:
    type: str = "None"
    param: tuple[Any, ...] = ()


@dataclass(frozen=True)
class CenteringConfig:
    type: str = "basic"
    function: str = "empca"
    iter_max: int = 10**6
    tol: float = 1e-7
    func: str = "func_mean_zero_sum_V_transform_corrected"
    dfunc: str = "dfunc_mean_zero_sum_V_transform_corrected"
    offsets_epoch_imposed: tuple[float, ...] = ()
    Vimposed: CenteringVImposedConfig = field(default_factory=CenteringVImposedConfig)
    Ustart: tuple[Any, ...] = ()
    Sstart: tuple[Any, ...] = ()
    Vstart: tuple[Any, ...] = ()


@dataclass(frozen=True)
class FlagsConfig:
    flag_detrend: int = 0
    flag_disp: int = 1
    flag_decomp_err: int = 1
    flag_fault_model: int = 1
    flag_load_rand_guess: int = 0
    flag_load_rand: int = 0
    mixnet_flag: int = 0
    flag_visible: int = 1
    flag_whitening: int = 0
    flag_plot_ts: int = 1
    flag_plot_ts_offsets: int = 0
    flag_rm_offsets: int = 0
    flag_ICA_decomp: int = 1
    flag_invert_ICs: int = 0
    flag_invert_offsets: int = 0


@dataclass(frozen=True)
class VelocityConfig:
    file: Path = Path("Data/California/SoCal/NGL/NA12/vel/MIDAS/vel_MIDASformat_NA12_2018-01-22.txt")
    format: str = "MIDAS"


@dataclass(frozen=True)
class OutlierCenteringConfig:
    type: str = "basic"
    function: str = "decomp_empca"
    n_components: int = 2
    n_comp_mean: int | None = None
    iter_max: int = 10**6
    tol: float = 1e-7
    func: str = "func_mean_zero_sum_V_transform_corrected"
    dfunc: str = "dfunc_mean_zero_sum_V_transform_corrected"
    Vimposed: CenteringVImposedConfig = field(default_factory=CenteringVImposedConfig)


@dataclass(frozen=True)
class OutlierPCADecompositionConfig:
    iter_max_decomp: int = 5 * 10**5
    tol_decomp: float = 1e-7
    decomp_fcn: str = "empca"
    rand_missingdata: int = 0
    rand_init: int = 0


@dataclass(frozen=True)
class OutliersConfig:
    blunder_threshold_unit: str = "mm"
    blunder_threshold_hor: float = 10.0
    blunder_threshold_ver: float = 30.0
    outlier_threshold: float = 5.0
    centering: OutlierCenteringConfig = field(default_factory=OutlierCenteringConfig)
    decompositionPCA: OutlierPCADecompositionConfig = field(default_factory=OutlierPCADecompositionConfig)


@dataclass(frozen=True)
class PCADecompositionConfig:
    decomp_fcn: str = "empca"
    iter_max_decomp: int = 5 * 10**5
    tol_decomp: float = 1e-7
    rand_missingdata: int = 0
    rand_init: int = 0


@dataclass(frozen=True)
class ICAMixConfig:
    b_alpha_0: float | tuple[float, ...] = 1e3
    c_alpha_0: float | tuple[float, ...] = 1e-3


@dataclass(frozen=True)
class ICANoiseConfig:
    b_Lam_0: float = 1e1
    c_Lam_0: float = 1e-1
    mb0: float = 1.0
    mn0: float = 0.0


@dataclass(frozen=True)
class ICASourceConfig:
    m_0: float | tuple[float, ...] = 0.0
    tau_0: float | tuple[float, ...] = 1.0
    b_0: float | tuple[float, ...] = 1e1
    c_0: float | tuple[float, ...] = 1e-1
    lambda_0: float | tuple[float, ...] | None = None
    setSource: int = 1


@dataclass(frozen=True)
class ICADecompositionConfig:
    source_type: str = "g"
    learning_percent: int = 100
    ICA_num: int = 1
    n_mixed_pdfs: int | tuple[int, ...] = 4
    states: int | tuple[int, ...] | None = None
    mix_prior_preset: str | None = None
    source_prior_preset: str | None = None
    net_init: str = "SVD"
    source_init: str = "kmeans"
    max_steps: int = 500
    isonoise: int = 1
    ARD: int = 1
    tol: float = 1e-8
    eta: float = 1.0
    mix: ICAMixConfig = field(default_factory=ICAMixConfig)
    noise: ICANoiseConfig = field(default_factory=ICANoiseConfig)
    source: ICASourceConfig = field(default_factory=ICASourceConfig)


@dataclass(frozen=True)
class Config:
    repo_root: Path
    case_dir: Path
    data_input_file: Path
    first_epoch: float = 2010.0
    last_epoch: float = 2019.26164336
    n_components: int = 2
    decomposition_mode: str = "t"
    unit_output: str = "mm"
    skip_epochs: tuple[float, ...] = ()
    threshold_ts_missingdata: float = 80.0
    threshold_epochs_missingdata: float = 100.0
    select_origin_lon: float = 12.0151
    select_origin_lat: float = 45.9753
    select_radius_km: float = 95000.0
    velocity: VelocityConfig = field(default_factory=VelocityConfig)
    centering: CenteringConfig = field(default_factory=CenteringConfig)
    decompositionPCA: PCADecompositionConfig = field(default_factory=PCADecompositionConfig)
    decompositionICA: ICADecompositionConfig = field(default_factory=ICADecompositionConfig)
    flags: FlagsConfig = field(default_factory=FlagsConfig)
    outliers: OutliersConfig = field(default_factory=OutliersConfig)


def case1_dir(repo_root: str | Path) -> Path:
    return Path(repo_root).resolve() / CASE1_RELATIVE_DIR


def case1_python_port_dir(repo_root: str | Path) -> Path:
    return case1_dir(repo_root) / CASE1_PYTHON_PORT_DIRNAME


def case1_python_port_config_dir(repo_root: str | Path) -> Path:
    return case1_python_port_dir(repo_root) / CASE1_PYTHON_CONFIG_DIRNAME


def case1_default_config_file(repo_root: str | Path) -> Path:
    return case1_python_port_config_dir(repo_root) / DEFAULT_CASE1_CONFIG_FILENAME


def case1_python_port_output_root(repo_root: str | Path) -> Path:
    return case1_python_port_dir(repo_root) / "output"


def case1_python_port_batch_output_root(repo_root: str | Path) -> Path:
    return case1_python_port_dir(repo_root) / "output_batch"


def case1_python_port_compare_batch_output_root(repo_root: str | Path) -> Path:
    return case1_python_port_dir(repo_root) / "output_compare_batch"


def case1_python_port_bundle_dir(repo_root: str | Path) -> Path:
    return case1_python_port_dir(repo_root) / "bundle_case1"


def _first_existing_path(candidates: list[Path]) -> Path | None:
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            return resolved
    return None


def _case1_named_file_matches(candidate: Path, raw_name: str) -> bool:
    candidate_name = candidate.name
    candidate_stem = candidate.stem
    normalized = raw_name
    normalized_stem = Path(raw_name).stem if Path(raw_name).suffix else raw_name

    return any(
        value == target or value.endswith(f".{target}")
        for value in (candidate_name, candidate_stem)
        for target in (normalized, normalized_stem)
    )


def _find_unique_case1_named_file_match(search_dir: Path, raw_name: str) -> Path | None:
    matches = sorted(
        candidate.resolve()
        for candidate in search_dir.iterdir()
        if candidate.is_file() and _case1_named_file_matches(candidate, raw_name)
    )
    if not matches:
        return None
    if len(matches) > 1:
        joined = ", ".join(str(path.name) for path in matches)
        raise ValueError(f"Ambiguous short name '{raw_name}' in {search_dir}: {joined}")
    return matches[0]


def _resolve_case1_named_file(
    raw_path: str | Path,
    repo_root: str | Path,
    *,
    search_dir: Path,
    prefixed_names: tuple[str, ...] = (),
    default_suffix: str | None = None,
    allow_fuzzy_match: bool = True,
) -> Path:
    repo_root = Path(repo_root).resolve()
    raw = Path(raw_path)

    if raw.exists():
        if raw.is_dir():
            raise IsADirectoryError(raw)
        return raw.resolve()

    repo_candidate = resolve_repo_path(raw, repo_root)
    if repo_candidate.exists():
        if repo_candidate.is_dir():
            raise IsADirectoryError(repo_candidate)
        return repo_candidate

    base_name = raw.name
    stem = raw.stem if raw.suffix else raw.name
    candidates = [search_dir / base_name]
    if default_suffix and not raw.name.lower().endswith(default_suffix.lower()):
        candidates.append(search_dir / f"{base_name}{default_suffix}")
    for prefix in prefixed_names:
        if stem.startswith(prefix):
            continue
        if raw.suffix:
            candidates.append(search_dir / f"{prefix}{stem}{raw.suffix}")
        elif default_suffix:
            candidates.append(search_dir / f"{prefix}{stem}{default_suffix}")
        else:
            candidates.append(search_dir / f"{prefix}{stem}")

    match = _first_existing_path(candidates)
    if match is not None:
        return match

    if allow_fuzzy_match:
        fuzzy_match = _find_unique_case1_named_file_match(search_dir, base_name)
        if fuzzy_match is not None:
            return fuzzy_match

    fallback = candidates[0]
    if raw.is_absolute():
        return raw.resolve()
    return fallback.resolve()


def resolve_case1_data_input_file(raw_path: str | Path, repo_root: str | Path) -> Path:
    repo_root = Path(repo_root).resolve()
    dataset_dir = case1_dir(repo_root) / "dataset"
    raw = Path(raw_path)
    name = raw.name
    stem = raw.stem if raw.suffix else raw.name
    candidates = [dataset_dir / name]
    if not raw.suffix:
        candidates.extend(
            [
                dataset_dir / f"{name}.txt",
                dataset_dir / f"data_input_{stem}.txt",
            ]
        )

    if raw.exists():
        if raw.is_dir():
            raise IsADirectoryError(raw)
        return raw.resolve()

    repo_candidate = resolve_repo_path(raw, repo_root)
    if repo_candidate.exists():
        if repo_candidate.is_dir():
            raise IsADirectoryError(repo_candidate)
        return repo_candidate

    match = _first_existing_path(candidates)
    if match is not None:
        return match

    if raw.is_absolute():
        return raw.resolve()
    return (dataset_dir / name).resolve()


def resolve_case1_config_file(raw_path: str | Path, repo_root: str | Path) -> Path:
    resolved = _resolve_case1_named_file(
        raw_path,
        repo_root,
        search_dir=case1_python_port_config_dir(repo_root),
        default_suffix=".json",
        allow_fuzzy_match=False,
    )
    if resolved.exists():
        return resolved

    available = sorted(path.name for path in case1_python_port_config_dir(repo_root).glob("config*.json"))
    available_list = ", ".join(available) if available else "<none>"
    raise FileNotFoundError(
        "Configuration file "
        f"'{raw_path}' not found. Pass an absolute path, a repo-relative path, or the exact config file name "
        "(optionally without .json), for example 'config.atf2026' or 'config.case1.verify.quick.basic'. "
        f"Short aliases like 'basic' are not accepted. Available configs: {available_list}"
    )


def resolve_case1_batch_file(raw_path: str | Path, repo_root: str | Path) -> Path:
    return _resolve_case1_named_file(
        raw_path,
        repo_root,
        search_dir=case1_python_port_config_dir(repo_root),
        prefixed_names=("batch.case1.search.", "batch."),
        default_suffix=".json",
    )


def resolve_case1_results_file(raw_path: str | Path, repo_root: str | Path) -> Path:
    repo_root = Path(repo_root).resolve()
    raw = Path(raw_path)

    def _resolve_existing(path: Path) -> Path | None:
        if not path.exists():
            return None
        if path.is_dir():
            for filename in ("all_python.npz", "all_python.mat"):
                candidate = path / filename
                if candidate.exists():
                    return candidate.resolve()
            raise FileNotFoundError(f"No all_python.npz or all_python.mat inside {path}")
        return path.resolve()

    direct_match = _resolve_existing(raw)
    if direct_match is not None:
        return direct_match

    repo_match = _resolve_existing(resolve_repo_path(raw, repo_root))
    if repo_match is not None:
        return repo_match

    candidates = [
        case1_python_port_output_root(repo_root) / raw / "all_python.npz",
        case1_python_port_output_root(repo_root) / raw / "all_python.mat",
    ]
    match = _first_existing_path(candidates)
    if match is not None:
        return match

    if raw.is_absolute():
        return raw.resolve()
    return candidates[0].resolve()


def _fallback_dataset_label(data_input_file: Path) -> str:
    name = data_input_file.name
    if name.startswith("data_input_") and data_input_file.suffix.lower() == ".txt":
        return name[len("data_input_") : -len(".txt")]
    if data_input_file.suffix:
        return data_input_file.stem
    return name


def infer_case1_dataset_label(data_input_file: str | Path, repo_root: str | Path) -> str:
    repo_root = Path(repo_root).resolve()
    resolved_data_input = resolve_case1_data_input_file(data_input_file, repo_root)

    labels: list[str] = []
    try:
        datasets = [entry for entry in parse_data_input(resolved_data_input, repo_root) if entry["instruction"] == "decomp"]
        for dataset in datasets:
            list_path = Path(dataset["list_path"]).resolve()
            label = list_path.stem
            try:
                stations = parse_station_list(list_path, repo_root)
            except Exception:
                stations = []
            if stations:
                parent_name = Path(stations[0]["file"]).resolve().parent.name
                if parent_name:
                    label = parent_name
            if label and label not in labels:
                labels.append(label)
    except Exception:
        labels = []

    if labels:
        return "__".join(labels)
    return _fallback_dataset_label(resolved_data_input)


def default_case1_output_dir(cfg: Config) -> Path:
    dataset_label = infer_case1_dataset_label(cfg.data_input_file, cfg.repo_root)
    return case1_python_port_output_root(cfg.repo_root) / dataset_label


def default_case1_batch_output_dir(repo_root: str | Path, batch_file: str | Path) -> Path:
    return case1_python_port_batch_output_root(repo_root) / Path(batch_file).stem


def default_case1_compare_batch_output_dir(repo_root: str | Path, batch_file: str | Path) -> Path:
    return case1_python_port_compare_batch_output_root(repo_root) / Path(batch_file).stem


def default_config(repo_root: str | Path, data_input_file: str | Path | None = None) -> Config:
    repo_root = Path(repo_root).resolve()
    case_dir = case1_dir(repo_root)
    default_config_file = case1_default_config_file(repo_root)
    bootstrap_data_input = case_dir / "dataset" / DEFAULT_CONFIG_BOOTSTRAP_DATA_INPUT_NAME
    cfg = Config(
        repo_root=repo_root,
        case_dir=case_dir,
        data_input_file=bootstrap_data_input,
    )
    cfg = apply_config_overrides(cfg, load_config_overrides(default_config_file))
    if cfg.data_input_file == bootstrap_data_input:
        raise ValueError(f"Default case1 config must define 'data_input_file': {default_config_file}")
    if data_input_file is not None:
        cfg = replace(cfg, data_input_file=resolve_case1_data_input_file(data_input_file, repo_root))
    return cfg


def load_config_overrides(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    try:
        payload = json.loads(resolved.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in configuration file {resolved}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration file must contain a JSON object: {resolved}")
    return payload


def _coerce_tuple(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Expected a JSON array for tuple-like config value, got {type(value).__name__}")
    coerced: list[Any] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            coerced.append(_coerce_tuple(item))
        else:
            coerced.append(item)
    return tuple(coerced)


def _coerce_override_value(current_value: Any, override_value: Any, repo_root: Path, key_path: str) -> Any:
    if is_dataclass(current_value):
        if not isinstance(override_value, dict):
            raise TypeError(f"Expected an object for config section '{key_path}'")
        return _merge_config_overrides(current_value, override_value, repo_root, key_path)
    if isinstance(current_value, Path):
        if key_path == "config.data_input_file":
            return resolve_case1_data_input_file(override_value, repo_root)
        return resolve_repo_path(override_value, repo_root)
    if isinstance(current_value, tuple):
        return _coerce_tuple(override_value)
    return override_value


def _merge_config_overrides(instance: Any, overrides: dict[str, Any], repo_root: Path, key_path: str) -> Any:
    if not is_dataclass(instance):
        raise TypeError(f"Cannot merge overrides into non-dataclass value at '{key_path}'")

    known_fields = {field.name for field in fields(instance)}
    updates: dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in known_fields:
            raise KeyError(f"Unknown config key '{key_path}.{key}'")
        if key in {"repo_root", "case_dir"}:
            raise KeyError(f"Config key '{key_path}.{key}' is derived and cannot be overridden")
        current_value = getattr(instance, key)
        updates[key] = _coerce_override_value(current_value, value, repo_root, f"{key_path}.{key}")
    return replace(instance, **updates)


def apply_config_overrides(cfg: Config, overrides: dict[str, Any]) -> Config:
    updated = _merge_config_overrides(cfg, overrides, cfg.repo_root, "config")
    if not isinstance(updated, Config):
        raise TypeError("Internal error while applying config overrides")
    return updated


def build_effective_config(
    repo_root: str | Path,
    data_input_file: str | Path | None = None,
    config_file: str | Path | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> Config:
    cfg = default_config(repo_root)
    if config_file is not None:
        cfg = apply_config_overrides(cfg, load_config_overrides(resolve_case1_config_file(config_file, cfg.repo_root)))
    if config_overrides is not None:
        cfg = apply_config_overrides(cfg, config_overrides)
    if data_input_file is not None:
        cfg = replace(cfg, data_input_file=resolve_case1_data_input_file(data_input_file, cfg.repo_root))
    return cfg


def effective_config_payload(cfg: Config) -> dict[str, Any]:
    payload = json_ready(cfg)
    if not isinstance(payload, dict):
        raise TypeError("Internal error while serializing the effective config")
    payload.pop("repo_root", None)
    payload.pop("case_dir", None)
    return payload


LEGACY_CENTERING_FUNCTION_ALIASES = {
    "empca": "empca",
    "decomp_empca": "empca",
    "decomp_CG_means": "decomp_CG_means",
}

LEGACY_PCA_DECOMP_ALIASES = {
    "empca": "empca",
    "decomp_empca": "empca",
    "decomp_srebro_CG_simultaneous": "decomp_srebro_CG_simultaneous",
}

DECOMPOSITION_MODE_ALIASES = {
    "t": "t",
    "t-mode": "t",
    "t_mode": "t",
    "temporal": "t",
    "time": "t",
    "s": "s",
    "s-mode": "s",
    "s_mode": "s",
    "spatial": "s",
    "space": "s",
}

LEGACY_ICA_SOURCE_PRIOR_PRESETS: dict[str, dict[str, float]] = {
    "legacy_o1": {"b_0": 1e3, "c_0": 1e-3},
    "legacy_o2": {"b_0": 1e1, "c_0": 1e-1},
    "legacy_o3": {"b_0": 1e-1, "c_0": 1e1},
    "legacy_o4": {"b_0": 1e-3, "c_0": 1e3},
}

LEGACY_ICA_MIX_PRIOR_PRESETS: dict[str, dict[str, float]] = {
    "legacy_r1": {"b_alpha_0": 1e5, "c_alpha_0": 1e-1},
    "legacy_r2": {"b_alpha_0": 1e1, "c_alpha_0": 1e-1},
    "legacy_r3": {"b_alpha_0": 1e-1, "c_alpha_0": 1e1},
    "legacy_r4": {"b_alpha_0": 1e-3, "c_alpha_0": 1e3},
}


def normalize_centering_function_name(name: str) -> str:
    return LEGACY_CENTERING_FUNCTION_ALIASES.get(name, name)


def normalize_pca_decomp_name(name: str) -> str:
    return LEGACY_PCA_DECOMP_ALIASES.get(name, name)


def normalize_decomposition_mode(mode: str) -> str:
    return DECOMPOSITION_MODE_ALIASES.get(str(mode).strip().lower(), str(mode).strip().lower())


def decomposition_is_s_mode(cfg_or_mode: Any) -> bool:
    mode = getattr(cfg_or_mode, "decomposition_mode", cfg_or_mode)
    return normalize_decomposition_mode(mode) == "s"


def legacy_combination_is_compatible(cfg: Config) -> tuple[bool, str | None]:
    centering_type = cfg.centering.type
    centering_function = normalize_centering_function_name(cfg.centering.function)
    if centering_type == "advanced" and centering_function == "empca":
        return False, "legacy centering does not allow advanced centering with decomp_empca/empca"
    return True, None


def _expand_component_values(value: Any, n_components: int, key_path: str, dtype: type[float] | type[int]) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=dtype).reshape(-1)
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value, dtype=dtype).reshape(-1)
    else:
        array = np.repeat(dtype(value), n_components)

    if array.size == 1 and n_components != 1:
        array = np.repeat(array.item(), n_components)
    if array.size != n_components:
        raise ValueError(f"Config key '{key_path}' must have length 1 or {n_components}, got {array.size}")
    return array.astype(dtype)


def _resolve_ica_source_prior_preset(cfg: ICADecompositionConfig) -> ICASourceConfig:
    preset_name = cfg.source_prior_preset
    if not preset_name:
        return cfg.source
    if preset_name not in LEGACY_ICA_SOURCE_PRIOR_PRESETS:
        supported = ", ".join(sorted(LEGACY_ICA_SOURCE_PRIOR_PRESETS))
        raise ValueError(f"Unsupported decompositionICA.source_prior_preset='{preset_name}'. Supported values: {supported}")
    preset = LEGACY_ICA_SOURCE_PRIOR_PRESETS[preset_name]
    return replace(cfg.source, **preset)


def _resolve_ica_mix_prior_preset(cfg: ICADecompositionConfig) -> ICAMixConfig:
    preset_name = cfg.mix_prior_preset
    if not preset_name:
        return cfg.mix
    if preset_name not in LEGACY_ICA_MIX_PRIOR_PRESETS:
        supported = ", ".join(sorted(LEGACY_ICA_MIX_PRIOR_PRESETS))
        raise ValueError(f"Unsupported decompositionICA.mix_prior_preset='{preset_name}'. Supported values: {supported}")
    preset = LEGACY_ICA_MIX_PRIOR_PRESETS[preset_name]
    return replace(cfg.mix, **preset)


def validate_and_describe_config(cfg: Config) -> list[str]:
    notes: list[str] = []

    supported_centering_types = {"basic", "advanced"}
    supported_centering_functions = set(LEGACY_CENTERING_FUNCTION_ALIASES)
    supported_vimposed_types = {"None", "Heaviside", "Linear", "V"}
    supported_pca_functions = set(LEGACY_PCA_DECOMP_ALIASES)
    supported_decomposition_modes = {"t", "s"}

    decomposition_mode = normalize_decomposition_mode(cfg.decomposition_mode)
    if decomposition_mode not in supported_decomposition_modes:
        supported = ", ".join(sorted(supported_decomposition_modes))
        raise ValueError(f"Unsupported decomposition_mode='{cfg.decomposition_mode}'. Supported values: {supported}")
    if cfg.decomposition_mode != decomposition_mode:
        notes.append(f"decomposition_mode='{cfg.decomposition_mode}' is normalized to '{decomposition_mode}'.")

    centering_type = cfg.centering.type
    if centering_type not in supported_centering_types:
        supported = ", ".join(sorted(supported_centering_types))
        raise ValueError(f"Unsupported centering.type='{centering_type}'. Supported values: {supported}")

    centering_function_raw = cfg.centering.function
    if centering_function_raw not in supported_centering_functions:
        supported = ", ".join(sorted(supported_centering_functions))
        raise ValueError(f"Unsupported centering.function='{centering_function_raw}'. Supported values: {supported}")
    centering_function = normalize_centering_function_name(centering_function_raw)

    vimposed_type = cfg.centering.Vimposed.type
    if vimposed_type not in supported_vimposed_types:
        supported = ", ".join(sorted(supported_vimposed_types))
        raise ValueError(f"Unsupported centering.Vimposed.type='{vimposed_type}'. Supported values: {supported}")

    if cfg.centering.offsets_epoch_imposed:
        if vimposed_type not in {"None", "Heaviside"}:
            raise ValueError(
                "centering.offsets_epoch_imposed is only compatible with centering.Vimposed.type='None' or 'Heaviside'."
            )
        if cfg.centering.Vimposed.param and vimposed_type == "Heaviside":
            raise ValueError(
                "Specify either centering.offsets_epoch_imposed or centering.Vimposed.param for Heaviside centering, not both."
            )
        notes.append(
            "centering.offsets_epoch_imposed is treated as a compatibility alias for centering.Vimposed.type='Heaviside'."
        )

    centering_initial_guess = (cfg.centering.Ustart, cfg.centering.Sstart, cfg.centering.Vstart)
    if any(value for value in centering_initial_guess) and not all(value for value in centering_initial_guess):
        raise ValueError("centering.Ustart, centering.Sstart, and centering.Vstart must be provided together.")
    if all(value for value in centering_initial_guess):
        notes.append("Advanced centering will use centering.Ustart/Sstart/Vstart as the initial decomposition guess.")

    compatible, reason = legacy_combination_is_compatible(cfg)
    if not compatible:
        raise ValueError(f"Incompatible configuration: {reason}.")

    if centering_type == "advanced" and centering_function != "decomp_CG_means":
        raise ValueError("Incompatible configuration: legacy advanced centering requires centering.function='decomp_CG_means'.")

    decomp_fcn_raw = cfg.decompositionPCA.decomp_fcn
    if decomp_fcn_raw not in supported_pca_functions:
        supported = ", ".join(sorted(supported_pca_functions))
        raise NotImplementedError(
            f"The Python port does not implement decompositionPCA.decomp_fcn='{decomp_fcn_raw}'. Supported values: {supported}."
        )
    decomp_fcn = normalize_pca_decomp_name(decomp_fcn_raw)
    if decomp_fcn not in {"empca", "decomp_srebro_CG_simultaneous"}:
        supported = "decomp_srebro_CG_simultaneous, empca"
        raise NotImplementedError(
            f"The Python port does not implement decompositionPCA.decomp_fcn='{decomp_fcn_raw}'. Supported values: {supported}."
        )
    if decomposition_mode == "s" and decomp_fcn == "decomp_srebro_CG_simultaneous":
        has_imposed_v = vimposed_type != "None" or bool(cfg.centering.offsets_epoch_imposed)
        if has_imposed_v:
            raise NotImplementedError(
                "S-mode with decompositionPCA.decomp_fcn='decomp_srebro_CG_simultaneous' does not support time-based "
                "centering.Vimposed/offsets_epoch_imposed constraints. Use decompositionPCA.decomp_fcn='empca' or remove "
                "the imposed temporal component."
            )

    if cfg.decompositionPCA.rand_init != 0:
        raise NotImplementedError("The Python port does not implement decompositionPCA.rand_init != 0 yet.")

    source_type = cfg.decompositionICA.source_type
    if source_type != "g":
        raise NotImplementedError(
            f"The Python port does not implement decompositionICA.source_type='{source_type}'. Supported value: 'g'."
        )

    if cfg.decompositionICA.ICA_num != 1:
        raise NotImplementedError(
            f"The Python port does not implement decompositionICA.ICA_num={cfg.decompositionICA.ICA_num}. Supported value: 1."
        )

    source_init = cfg.decompositionICA.source_init
    if source_init != "kmeans":
        raise NotImplementedError(
            f"The Python port does not implement decompositionICA.source_init='{source_init}'. Supported value: 'kmeans'."
        )

    net_init = cfg.decompositionICA.net_init
    supported_net_init = {"SVD", "SVD_S&J"}
    if net_init not in supported_net_init:
        supported = ", ".join(sorted(supported_net_init))
        raise NotImplementedError(
            f"The Python port does not implement decompositionICA.net_init='{net_init}'. Supported values: {supported}."
        )

    if cfg.decompositionICA.mix_prior_preset is not None:
        _resolve_ica_mix_prior_preset(cfg.decompositionICA)
    if cfg.decompositionICA.source_prior_preset is not None:
        _resolve_ica_source_prior_preset(cfg.decompositionICA)

    notes.append(
        "run_decomposition currently uses scen/selection, centering, decompositionPCA, and decompositionICA directly; "
        "flags and outliers are preserved for compatibility, while velocity is retained mainly for the MATLAB clean "
        "legacy path."
    )

    return notes


def find_repo_root(start: str | Path) -> Path:
    start_path = Path(start).resolve()
    candidates = [start_path] + list(start_path.parents)
    for candidate in candidates:
        if (candidate / CASE1_RELATIVE_DIR).exists():
            return candidate
    raise FileNotFoundError(f"Cannot infer repo root starting from {start_path}")


def resolve_repo_path(raw_path: str | Path, repo_root: str | Path) -> Path:
    repo_root = Path(repo_root).resolve()
    raw_path = Path(raw_path)

    if raw_path.exists():
        return raw_path.resolve()

    if not raw_path.is_absolute():
        candidate = (repo_root / raw_path).resolve()
        if candidate.exists():
            return candidate

    raw_str = str(raw_path)
    for marker in ("/Scenarios/", "/Data/", "/rewrite/"):
        if marker in raw_str:
            suffix = raw_str.split(marker, 1)[1]
            candidate = (repo_root / marker.strip("/") / suffix).resolve()
            if candidate.exists():
                return candidate

    return raw_path.resolve()


def unit_scale(unit_input: str, unit_output: str) -> float:
    units = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
    try:
        return units[unit_input] / units[unit_output]
    except KeyError as exc:
        raise ValueError(f"Unsupported unit conversion: {unit_input} -> {unit_output}") from exc


def parse_pipe_line(line: str, expected_parts: int) -> list[str]:
    parts = [part.strip() for part in line.split("|")]
    if len(parts) < expected_parts:
        raise ValueError(f"Malformed line: {line}")
    return parts


def parse_data_input(path: Path, repo_root: Path) -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = parse_pipe_line(line, 5)
        datasets.append(
            {
                "list_path": resolve_repo_path(parts[0], repo_root),
                "type": parts[1],
                "timeunit": parts[2],
                "unit_input": parts[3],
                "instruction": parts[4],
            }
        )
    return datasets


def parse_station_list(path: Path, repo_root: Path) -> list[dict[str, Any]]:
    stations: list[dict[str, Any]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = parse_pipe_line(line, 6)
        stations.append(
            {
                "name": parts[0],
                "file": resolve_repo_path(parts[1], repo_root),
                "format": parts[2],
                "lon": float(parts[3]),
                "lat": float(parts[4]),
                "height": float(parts[5]) * 1e-3,
            }
        )
    return stations


def _flatten_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, bytes):
        return [value.decode()]
    if isinstance(value, str):
        return [value]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _flatten_text_values(value.item())
        values: list[str] = []
        for item in value.reshape(-1):
            values.extend(_flatten_text_values(item))
        return values
    if isinstance(value, (list, tuple)):
        values: list[str] = []
        for item in value:
            values.extend(_flatten_text_values(item))
        return values
    return [str(value)]


def normalize_gps_dataset_type(value: Any) -> str:
    normalized = _flatten_text_values(value)
    if not normalized:
        raise ValueError("Dataset type is empty.")
    dataset_type = normalized[0].strip().upper()
    if dataset_type not in SUPPORTED_GPS_COMPONENTS:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    return dataset_type


def gps_components_for_type(value: Any) -> tuple[str, ...]:
    return SUPPORTED_GPS_COMPONENTS[normalize_gps_dataset_type(value)]


def infer_gps_layout(type_value: Any, n_rows: int | None = None) -> tuple[str, tuple[str, ...]]:
    for text in _flatten_text_values(type_value):
        candidate = text.strip().upper()
        if len(candidate) >= 4 and candidate[:4] in SUPPORTED_GPS_COMPONENTS:
            dataset_type = candidate[:4]
            components = SUPPORTED_GPS_COMPONENTS[dataset_type]
            if n_rows is not None and n_rows % len(components) != 0:
                raise ValueError(
                    f"Series count {n_rows} is not compatible with dataset type {dataset_type} "
                    f"({len(components)} components per station)."
                )
            return dataset_type, components
    raise ValueError(f"Unsupported dataset type information: {type_value!r}")


def station_names_from_series(series_names: Any, type_value: Any) -> list[str]:
    names = _flatten_text_values(series_names)
    if not names:
        return []
    _, components = infer_gps_layout(type_value, len(names))
    component_size = len(components)
    return [name[:4] for name in names[::component_size]]


def load_tseri_series(input_file: Path, unit_input: str, unit_output: str) -> dict[str, Any]:
    data = np.genfromtxt(input_file, skip_header=2, dtype=None, encoding="utf-8")
    scale = unit_scale(unit_input, unit_output)
    timeline = np.asarray(data["f0"], dtype=float)
    return {
        "timeline": timeline,
        "pos": {
            "e": scale * np.asarray(data["f1"], dtype=float),
            "n": scale * np.asarray(data["f2"], dtype=float),
            "u": scale * np.asarray(data["f6"], dtype=float),
            "var_e": scale * np.asarray(data["f3"], dtype=float),
            "var_n": scale * np.asarray(data["f4"], dtype=float),
            "var_u": scale * np.asarray(data["f7"], dtype=float),
        },
    }


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0
    dlat = np.deg2rad(lat2 - lat1)
    dlon = np.deg2rad(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * earth_radius_km * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


def select_stations(stations: list[dict[str, Any]], origin_lon: float, origin_lat: float, radius_km: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for station in stations:
        distance = haversine_km(origin_lat, origin_lon, station["lat"], station["lon"])
        keep = distance <= radius_km if radius_km >= 0 else distance > abs(radius_km)
        if keep:
            selected.append(station)
    return selected


def empty_xcalc() -> dict[str, Any]:
    return {
        "name": [],
        "type": [],
        "llh": np.empty((0, 3), dtype=float),
        "timeline": np.empty((0,), dtype=float),
        "ts": np.empty((0, 0), dtype=float),
        "var_ts": np.empty((0, 0), dtype=float),
    }


def build_xcalc(
    stations: list[dict[str, Any]],
    first_epoch: float,
    last_epoch: float,
    skip_epochs: tuple[float, ...],
    threshold_ts_missingdata: float,
    threshold_epochs_missingdata: float,
) -> dict[str, Any]:
    if not stations:
        return empty_xcalc()

    all_timeline = np.concatenate([station["timeline"] for station in stations])
    timeline = np.unique(np.sort(all_timeline))
    timeline = timeline[(timeline >= first_epoch) & (timeline <= last_epoch)]

    n_series = sum(len(gps_components_for_type(station["type"])) for station in stations)
    nt = timeline.size
    ts = np.full((n_series, nt), np.nan, dtype=float)
    var_ts = np.full((n_series, nt), np.inf, dtype=float)
    llh = np.full((n_series, 3), np.nan, dtype=float)
    names: list[str] = []
    types: list[str] = []

    row = 0
    for station in stations:
        index_map = {float(epoch): idx for idx, epoch in enumerate(timeline)}
        llh_row = np.array([station["lon"], station["lat"], station["height"]], dtype=float)
        station_type = normalize_gps_dataset_type(station["type"])
        for component in gps_components_for_type(station_type):
            names.append(f"{station['name']}{component}")
            types.append(f"{station_type}{component}")
            llh[row, :] = llh_row
            for src_idx, epoch in enumerate(station["timeline"]):
                dst_idx = index_map.get(float(epoch))
                if dst_idx is None:
                    continue
                ts[row, dst_idx] = station["pos"][component][src_idx]
                var_ts[row, dst_idx] = station["pos"][f"var_{component}"][src_idx]
            row += 1

    xd = {
        "name": names,
        "type": types,
        "llh": llh,
        "timeline": timeline.copy(),
        "ts": ts,
        "var_ts": var_ts,
    }
    return apply_filters(xd, skip_epochs, threshold_ts_missingdata, threshold_epochs_missingdata)


def apply_filters(
    xd: dict[str, Any],
    skip_epochs: tuple[float, ...],
    threshold_ts_missingdata: float,
    threshold_epochs_missingdata: float,
) -> dict[str, Any]:
    if skip_epochs:
        skip_mask = np.isin(xd["timeline"], np.asarray(skip_epochs, dtype=float))
        xd["timeline"] = xd["timeline"][~skip_mask]
        xd["ts"] = xd["ts"][:, ~skip_mask]
        xd["var_ts"] = xd["var_ts"][:, ~skip_mask]

    nts, nt = xd["var_ts"].shape
    ts_keep = np.ones(nts, dtype=bool)
    for idx in range(nts):
        if np.isinf(xd["var_ts"][idx]).sum() >= threshold_ts_missingdata * nt / 100.0:
            ts_keep[idx] = False

    xd["name"] = [name for idx, name in enumerate(xd["name"]) if ts_keep[idx]]
    xd["type"] = [value for idx, value in enumerate(xd["type"]) if ts_keep[idx]]
    xd["llh"] = xd["llh"][ts_keep, :]
    xd["ts"] = xd["ts"][ts_keep, :]
    xd["var_ts"] = xd["var_ts"][ts_keep, :]

    nts, nt = xd["var_ts"].shape
    epoch_keep = np.ones(nt, dtype=bool)
    for idx in range(nt):
        if np.isinf(xd["var_ts"][:, idx]).sum() >= threshold_epochs_missingdata * nts / 100.0:
            epoch_keep[idx] = False

    xd["timeline"] = xd["timeline"][epoch_keep]
    xd["ts"] = xd["ts"][:, epoch_keep]
    xd["var_ts"] = xd["var_ts"][:, epoch_keep]
    return xd


def load_case_dataset(cfg: Config) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    datasets = [entry for entry in parse_data_input(cfg.data_input_file, cfg.repo_root) if entry["instruction"] == "decomp"]
    if not datasets:
        raise RuntimeError(f"No dataset marked as 'decomp' in {cfg.data_input_file}")

    stations: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_type = normalize_gps_dataset_type(dataset["type"])
        for listed in parse_station_list(dataset["list_path"], cfg.repo_root):
            series = load_tseri_series(listed["file"], dataset["unit_input"], cfg.unit_output)
            station = dict(listed)
            station["type"] = dataset_type
            station["timeline"] = series["timeline"]
            station["pos"] = series["pos"]
            stations.append(station)

    stations = select_stations(
        stations,
        origin_lon=cfg.select_origin_lon,
        origin_lat=cfg.select_origin_lat,
        radius_km=cfg.select_radius_km,
    )
    xd = build_xcalc(
        stations,
        first_epoch=cfg.first_epoch,
        last_epoch=cfg.last_epoch,
        skip_epochs=cfg.skip_epochs,
        threshold_ts_missingdata=cfg.threshold_ts_missingdata,
        threshold_epochs_missingdata=cfg.threshold_epochs_missingdata,
    )
    return stations, xd


def _fill_missing_row(row: np.ndarray, var_row: np.ndarray) -> np.ndarray:
    filled = row.copy()
    missing = np.isinf(var_row) | ~np.isfinite(filled)
    measured = np.flatnonzero(~missing)
    if measured.size == 0:
        filled[:] = 0.0
        return filled

    first = measured[0]
    last = measured[-1]
    before = np.flatnonzero(missing & (np.arange(filled.size) < first))
    between = np.flatnonzero(missing & (np.arange(filled.size) > first) & (np.arange(filled.size) < last))
    after = np.flatnonzero(missing & (np.arange(filled.size) > last))
    filled[before] = filled[first]
    if between.size:
        filled[between] = np.interp(between, measured, filled[measured])
    filled[after] = filled[last]
    return filled


def center_basic(xd: dict[str, Any], cfg: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    imposed_v, _ = build_imposed_v(xd["timeline"], cfg)
    return center_basic_legacy(xd, cfg.n_components, imposed_v)


def center_data(xd: dict[str, Any], cfg: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    imposed_v, _ = build_imposed_v(xd["timeline"], cfg)
    if cfg.centering.type == "basic":
        return center_basic_legacy(xd, cfg.n_components, imposed_v)
    if cfg.centering.type == "advanced":
        return center_advanced_legacy(xd, cfg, imposed_v)
    raise ValueError(f"Unsupported centering.type='{cfg.centering.type}'")


def matlab_var(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    if axis is None:
        count = x.size
    else:
        count = x.shape[axis]
    ddof = 1 if count > 1 else 0
    return np.var(x, axis=axis, ddof=ddof)


def matlab_std(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    if axis is None:
        count = x.size
    else:
        count = x.shape[axis]
    ddof = 1 if count > 1 else 0
    return np.std(x, axis=axis, ddof=ddof)


def empca(
    xdat: np.ndarray,
    xweight: np.ndarray,
    n_comp: int,
    tol_chi2: float,
    rand_missingdata: int = 0,
    rand_init: int = 0,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float], int]:
    x = np.array(xdat, dtype=float, copy=True)
    m, t = x.shape
    xw = np.array(xweight, dtype=float, copy=True)

    weighted_mean = np.zeros(m, dtype=float)
    for idx in range(m):
        weighted_mean[idx] = np.sum(xw[idx] * x[idx]) / np.sum(xw[idx])
    mu_x = np.repeat(weighted_mean[:, None], t, axis=1)
    x = x - mu_x
    xdat_centered = x.copy()

    ind_missing = xw == 0.0
    if rand_missingdata:
        rng = np.random.default_rng(0)
        for idx in range(m):
            row_missing = ind_missing[idx]
            row_present = ~row_missing
            if not np.any(row_missing):
                continue
            variance = matlab_var(x[idx, row_present])
            x[idx, row_missing] = np.sqrt(variance) * rng.standard_normal(row_missing.sum())
    else:
        x[ind_missing] = 0.0

    if rand_init:
        raise NotImplementedError("rand_init=1 is not supported in the Python port")
    u0, singular_values0, _ = np.linalg.svd(x, full_matrices=False)
    u = u0[:, :n_comp]
    c = np.diag(singular_values0[:n_comp]) @ np.linalg.svd(x, full_matrices=False)[2][:n_comp, :]

    xpca0 = u @ c
    chi2_path = [float(np.sum(xw * ((xdat_centered - xpca0) ** 2)))]

    c = np.zeros((n_comp, t), dtype=float)
    for col in range(t):
        vec_w = xw[:, col]
        a = np.zeros((n_comp, n_comp), dtype=float)
        for ii in range(n_comp):
            for ll in range(n_comp):
                a[ii, ll] = float((vec_w * u[:, ii]).T @ u[:, ll])
        b = u.T @ (vec_w * x[:, col])
        c[:, col] = solve_linear(a, b)

    n = (xw * x) @ c.T
    d = xw @ ((c**2).T)
    u = n / d
    x_work = x.copy()
    for ii in range(n_comp):
        norm_u = np.linalg.norm(u[:, ii])
        u[:, ii] = u[:, ii] / norm_u
        c[ii, :] = c[ii, :] * norm_u
        x_hat = np.outer(u[:, ii], c[ii, :])
        x_work = x_work - x_hat
    u[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])
    for ii in range(1, n_comp):
        for iii in range(ii):
            k = np.sum(u[:, ii] * u[:, iii])
            u[:, ii] = u[:, ii] - k * u[:, iii]
        u[:, ii] = u[:, ii] / np.linalg.norm(u[:, ii])

    xpca = u @ c
    chi2_path.append(float(np.sum(xw * ((xdat_centered - xpca) ** 2))))
    delta_chi2 = chi2_path[-2] - chi2_path[-1]
    u_tmp = u.copy()
    cc = 1

    for _ in range(max_iter):
        u_tmp = u.copy()
        for col in range(t):
            vec_w = xw[:, col]
            a = np.zeros((n_comp, n_comp), dtype=float)
            for ii in range(n_comp):
                for ll in range(n_comp):
                    a[ii, ll] = float((vec_w * u[:, ii]).T @ u[:, ll])
            b = u.T @ (vec_w * x[:, col])
            c[:, col] = solve_linear(a, b)

        n = (xw * x) @ c.T
        d = xw @ ((c**2).T)
        u = n / d
        x_work = x.copy()
        for ii in range(n_comp):
            norm_u = np.linalg.norm(u[:, ii])
            u[:, ii] = u[:, ii] / norm_u
            c[ii, :] = c[ii, :] * norm_u
            x_hat = np.outer(u[:, ii], c[ii, :])
            x_work = x_work - x_hat

        u[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])
        for ii in range(1, n_comp):
            for iii in range(ii):
                k = np.sum(u[:, ii] * u[:, iii])
                u[:, ii] = u[:, ii] - k * u[:, iii]
            u[:, ii] = u[:, ii] / np.linalg.norm(u[:, ii])

        xpca = u @ c
        cc += 1
        chi2_new = float(np.sum(xw * ((xdat_centered - xpca) ** 2)))
        chi2_path.append(chi2_new)
        delta_chi2 = chi2_path[-2] - chi2_new
        if delta_chi2 < 0:
            break
        if delta_chi2 <= tol_chi2:
            break

    if delta_chi2 < 0:
        u = u_tmp
        c_final = c.copy()
    else:
        c_final = c

    s = np.zeros((n_comp, n_comp), dtype=float)
    c_norm = np.zeros_like(c_final)
    for idx in range(n_comp):
        norm_c = np.linalg.norm(c_final[idx, :])
        s[idx, idx] = norm_c
        c_norm[idx, :] = c_final[idx, :] / norm_c
    v = c_norm.T
    return u, s, v, mu_x, chi2_path, cc


def solve_linear(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(a, b, rcond=None)[0]


def calc_displ_lincomp(decomp: dict[str, Any], nn: int) -> tuple[np.ndarray, np.ndarray]:
    u = decomp["U"][:, [nn]]
    var_u = decomp["var_U"][:, [nn]]
    s = float(decomp["S"][nn, nn])
    v = decomp["V"][:, [nn]]
    var_v = decomp["var_V"][:, [nn]]
    displ = u * s @ v.T
    var_mod1 = var_u * (s * s) @ var_v.T
    var_mod2 = var_u * (s * s) @ (v**2).T
    var_mod3 = (u**2) * (s * s) @ var_v.T
    var_displ = var_mod1 + var_mod2 + var_mod3
    return displ, var_displ


def calc_chi2_matrix(data: np.ndarray, wdata: np.ndarray, fitdata: np.ndarray) -> float:
    return float(np.sum(((data - fitdata) ** 2) * wdata))


def effective_observation_count(var_ts: np.ndarray) -> int:
    return int(np.count_nonzero(np.isfinite(var_ts)))


def missing_data_fraction(var_ts: np.ndarray) -> float:
    total = int(np.size(var_ts))
    if total == 0:
        return float("nan")
    return 1.0 - effective_observation_count(var_ts) / total


def pca_parameter_count(n_series: int, n_epochs: int, n_components: int) -> int:
    return int(n_components * (n_series + n_epochs - 1))


def ica_parameter_count(n_sensors: int, n_points: int, n_components: int) -> int:
    return int(8 * n_points * n_components + 2 * n_sensors * n_components + 2 * n_components + 4 * n_sensors)


def reduced_chi2(chi2: float, n_observations: int, n_parameters: int) -> float:
    dof = int(n_observations - n_parameters)
    if dof <= 0:
        return float("nan")
    return float(chi2 / dof)


def weighted_rms(chi2: float, n_observations: int) -> float:
    if n_observations <= 0:
        return float("nan")
    return float(np.sqrt(chi2 / n_observations))


def calc_variance_explained(data: np.ndarray, var_ts: np.ndarray, fitdata: np.ndarray) -> float:
    valid = np.isfinite(data) & np.isfinite(var_ts) & np.isfinite(fitdata) & (var_ts > 0.0)
    if not np.any(valid):
        return float("nan")
    numerator = np.sum(((data[valid] - fitdata[valid]) ** 2) / var_ts[valid])
    denominator = np.sum((data[valid] ** 2) / var_ts[valid])
    if denominator == 0.0:
        return float("nan")
    return float(100.0 * (1.0 - numerator / denominator))


def singular_values_from_s(s: np.ndarray) -> np.ndarray:
    return np.asarray(np.diag(np.asarray(s, dtype=float)), dtype=float)


def compute_quality_metrics(results: dict[str, Any]) -> dict[str, Any]:
    xd = results["Xd"]
    pca = results["PCA"]
    ica = results["ICA"]
    n_series, n_epochs = xd["ts"].shape
    n_components = pca["U"].shape[1]
    n_observations = effective_observation_count(xd["var_ts"])
    cfg = results.get("cfg")
    if isinstance(cfg, Config):
        decomposition_mode = normalize_decomposition_mode(cfg.decomposition_mode)
    else:
        decomposition_mode = normalize_decomposition_mode(
            pca.get("decomposition_mode", ica.get("decomposition_mode", "t"))
        )
    p_pca = pca_parameter_count(n_series, n_epochs, n_components)
    if decomposition_mode == "s":
        p_ica = ica_parameter_count(n_epochs, n_series, n_components)
    else:
        p_ica = ica_parameter_count(n_series, n_epochs, n_components)
    metrics = results.get("metrics", {})

    chi2_pca = float(metrics.get("chi2_PCA", calc_chi2_matrix(xd["ts"], (1.0 / xd["var_ts"]) ** 2, pca["ts"])))
    chi2_ica = float(metrics.get("chi2_ICA", calc_chi2_matrix(xd["ts"], (1.0 / xd["var_ts"]) ** 2, ica["ts"])))
    var_explained_pca = float(metrics.get("variance_explained_PCA", calc_variance_explained(xd["ts"], xd["var_ts"], pca["ts"])))
    var_explained_ica = float(metrics.get("variance_explained_ICA", calc_variance_explained(xd["ts"], xd["var_ts"], ica["ts"])))

    ard_weights = metrics.get("ard")
    if ard_weights is None and isinstance(ica.get("net"), dict) and "alphas" in ica["net"]:
        alphas = np.asarray(ica["net"]["alphas"], dtype=float).reshape(-1)
        ard_weights = (1.0 / alphas) / np.sum(1.0 / alphas)
    ard_weights_array = np.asarray(ard_weights, dtype=float).reshape(-1) if ard_weights is not None else np.empty((0,), dtype=float)

    alphas_array = np.empty((0,), dtype=float)
    energy = float("nan")
    energy_path = np.empty((0,), dtype=float)
    if isinstance(ica.get("net"), dict):
        if "alphas" in ica["net"]:
            alphas_array = np.asarray(ica["net"]["alphas"], dtype=float).reshape(-1)
        if "energy" in ica["net"]:
            energy = float(ica["net"]["energy"])
        if "energy_path" in ica["net"]:
            energy_path = np.asarray(ica["net"]["energy_path"], dtype=float).reshape(-1)

    ard_ratio = float("nan")
    if alphas_array.size > 0 and np.all(np.isfinite(alphas_array)) and np.min(alphas_array) > 0.0:
        ard_ratio = float(np.max(alphas_array) / np.min(alphas_array))

    pca_singular = singular_values_from_s(pca["S"])
    ica_singular = singular_values_from_s(ica["S"])
    pca_singular_rel = pca_singular / np.sum(pca_singular) if np.sum(pca_singular) > 0.0 else np.full_like(pca_singular, np.nan)
    ica_singular_rel = ica_singular / np.sum(ica_singular) if np.sum(ica_singular) > 0.0 else np.full_like(ica_singular, np.nan)

    pca_fit = pca.get("fit", {}) if isinstance(pca.get("fit"), dict) else {}
    ica_iterations = int(energy_path.size) if energy_path.size else None
    last_energy_delta = float("nan")
    if energy_path.size >= 2:
        last_energy_delta = float(energy_path[-1] - energy_path[-2])

    return {
        "data": {
            "n_series": int(n_series),
            "n_epochs": int(n_epochs),
            "n_components": int(n_components),
            "decomposition_mode": decomposition_mode,
            "n_observations": int(n_observations),
            "missing_data_fraction": float(missing_data_fraction(xd["var_ts"])),
        },
        "PCA": {
            "n_parameters": int(p_pca),
            "degrees_of_freedom": int(n_observations - p_pca),
            "chi2": float(chi2_pca),
            "reduced_chi2": float(reduced_chi2(chi2_pca, n_observations, p_pca)),
            "weighted_rms": float(weighted_rms(chi2_pca, n_observations)),
            "variance_explained": float(var_explained_pca),
            "singular_values": pca_singular,
            "singular_values_relative": pca_singular_rel,
            "fit_method": pca_fit.get("method"),
            "fit_iterations": pca_fit.get("iterations"),
            "fit_objective_name": pca_fit.get("objective_name"),
            "fit_objective_final": pca_fit.get("objective_final"),
        },
        "ICA": {
            "n_parameters": int(p_ica),
            "degrees_of_freedom": int(n_observations - p_ica),
            "chi2": float(chi2_ica),
            "reduced_chi2": float(reduced_chi2(chi2_ica, n_observations, p_ica)),
            "weighted_rms": float(weighted_rms(chi2_ica, n_observations)),
            "variance_explained": float(var_explained_ica),
            "chi2_gain_vs_PCA_pct": float(100.0 * (chi2_pca - chi2_ica) / chi2_pca) if chi2_pca != 0.0 else float("nan"),
            "variance_explained_gain_vs_PCA": float(var_explained_ica - var_explained_pca),
            "singular_values": ica_singular,
            "singular_values_relative": ica_singular_rel,
            "energy": float(energy),
            "energy_path": energy_path,
            "energy_delta_last": last_energy_delta,
            "iterations": ica_iterations,
            "alphas": alphas_array,
            "ard_weights": ard_weights_array,
            "ard_ratio": float(ard_ratio),
            "ard_noise_threshold": 10.0,
            "ard_suggests_too_many_components": bool(ard_ratio >= 10.0) if np.isfinite(ard_ratio) else None,
        },
    }


def flat_quality_metrics(quality: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_observations": quality["data"]["n_observations"],
        "missing_data_fraction": quality["data"]["missing_data_fraction"],
        "reduced_chi2_PCA": quality["PCA"]["reduced_chi2"],
        "reduced_chi2_ICA": quality["ICA"]["reduced_chi2"],
        "weighted_rms_PCA": quality["PCA"]["weighted_rms"],
        "weighted_rms_ICA": quality["ICA"]["weighted_rms"],
        "chi2_gain_ICA_vs_PCA_pct": quality["ICA"]["chi2_gain_vs_PCA_pct"],
        "var_explained_gain_ICA_vs_PCA": quality["ICA"]["variance_explained_gain_vs_PCA"],
        "ICA_energy": quality["ICA"]["energy"],
        "ICA_iterations": quality["ICA"]["iterations"],
        "ARD_ratio": quality["ICA"]["ard_ratio"],
    }


def build_pca(xd: dict[str, Any], cfg: Config, pca_4cen: dict[str, Any] | None = None) -> dict[str, Any]:
    decomp_fcn = normalize_pca_decomp_name(cfg.decompositionPCA.decomp_fcn)
    decomposition_mode = normalize_decomposition_mode(cfg.decomposition_mode)
    work_ts = xd["ts"].T if decomposition_mode == "s" else xd["ts"]
    work_var_ts = xd["var_ts"].T if decomposition_mode == "s" else xd["var_ts"]
    weights = (1.0 / work_var_ts) ** 2
    weights[~np.isfinite(weights)] = 0.0
    if decomp_fcn == "empca":
        u, s, v, _, chi2_path, fit_iterations = empca(
            xdat=work_ts,
            xweight=weights,
            n_comp=cfg.n_components,
            tol_chi2=cfg.decompositionPCA.tol_decomp,
            rand_missingdata=cfg.decompositionPCA.rand_missingdata,
            rand_init=cfg.decompositionPCA.rand_init,
            max_iter=cfg.decompositionPCA.iter_max_decomp,
        )
        fit = {
            "method": decomp_fcn,
            "iterations": int(fit_iterations),
            "objective_name": "chi2",
            "objective_final": float(chi2_path[-1]),
            "objective_path": np.asarray(chi2_path, dtype=float),
        }
    else:
        if decomposition_mode == "s":
            imposed_v = np.empty((work_ts.shape[1], 0), dtype=float)
            heaviside_v = np.empty((0,), dtype=int)
            u0 = s0 = v0 = None
        else:
            imposed_v, heaviside_v = build_imposed_v(xd["timeline"], cfg)
            if pca_4cen is None:
                u0 = s0 = v0 = None
            else:
                u0 = pca_4cen.get("U")
                s0 = pca_4cen.get("S")
                v0 = pca_4cen.get("V")
        u, s, v, residual, fit_iterations = decomp_srebro_cg_simultaneous(
            x_dat=work_ts,
            x_weight=weights,
            n_comp=cfg.n_components,
            iter_max=cfg.decompositionPCA.iter_max_decomp,
            tol=cfg.decompositionPCA.tol_decomp,
            imposed_v=imposed_v,
            heaviside_v=heaviside_v,
            u0=u0,
            s0=s0,
            v0=v0,
        )
        fit = {
            "method": decomp_fcn,
            "iterations": int(fit_iterations),
            "objective_name": "residual",
            "objective_final": float(residual),
        }
    if decomposition_mode == "s":
        u, v = v, u
    pca = {
        "name": list(xd["name"]),
        "llh": xd["llh"],
        "timeline": xd["timeline"],
        "decomposition_mode": decomposition_mode,
        "U": u,
        "S": s,
        "V": v,
        "var_U": np.full_like(u, np.nan),
        "var_V": np.full_like(v, np.nan),
        "ts": u @ s @ v.T,
        "var_ts": np.full((u.shape[0], v.shape[0]), np.nan),
        "type": list(xd["type"]),
        "fit": fit,
    }
    pca["displ"] = []
    pca["var_displ"] = []
    for nn in range(cfg.n_components):
        displ, var_displ = calc_displ_lincomp(pca, nn)
        pca["displ"].append(displ)
        pca["var_displ"].append(var_displ)
    return pca


def build_ica_init_parameters(cfg: Config, pca: dict[str, Any]) -> dict[str, Any]:
    decomposition_mode = normalize_decomposition_mode(cfg.decomposition_mode)
    if decomposition_mode == "s":
        init_u = pca["V"]
        init_v = pca["U"]
    else:
        init_u = pca["U"]
        init_v = pca["V"]
    nt = init_v.shape[0]
    n_comp = pca["U"].shape[1]
    ica_cfg = cfg.decompositionICA
    mix_cfg = _resolve_ica_mix_prior_preset(ica_cfg)
    source_cfg = _resolve_ica_source_prior_preset(ica_cfg)

    if ica_cfg.states is None:
        states = _expand_component_values(ica_cfg.n_mixed_pdfs, n_comp, "config.decompositionICA.n_mixed_pdfs", int)
    else:
        states = _expand_component_values(ica_cfg.states, n_comp, "config.decompositionICA.states", int)

    lambda_0 = source_cfg.lambda_0
    if lambda_0 is None:
        lambda_0_values = np.ones(n_comp, dtype=float) * round((nt / 100.0 + nt / 10.0) / 2.0)
    else:
        lambda_0_values = _expand_component_values(lambda_0, n_comp, "config.decompositionICA.source.lambda_0", float)

    return {
        "source_type": ica_cfg.source_type,
        "src_type": ica_cfg.source_type,
        "HMM": 1 if len(ica_cfg.source_type) == 2 else 0,
        "learning_percent": ica_cfg.learning_percent,
        "ICA_num": ica_cfg.ICA_num,
        "n_comp_ICA": [n_comp],
        "states": states,
        "mix": {
            "b_alpha_0": _expand_component_values(
                mix_cfg.b_alpha_0, n_comp, "config.decompositionICA.mix.b_alpha_0", float
            ),
            "c_alpha_0": _expand_component_values(
                mix_cfg.c_alpha_0, n_comp, "config.decompositionICA.mix.c_alpha_0", float
            ),
        },
        "noise": {
            "b_Lam_0": float(ica_cfg.noise.b_Lam_0),
            "c_Lam_0": float(ica_cfg.noise.c_Lam_0),
            "mb0": float(ica_cfg.noise.mb0),
            "mn0": float(ica_cfg.noise.mn0),
        },
        "source": {
            "m_0": _expand_component_values(source_cfg.m_0, n_comp, "config.decompositionICA.source.m_0", float),
            "tau_0": _expand_component_values(source_cfg.tau_0, n_comp, "config.decompositionICA.source.tau_0", float),
            "b_0": _expand_component_values(source_cfg.b_0, n_comp, "config.decompositionICA.source.b_0", float),
            "c_0": _expand_component_values(source_cfg.c_0, n_comp, "config.decompositionICA.source.c_0", float),
            "lambda_0": lambda_0_values,
            "setSource": int(source_cfg.setSource),
        },
        "net_init": ica_cfg.net_init,
        "source_init": ica_cfg.source_init,
        "decomposition_mode": decomposition_mode,
        "U": init_u,
        "S": pca["S"],
        "V": init_v,
        "max_steps": int(ica_cfg.max_steps),
        "isonoise": int(ica_cfg.isonoise),
        "ARD": int(ica_cfg.ARD),
        "n": 1,
        "mixmatrix_guess": np.empty((0, 0), dtype=float),
        "sources_guess": np.empty((0, 0), dtype=float),
        "net": {},
        "job_number": np.empty((0, 0), dtype=float),
        "check_progress": 0,
        "tol": float(ica_cfg.tol),
        "eta": float(ica_cfg.eta),
    }


def savemat_ready(value: Any) -> Any:
    if value is None:
        return np.empty((0, 0), dtype=float)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {field.name: savemat_ready(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key: savemat_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [savemat_ready(item) for item in value]
    if isinstance(value, tuple):
        return [savemat_ready(item) for item in value]
    return value


def save_results_mat(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mat_results = dict(results)
    mat_results.pop("quality_metrics", None)
    savemat(path, savemat_ready(mat_results), do_compression=False, oned_as="row")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {field.name: json_ready(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), indent=2, sort_keys=True) + "\n")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _npz_scalar_or_array(payload: dict[str, Any], key: str, default: Any = None) -> Any:
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def results_from_npz_payload(payload: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {
        "Xd": {
            "ts": payload["Xd_ts"],
            "var_ts": payload["Xd_var_ts"],
            "timeline": payload["Xd_timeline"],
            "llh": payload["Xd_llh"],
            "centering_offsets": payload["Xd_centering_offsets"],
            "name": _npz_scalar_or_array(payload, "Xd_name"),
            "type": _npz_scalar_or_array(payload, "Xd_type"),
        },
        "PCA": {
            "U": payload["PCA_U"],
            "S": payload["PCA_S"],
            "V": payload["PCA_V"],
            "ts": payload["PCA_ts"],
        },
        "ICA": {
            "U": payload["ICA_U"],
            "S": payload["ICA_S"],
            "V": payload["ICA_V"],
            "ts": payload["ICA_ts"],
        },
        "STATIONS_name": _npz_scalar_or_array(payload, "STATIONS_name"),
    }
    if "config_json" in payload:
        results["config"] = json.loads(_npz_scalar_or_array(payload, "config_json"))
    if "resolved_config_json" in payload:
        results["resolved_config"] = json.loads(_npz_scalar_or_array(payload, "resolved_config_json"))
    if "quality_metrics_json" in payload:
        results["quality_metrics"] = json.loads(_npz_scalar_or_array(payload, "quality_metrics_json"))
    if "PCA_fit_method" in payload:
        results["PCA"]["fit"] = {
            "method": _npz_scalar_or_array(payload, "PCA_fit_method"),
            "iterations": int(_npz_scalar_or_array(payload, "PCA_fit_iterations", 0)),
            "objective_name": _npz_scalar_or_array(payload, "PCA_fit_objective_name"),
            "objective_final": float(_npz_scalar_or_array(payload, "PCA_fit_objective_final", np.nan)),
        }
        if "PCA_fit_objective_path" in payload:
            results["PCA"]["fit"]["objective_path"] = payload["PCA_fit_objective_path"]
    if "PCA_decomposition_mode" in payload:
        results["PCA"]["decomposition_mode"] = _npz_scalar_or_array(payload, "PCA_decomposition_mode")
    if "ICA_decomposition_mode" in payload:
        results["ICA"]["decomposition_mode"] = _npz_scalar_or_array(payload, "ICA_decomposition_mode")
    if "ICA_llh" in payload:
        results["ICA"]["llh"] = payload["ICA_llh"]
    if "ICA_timeline" in payload:
        results["ICA"]["timeline"] = payload["ICA_timeline"]
    if "ICA_name" in payload:
        results["ICA"]["name"] = _npz_scalar_or_array(payload, "ICA_name")
    if "ICA_type" in payload:
        results["ICA"]["type"] = _npz_scalar_or_array(payload, "ICA_type")
    if any(key in payload for key in ("ICA_net_energy", "ICA_net_alphas", "ICA_net_energy_path")):
        results["ICA"]["net"] = {}
        if "ICA_net_energy" in payload:
            results["ICA"]["net"]["energy"] = float(_npz_scalar_or_array(payload, "ICA_net_energy", np.nan))
        if "ICA_net_alphas" in payload:
            results["ICA"]["net"]["alphas"] = payload["ICA_net_alphas"]
        if "ICA_net_energy_path" in payload:
            results["ICA"]["net"]["energy_path"] = payload["ICA_net_energy_path"]
    metric_keys = (
        "chi2_PCA",
        "chi2_ICA",
        "variance_explained_PCA",
        "variance_explained_ICA",
        "ard",
        "n_observations",
        "missing_data_fraction",
        "reduced_chi2_PCA",
        "reduced_chi2_ICA",
        "weighted_rms_PCA",
        "weighted_rms_ICA",
        "chi2_gain_ICA_vs_PCA_pct",
        "var_explained_gain_ICA_vs_PCA",
        "ICA_energy",
        "ICA_iterations",
        "ARD_ratio",
    )
    loaded_metrics: dict[str, Any] = {}
    for key in metric_keys:
        if key in payload:
            value = _npz_scalar_or_array(payload, key)
            if isinstance(value, np.ndarray):
                loaded_metrics[key] = value
            elif key in {"ICA_iterations", "n_observations"} and value is not None and np.isfinite(value):
                loaded_metrics[key] = int(value)
            else:
                loaded_metrics[key] = float(value) if isinstance(value, (int, float, np.generic)) else value
    if loaded_metrics:
        results["metrics"] = loaded_metrics
    return results


def load_results_file(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if resolved.suffix.lower() == ".npz":
        with np.load(resolved, allow_pickle=True) as payload:
            return results_from_npz_payload({key: payload[key] for key in payload.files})
    return loadmat(resolved, simplify_cells=True)


def build_results_npz_payload(results: dict[str, Any]) -> dict[str, Any]:
    xd = results["Xd"]
    pca = results["PCA"]
    ica = results["ICA"]
    payload: dict[str, Any] = {
        "Xd_ts": xd["ts"],
        "Xd_var_ts": xd["var_ts"],
        "Xd_timeline": xd["timeline"],
        "Xd_llh": xd["llh"],
        "Xd_centering_offsets": xd["centering_offsets"],
        "Xd_name": np.asarray(xd["name"], dtype=object),
        "Xd_type": np.asarray(xd["type"], dtype=object),
        "PCA_U": pca["U"],
        "PCA_S": pca["S"],
        "PCA_V": pca["V"],
        "PCA_ts": pca["ts"],
        "PCA_decomposition_mode": np.asarray(pca.get("decomposition_mode", "t")),
        "ICA_U": ica["U"],
        "ICA_S": ica["S"],
        "ICA_V": ica["V"],
        "ICA_ts": ica["ts"],
        "ICA_decomposition_mode": np.asarray(ica.get("decomposition_mode", pca.get("decomposition_mode", "t"))),
        "ICA_llh": ica["llh"],
        "ICA_timeline": ica["timeline"],
        "ICA_name": np.asarray(ica["name"], dtype=object),
        "ICA_type": np.asarray(ica["type"], dtype=object),
        "A_recon": results["A_recon"],
        "S_recon": results["S_recon"],
        "var_A_recon": results["var_A_recon"],
        "var_S_recon": results["var_S_recon"],
        "data_mask": results["data_mask"],
        "ind_missing_data": results["ind_missing_data"],
        "STATIONS_name": np.asarray(results["STATIONS_name"], dtype=object),
    }
    for key, value in results.get("metrics", {}).items():
        payload[key] = np.asarray(value)
    if isinstance(pca.get("fit"), dict):
        payload["PCA_fit_method"] = np.asarray(pca["fit"].get("method"))
        payload["PCA_fit_iterations"] = np.asarray(pca["fit"].get("iterations", -1), dtype=int)
        payload["PCA_fit_objective_name"] = np.asarray(pca["fit"].get("objective_name"))
        payload["PCA_fit_objective_final"] = np.asarray(pca["fit"].get("objective_final", np.nan), dtype=float)
        if "objective_path" in pca["fit"]:
            payload["PCA_fit_objective_path"] = np.asarray(pca["fit"]["objective_path"], dtype=float)
    if isinstance(ica.get("net"), dict):
        if "energy" in ica["net"]:
            payload["ICA_net_energy"] = np.asarray(ica["net"]["energy"], dtype=float)
        if "energy_path" in ica["net"]:
            payload["ICA_net_energy_path"] = np.asarray(ica["net"]["energy_path"], dtype=float)
        if "alphas" in ica["net"]:
            payload["ICA_net_alphas"] = np.asarray(ica["net"]["alphas"], dtype=float)
    if "cfg" in results and is_dataclass(results["cfg"]):
        payload["config_json"] = np.asarray(json.dumps(json_ready(results["cfg"])))
    resolved_config = build_resolved_config_summary(results)
    if resolved_config is not None:
        payload["resolved_config_json"] = np.asarray(json.dumps(resolved_config))
    if "quality_metrics" in results:
        payload["quality_metrics_json"] = np.asarray(json.dumps(json_ready(results["quality_metrics"])))
    return payload


def save_results_npz(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **build_results_npz_payload(results))


def build_resolved_config_summary(results: dict[str, Any]) -> dict[str, Any] | None:
    cfg = results.get("cfg")
    init_parameters = results.get("init_parameters")
    if not isinstance(cfg, Config) or not isinstance(init_parameters, dict):
        return None

    resolved = json_ready(cfg)
    resolved["centering"]["function_resolved"] = normalize_centering_function_name(cfg.centering.function)
    if cfg.centering.offsets_epoch_imposed:
        resolved["centering"]["Vimposed_compatibility_resolved"] = {
            "type": "Heaviside",
            "param": json_ready(cfg.centering.offsets_epoch_imposed),
        }
    resolved["decompositionPCA"]["decomp_fcn_resolved"] = normalize_pca_decomp_name(cfg.decompositionPCA.decomp_fcn)
    resolved["decomposition_mode_resolved"] = normalize_decomposition_mode(cfg.decomposition_mode)
    resolved["decompositionICA"]["states_resolved"] = json_ready(np.asarray(init_parameters["states"], dtype=int))
    resolved["decompositionICA"]["mix_resolved"] = json_ready(init_parameters["mix"])
    resolved["decompositionICA"]["noise_resolved"] = json_ready(init_parameters["noise"])
    resolved["decompositionICA"]["source_resolved"] = json_ready(init_parameters["source"])
    return resolved


def build_summary(results: dict[str, Any], generated_files: dict[str, str] | None = None) -> dict[str, Any]:
    xd = results["Xd"]
    pca = results["PCA"]
    ica = results["ICA"]
    return {
        "case": "case1",
        "formats": generated_files or {},
        "run_metadata": results.get("run_metadata"),
        "config": json_ready(results["cfg"]) if is_dataclass(results["cfg"]) else None,
        "resolved_config": build_resolved_config_summary(results),
        "config_notes": list(results.get("config_notes", [])),
        "dimensions": {
            "n_series": int(xd["ts"].shape[0]),
            "n_epochs": int(xd["ts"].shape[1]),
            "n_components": int(pca["U"].shape[1]),
            "decomposition_mode": pca.get("decomposition_mode", "t"),
            "n_stations": int(len(results["STATIONS_name"])),
        },
        "stations": list(results["STATIONS_name"]),
        "metrics": results["metrics"],
        "quality_metrics": results.get("quality_metrics"),
    }


def align_components(u_test: np.ndarray, v_test: np.ndarray, u_ref: np.ndarray, v_ref: np.ndarray) -> tuple[tuple[int, ...], np.ndarray]:
    n_comp = u_ref.shape[1]
    best_perm: tuple[int, ...] | None = None
    best_score = -np.inf
    best_signs = np.ones(n_comp, dtype=float)
    for perm in permutations(range(n_comp)):
        perm = tuple(perm)
        score = 0.0
        signs = np.ones(n_comp, dtype=float)
        for idx_ref, idx_test in enumerate(perm):
            corr_u = float(np.dot(u_test[:, idx_test], u_ref[:, idx_ref]))
            corr_v = float(np.dot(v_test[:, idx_test], v_ref[:, idx_ref]))
            if abs(corr_v) >= abs(corr_u):
                sign = np.sign(corr_v) if corr_v != 0 else 1.0
                score += abs(corr_v)
            else:
                sign = np.sign(corr_u) if corr_u != 0 else 1.0
                score += abs(corr_u)
            signs[idx_ref] = sign
        if score > best_score:
            best_score = score
            best_perm = perm
            best_signs = signs
    if best_perm is None:
        raise RuntimeError("Cannot align empty component set")
    return best_perm, best_signs


def reorder_decomp(decomp: dict[str, Any], perm: tuple[int, ...], signs: np.ndarray, recompute_ts: bool = True) -> dict[str, Any]:
    reordered = dict(decomp)
    reordered["U"] = decomp["U"][:, perm] * signs[None, :]
    reordered["S"] = decomp["S"][np.ix_(perm, perm)]
    reordered["V"] = decomp["V"][:, perm] * signs[None, :]
    if "var_U" in decomp:
        reordered["var_U"] = decomp["var_U"][:, perm]
    if "var_V" in decomp:
        reordered["var_V"] = decomp["var_V"][:, perm]
    if recompute_ts and "ts" in decomp:
        reordered["ts"] = reordered["U"] @ reordered["S"] @ reordered["V"].T
    return reordered
