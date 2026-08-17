from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import block_diag


EPS = np.finfo(float).eps


def _matlab_vec(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=float).reshape(-1, order="F")


def fill_missing_rows(ts: np.ndarray, var_ts: np.ndarray) -> np.ndarray:
    filled = np.asarray(ts, dtype=float).copy()
    n_epochs = filled.shape[1]
    epochs = np.arange(n_epochs)
    for idx in range(filled.shape[0]):
        missing = np.isinf(var_ts[idx]) | ~np.isfinite(filled[idx])
        measured = np.flatnonzero(~missing)
        if measured.size == 0:
            filled[idx] = 0.0
            continue
        first = measured[0]
        last = measured[-1]
        before = np.flatnonzero(missing & (epochs < first))
        between = np.flatnonzero(missing & (epochs > first) & (epochs < last))
        after = np.flatnonzero(missing & (epochs > last))
        filled[idx, before] = filled[idx, first]
        if between.size:
            filled[idx, between] = np.interp(between, measured, filled[idx, measured])
        filled[idx, after] = filled[idx, last]
    return filled


def weighted_row_means(data: np.ndarray, weight: np.ndarray) -> np.ndarray:
    means = np.zeros(data.shape[0], dtype=float)
    for idx in range(data.shape[0]):
        valid = (weight[idx] > 0.0) & np.isfinite(data[idx])
        if np.any(valid):
            means[idx] = float(np.sum(weight[idx, valid] * data[idx, valid]) / np.sum(weight[idx, valid]))
    return means


def normalize_v(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return np.empty((v.shape[0], 0), dtype=float)
    means = np.mean(v, axis=0, keepdims=True)
    centered = v.copy() if np.isnan(means).any() else v - means
    norms = np.sqrt(np.diag(centered.T @ centered))
    normalized = centered.copy()
    for idx, norm in enumerate(norms):
        if norm > 0.0:
            normalized[:, idx] /= norm
    return normalized


def build_imposed_v(timeline: np.ndarray, cfg: Any) -> tuple[np.ndarray, np.ndarray]:
    vimposed_type = cfg.centering.Vimposed.type
    params = cfg.centering.Vimposed.param
    offsets_epoch_imposed = getattr(cfg.centering, "offsets_epoch_imposed", ())
    n_epochs = timeline.size

    if offsets_epoch_imposed:
        if vimposed_type not in {"None", "Heaviside"}:
            raise ValueError(
                "centering.offsets_epoch_imposed is only compatible with centering.Vimposed.type='None' or 'Heaviside'"
            )
        if params and vimposed_type == "Heaviside":
            raise ValueError(
                "Specify either centering.offsets_epoch_imposed or centering.Vimposed.param for Heaviside centering, not both"
            )
        vimposed_type = "Heaviside"
        params = offsets_epoch_imposed

    if vimposed_type == "None":
        return np.empty((n_epochs, 0), dtype=float), np.empty((0,), dtype=int)

    if vimposed_type == "Heaviside":
        offsets = np.asarray(params, dtype=float).reshape(-1)
        imposed = np.zeros((n_epochs, offsets.size), dtype=float)
        for idx, offset in enumerate(offsets):
            imposed[:, idx] = np.heaviside(timeline - offset, 0.5)
        return normalize_v(imposed), np.arange(offsets.size, dtype=int)

    if vimposed_type == "Linear":
        imposed = np.arange(1, n_epochs + 1, dtype=float).reshape(-1, 1)
        return normalize_v(imposed), np.empty((0,), dtype=int)

    if vimposed_type == "V":
        imposed = np.asarray(params, dtype=float)
        if imposed.ndim == 1:
            imposed = imposed.reshape(-1, 1)
        if imposed.shape[0] != n_epochs:
            raise ValueError(
                f"centering.Vimposed.param for type='V' must have {n_epochs} rows, got {imposed.shape[0]}"
            )
        return normalize_v(imposed), np.empty((0,), dtype=int)

    raise ValueError(f"Unsupported centering.Vimposed.type='{vimposed_type}'")


def _config_matrix(value: Any, field_name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.size == 0:
        return np.empty((0, 0), dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{field_name} must be a 2D array when provided")
    return matrix


def _centering_initial_guess(cfg: Any, n_tseries: int, n_epochs: int, n_comp: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    u_start = _config_matrix(getattr(cfg.centering, "Ustart", ()), "centering.Ustart")
    s_start = _config_matrix(getattr(cfg.centering, "Sstart", ()), "centering.Sstart")
    v_start = _config_matrix(getattr(cfg.centering, "Vstart", ()), "centering.Vstart")

    if u_start.size == 0 and s_start.size == 0 and v_start.size == 0:
        return None
    if u_start.size == 0 or s_start.size == 0 or v_start.size == 0:
        raise ValueError("centering.Ustart, centering.Sstart, and centering.Vstart must all be provided together")

    expected_u = (n_tseries, n_comp)
    expected_s = (n_comp, n_comp)
    expected_v = (n_epochs, n_comp)
    if u_start.shape != expected_u:
        raise ValueError(f"centering.Ustart must have shape {expected_u}, got {u_start.shape}")
    if s_start.shape != expected_s:
        raise ValueError(f"centering.Sstart must have shape {expected_s}, got {s_start.shape}")
    if v_start.shape != expected_v:
        raise ValueError(f"centering.Vstart must have shape {expected_v}, got {v_start.shape}")
    return u_start, s_start, v_start


def _transformation_matrix(n_epochs: int) -> tuple[np.ndarray, np.ndarray]:
    epoch_nums = np.arange(1, n_epochs, dtype=float).reshape(-1, 1)
    neg_ones = -np.ones((n_epochs - 1, n_epochs - 1), dtype=float)
    scale = np.repeat(1.0 / np.sqrt(epoch_nums**2 + epoch_nums), n_epochs, axis=1)
    left = np.diagflat(epoch_nums[:, 0]) + np.tril(neg_ones, -1)
    transform = np.hstack([left, -np.ones((n_epochs - 1, 1), dtype=float)]) * scale
    return transform, np.linalg.pinv(transform)


def _reconstruct_multi_components(
    x: np.ndarray,
    n_tseries: int,
    n_epochs: int,
    n_comp: int,
    imposed_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_imposed = imposed_v.shape[1]
    u = np.asarray(x[: n_tseries * n_comp], dtype=float).reshape((n_tseries, n_comp), order="F")
    n_free = n_comp - n_imposed
    if n_free > 0:
        v_free = np.asarray(x[n_tseries * n_comp :], dtype=float).reshape((n_epochs, n_free), order="F")
        v = np.hstack([imposed_v, v_free]) if n_imposed > 0 else v_free
    else:
        v = imposed_v.copy()
    return u, v


def _reconstruct_multi_direction(
    direction: np.ndarray,
    n_tseries: int,
    n_epochs: int,
    n_comp: int,
    imposed_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_imposed = imposed_v.shape[1]
    u_dir = np.asarray(direction[: n_tseries * n_comp], dtype=float).reshape((n_tseries, n_comp), order="F")
    n_free = n_comp - n_imposed
    if n_free > 0:
        v_free = np.asarray(direction[n_tseries * n_comp :], dtype=float).reshape((n_epochs, n_free), order="F")
        zeros_imposed = np.zeros((n_epochs, n_imposed), dtype=float)
        v_dir = np.hstack([zeros_imposed, v_free]) if n_imposed > 0 else v_free
    else:
        v_dir = np.zeros((n_epochs, n_comp), dtype=float)
    return u_dir, v_dir


def _func_multi_component(
    x: np.ndarray,
    x_dat: np.ndarray,
    x_weight: np.ndarray,
    n_tseries: int,
    n_epochs: int,
    n_comp: int,
    imposed_v: np.ndarray,
) -> float:
    u, v = _reconstruct_multi_components(x, n_tseries, n_epochs, n_comp, imposed_v)
    return float(np.sum(((u @ v.T - x_dat) ** 2) * x_weight))


def _grad_multi_component(
    x: np.ndarray,
    x_dat: np.ndarray,
    x_weight: np.ndarray,
    n_tseries: int,
    n_epochs: int,
    n_comp: int,
    imposed_v: np.ndarray,
) -> np.ndarray:
    n_imposed = imposed_v.shape[1]
    u, v = _reconstruct_multi_components(x, n_tseries, n_epochs, n_comp, imposed_v)
    weighted_residual = (u @ v.T - x_dat) * x_weight
    grad_u = 2.0 * weighted_residual @ v
    pieces = [_matlab_vec(grad_u)]
    if n_comp > n_imposed:
        grad_v = 2.0 * weighted_residual.T @ u[:, n_imposed:]
        pieces.append(_matlab_vec(grad_v))
    return np.concatenate(pieces) if pieces else np.empty((0,), dtype=float)


def _calc_abg_multi_component(
    x: np.ndarray,
    direction: np.ndarray,
    x_dat: np.ndarray,
    n_tseries: int,
    n_epochs: int,
    n_comp: int,
    imposed_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_x, v_x = _reconstruct_multi_components(x, n_tseries, n_epochs, n_comp, imposed_v)
    u_r, v_r = _reconstruct_multi_direction(direction, n_tseries, n_epochs, n_comp, imposed_v)
    alpha = _matlab_vec(u_r @ v_r.T)
    beta = _matlab_vec(u_r @ v_x.T + u_x @ v_r.T)
    gamma = _matlab_vec(u_x @ v_x.T - x_dat)
    return alpha, beta, gamma


def _reconstruct_zero_sum_components(
    x: np.ndarray,
    n_tseries: int,
    n_epochs: int,
    n_comp: int,
    imposed_v: np.ndarray,
    transform_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_imposed = imposed_v.shape[1]
    means = np.asarray(x[:n_tseries], dtype=float)
    u_start = n_tseries
    u_stop = n_tseries * (n_comp + 1)
    u = np.asarray(x[u_start:u_stop], dtype=float).reshape((n_tseries, n_comp), order="F")
    n_free = n_comp - n_imposed
    if n_free > 0:
        w = np.asarray(x[u_stop:], dtype=float).reshape((n_epochs - 1, n_free), order="F")
        v_free = transform_inv @ w
        v = np.hstack([imposed_v, v_free]) if n_imposed > 0 else v_free
    else:
        v = imposed_v.copy()
    return means, u, v


def _reconstruct_zero_sum_direction(
    direction: np.ndarray,
    n_tseries: int,
    n_epochs: int,
    n_comp: int,
    imposed_v: np.ndarray,
    transform_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_imposed = imposed_v.shape[1]
    means_dir = np.asarray(direction[:n_tseries], dtype=float)
    u_start = n_tseries
    u_stop = n_tseries * (n_comp + 1)
    u_dir = np.asarray(direction[u_start:u_stop], dtype=float).reshape((n_tseries, n_comp), order="F")
    n_free = n_comp - n_imposed
    if n_free > 0:
        w_dir = np.asarray(direction[u_stop:], dtype=float).reshape((n_epochs - 1, n_free), order="F")
        v_free = transform_inv @ w_dir
        zeros_imposed = np.zeros((n_epochs, n_imposed), dtype=float)
        v_dir = np.hstack([zeros_imposed, v_free]) if n_imposed > 0 else v_free
    else:
        v_dir = np.zeros((n_epochs, n_comp), dtype=float)
    return means_dir, u_dir, v_dir


def _func_zero_sum(
    x: np.ndarray,
    x_dat: np.ndarray,
    x_weight: np.ndarray,
    n_tseries: int,
    n_epochs: int,
    n_comp: int,
    imposed_v: np.ndarray,
    transform_inv: np.ndarray,
) -> float:
    means, u, v = _reconstruct_zero_sum_components(x, n_tseries, n_epochs, n_comp, imposed_v, transform_inv)
    return float(np.sum(((u @ v.T - x_dat + means[:, None]) ** 2) * x_weight))


def _grad_zero_sum(
    x: np.ndarray,
    x_dat: np.ndarray,
    x_weight: np.ndarray,
    n_tseries: int,
    n_epochs: int,
    n_comp: int,
    imposed_v: np.ndarray,
    transform: np.ndarray,
    transform_inv: np.ndarray,
) -> np.ndarray:
    n_imposed = imposed_v.shape[1]
    means, u, v = _reconstruct_zero_sum_components(x, n_tseries, n_epochs, n_comp, imposed_v, transform_inv)
    weighted_residual = (u @ v.T - x_dat + means[:, None]) * x_weight
    grad_means = 2.0 * np.sum(weighted_residual, axis=1)
    grad_u = 2.0 * weighted_residual @ v
    pieces = [_matlab_vec(grad_means), _matlab_vec(grad_u)]
    if n_comp > n_imposed:
        grad_v = 2.0 * weighted_residual.T @ u[:, n_imposed:]
        grad_w = transform @ grad_v
        pieces.append(_matlab_vec(grad_w))
    return np.concatenate(pieces) if pieces else np.empty((0,), dtype=float)


def _calc_abg_zero_sum(
    x: np.ndarray,
    direction: np.ndarray,
    x_dat: np.ndarray,
    n_tseries: int,
    n_epochs: int,
    n_comp: int,
    imposed_v: np.ndarray,
    transform_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means_x, u_x, v_x = _reconstruct_zero_sum_components(x, n_tseries, n_epochs, n_comp, imposed_v, transform_inv)
    means_r, u_r, v_r = _reconstruct_zero_sum_direction(
        direction, n_tseries, n_epochs, n_comp, imposed_v, transform_inv
    )
    alpha = _matlab_vec(u_r @ v_r.T)
    beta = _matlab_vec(u_r @ v_x.T + u_x @ v_r.T + means_r[:, None])
    gamma = _matlab_vec(u_x @ v_x.T + means_x[:, None] - x_dat)
    return alpha, beta, gamma


def conjugate_gradient_legacy(
    x0: np.ndarray,
    func: Any,
    dfunc: Any,
    calc_abg: Any,
    x_weight: np.ndarray,
    iter_max: int,
    ftol: float,
) -> tuple[np.ndarray, float, int]:
    x = np.asarray(x0, dtype=float).copy()
    f_value = float(func(x))
    f_prime = np.asarray(dfunc(x), dtype=float)
    weight_col = _matlab_vec(x_weight)
    residual = -f_prime
    direction = residual.copy()
    delta_new = float(residual @ residual)
    fp = f_value

    for iteration in range(1, iter_max + 1):
        alpha, beta, gamma = calc_abg(x, direction)
        cubic_term = 4.0 * np.sum((alpha**2) * weight_col)
        quadratic_term = 3.0 * np.sum((2.0 * alpha * beta) * weight_col)
        linear_term = 2.0 * np.sum((2.0 * alpha * gamma + beta**2) * weight_col)
        constant_term = np.sum((2.0 * beta * gamma) * weight_col)
        derivative_roots = np.roots(np.asarray([cubic_term, quadratic_term, linear_term, constant_term], dtype=float))

        f_at_roots: list[float] = []
        real_roots: list[float] = []
        for root in derivative_roots:
            if abs(root.imag) <= 1e-12:
                root_real = float(root.real)
                real_roots.append(root_real)
                f_at_roots.append(float(func(x + root_real * residual)))
        if not real_roots:
            real_roots = [0.0]
            f_at_roots = [f_value]
        best_idx = int(np.argmin(f_at_roots))
        f_value = f_at_roots[best_idx]
        delta_d = real_roots[best_idx]

        x = x + delta_d * direction
        f_prime = np.asarray(dfunc(x), dtype=float)
        residual = -f_prime
        delta_old = delta_new
        delta_new = float(residual @ residual)
        beta_coeff = delta_new / delta_old if delta_old > 0.0 else 0.0
        direction = residual + beta_coeff * direction
        if float(residual @ direction) <= 0.0:
            direction = residual.copy()
        if 2.0 * abs(f_value - fp) < ftol * (abs(f_value) + abs(fp) + 1e-10):
            return x, f_value, iteration
        fp = f_value

    return x, f_value, iter_max


def _initial_guess_srebro(x_data_matrix: np.ndarray, n_comp: int) -> np.ndarray:
    if n_comp == 0:
        return np.empty((0,), dtype=float)
    u, singular_values, vh = np.linalg.svd(x_data_matrix, full_matrices=False)
    u_guess = u[:, :n_comp] @ np.diag(np.sqrt(singular_values[:n_comp]))
    v_guess = np.diag(np.sqrt(singular_values[:n_comp])) @ vh[:n_comp, :]
    return np.concatenate([_matlab_vec(u_guess), _matlab_vec(v_guess.T)])


def _take_svd(matrix: np.ndarray, n_comp: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_comp == 0:
        return (
            np.empty((matrix.shape[0], 0), dtype=float),
            np.empty((0, 0), dtype=float),
            np.empty((matrix.shape[1], 0), dtype=float),
        )
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    return u[:, :n_comp], np.diag(singular_values[:n_comp]), vh[:n_comp, :].T


def decomp_srebro_cg_simultaneous(
    x_dat: np.ndarray,
    x_weight: np.ndarray,
    n_comp: int,
    iter_max: int,
    tol: float,
    imposed_v: np.ndarray,
    heaviside_v: np.ndarray,
    u0: np.ndarray | None = None,
    s0: np.ndarray | None = None,
    v0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    n_tseries, n_epochs = x_dat.shape
    n_imposed = imposed_v.shape[1]
    n_free = n_comp - n_imposed
    if n_free < 0:
        raise ValueError(f"n_comp={n_comp} is smaller than the number of imposed components ({n_imposed})")

    if n_imposed > 0:
        us_imposed_guess = x_dat @ imposed_v
        s_imposed_guess = np.diag(np.diag(us_imposed_guess.T @ us_imposed_guess))
        u_imposed_guess = us_imposed_guess.copy()
        for idx in range(n_imposed):
            if s_imposed_guess[idx, idx] > 0.0:
                u_imposed_guess[:, idx] /= s_imposed_guess[idx, idx]
        x_minus_imposed = x_dat - u_imposed_guess @ s_imposed_guess @ imposed_v.T
    else:
        us_imposed_guess = np.empty((n_tseries, 0), dtype=float)
        x_minus_imposed = x_dat

    if u0 is None or s0 is None or v0 is None or u0.size == 0 or s0.size == 0 or v0.size == 0:
        x_guess = np.concatenate([_matlab_vec(us_imposed_guess), _initial_guess_srebro(x_minus_imposed, n_free)])
    else:
        n_comp_guess = u0.shape[1]
        if n_comp < n_comp_guess:
            raise ValueError("Initial decomposition guess has more components than requested n_comp")
        n_comp_min = min(n_comp_guess, n_comp)
        pieces: list[np.ndarray] = []
        if n_imposed > 0:
            pieces.append(_matlab_vec(u0[:, :n_imposed] @ s0[:n_imposed, :n_imposed]))
        if n_comp_min > n_imposed:
            s_guess = np.sqrt(s0[n_imposed:n_comp_min, n_imposed:n_comp_min])
            pieces.append(_matlab_vec(u0[:, n_imposed:n_comp_min] @ s_guess))
        if n_comp > n_comp_min:
            remainder = x_dat - u0[:, :n_comp_min] @ s0[:n_comp_min, :n_comp_min] @ v0[:, :n_comp_min].T
            extra_guess = _initial_guess_srebro(remainder, n_comp - n_comp_min)
            extra_u_size = n_tseries * (n_comp - n_comp_min)
            pieces.append(extra_guess[:extra_u_size])
        if n_comp_min > n_imposed:
            s_guess = np.sqrt(s0[n_imposed:n_comp_min, n_imposed:n_comp_min])
            pieces.append(_matlab_vec(v0[:, n_imposed:n_comp_min] @ s_guess))
        if n_comp > n_comp_min:
            extra_guess = _initial_guess_srebro(remainder, n_comp - n_comp_min)
            extra_u_size = n_tseries * (n_comp - n_comp_min)
            pieces.append(extra_guess[extra_u_size:])
        x_guess = np.concatenate(pieces) if pieces else np.empty((0,), dtype=float)

    func = lambda vector: _func_multi_component(vector, x_dat, x_weight, n_tseries, n_epochs, n_comp, imposed_v)
    dfunc = lambda vector: _grad_multi_component(vector, x_dat, x_weight, n_tseries, n_epochs, n_comp, imposed_v)
    calc_abg = lambda vector, direction: _calc_abg_multi_component(
        vector, direction, x_dat, n_tseries, n_epochs, n_comp, imposed_v
    )
    x_final, residual, iter_num = conjugate_gradient_legacy(x_guess, func, dfunc, calc_abg, x_weight, iter_max, tol)

    u = x_final[: n_comp * n_tseries].reshape((n_tseries, n_comp), order="F")
    if n_free > 0:
        v = x_final[n_comp * n_tseries :].reshape((n_epochs, n_free), order="F")
        u_free, s_free, v_free = _take_svd(u[:, n_imposed:] @ v.T, n_free)
    else:
        u_free = np.empty((n_tseries, 0), dtype=float)
        s_free = np.empty((0, 0), dtype=float)
        v_free = np.empty((n_epochs, 0), dtype=float)

    imposed_u = u[:, :n_imposed]
    if n_imposed > 0:
        imposed_s = np.diag(np.sqrt(np.maximum(np.diag(imposed_u.T @ imposed_u), 0.0)))
        normalized_imposed_u = imposed_u.copy()
        for idx in range(n_imposed):
            if imposed_s[idx, idx] > 0.0:
                normalized_imposed_u[:, idx] /= imposed_s[idx, idx]
    else:
        imposed_s = np.empty((0, 0), dtype=float)
        normalized_imposed_u = np.empty((n_tseries, 0), dtype=float)

    u_out = np.hstack([normalized_imposed_u, u_free]) if n_imposed > 0 else u_free
    s_out = block_diag(imposed_s, s_free) if n_imposed > 0 else s_free
    v_out = np.hstack([imposed_v, v_free]) if n_imposed > 0 else v_free

    initial_product = u_out @ s_out @ v_out.T if s_out.size else np.zeros_like(x_dat)
    epsilon = 1e-10
    if heaviside_v.size:
        for current_component in heaviside_v:
            step_epochs = np.flatnonzero(np.abs(np.diff(v_out[:, current_component])) > epsilon)
            if step_epochs.size == 0:
                continue
            step_epoch = int(step_epochs[0])
            heaviside_diff = v_out[-1, current_component] - v_out[0, current_component]
            if abs(heaviside_diff) <= epsilon:
                continue
            for idx in range(n_comp):
                if idx == current_component:
                    continue
                c_i = -(v_out[step_epoch + 1, idx] - v_out[step_epoch, idx])
                step_fcn = np.concatenate(
                    [np.zeros(step_epoch + 1, dtype=float), np.full(n_epochs - step_epoch - 1, c_i, dtype=float)]
                )
                step_fcn -= np.mean(step_fcn)
                v_out[:, idx] += step_fcn
                us = u_out @ s_out
                us[:, current_component] -= us[:, idx] * c_i / heaviside_diff
                s_prime = np.diag(np.sqrt(np.maximum(np.diag(us.T @ us), 0.0)))
                u_prime = us.copy()
                for jdx in range(s_prime.shape[0]):
                    if s_prime[jdx, jdx] > 0.0:
                        u_prime[:, jdx] /= s_prime[jdx, jdx]
                s_out = s_prime
                u_out = u_prime
    final_product = u_out @ s_out @ v_out.T if s_out.size else np.zeros_like(x_dat)
    if np.linalg.norm(initial_product - final_product) > epsilon:
        raise RuntimeError("Heaviside post-processing changed the reconstruction unexpectedly")

    return u_out, s_out, v_out, residual, iter_num


def center_basic_legacy(xd: dict[str, Any], n_components: int, imposed_v: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    filled_ts = fill_missing_rows(xd["ts"], xd["var_ts"])
    weights = (1.0 / xd["var_ts"]) ** 2
    weights[~np.isfinite(weights)] = 0.0
    offsets = weighted_row_means(filled_ts, weights)

    centered = dict(xd)
    centered["ts"] = filled_ts - offsets[:, None]
    centered["centering_offsets"] = offsets

    u, s, v = _take_svd(np.nan_to_num(centered["ts"], nan=0.0, posinf=0.0, neginf=0.0), n_components)
    pca_4cen = {
        "llh": xd["llh"],
        "timeline": xd["timeline"],
        "U": u,
        "S": s,
        "V": v,
        "V_imposed": imposed_v,
    }
    return centered, pca_4cen


def center_advanced_legacy(xd: dict[str, Any], cfg: Any, imposed_v: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    if cfg.centering.function != "decomp_CG_means":
        raise ValueError("Legacy advanced centering only supports centering.function='decomp_CG_means'")

    filled_ts = fill_missing_rows(xd["ts"], xd["var_ts"])
    weights = (1.0 / xd["var_ts"]) ** 2
    weights[~np.isfinite(weights)] = 0.0

    mean_offsets_rough = weighted_row_means(filled_ts, weights)
    x_dat_temp = filled_ts - mean_offsets_rough[:, None]

    n_tseries, n_epochs = x_dat_temp.shape
    n_comp = int(cfg.n_components)
    n_imposed = imposed_v.shape[1]
    n_free = n_comp - n_imposed
    if n_free < 0:
        raise ValueError(f"n_components={n_comp} is smaller than the number of imposed centering components ({n_imposed})")

    if n_imposed > 0:
        us_imposed_guess = x_dat_temp @ imposed_v
        s_imposed_guess = np.diag(np.diag(us_imposed_guess.T @ us_imposed_guess))
        u_imposed_guess = us_imposed_guess.copy()
        for idx in range(n_imposed):
            if s_imposed_guess[idx, idx] > 0.0:
                u_imposed_guess[:, idx] /= s_imposed_guess[idx, idx]
        x_minus_imposed = x_dat_temp - u_imposed_guess @ s_imposed_guess @ imposed_v.T
    else:
        u_imposed_guess = np.empty((n_tseries, 0), dtype=float)
        s_imposed_guess = np.empty((0, 0), dtype=float)
        x_minus_imposed = x_dat_temp

    start_guess = _centering_initial_guess(cfg, n_tseries, n_epochs, n_comp)
    if start_guess is None:
        u_guess, s_guess, v_guess = _take_svd(x_minus_imposed, n_free)
        u_augmented = np.hstack([u_imposed_guess, u_guess]) if n_imposed > 0 else u_guess
        s_augmented = block_diag(s_imposed_guess, s_guess) if n_imposed > 0 else s_guess
        v_augmented = np.hstack([imposed_v, v_guess]) if n_imposed > 0 else v_guess
    else:
        u_augmented, s_augmented, v_augmented = start_guess

    transform, transform_inv = _transformation_matrix(n_epochs)
    means = np.zeros(n_tseries, dtype=float)
    sqrt_s = np.sqrt(np.maximum(s_augmented, 0.0))
    v_means = np.mean(v_augmented, axis=0, keepdims=True)
    v_zero_sum = v_augmented - v_means
    v_zero_sum[-1, :] = -np.sum(v_zero_sum[:-1, :], axis=0)
    means = (u_augmented @ s_augmented @ v_means.T).reshape(-1)
    u_scaled = u_augmented @ sqrt_s
    v_scaled = v_zero_sum @ sqrt_s
    pieces = [_matlab_vec(means), _matlab_vec(u_scaled)]
    if n_free > 0:
        w = transform @ v_scaled[:, n_imposed:]
        pieces.append(_matlab_vec(w))
    x0 = np.concatenate(pieces)

    func = lambda vector: _func_zero_sum(
        vector, x_dat_temp, weights, n_tseries, n_epochs, n_comp, imposed_v, transform_inv
    )
    dfunc = lambda vector: _grad_zero_sum(
        vector, x_dat_temp, weights, n_tseries, n_epochs, n_comp, imposed_v, transform, transform_inv
    )
    calc_abg = lambda vector, direction: _calc_abg_zero_sum(
        vector, direction, x_dat_temp, n_tseries, n_epochs, n_comp, imposed_v, transform_inv
    )
    x_final, _, _ = conjugate_gradient_legacy(x0, func, dfunc, calc_abg, weights, int(cfg.centering.iter_max), float(cfg.centering.tol))

    means_fine = np.asarray(x_final[:n_tseries], dtype=float)
    us = np.asarray(x_final[n_tseries : n_tseries * (n_comp + 1)], dtype=float).reshape((n_tseries, n_comp), order="F")
    if n_free > 0:
        w_final = np.asarray(x_final[n_tseries * (n_comp + 1) :], dtype=float).reshape((n_epochs - 1, n_free), order="F")
        v_free_unscaled = transform_inv @ w_final
        v_unscaled = np.hstack([imposed_v, v_free_unscaled]) if n_imposed > 0 else v_free_unscaled
    else:
        v_unscaled = imposed_v.copy()

    us_norms = np.sqrt(np.maximum(np.diag(us.T @ us), 0.0))
    v_norms = np.sqrt(np.maximum(np.diag(v_unscaled.T @ v_unscaled), 0.0))
    u = us.copy()
    v = v_unscaled.copy()
    for idx, norm in enumerate(us_norms):
        if norm > 0.0:
            u[:, idx] /= norm
    for idx, norm in enumerate(v_norms):
        if norm > 0.0:
            v[:, idx] /= norm
    s = np.diag(us_norms) @ np.diag(v_norms)

    final_offsets = mean_offsets_rough + means_fine
    centered = dict(xd)
    centered["ts"] = filled_ts - final_offsets[:, None]
    centered["centering_offsets"] = final_offsets
    pca_4cen = {
        "llh": xd["llh"],
        "timeline": xd["timeline"],
        "U": u,
        "S": s,
        "V": v,
        "V_imposed": imposed_v,
    }
    return centered, pca_4cen
