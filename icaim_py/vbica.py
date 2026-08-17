from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import digamma, gammaln

from .common import (
    EPS,
    calc_chi2_matrix,
    calc_displ_lincomp,
    calc_variance_explained,
    matlab_std,
    matlab_var,
    normalize_decomposition_mode,
    solve_linear,
)


def scalexp(log_eta: np.ndarray, all_values: bool = False) -> np.ndarray:
    comps, points = log_eta.shape
    if all_values:
        scale = np.max(log_eta)
        index = log_eta - scale
        bit1 = np.sum(np.exp(index))
        z = np.log(bit1) + scale
        return np.exp(log_eta - z)
    scale = np.max(log_eta, axis=0, keepdims=True)
    index = log_eta - scale
    bit1 = np.sum(np.exp(index), axis=0, keepdims=True)
    z = np.log(bit1 + EPS) + scale
    return np.exp(log_eta - z)


def kmeans1(k: int, y: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=float).reshape(-1)
    n = y.size
    x = np.sort(y)
    seeds = np.ceil(np.cumsum(np.array([1.0] + [2.0] * (k - 1)) * n / (2.0 * k))).astype(int) - 1
    seeds = np.clip(seeds, 0, n - 1)

    last_i = np.ones(n, dtype=int)
    m = x[seeds].copy()
    eta = np.ones(n, dtype=float)

    loops = 0
    for loops in range(1, 101):
        d = np.vstack([(y - centre) ** 2 for centre in m])
        i = np.argmin(d, axis=0)
        if np.sum(i - last_i) == 0:
            break
        for j in range(k):
            mask = i == j
            if np.sum(eta[mask]) == 0:
                m[j] = np.sum(eta[mask] * y[mask])
            else:
                m[j] = np.sum(eta[mask] * y[mask]) / np.sum(eta[mask])
        last_i = i

    v = np.zeros(k, dtype=float)
    mix_prob = np.zeros(k, dtype=float)
    gammas = np.zeros((k, n), dtype=float)
    for j in range(k):
        mask = i == j
        denom = np.sum(eta[mask]) + EPS
        v[j] = np.sum(eta[mask] * (y[mask] - m[j]) ** 2) / denom
        if v[j] == 0:
            v[j] = 1000.0
        mix_prob[j] = np.sum(mask) / n
        gammas[j, :] = (1.0 / (2.0 * np.pi * v[j])) * np.exp(-((y - m[j]) ** 2) / (2.0 * v[j])) * mix_prob[j]
    gammas /= np.sum(gammas, axis=0, keepdims=True)
    return {"gammas": gammas, "v": v, "m": m, "pi": mix_prob / np.sum(mix_prob), "k": k, "nloops": loops, "last_i": last_i}


def init_mog1(x: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    mx = np.mean(x)
    vx = matlab_var(x)
    bx = 1.0 / vx
    mean_precision = bx
    var_precision = bx / n
    b_0 = var_precision / mean_precision
    c_0 = (mean_precision**2) / var_precision
    m_0 = mx
    v_0 = (0.3 * (np.max(x) - np.min(x))) ** 2
    tau_0 = 1.0 / v_0
    b = np.array([b_0], dtype=float)
    c = np.array([c_0], dtype=float)
    mm = np.array([m_0], dtype=float)
    tau = np.array([tau_0], dtype=float)
    lam = np.array([float(n)], dtype=float)
    return {
        "type": "g",
        "m": 1,
        "priors": {"lambda_0": float(n), "m_0": m_0, "tau_0": tau_0, "b_0": b_0, "c_0": c_0},
        "posts": {"lambda": lam, "mm": mm, "tau": tau, "b": b, "c": c},
        "pi": np.array([1.0], dtype=float),
        "centres": mm.copy(),
        "precs": b * c,
        "gammas": np.ones((1, n), dtype=float),
    }


def init_mog(x: np.ndarray, m: int, init_method: str, n_source: int, priors: dict[str, Any]) -> dict[str, Any]:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    if m == 1:
        return init_mog1(x)

    set_priors = int(priors.get("setSource", 0)) == 1
    lambda_0 = 5.0
    if set_priors:
        lambda_0 = float(priors["lambda_0"][n_source - 1])
    m_0 = np.mean(x)
    v_0 = (0.3 * (np.max(x) - np.min(x))) ** 2
    tau_0 = 1.0 / v_0

    if init_method != "kmeans":
        raise NotImplementedError(f"Unsupported source_init for active port: {init_method}")

    kmeans = kmeans1(m, x)
    gammas = kmeans["gammas"]
    kv = kmeans["v"]
    mean_precision = np.mean(1.0 / kv)
    var_precision = matlab_std(1.0 / kv) ** 2
    b_0 = float(np.mean(var_precision / mean_precision))
    c_0 = float(np.mean((mean_precision**2) / var_precision))
    lam = kmeans["pi"]
    mm = kmeans["m"].copy()
    v = np.mean(kv) * np.arange(1, m + 1, dtype=float) / (n / m)
    b = np.zeros(m, dtype=float)
    c = np.zeros(m, dtype=float)
    for idx in range(m):
        precision = 1.0 / kmeans["v"][idx] if kmeans["v"][idx] > 0 else mean_precision
        b[idx] = var_precision / precision
        c[idx] = (precision**2) / var_precision

    if set_priors:
        m_0 = float(priors["m_0"][n_source - 1])
        b_0 = float(priors["b_0"][n_source - 1])
        c_0 = float(priors["c_0"][n_source - 1])
        tau_0 = float(priors["tau_0"][n_source - 1])
        lambda_0 = float(priors["lambda_0"][n_source - 1])

    return {
        "type": "g",
        "m": m,
        "priors": {"lambda_0": lambda_0, "m_0": m_0, "tau_0": tau_0, "b_0": b_0, "c_0": c_0},
        "posts": {"lambda": lam, "mm": mm, "tau": 1.0 / v, "b": b, "c": c},
        "pi": lam / np.sum(lam),
        "centres": mm.copy(),
        "precs": b * c,
        "gammas": gammas,
    }


def initialise_mix1d(x: np.ndarray, m: int, source_type: str, init_method: str, n_source: int, priors: dict[str, Any]) -> dict[str, Any]:
    if source_type != "g":
        raise NotImplementedError(f"Unsupported source_type in Python port: {source_type}")
    return init_mog(x, m, init_method, n_source, priors)


def log_ptilde2(y: np.ndarray, y_prec: np.ndarray, src: dict[str, Any]) -> np.ndarray:
    m, n = y.shape
    b = np.asarray(src["posts"]["b"], dtype=float)
    c = np.asarray(src["posts"]["c"], dtype=float)
    mm = np.asarray(src["centres"], dtype=float)
    tau = np.asarray(src["posts"]["tau"], dtype=float)
    mm_sq = (mm**2 + 1.0 / tau)[:, None]
    log_tilde_beta = (digamma(c) + np.log(b + EPS))[:, None]
    mean_beta = c * b
    databit = 0.5 * (y * y * y_prec - mm_sq * mean_beta[:, None] - np.log(y_prec))
    return 0.5 * log_tilde_beta + databit


def gammas(src: dict[str, Any], y: np.ndarray, y_sq: np.ndarray, algorithm: int) -> tuple[np.ndarray, int]:
    if algorithm != 2:
        raise NotImplementedError("Only vbICA2 E-step is implemented in the Python port")
    lam = np.asarray(src["posts"]["lambda"], dtype=float)
    log_obslike_tilde = log_ptilde2(y, y_sq, src)
    log_tilde_pi = digamma(lam) - digamma(np.sum(lam))
    log_gam = log_tilde_pi[:, None] + log_obslike_tilde
    if log_gam.shape[0] == 1:
        return scalexp(log_gam, all_values=True), 0
    return scalexp(log_gam), 0


def learn_mix1d(src: dict[str, Any], x: np.ndarray, x_sq: np.ndarray, tol: float, max_steps: int) -> dict[str, Any]:
    src_type = src["type"]
    if src_type != "g":
        raise NotImplementedError("Only Gaussian source mixtures are implemented in the Python port")

    m, n = x.shape
    if m != int(src["m"]):
        raise ValueError("Source signal dimensionality and source model mismatch")

    gamma = src["gammas"].copy()
    m_q = x.copy()
    b_q = x_sq.copy()
    x_true = m_q
    x_sq_true = m_q**2 + 1.0 / b_q

    lambda_0 = float(src["priors"]["lambda_0"])
    b_0 = float(src["priors"]["b_0"])
    c_0 = float(src["priors"]["c_0"])
    m_0 = float(src["priors"]["m_0"])
    tau_0 = float(src["priors"]["tau_0"])

    b = np.asarray(src["posts"]["b"], dtype=float).copy()
    c = np.asarray(src["posts"]["c"], dtype=float).copy()
    mm = np.asarray(src["posts"]["mm"], dtype=float).copy()
    tau = np.asarray(src["posts"]["tau"], dtype=float).copy()

    ftot = 0.0
    outer_steps = 1
    inner_steps = max_steps

    for _ in range(outer_steps):
        if m == 1:
            gamma = np.ones((1, n), dtype=float)
        else:
            gamma, _ = gammas(src, m_q, b_q, 2)

        err = np.inf
        for _ in range(inner_steps):
            gamma_sum = np.sum(gamma, axis=1)
            lam = lambda_0 + gamma_sum
            src["pi"] = lam / np.sum(lam)
            src["posts"]["lambda"] = lam

            lambda_p = lambda_0 * np.ones(m, dtype=float)
            dir1 = np.sum(gammaln(lam + EPS) - gammaln(lambda_p + EPS))
            dir2 = gammaln(np.sum(lam + EPS)) - gammaln(np.sum(lambda_p + EPS))
            f_dir = dir1 - dir2
            ent_gam = -np.sum(gamma * np.log(gamma + EPS))

            mean_xsq = np.sum(gamma * x_sq_true, axis=1)
            mean_x = np.sum(gamma * x_true, axis=1)
            mu_sq = gamma_sum * (mm**2 + 1.0 / tau)
            data_bit = mean_xsq - 2.0 * mm * mean_x + mu_sq
            b = 1.0 / ((1.0 / b_0) + 0.5 * data_bit)
            c = c_0 + 0.5 * gamma_sum
            mean_beta = b * c
            src["posts"]["b"] = b
            src["posts"]["c"] = c
            f_beta = np.sum(gammaln(c) - gammaln(c_0) + c * np.log(b) - c_0 * np.log(b_0))

            tau = tau_0 + mean_beta * gamma_sum
            mm = (m_0 + mean_beta * mean_x) / tau
            src["posts"]["mm"] = mm
            src["posts"]["tau"] = tau

            b_ratio = tau_0 / tau
            f_gauss = -0.5 * np.sum(-np.log(b_ratio) + b_ratio - 1.0 + tau_0 * (mm - m_0) ** 2)

            old_fm = ftot
            f_hidd = ent_gam - n / 2.0 * np.log(2.0 * np.pi)
            f_params = f_gauss + f_beta + f_dir
            ftot = f_hidd + f_params
            err = abs((old_fm - ftot) / ftot) if ftot != 0 else np.inf
            if err < tol:
                break
        if err < tol:
            break

    src["centres"] = mm
    src["precs"] = b * c
    src["gammas"] = gamma
    src["f_hidd"] = f_hidd
    src["f_params"] = f_params
    src["ftot"] = ftot
    return src


def recon_source2(
    data: np.ndarray,
    data_mask: np.ndarray,
    old_x: np.ndarray,
    h: np.ndarray,
    h_sq: np.ndarray,
    lambda_hat: np.ndarray,
    source: list[dict[str, Any]],
    tol: float,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], float]:
    sensors, sources = h.shape
    _, points = data.shape

    preweight = np.zeros((sources, points), dtype=float)
    for t in range(points):
        preweight[:, t] = h_sq.T @ (data_mask[:, t] * lambda_hat)
    weight_h = preweight

    gammas_list: list[np.ndarray] = []
    betas: list[np.ndarray] = []
    mean_prior: list[np.ndarray] = []
    beta_x_giv_q: list[np.ndarray] = []
    for idx in range(sources):
        gammas_list.append(np.asarray(source[idx]["gammas"], dtype=float))
        precs = np.asarray(source[idx]["precs"], dtype=float)
        mus = np.asarray(source[idx]["centres"], dtype=float)
        betas.append(precs)
        mean_prior.append(precs * mus)
        beta_x_giv_q.append(precs[:, None] + weight_h[idx][None, :])

    noise_h = lambda_hat[:, None] * h
    data_bit = noise_h.T @ (data_mask * data)

    err_x = tol + 1.0
    x = old_x.copy()
    x_q: list[np.ndarray] = []
    mu_x_giv_q: list[np.ndarray] = []
    steps = 0
    while err_x > tol:
        steps += 1
        x_q = []
        mu_x_giv_q = []
        for idx in range(sources):
            h_not_i = h.copy()
            h_not_i[:, idx] = 0.0
            prior = mean_prior[idx][:, None]
            data_mean = data_bit[idx] - noise_h[:, idx].T @ (data_mask * (h_not_i @ old_x))
            bracketbit = prior + data_mean[None, :]
            mu = bracketbit / beta_x_giv_q[idx]
            mu_x_giv_q.append(mu)
            x_q.append(mu)
            x[idx, :] = np.sum(gammas_list[idx] * mu, axis=0)
        err_x = np.linalg.norm(x - old_x)
        if steps == max_steps:
            break
        old_x = x.copy()

    x_sq_q: list[np.ndarray] = []
    x_sq = np.zeros((sources, points), dtype=float)
    preentx = np.zeros((sources, points), dtype=float)
    for idx in range(sources):
        x_sq_local = mu_x_giv_q[idx] ** 2 + 1.0 / beta_x_giv_q[idx]
        x_sq_q.append(x_sq_local)
        x_sq[idx, :] = np.sum(gammas_list[idx] * x_sq_local, axis=0)
        preentx[idx, :] = -np.sum(gammas_list[idx] * np.log(beta_x_giv_q[idx]), axis=0)
    entx = 0.5 * np.sum(preentx) + sources * points / 2.0
    return x, x_sq, x_q, x_sq_q, mu_x_giv_q, beta_x_giv_q, float(entx)


def learn_matrix(
    data: np.ndarray,
    data_mask: np.ndarray,
    old_h: np.ndarray,
    x: np.ndarray,
    x_sq: np.ndarray,
    lambda_hat: np.ndarray,
    alpha_prec: np.ndarray,
    alpha_mean: np.ndarray,
    tol: float,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    sensors, points = data.shape
    _, sources = old_h.shape

    sum_x_sq = np.zeros((sensors, sources), dtype=float)
    for j in range(sensors):
        sum_x_sq[j, :] = np.sum(data_mask[j][None, :] * x_sq, axis=1)
    network_term = lambda_hat[:, None] * sum_x_sq
    alpha_hat = alpha_prec[None, :] + network_term

    coefficient = 1.0 / alpha_hat
    big_noise = lambda_hat[:, None]

    x1 = np.zeros((sensors, sources), dtype=float)
    for j in range(sensors):
        x_mask = data_mask[j][None, :] * x
        x1[j, :] = np.diag(x @ x_mask.T)

    h = old_h.copy()
    m_h_hat = h.copy()
    for _ in range(max_steps):
        x2 = (data_mask * (data - old_h @ x)) @ x.T
        x3 = x1 * old_h + x2
        m_h_hat = coefficient * (-alpha_mean[None, :] + big_noise * x3)
        h = m_h_hat.copy()
        err_h = np.linalg.norm(h - old_h)
        if err_h < tol:
            break
        old_h = h.copy()

    h_sq = m_h_hat**2 + 1.0 / alpha_hat
    f_mix = 0.5 * sensors * sources - 0.5 * np.sum(np.log(alpha_hat))
    return h, h_sq, m_h_hat, alpha_hat, float(f_mix)


def update_alpha(priors: dict[str, Any], h_sq_column: np.ndarray, ard: int, num_source: int) -> tuple[float, float, float]:
    b_alpha_0 = float(priors["b_alpha_0"][num_source])
    c_alpha_0 = float(priors["c_alpha_0"][num_source])
    sum_mix_sq = float(np.sum(h_sq_column))
    if ard:
        b_alpha_hat = 1.0 / ((1.0 / b_alpha_0) + 0.5 * sum_mix_sq)
        c_alpha_hat = c_alpha_0 + h_sq_column.size * 0.5
        fa1 = gammaln(c_alpha_hat) - gammaln(c_alpha_0)
    else:
        total_sq = sum_mix_sq
        b_alpha_hat = 1.0 / ((1.0 / b_alpha_0) + 0.5 * total_sq)
        c_alpha_hat = c_alpha_0 + h_sq_column.size / 2.0
        fa1 = gammaln(c_alpha_hat) - gammaln(c_alpha_0)
    fa2 = c_alpha_hat * np.log(b_alpha_hat + EPS) - c_alpha_0 * np.log(b_alpha_0 + EPS)
    return float(b_alpha_hat), float(c_alpha_hat), float(fa1 + fa2)


def update_mean(priors: dict[str, Any], data: np.ndarray, data_mask: np.ndarray, h: np.ndarray, x: np.ndarray, lambda_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    sensors, _ = data.shape
    sumeta = np.sum(data_mask, axis=1)
    mb0 = float(priors["mb0"])
    mn0 = float(priors["mn0"])
    mb = mb0 + sumeta * lambda_hat
    ratio = lambda_hat / mb
    diff = mb0 * mn0 + np.sum(data_mask * (data - h @ x), axis=1)
    mn = ratio * diff
    pratio = mb0 / mb
    bit1 = pratio - 1.0 - np.log(pratio)
    bit2 = mb0 * (mn - mn0) ** 2
    f_dc = -0.5 * np.sum(bit1 + bit2)
    return mn, mb, float(f_dc)


def learn_noise(
    priors: dict[str, Any],
    data: np.ndarray,
    data_mask: np.ndarray,
    mb: np.ndarray,
    h: np.ndarray,
    x: np.ndarray,
    h_sq: np.ndarray,
    x_sq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    sensors, points = data.shape
    b_lam_0 = float(priors["b_Lam_0"])
    c_lam_0 = float(priors["c_Lam_0"])
    sumeta = np.sum(data_mask, axis=1)

    data_hat = h @ x
    dd = data**2
    ddh = data * data_hat
    data_hat_var = h_sq @ x_sq - (h**2) @ (x**2)
    dhdh = data_hat**2 + data_hat_var
    mean_bit = (1.0 / mb)[:, None]
    ed = np.sum(data_mask * (dd - 2.0 * ddh + dhdh + mean_bit), axis=1)

    b_lam_hat = 1.0 / ((1.0 / b_lam_0) + 0.5 * ed)
    c_lam_hat = c_lam_0 + sumeta / 2.0
    f_lambda = np.sum(gammaln(c_lam_hat) - gammaln(c_lam_0) + c_lam_hat * np.log(b_lam_hat) - c_lam_0 * np.log(b_lam_0))
    return b_lam_hat * c_lam_hat, b_lam_hat, float(f_lambda)


def init_vbica(data: np.ndarray, init_parameters: dict[str, Any], priors: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray]:
    source_type = init_parameters["source_type"]
    source_init = init_parameters["source_init"]
    net_init = init_parameters["net_init"]
    states = np.asarray(init_parameters["states"], dtype=int)
    sources = states.size
    sensors, points = data.shape
    init_data = data
    mn = np.zeros(sensors, dtype=float)

    if source_type != "g":
        raise NotImplementedError("The Python port currently supports only Gaussian sources")

    if net_init == "SVD":
        _, singular_values, vh_svd = np.linalg.svd(init_data.T, full_matrices=False)
        h = vh_svd.T[:, :sources]
        m_h_hat = h.copy()
        if sensors == sources:
            var_noise = matlab_var(init_data.T, axis=0) / 5.0
            isovalue = np.mean(1.0 / var_noise)
        else:
            unexplained = singular_values[sources : min(sensors, points)]
            mean_crap = float(np.mean(unexplained) ** 2 / points)
            isovalue = 1.0 / mean_crap
    elif net_init == "SVD_S&J":
        h = np.asarray(init_parameters["U"], dtype=float)[:, :sources]
        m_h_hat = h.copy()
        if sensors == sources:
            var_noise = matlab_var(init_data.T, axis=0) / 5.0
            isovalue = np.mean(1.0 / var_noise)
        else:
            _, singular_values, _ = np.linalg.svd(init_data, full_matrices=False)
            unexplained = singular_values[sources : min(sensors, points)]
            mean_crap = float(np.mean(unexplained) ** 2 / points)
            isovalue = 1.0 / mean_crap
    else:
        raise NotImplementedError(f"Unsupported net_init in Python port: {net_init}")
    lambda_hat = np.ones(sensors, dtype=float) * isovalue
    x = solve_linear(h.T @ h, h.T @ init_data)
    mu_x_hat = x.copy()

    source_models = []
    for idx in range(sources):
        source_models.append(
            initialise_mix1d(
                x[idx, :],
                int(states[idx]),
                source_type,
                source_init,
                idx + 1,
                priors["source"],
            )
        )
    return lambda_hat, m_h_hat, mu_x_hat, source_models, mn


def initialise_ica(data: np.ndarray, init_parameters: dict[str, Any]) -> dict[str, Any]:
    source_type = init_parameters["source_type"]
    states = np.asarray(init_parameters["states"], dtype=int)
    sources = states.size
    sensors, _ = data.shape

    priors = {
        "mix": {
            "b_alpha_0": np.asarray(init_parameters["mix"]["b_alpha_0"], dtype=float),
            "c_alpha_0": np.asarray(init_parameters["mix"]["c_alpha_0"], dtype=float),
        },
        "noise": {
            "b_Lam_0": float(init_parameters["noise"]["b_Lam_0"]),
            "c_Lam_0": float(init_parameters["noise"]["c_Lam_0"]),
            "mb0": float(init_parameters["noise"]["mb0"]),
            "mn0": float(init_parameters["noise"]["mn0"]),
        },
        "source": {
            "m_0": np.asarray(init_parameters["source"]["m_0"], dtype=float),
            "tau_0": np.asarray(init_parameters["source"]["tau_0"], dtype=float),
            "b_0": np.asarray(init_parameters["source"]["b_0"], dtype=float),
            "c_0": np.asarray(init_parameters["source"]["c_0"], dtype=float),
            "lambda_0": np.asarray(init_parameters["source"]["lambda_0"], dtype=float),
            "setSource": int(init_parameters["source"]["setSource"]),
        },
    }

    lambda_hat, m_h_hat, mu_x_hat, source_models, mn = init_vbica(data, init_parameters, priors)
    mb = np.repeat(priors["noise"]["mb0"], sensors)
    beta_x_hat = 1.0 / ((1e-4) * np.ones_like(mu_x_hat))
    b_alpha_hat = np.asarray(priors["mix"]["b_alpha_0"], dtype=float)
    c_alpha_hat = np.asarray(priors["mix"]["c_alpha_0"], dtype=float)
    alpha = b_alpha_hat * c_alpha_hat
    alpha_hat = np.repeat(alpha[None, :], sensors, axis=0)
    points = data.shape[1]
    c_noise = points * sensors / 2.0

    return {
        "source_type": source_type,
        "source_states": states,
        "sources": source_models,
        "alphas": alpha,
        "noise": lambda_hat,
        "mean": mn,
        "priors": priors,
        "posts": {
            "x_mean": mu_x_hat,
            "x_prec": beta_x_hat,
            "mix": {"H_mean": m_h_hat, "H_prec": alpha_hat},
            "alpha": {"b": b_alpha_hat, "c": c_alpha_hat},
            "noise": {"b": lambda_hat / c_noise, "c": np.repeat(c_noise, sensors), "mn": mn, "mb": mb},
        },
        "net_init": init_parameters["net_init"],
        "source_init": init_parameters["source_init"],
    }


def learn_ica2(data: np.ndarray, init_parameters: dict[str, Any], network: dict[str, Any], data_mask: np.ndarray) -> dict[str, Any]:
    source = network["sources"]
    priors = network["priors"]
    alpha = np.asarray(network["alphas"], dtype=float)
    mu_x_hat = np.asarray(network["posts"]["x_mean"], dtype=float)
    beta_x_hat = np.asarray(network["posts"]["x_prec"], dtype=float)
    m_h_hat = np.asarray(network["posts"]["mix"]["H_mean"], dtype=float)
    alpha_hat = np.asarray(network["posts"]["mix"]["H_prec"], dtype=float)
    lambda_hat = np.asarray(network["noise"], dtype=float)
    mn = np.asarray(network["posts"]["noise"]["mn"], dtype=float)
    mb = np.asarray(network["posts"]["noise"]["mb"], dtype=float)

    x = mu_x_hat.copy()
    x_sq = mu_x_hat**2 + 1.0 / beta_x_hat
    h = m_h_hat.copy()
    h_sq = m_h_hat**2 + 1.0 / alpha_hat

    sensors, points = data.shape
    sources_n = len(source)
    dc = np.repeat(mn[:, None], points, axis=1)
    sumeta = points

    fact_source = 1.0
    fact2 = 1e10
    factx = 1e10
    facth = 1e10
    src_steps = int(init_parameters["max_steps"])
    x_steps = int(init_parameters["max_steps"])
    h_steps = int(init_parameters["max_steps"])
    max2_steps = int(init_parameters["max_steps"])
    tol = float(init_parameters["tol"])
    ard = int(init_parameters["ARD"])

    steps = 0
    err = tol + 1.0
    ftot = 0.0
    f_lambda = 0.0
    f_h = 0.0
    f_dc = 0.0
    energy_path: list[float] = []
    mu_q: list[np.ndarray] = []
    beta_q: list[np.ndarray] = []

    while err > tol:
        steps += 1
        iter2 = 0
        err2 = fact2 * tol + 1.0

        while err2 > fact2 * tol:
            iter2 += 1
            old_x = x.copy()
            x, x_sq, x_q, x_sq_q, mu_q, beta_q, ent_x = recon_source2(
                data=data - dc,
                data_mask=data_mask,
                old_x=x,
                h=h,
                h_sq=h_sq,
                lambda_hat=lambda_hat,
                source=source,
                tol=factx * tol,
                max_steps=x_steps,
            )

            alpha_prec = alpha.copy()
            alpha_mean = np.zeros_like(alpha_prec)
            old_h = h.copy()
            h, h_sq, m_h_hat, alpha_hat, f_h = learn_matrix(
                data=data - dc,
                data_mask=data_mask,
                old_h=h,
                x=x,
                x_sq=x_sq,
                lambda_hat=lambda_hat,
                alpha_prec=alpha_prec,
                alpha_mean=alpha_mean,
                tol=facth * tol,
                max_steps=h_steps,
            )

            b_alpha_hat = np.zeros(sources_n, dtype=float)
            c_alpha_hat = np.zeros(sources_n, dtype=float)
            f_alph_terms = np.zeros(sources_n, dtype=float)
            for idx in range(sources_n):
                b_alpha_hat[idx], c_alpha_hat[idx], f_alph_terms[idx] = update_alpha(priors["mix"], h_sq[:, idx], ard, idx)
            f_alph = float(np.sum(f_alph_terms))
            alpha = b_alpha_hat * c_alpha_hat

            if iter2 == max2_steps:
                break
            errx2 = np.linalg.norm(x - old_x) / points
            errh2 = np.linalg.norm(h - old_h)
            err2 = errx2 + errh2

        mn, mb, f_dc = update_mean(priors["noise"], data, data_mask, h, x, lambda_hat)
        dc = np.repeat(mn[:, None], points, axis=1)

        lambda_hat, b_lam_hat, f_lambda = learn_noise(priors["noise"], data - dc, data_mask, mb, h, x, h_sq, x_sq)
        c_lam_hat = lambda_hat / b_lam_hat

        fsp = np.zeros(sources_n, dtype=float)
        fsh = np.zeros(sources_n, dtype=float)
        for idx in range(sources_n):
            source[idx] = learn_mix1d(source[idx], mu_q[idx], beta_q[idx], fact_source * tol, src_steps)
            fsp[idx] = source[idx]["f_params"]
            fsh[idx] = source[idx]["f_hidd"] + points / 2.0 * np.log(2.0 * np.pi)

        f_mix = f_alph + f_h
        f_noise = f_dc + f_lambda
        f_params = np.sum(fsp) + f_mix + f_noise
        f_hidd = np.sum(fsh) + ent_x - 0.5 * sumeta * sensors * np.log(2.0 * np.pi)
        oldf = ftot
        ftot = float(f_params + f_hidd)
        energy_path.append(ftot)
        err = abs((ftot - oldf) / ftot) if ftot != 0 else np.inf
        if steps == int(init_parameters["max_steps"]):
            break

    state_paths = np.zeros((sources_n, points), dtype=int)
    for idx in range(sources_n):
        state_paths[idx, :] = np.argmax(source[idx]["gammas"], axis=0) + 1

    x_mean = np.zeros_like(x)
    for idx in range(sources_n):
        x_mean[idx, :] = np.sum(source[idx]["gammas"] * mu_q[idx], axis=0)
    beta_x_hat = 1.0 / ((x_sq - x**2) + EPS)

    network["state_paths"] = state_paths
    network["recon"] = x
    network["sources"] = source
    network["mixmatrix"] = h
    network["alphas"] = alpha
    network["noise"] = lambda_hat
    network["mean"] = mn
    network["posts"]["x_mean_q"] = mu_q
    network["posts"]["x_prec_q"] = beta_q
    network["posts"]["x_mean"] = x_mean
    network["posts"]["x_prec"] = beta_x_hat
    network["posts"]["mix"]["H_mean"] = m_h_hat
    network["posts"]["mix"]["H_prec"] = alpha_hat
    network["posts"]["alpha"]["b"] = b_alpha_hat
    network["posts"]["alpha"]["c"] = c_alpha_hat
    network["posts"]["noise"]["b"] = b_lam_hat
    network["posts"]["noise"]["c"] = c_lam_hat
    network["posts"]["noise"]["mn"] = mn
    network["posts"]["noise"]["mb"] = mb
    network["energy_path"] = np.asarray(energy_path, dtype=float)
    network["f_params"] = float(f_params)
    network["f_hidd"] = float(f_hidd)
    network["energy"] = float(energy_path[-1])
    network["model"] = "vbICA2"
    network["factSource"] = fact_source
    network["fact2_start"] = 1e10
    network["factx_start"] = 1e10
    network["facth_start"] = 1e10
    network["fact2"] = fact2
    network["factx"] = factx
    network["facth"] = facth
    network["n_steps_new_fact"] = 50
    return network


def vbica2(data: np.ndarray, init_parameters: dict[str, Any], data_mask: np.ndarray) -> dict[str, Any]:
    network = initialise_ica(data, init_parameters)
    return learn_ica2(data, init_parameters, network, data_mask)


def sort_components(decomp: dict[str, Any]) -> dict[str, Any]:
    order = tuple(np.argsort(np.diag(decomp["S"]))[::-1])
    decomp["U"] = decomp["U"][:, order]
    decomp["S"] = np.diag(np.diag(decomp["S"])[list(order)])
    decomp["V"] = decomp["V"][:, order]
    decomp["var_U"] = decomp["var_U"][:, order]
    decomp["var_V"] = decomp["var_V"][:, order]
    decomp["ind_sorting"] = np.asarray(order, dtype=int) + 1
    return decomp


def decompose_ica(xd: dict[str, Any], pca: dict[str, Any], init_parameters: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    decomposition_mode = normalize_decomposition_mode(init_parameters.get("decomposition_mode", "t"))
    source_ts = xd["ts"]
    source_var_ts = xd["var_ts"]
    xdat = source_ts.T if decomposition_mode == "s" else source_ts
    var_ts = source_var_ts.T if decomposition_mode == "s" else source_var_ts
    xweight = (1.0 / var_ts) ** 2
    xweight[~np.isfinite(xweight)] = 0.0
    data_mask = np.ones_like(xdat, dtype=float)
    ind_missing = np.isinf(var_ts)
    data_mask[ind_missing] = 0.0

    net = vbica2(xdat, init_parameters, data_mask)
    a_recon_work = net["mixmatrix"]
    s_recon_work = net["recon"]
    t = xdat.shape[1]
    adjusting_offset = np.repeat(net["mean"][:, None], t, axis=1)
    x_ica_work = a_recon_work @ s_recon_work + adjusting_offset
    a_recon_var_work = 1.0 / net["posts"]["mix"]["H_prec"]
    s_recon_var_work = 1.0 / net["posts"]["x_prec"]
    x_ica_var_work = (
        a_recon_var_work @ s_recon_var_work
        + a_recon_var_work @ (s_recon_work**2)
        + (a_recon_work**2) @ s_recon_var_work
    )

    a_col_norm = np.sqrt(np.sum(a_recon_work**2, axis=0))
    lambda_a = np.diag(a_col_norm)
    s_row_norm = np.sqrt(np.sum(s_recon_work**2, axis=1))
    lambda_s = np.diag(s_row_norm)
    a_norm = a_recon_work / a_col_norm[None, :]
    s_norm = s_recon_work / s_row_norm[:, None]
    u_work = a_norm
    s_ica = lambda_a @ lambda_s
    v_work = s_norm.T
    u_var_work = a_recon_var_work * ((1.0 / a_col_norm) ** 2)[None, :]
    v_var_work = (((1.0 / s_row_norm) ** 2)[:, None] * s_recon_var_work).T

    if decomposition_mode == "s":
        u_ica = v_work
        v_ica = u_work
        u_var_ica = v_var_work
        v_var_ica = u_var_work
        x_ica = x_ica_work.T
        x_ica_var = x_ica_var_work.T
        a_recon = s_recon_work.T
        s_recon = a_recon_work.T
        a_recon_var = s_recon_var_work.T
        s_recon_var = a_recon_var_work.T
        data_mask_out = data_mask.T
        ind_missing_out = np.isinf(source_var_ts)
    else:
        u_ica = u_work
        v_ica = v_work
        u_var_ica = u_var_work
        v_var_ica = v_var_work
        x_ica = x_ica_work
        x_ica_var = x_ica_var_work
        a_recon = a_recon_work
        s_recon = s_recon_work
        a_recon_var = a_recon_var_work
        s_recon_var = s_recon_var_work
        data_mask_out = data_mask
        ind_missing_out = ind_missing

    ica = {
        "name": list(xd["name"]),
        "llh": xd["llh"],
        "timeline": xd["timeline"],
        "decomposition_mode": decomposition_mode,
        "U": u_ica,
        "S": s_ica,
        "V": v_ica,
        "var_U": u_var_ica,
        "var_V": v_var_ica,
        "ts": x_ica,
        "var_ts": x_ica_var,
        "net": net,
        "type": list(xd["type"]),
    }

    ica = sort_components(ica)
    sign_v = np.sign(ica["V"][-1, :] - ica["V"][0, :])
    sign_v[sign_v == 0] = 1.0
    ica["V"] = ica["V"] * sign_v[None, :]
    ica["U"] = ica["U"] * sign_v[None, :]

    ica["ts_comp"] = []
    ica["var_ts_comp"] = []
    for nn in range(ica["S"].shape[0]):
        displ, var_displ = calc_displ_lincomp(ica, nn)
        ica["ts_comp"].append(displ)
        ica["var_ts_comp"].append(var_displ)

    ard = (1.0 / ica["net"]["alphas"]) / np.sum(1.0 / ica["net"]["alphas"])
    source_weight = (1.0 / source_var_ts) ** 2
    source_weight[~np.isfinite(source_weight)] = 0.0
    chi2_pca = calc_chi2_matrix(source_ts, source_weight, pca["ts"])
    chi2_ica = calc_chi2_matrix(source_ts, source_weight, ica["ts"])
    var_explained_pca = calc_variance_explained(source_ts, source_var_ts, pca["ts"])
    var_explained_ica = calc_variance_explained(source_ts, source_var_ts, ica["ts"])
    metrics = {
        "ard": ard,
        "chi2_PCA": float(chi2_pca),
        "chi2_ICA": float(chi2_ica),
        "variance_explained_PCA": float(var_explained_pca),
        "variance_explained_ICA": float(var_explained_ica),
    }
    aux = {
        "A_recon": a_recon,
        "S_recon": s_recon,
        "var_A_recon": a_recon_var,
        "var_S_recon": s_recon_var,
        "data_mask": data_mask_out,
        "ind_missing_data": np.flatnonzero(ind_missing_out) + 1,
    }
    return ica, {**metrics, **aux}
