from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .joint_tuning import (
    JointTuningConfig,
    JointTuningResult,
    collect_joint_candidate_cache,
    select_joint_from_cache,
    weight_from_joint_cache,
)


@dataclass
class PPCIResult:
    method: str
    theta_hat: float
    se: float
    ci_low: float
    ci_high: float
    J_hat: float
    V_hat: float
    h: float = np.nan
    lambda_value: float = np.nan
    h_mean: float = np.nan
    lambda_mean: float = np.nan
    h_1: float = np.nan
    h_2: float = np.nan
    lambda_1: float = np.nan
    lambda_2: float = np.nan
    tuning_status: str = ""
    den: float = np.nan
    sigma2_Y_minus_f: float = np.nan
    sigma2_f: float = np.nan
    sigma2_Y: float = np.nan
    h_factor_1: float = np.nan
    h_factor_2: float = np.nan
    lambda_factor_1: float = np.nan
    lambda_factor_2: float = np.nan
    ess0_1: float = np.nan
    ess0_2: float = np.nan
    op_score_1: float = np.nan
    op_score_2: float = np.nan
    loc_score_1: float = np.nan
    loc_score_2: float = np.nan
    h_factor: float = np.nan
    lambda_factor: float = np.nan
    op_score: float = np.nan
    loc_score: float = np.nan
    h_mode: str = ""
    lambda_selection: str = ""
    omega: float = np.nan
    omega_1: float = np.nan
    omega_2: float = np.nan
    omega_raw_1: float = np.nan
    omega_raw_2: float = np.nan
    omega_clipped_rate: float = np.nan
    omega_sd: float = np.nan
    omega_min: float = np.nan
    omega_max: float = np.nan
    omega_folds: int = 0
    V_labeled: float = np.nan
    V_unlabeled: float = np.nan

    def as_dict(self):
        return self.__dict__.copy()


def _safe_var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    return float(np.var(x, ddof=1)) if x.size > 1 else 0.0


def _ci(theta: float, se: float, z: float) -> tuple[float, float]:
    return float(theta - z * se), float(theta + z * se)


def ppi_global_mean(Y_l: np.ndarray, f_l: np.ndarray, f_u: np.ndarray, z_alpha: float = 1.96) -> PPCIResult:
    Y_l = np.asarray(Y_l, dtype=float).ravel()
    f_l = np.asarray(f_l, dtype=float).ravel()
    f_u = np.asarray(f_u, dtype=float).ravel()
    n, N = Y_l.size, f_u.size
    theta_hat = float(np.mean(Y_l - f_l) + np.mean(f_u))
    V_hat = _safe_var(Y_l - f_l) / max(n, 1) + _safe_var(f_u - theta_hat) / max(N, 1)
    se = float(np.sqrt(max(V_hat, 0.0)))
    lo, hi = _ci(theta_hat, se, z_alpha)
    return PPCIResult("PPI", theta_hat, se, lo, hi, J_hat=-1.0, V_hat=V_hat)


def lo_mean_from_weights(X_l: np.ndarray, Y_l: np.ndarray, w_l: np.ndarray, z_alpha: float = 1.96) -> PPCIResult:
    Y_l = np.asarray(Y_l, dtype=float).ravel()
    w_l = np.asarray(w_l, dtype=float).ravel()
    n = Y_l.size
    den = float(np.mean(w_l))
    theta_hat = float(np.mean(w_l * Y_l) / (den + 1e-16))
    psi = w_l * (Y_l - theta_hat)
    V_hat = _safe_var(psi) / max(n, 1)
    J_hat = -den
    se = float(np.sqrt(max(V_hat, 0.0)) / max(abs(J_hat), 1e-16))
    lo, hi = _ci(theta_hat, se, z_alpha)
    return PPCIResult("LO", theta_hat, se, lo, hi, J_hat=J_hat, V_hat=V_hat, den=den, sigma2_Y=_safe_var(psi))


def ppci_mean_from_weight_values(
    Y_l: np.ndarray,
    f_l: np.ndarray,
    f_u: np.ndarray,
    w_l: np.ndarray,
    w_u: np.ndarray,
    z_alpha: float = 1.96,
    method: str = "PPCI",
) -> PPCIResult:
    Y_l = np.asarray(Y_l, dtype=float).ravel()
    f_l = np.asarray(f_l, dtype=float).ravel()
    f_u = np.asarray(f_u, dtype=float).ravel()
    w_l = np.asarray(w_l, dtype=float).ravel()
    w_u = np.asarray(w_u, dtype=float).ravel()
    n, N = Y_l.size, f_u.size
    den = float(np.mean(w_u))
    numerator = float(np.mean(w_l * (Y_l - f_l)) + np.mean(w_u * f_u))
    theta_hat = numerator / (den + 1e-16)
    psi_Y = w_l * (Y_l - theta_hat)
    psi_R = w_l * (Y_l - f_l)
    psi_U = w_u * (f_u - theta_hat)
    sigma2_Y = _safe_var(psi_Y)
    sigma2_R = _safe_var(psi_R)
    sigma2_U = _safe_var(psi_U)
    V_hat = sigma2_R / max(n, 1) + sigma2_U / max(N, 1)
    J_hat = -den
    se = float(np.sqrt(max(V_hat, 0.0)) / max(abs(J_hat), 1e-16))
    lo, hi = _ci(theta_hat, se, z_alpha)
    return PPCIResult(method, theta_hat, se, lo, hi, J_hat=J_hat, V_hat=V_hat, den=den,
                      sigma2_Y_minus_f=sigma2_R, sigma2_f=sigma2_U, sigma2_Y=sigma2_Y,
                      omega=1.0, V_labeled=sigma2_R / max(n, 1),
                      V_unlabeled=sigma2_U / max(N, 1))


def _validate_mean_inputs(
    Y_l: np.ndarray,
    f_l: np.ndarray,
    f_u: np.ndarray,
    w_l: np.ndarray,
    w_u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(np.asarray(x, dtype=float).ravel() for x in (Y_l, f_l, f_u, w_l, w_u))
    Y_l, f_l, f_u, w_l, w_u = arrays
    if not (Y_l.size == f_l.size == w_l.size):
        raise ValueError("Y_l, f_l, and w_l must have the same length.")
    if f_u.size != w_u.size:
        raise ValueError("f_u and w_u must have the same length.")
    if Y_l.size < 2 or f_u.size < 2:
        raise ValueError("PPCI++ requires at least two labeled and two unlabeled observations.")
    return Y_l, f_l, f_u, w_l, w_u


def ppci_plus_mean_given_omegas(
    Y_l: np.ndarray,
    f_l: np.ndarray,
    f_u: np.ndarray,
    w_l: np.ndarray,
    w_u: np.ndarray,
    fold_ids: np.ndarray,
    omegas: np.ndarray,
    z_alpha: float = 1.96,
    method: str = "PPCI++",
    omega_raw: Optional[np.ndarray] = None,
) -> PPCIResult:
    """Evaluate the cross-fitted PPCI++ moment for fixed fold-specific weights."""
    Y_l, f_l, f_u, w_l, w_u = _validate_mean_inputs(Y_l, f_l, f_u, w_l, w_u)
    fold_ids = np.asarray(fold_ids, dtype=int).ravel()
    omegas = np.asarray(omegas, dtype=float).ravel()
    if fold_ids.size != Y_l.size:
        raise ValueError("fold_ids must have one entry per labeled observation.")
    fold_values = np.unique(fold_ids)
    if fold_values.size != omegas.size:
        raise ValueError("omegas must have one entry per distinct labeled fold.")
    if not np.array_equal(fold_values, np.arange(fold_values.size)):
        raise ValueError("fold_ids must be consecutive integers starting at zero.")
    if not np.isfinite(omegas).all() or np.any((omegas < 0.0) | (omegas > 1.0)):
        raise ValueError("omegas must be finite and lie in [0, 1].")

    n, N = Y_l.size, f_u.size
    pi = np.array([np.mean(fold_ids == k) for k in fold_values], dtype=float)
    bar_omega = float(np.dot(pi, omegas))

    if np.all(omegas == 0.0):
        result = lo_mean_from_weights(np.empty((n, 0)), Y_l, w_l, z_alpha=z_alpha)
        result.method = method
        result.omega = result.omega_1 = result.omega_2 = 0.0
        result.omega_clipped_rate = 0.0
        result.V_labeled = result.V_hat
        result.V_unlabeled = 0.0
        return result
    if np.all(omegas == 1.0):
        result = ppci_mean_from_weight_values(Y_l, f_l, f_u, w_l, w_u, z_alpha=z_alpha, method=method)
        result.omega_1 = result.omega_2 = 1.0
        result.omega_clipped_rate = 0.0
        return result

    numerator = bar_omega * float(np.mean(w_u * f_u))
    den = bar_omega * float(np.mean(w_u))
    for k, omega_k in enumerate(omegas):
        idx = fold_ids == k
        numerator += pi[k] * float(np.mean(w_l[idx] * (Y_l[idx] - omega_k * f_l[idx])))
        den += pi[k] * (1.0 - omega_k) * float(np.mean(w_l[idx]))
    theta_hat = float(numerator / (den + 1e-16))

    V_labeled = 0.0
    sigma2_labeled = 0.0
    for k, omega_k in enumerate(omegas):
        idx = fold_ids == k
        z_k = w_l[idx] * ((Y_l[idx] - theta_hat) - omega_k * (f_l[idx] - theta_hat))
        var_k = _safe_var(z_k)
        V_labeled += pi[k] ** 2 * var_k / max(int(np.sum(idx)), 1)
        sigma2_labeled += pi[k] * var_k
    u = w_u * (f_u - theta_hat)
    sigma2_u = _safe_var(u)
    V_unlabeled = bar_omega**2 * sigma2_u / max(N, 1)
    V_hat = float(V_labeled + V_unlabeled)
    J_hat = -float(den)
    se = float(np.sqrt(max(V_hat, 0.0)) / max(abs(J_hat), 1e-16))
    lo, hi = _ci(theta_hat, se, z_alpha)
    result = PPCIResult(
        method, theta_hat, se, lo, hi, J_hat=J_hat, V_hat=V_hat, den=den,
        sigma2_Y_minus_f=sigma2_labeled, sigma2_f=sigma2_u,
        sigma2_Y=_safe_var(w_l * (Y_l - theta_hat)), omega=bar_omega,
        V_labeled=V_labeled, V_unlabeled=V_unlabeled,
    )
    result.omega_1 = float(omegas[0])
    result.omega_2 = float(omegas[1]) if omegas.size > 1 else float(omegas[0])
    result.omega_sd = float(np.std(omegas, ddof=1)) if omegas.size > 1 else 0.0
    result.omega_min = float(np.min(omegas))
    result.omega_max = float(np.max(omegas))
    result.omega_folds = int(omegas.size)
    if omega_raw is not None:
        raw = np.asarray(omega_raw, dtype=float).ravel()
        result.omega_raw_1 = float(raw[0])
        result.omega_raw_2 = float(raw[1]) if raw.size > 1 else float(raw[0])
        result.omega_clipped_rate = float(np.mean(np.abs(raw - omegas) > 1e-12))
    return result


def ppci_plus_mean_from_weight_values(
    Y_l: np.ndarray,
    f_l: np.ndarray,
    f_u: np.ndarray,
    w_l: np.ndarray,
    w_u: np.ndarray,
    rng: np.random.Generator,
    z_alpha: float = 1.96,
    omega_ridge: float = 1e-6,
    omega_folds: int = 5,
    method: str = "PPCI++",
) -> PPCIResult:
    """K-fold labeled cross-fitting for the data-driven PPCI++ coefficient."""
    Y_l, f_l, f_u, w_l, w_u = _validate_mean_inputs(Y_l, f_l, f_u, w_l, w_u)
    n, N = Y_l.size, f_u.size
    omega_folds = int(omega_folds)
    if omega_folds < 2 or omega_folds > n:
        raise ValueError("omega_folds must be between 2 and n.")
    perm = rng.permutation(n)
    fold_ids = np.empty(n, dtype=int)
    for k, indices in enumerate(np.array_split(perm, omega_folds)):
        fold_ids[indices] = k
    raw = np.empty(omega_folds, dtype=float)
    for k in range(omega_folds):
        train = fold_ids != k
        theta_pilot = lo_mean_from_weights(
            np.empty((int(np.sum(train)), 0)), Y_l[train], w_l[train]
        ).theta_hat
        a = w_l[train] * (Y_l[train] - theta_pilot)
        b = w_l[train] * (f_l[train] - theta_pilot)
        c = w_u * (f_u - theta_pilot)
        var_a = _safe_var(a)
        var_b = _safe_var(b)
        var_c = _safe_var(c)
        cov_ab = float(np.cov(a, b, ddof=1)[0, 1]) if a.size > 1 else 0.0
        scale = max(var_a, var_b, (n / max(N, 1)) * var_c, 1e-12)
        denominator = var_b + (n / max(N, 1)) * var_c + float(omega_ridge) * scale
        raw[k] = cov_ab / denominator if denominator > 0.0 else 0.0
    omegas = np.clip(raw, 0.0, 1.0)
    result = ppci_plus_mean_given_omegas(
        Y_l, f_l, f_u, w_l, w_u, fold_ids, omegas,
        z_alpha=z_alpha, method=method, omega_raw=raw,
    )
    return result


def _attach_joint_tuning(res: PPCIResult, tr1: JointTuningResult, tr2: JointTuningResult) -> None:
    res.h_1, res.h_2 = tr1.h, tr2.h
    res.lambda_1, res.lambda_2 = tr1.lam, tr2.lam
    res.h_mean = 0.5 * (tr1.h + tr2.h)
    res.lambda_mean = 0.5 * (tr1.lam + tr2.lam)
    res.h = res.h_mean
    res.lambda_value = res.lambda_mean
    res.tuning_status = f"fold1:{tr1.status};fold2:{tr2.status}"
    res.h_factor_1, res.h_factor_2 = tr1.h_factor_vs_median, tr2.h_factor_vs_median
    res.lambda_factor_1, res.lambda_factor_2 = tr1.lambda_factor, tr2.lambda_factor
    res.ess0_1, res.ess0_2 = tr1.ess0, tr2.ess0
    res.op_score_1, res.op_score_2 = tr1.op_score, tr2.op_score
    res.loc_score_1, res.loc_score_2 = tr1.loc_score, tr2.loc_score
    res.h_factor = 0.5 * (tr1.h_factor_vs_median + tr2.h_factor_vs_median)
    res.lambda_factor = 0.5 * (tr1.lambda_factor + tr2.lambda_factor)
    res.op_score = 0.5 * (tr1.op_score + tr2.op_score)
    res.loc_score = 0.5 * (tr1.loc_score + tr2.loc_score)
    res.h_mode = tr1.h_grid_mode if tr1.h_grid_mode == tr2.h_grid_mode else f"fold1:{tr1.h_grid_mode};fold2:{tr2.h_grid_mode}"
    res.lambda_selection = "joint_p1"


def ppci_mean_split(
    X_l: np.ndarray,
    Y_l: np.ndarray,
    f_l: np.ndarray,
    X_u: np.ndarray,
    f_u: np.ndarray,
    x0: np.ndarray,
    rng: np.random.Generator,
    tuning_cfg: Optional[JointTuningConfig] = None,
    z_alpha: float = 1.96,
) -> tuple[PPCIResult, PPCIResult, PPCIResult, dict]:
    """Two-fold cross-fitted PPCI, LO, and global PPI for conditional mean."""
    tuning_cfg = tuning_cfg or JointTuningConfig()
    X_l = np.asarray(X_l, dtype=float)
    X_u = np.asarray(X_u, dtype=float)
    N = X_u.shape[0]
    perm = rng.permutation(N)
    I1 = np.sort(perm[: N // 2])
    I2 = np.sort(perm[N // 2 :])
    cache1 = collect_joint_candidate_cache(X_u[I1], x0, n=X_l.shape[0], cfg=tuning_cfg)
    cache2 = collect_joint_candidate_cache(X_u[I2], x0, n=X_l.shape[0], cfg=tuning_cfg)
    tr1 = select_joint_from_cache(cache1, "GH", cfg=tuning_cfg)
    tr2 = select_joint_from_cache(cache2, "GH", cfg=tuning_cfg)
    w1 = weight_from_joint_cache(cache1, tr1, tuning_cfg)
    w2 = weight_from_joint_cache(cache2, tr2, tuning_cfg)
    w_l = 0.5 * (w1(X_l) + w2(X_l))
    w_u = np.zeros(N, dtype=float)
    # OOF: points in I1 evaluated by weight trained on I2, and vice versa.
    w_u[I1] = w2(X_u[I1])
    w_u[I2] = w1(X_u[I2])
    ppci = ppci_mean_from_weight_values(Y_l, f_l, f_u, w_l, w_u, z_alpha=z_alpha, method="PPCI")
    lo = lo_mean_from_weights(X_l, Y_l, w_l, z_alpha=z_alpha)
    ppi = ppi_global_mean(Y_l, f_l, f_u, z_alpha=z_alpha)
    for res in (ppci, lo):
        _attach_joint_tuning(res, tr1, tr2)
    info = {
        "I1": I1, "I2": I2,
        "tuning_1": tr1.as_dict(), "tuning_2": tr2.as_dict(),
        "w_u_mean": float(np.mean(w_u)), "w_l_mean": float(np.mean(w_l)),
    }
    return ppci, lo, ppi, info


def ppci_mean_nosplit(
    X_l: np.ndarray,
    Y_l: np.ndarray,
    f_l: np.ndarray,
    X_u: np.ndarray,
    f_u: np.ndarray,
    x0: np.ndarray,
    tuning_cfg: Optional[JointTuningConfig] = None,
    z_alpha: float = 1.96,
) -> tuple[PPCIResult, PPCIResult, PPCIResult, dict]:
    tuning_cfg = tuning_cfg or JointTuningConfig()
    X_l = np.asarray(X_l, dtype=float)
    X_u = np.asarray(X_u, dtype=float)
    # The no-split construction in the paper estimates the empirical operator
    # from all observed covariates, including the labeled designs.
    X_pool = np.vstack([X_l, X_u])
    cache = collect_joint_candidate_cache(X_pool, x0, n=X_l.shape[0], cfg=tuning_cfg)
    tr = select_joint_from_cache(cache, "GH", cfg=tuning_cfg)
    w = weight_from_joint_cache(cache, tr, tuning_cfg)
    w_l = w(X_l)
    w_u = w(X_u)
    ppci = ppci_mean_from_weight_values(Y_l, f_l, f_u, w_l, w_u, z_alpha=z_alpha, method="PPCI")
    lo = lo_mean_from_weights(X_l, Y_l, w_l, z_alpha=z_alpha)
    ppi = ppi_global_mean(Y_l, f_l, f_u, z_alpha=z_alpha)
    for res in (ppci, lo):
        res.h_mean = tr.h
        res.lambda_mean = tr.lam
        res.h = tr.h
        res.lambda_value = tr.lam
        res.h_1 = res.h_2 = tr.h
        res.lambda_1 = res.lambda_2 = tr.lam
        res.tuning_status = tr.status
        res.h_factor_1 = res.h_factor_2 = tr.h_factor_vs_median
        res.lambda_factor_1 = res.lambda_factor_2 = tr.lambda_factor
        res.ess0_1 = res.ess0_2 = tr.ess0
        res.op_score_1 = res.op_score_2 = tr.op_score
        res.loc_score_1 = res.loc_score_2 = tr.loc_score
        res.h_factor = tr.h_factor_vs_median
        res.lambda_factor = tr.lambda_factor
        res.op_score = tr.op_score
        res.loc_score = tr.loc_score
        res.h_mode = tr.h_grid_mode
        res.lambda_selection = "joint_p1_nosplit"
    info = {"tuning": tr.as_dict(), "w_u_mean": float(np.mean(w_u)), "w_l_mean": float(np.mean(w_l))}
    return ppci, lo, ppi, info


def fit_ppci_mean(
    X_l: np.ndarray,
    Y_l: np.ndarray,
    f_l: np.ndarray,
    X_u: np.ndarray,
    f_u: np.ndarray,
    x0: np.ndarray,
    split: str = "twofold",
    seed: int = 0,
    tuning_cfg: Optional[JointTuningConfig] = None,
    z_alpha: float = 1.96,
):
    rng = np.random.default_rng(seed)
    if split.lower() in {"twofold", "split", "cf", "crossfit"}:
        return ppci_mean_split(X_l, Y_l, f_l, X_u, f_u, x0, rng=rng, tuning_cfg=tuning_cfg, z_alpha=z_alpha)
    if split.lower() in {"nosplit", "no-split", "full"}:
        return ppci_mean_nosplit(X_l, Y_l, f_l, X_u, f_u, x0, tuning_cfg=tuning_cfg, z_alpha=z_alpha)
    raise ValueError("split must be 'twofold' or 'nosplit'.")
