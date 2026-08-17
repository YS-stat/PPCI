from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

from .kernels import get_kernel
from .weights import spectral_diagnostics, effective_dimension_from_spectrum, point_leverage_from_spectrum


@dataclass
class TuningConfig:
    """Covariate-only finite-sample tuning for RKHS localization PPCI weights.

    The defaults are deliberately practical rather than asymptotic-only.  They search
    for a covariate-only h rule with a stable RKHS localization weight and then choose a
    stable lambda on an O(1/n) grid.
    """

    # local-support search for h
    k_min_floor: int = 50
    k_max_frac: float = 0.50
    k_growth: float = 1.50
    ess_grid_size: int = 96
    min_h: float = 1e-8
    max_h_mult: float = 5.0
    h_mode: str = "ess_local"
    h_factors: tuple[float, ...] = (0.8, 1.0, 1.2)
    fixed_h_factor: float = 1.0

    # lambda grid: lambda = factor / n
    lambda_factor_min: float = 0.2
    lambda_factor_max: float = 20.0
    lambda_grid_size: int = 31
    lambda_selection: str = "smallest_stable"
    fixed_lambda_factor: float = 1.0

    # finite-sample stability thresholds.  These are empirical analogues of the
    # effective-dimension and pointwise-leverage conditions in the proofs.
    tau_op: float = 12.0
    tau_loc: float = 4.0

    # kernel/backend
    kernel: str = "matern52"
    backend: str = "cpu"


@dataclass
class TuningResult:
    h: float
    lam: float
    status: str
    M: int
    k_target: int
    ess0: float
    lambda_min: float
    lambda_max: float
    lambda_factor: float
    eff_dim: float
    point_leverage: float
    op_score: float
    loc_score: float
    n_stable: int
    h_anchor_median: float
    h_factor_vs_median: float
    h_mode: str
    h_factor: float
    lambda_selection: str

    def as_dict(self):
        return asdict(self)


def raw_ess_for_h(Z: np.ndarray, x0: np.ndarray, h: float, kernel: str = "matern52") -> float:
    kernel_fn = get_kernel(kernel)
    Z = np.asarray(Z, dtype=float)
    x0 = np.asarray(x0, dtype=float).reshape(1, -1)
    a = kernel_fn(Z, x0, float(h)).ravel()
    denom = float(np.sum(a * a))
    if denom <= 0:
        return 0.0
    return float((np.sum(a) ** 2) / denom)


def distance_median_anchor(Z: np.ndarray, x0: np.ndarray) -> float:
    Z = np.asarray(Z, dtype=float)
    x0 = np.asarray(x0, dtype=float).reshape(1, -1)
    d = np.linalg.norm(Z - x0, axis=1)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 1.0
    return max(float(np.median(d)), 1e-8)


def solve_h_for_ess(Z: np.ndarray, x0: np.ndarray, k_target: float, cfg: TuningConfig) -> float:
    """Find the smallest h with raw kernel ESS at least k_target."""
    Z = np.asarray(Z, dtype=float)
    x0 = np.asarray(x0, dtype=float).reshape(1, -1)
    d = np.linalg.norm(Z - x0, axis=1)
    positive = d[d > 1e-12]
    med = distance_median_anchor(Z, x0)
    low = cfg.min_h if positive.size == 0 else max(cfg.min_h, float(np.min(positive)) * 1e-3)
    high = max(med * cfg.max_h_mult, low * 10.0)
    high = max(high, float(np.max(d)) if d.size else high, 1.0)
    for _ in range(40):
        if raw_ess_for_h(Z, x0, high, cfg.kernel) >= k_target or high > 1e8:
            break
        high *= 2.0
    lo_log, hi_log = np.log(low), np.log(high)
    for _ in range(60):
        mid_log = 0.5 * (lo_log + hi_log)
        mid = float(np.exp(mid_log))
        if raw_ess_for_h(Z, x0, mid, cfg.kernel) >= k_target:
            hi_log = mid_log
        else:
            lo_log = mid_log
    return float(np.exp(hi_log))


def k_grid(M: int, cfg: TuningConfig) -> list[int]:
    k_max = max(2, int(np.floor(cfg.k_max_frac * M)))
    k_min = min(k_max, max(2, cfg.k_min_floor, int(np.ceil(np.sqrt(M)))))
    vals = []
    k = k_min
    while k < k_max:
        vals.append(int(k))
        k = int(np.ceil(k * cfg.k_growth))
        if vals and k <= vals[-1]:
            k = vals[-1] + 1
    vals.append(k_max)
    out = []
    for v in vals:
        if not out or v != out[-1]:
            out.append(int(v))
    return out


def lambda_grid(n: int, cfg: TuningConfig) -> np.ndarray:
    n_eff = max(int(n), 1)
    if str(cfg.lambda_selection).lower() == "fixed_factor":
        return np.array([float(cfg.fixed_lambda_factor) / float(n_eff)])
    factors = np.logspace(np.log10(cfg.lambda_factor_min), np.log10(cfg.lambda_factor_max), int(cfg.lambda_grid_size))
    return factors / float(n_eff)


def evaluate_stability(Z: np.ndarray, x0: np.ndarray, h: float, lam_grid: np.ndarray, n: int, cfg: TuningConfig):
    M = int(np.asarray(Z).shape[0])
    eigvals, tmp, k00 = spectral_diagnostics(Z, x0, h, cfg.kernel, backend=cfg.backend)
    rows = []
    for lam in np.asarray(lam_grid, dtype=float):
        eff = effective_dimension_from_spectrum(eigvals, M, lam)
        lev = point_leverage_from_spectrum(eigvals, tmp, k00, M, lam)
        op_score = eff * np.sqrt(np.log(max(M, 3)) / max(M, 1))
        loc_score = lev * np.log(max(n + M, 3)) / max(min(n, M), 1)
        rows.append({
            "lambda": float(lam),
            "lambda_factor": float(lam * max(n, 1)),
            "eff_dim": float(eff),
            "point_leverage": float(lev),
            "op_score": float(op_score),
            "loc_score": float(loc_score),
            "stable": bool((op_score <= cfg.tau_op) and (loc_score <= cfg.tau_loc)),
        })
    return rows


def _choose_lambda(stable_rows: list[dict], cfg: TuningConfig) -> dict:
    selection = str(cfg.lambda_selection).lower()
    if selection == "smallest_stable":
        return min(stable_rows, key=lambda r: r["lambda"])
    if selection == "largest_stable":
        return max(stable_rows, key=lambda r: r["lambda"])
    if selection == "fixed_factor":
        return stable_rows[0]
    raise ValueError("lambda_selection must be 'smallest_stable', 'largest_stable', or 'fixed_factor'.")


def _make_result(
    *,
    h: float,
    chosen: dict,
    status: str,
    M: int,
    k_target: int,
    ess0: float,
    lam_grid: np.ndarray,
    med: float,
    n_stable: int,
    cfg: TuningConfig,
) -> TuningResult:
    h_factor = float(h / med)
    return TuningResult(
        h=float(h), lam=float(chosen["lambda"]), status=status, M=M,
        k_target=int(k_target), ess0=float(ess0), lambda_min=float(np.min(lam_grid)),
        lambda_max=float(np.max(lam_grid)), lambda_factor=float(chosen["lambda_factor"]),
        eff_dim=float(chosen["eff_dim"]), point_leverage=float(chosen["point_leverage"]),
        op_score=float(chosen["op_score"]), loc_score=float(chosen["loc_score"]),
        n_stable=int(n_stable), h_anchor_median=float(med), h_factor_vs_median=h_factor,
        h_mode=str(cfg.h_mode), h_factor=h_factor, lambda_selection=str(cfg.lambda_selection),
    )


def _least_bad_row(rows: list[dict], cfg: TuningConfig) -> dict:
    return min(rows, key=lambda r: max(r["op_score"] / cfg.tau_op, r["loc_score"] / cfg.tau_loc))


def _validate_config(cfg: TuningConfig) -> None:
    if str(cfg.h_mode).lower() not in {"ess_local", "median_grid", "fixed_factor"}:
        raise ValueError("h_mode must be 'ess_local', 'median_grid', or 'fixed_factor'.")
    if str(cfg.lambda_selection).lower() not in {"smallest_stable", "largest_stable", "fixed_factor"}:
        raise ValueError("lambda_selection must be 'smallest_stable', 'largest_stable', or 'fixed_factor'.")
    if str(cfg.lambda_selection).lower() == "fixed_factor" and float(cfg.fixed_lambda_factor) <= 0:
        raise ValueError("fixed_lambda_factor must be positive.")


def tune_h_lambda_from_covariates(Z: np.ndarray, x0: np.ndarray, n: int, cfg: Optional[TuningConfig] = None) -> TuningResult:
    """Tune h and lambda using only covariates.

    Search k from small to large (more local to less local).  For the first h with at
    least one stable lambda on the O(1/n) grid, return the smallest stable lambda.
    This is a practical undersmoothing rule: localize as much as possible, and use the
    most undersmoothed lambda that keeps the empirical RKHS localization diagnostics stable.
    """
    cfg = cfg or TuningConfig()
    _validate_config(cfg)
    Z = np.asarray(Z, dtype=float)
    x0 = np.asarray(x0, dtype=float).ravel()
    M = int(Z.shape[0])
    if M < 4:
        raise ValueError("Need at least four covariates for tuning.")
    lam_grid = lambda_grid(n, cfg)
    med = distance_median_anchor(Z, x0)
    fallback = None
    h_mode = str(cfg.h_mode).lower()

    if h_mode in {"median_grid", "fixed_factor"}:
        if h_mode == "fixed_factor":
            factors = [float(cfg.fixed_h_factor)]
        else:
            factors = [float(f) for f in cfg.h_factors]
        if not factors:
            raise ValueError("h_factors must contain at least one value.")
        factors = [max(float(f), 1e-12) for f in factors]
        for factor in sorted(factors, key=lambda f: (abs(np.log(f)), f)):
            h = float(med * factor)
            ess0 = raw_ess_for_h(Z, x0, h, cfg.kernel)
            rows = evaluate_stability(Z, x0, h, lam_grid, n, cfg)
            stable = [r for r in rows if r["stable"]]
            if stable:
                chosen = _choose_lambda(stable, cfg)
                return _make_result(
                    h=h, chosen=chosen, status="stable", M=M, k_target=0, ess0=ess0,
                    lam_grid=lam_grid, med=med, n_stable=len(stable), cfg=cfg,
                )
            chosen = _least_bad_row(rows, cfg)
            badness = max(chosen["op_score"] / cfg.tau_op, chosen["loc_score"] / cfg.tau_loc)
            if fallback is None or badness < fallback[0]:
                fallback = (badness, h, ess0, chosen)
        _, h, ess0, chosen = fallback
        return _make_result(
            h=h, chosen=chosen, status="fallback_unstable", M=M, k_target=0, ess0=ess0,
            lam_grid=lam_grid, med=med, n_stable=0, cfg=cfg,
        )

    for k in k_grid(M, cfg):
        h = solve_h_for_ess(Z, x0, k, cfg)
        ess0 = raw_ess_for_h(Z, x0, h, cfg.kernel)
        rows = evaluate_stability(Z, x0, h, lam_grid, n, cfg)
        stable = [r for r in rows if r["stable"]]
        if stable:
            chosen = _choose_lambda(stable, cfg)
            return _make_result(
                h=h, chosen=chosen, status="stable", M=M, k_target=k, ess0=ess0,
                lam_grid=lam_grid, med=med, n_stable=len(stable), cfg=cfg,
            )
        # Keep the least bad value at this h for fallback diagnostics.
        rows_sorted = sorted(rows, key=lambda r: max(r["op_score"] / cfg.tau_op, r["loc_score"] / cfg.tau_loc))
        if fallback is None:
            fallback = (k, h, ess0, rows_sorted[0])
    # If no stable pair exists on the grid, use the least bad pair but mark it clearly.
    k, h, ess0, chosen = fallback
    return _make_result(
        h=h, chosen=chosen, status="fallback_unstable", M=M, k_target=k, ess0=ess0,
        lam_grid=lam_grid, med=med, n_stable=0, cfg=cfg,
    )
