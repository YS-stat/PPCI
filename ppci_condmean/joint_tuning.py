from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Optional
import math
import numpy as np
from scipy.linalg import eigh

from .kernels import get_kernel
from .tuning import raw_ess_for_h, solve_h_for_ess, k_grid, distance_median_anchor, TuningConfig


_EPS = 1e-12


def _torch_or_none():
    try:
        import torch
        return torch
    except Exception:
        return None


def _torch_kernel(X, Z, h: float, kernel: str):
    torch = _torch_or_none()
    if torch is None:
        raise RuntimeError("PyTorch is not available")
    h = float(max(h, 1e-12))
    X2 = torch.sum(X * X, dim=1, keepdim=True)
    Z2 = torch.sum(Z * Z, dim=1, keepdim=True).T
    d2 = torch.clamp(X2 + Z2 - 2.0 * (X @ Z.T), min=0.0)
    name = str(kernel).lower()
    if name in {"gaussian", "rbf"}:
        return torch.exp(-0.5 * d2 / (h * h))
    r = torch.sqrt(d2 + 1e-18)
    t = (5.0 ** 0.5) * r / h
    return (1.0 + t + (t * t) / 3.0) * torch.exp(-t)


@dataclass
class JointTuningConfig:
    """Covariate-only joint bandwidth and regularization tuning for PPCI.

    Families implemented in tune_joint_from_covariates:
      * INC: incumbent ESS-local + smallest stable.
      * PC: power-constrained joint tuning. Constraint:
            P_{h,lambda}(x0) <= pc_r * P_{h,lambda_min(h)}(x0),
            then choose the pair with the smallest covariate-only SE proxy.
      * MB: moment-budgeted / geometric-bias joint tuning. Constraint:
            sqrt(n) * B_geo(h,lambda) / s_w(h,lambda) <= mb_gamma.
      * GH: gamma-H RKHS self-normalized budget. Constraint:
            sqrt(n * lambda * [D_hat(x0;lambda) - V_w]_+ / V_w) <= gh_gamma.

    The paper-facing default is the P1 labelled-scale GH screen. The legacy PC and MB
    selectors are retained for sensitivity experiments. Every selector changes only the
    choice of ``(h, lambda)``; the PPCI estimator and confidence interval are unchanged.
    """

    # Candidate h grid G_h.  The paper-facing default is the median-distance grid:
    # h = a * median(||X-x0||), with a in h_factors.  The original ESS grid is
    # retained as an explicit compatibility option for old simulation reports.
    h_grid_mode: str = "median_grid"  # "ess" or "median_grid"
    h_factors: tuple[float, ...] = (0.8, 1.0, 1.15, 1.2)
    k_min_floor: int = 50
    k_max_frac: float = 0.80
    k_growth: float = 1.50
    min_h: float = 1e-8
    max_h_mult: float = 5.0

    # lambda grid. If lambda_grid_mode == "shrinking", lambda = c / (n loglog(n+e^e));
    # otherwise lambda = c / n. The paper experiments use the shrinking grid.
    lambda_factor_min: float = 0.05
    lambda_factor_max: float = 20.0
    lambda_grid_size: int = 41
    lambda_grid_mode: str = "shrinking"  # "n" or "shrinking"

    # stability screens
    tau_op: float = 12.0
    tau_loc: float = 4.0

    # family-specific default budgets
    pc_r: float = 1.04
    mb_gamma: float = 0.25
    # New paper notation: A_bias uses c_bias and one of the three bias screens.
    # P1 is the default labelled-scale A_bias screen in the paper.  "legacy"
    # preserves the old gh_gamma / adaptive GH behavior for old reports only.
    bias_screen: str = "p1_label"  # "p1_label", "p2_log", "p3_full", "legacy"
    c_bias: float = 0.18
    gh_gamma: float = 0.30
    gh_adaptive: bool = False
    gh_adaptive_rule: str = "legacy"  # "legacy" or "log_ratio"
    gh_c0: float = 0.18
    gh_gamma0: float = 0.30
    gh_rho: float = 0.15
    gh_ref_ratio: float = 5.0
    gh_edge_rho: float = 0.0
    gh_edge_ridge: float = 1e-6
    gh_a_tau: float = math.inf
    gh_pc_r: float = math.inf
    constraint_fallback: str = "least_violation"  # "min_sw" or "least_violation"

    # numerical safety
    min_abs_j: float = 1e-6
    kernel: str = "matern52"
    backend: str = "cpu"  # cpu or torch; torch uses cuda if available.


@dataclass
class JointTuningResult:
    method: str
    h: float
    lam: float
    status: str
    M: int
    n_label: int
    k_target: int
    ess0: float
    lambda_factor: float
    lambda_grid_mode: str
    eff_dim: float
    point_leverage: float
    op_score: float
    loc_score: float
    stable: bool
    n_stable_this_h: int
    n_feasible_total: int
    h_grid_mode: str
    h_anchor_median: float
    h_factor_vs_median: float
    J_w: float
    V_w: float
    Q_h: float
    D_h_point: float
    D_h_op: float
    sw_proxy: float
    power: float
    power_ref: float
    pc_ratio: float
    m1_norm: float
    M2_nuc: float
    B_geo: float
    R_MB: float
    R_GH: float
    bias_screen: str
    c_bias: float
    bias_score: float
    bias_score_label: float
    bias_score_full: float
    bias_budget: float
    gh_gamma_used: float
    gh_gamma_eff: float
    edge_score: float
    A_score: float
    M2_lambda_min: float

    def as_dict(self):
        return asdict(self)


def lambda_grid_joint(n: int, cfg: JointTuningConfig) -> np.ndarray:
    n_eff = max(int(n), 1)
    factors = np.logspace(np.log10(cfg.lambda_factor_min), np.log10(cfg.lambda_factor_max), int(cfg.lambda_grid_size))
    denom = float(n_eff)
    if str(cfg.lambda_grid_mode).lower() in {"shrinking", "inf", "theory"}:
        ell = math.log(math.log(n_eff + math.e ** math.e))
        denom *= max(ell, 1.0)
    return factors / denom


def lambda_factor_report(lam: float, n: int, cfg: JointTuningConfig) -> float:
    n_eff = max(int(n), 1)
    if str(cfg.lambda_grid_mode).lower() in {"shrinking", "inf", "theory"}:
        ell = math.log(math.log(n_eff + math.e ** math.e))
        return float(lam * n_eff * max(ell, 1.0))
    return float(lam * n_eff)


def _parse_h_factors(cfg: JointTuningConfig) -> tuple[float, ...]:
    vals = cfg.h_factors
    if isinstance(vals, str):
        out = tuple(float(x) for x in vals.split(",") if x.strip())
    else:
        out = tuple(float(x) for x in vals)
    if not out:
        raise ValueError("h_factors must contain at least one positive value.")
    return tuple(v for v in out if v > 0)


def h_grid_joint(Z: np.ndarray, x0: np.ndarray, cfg: JointTuningConfig) -> list[tuple[int, float, float]]:
    """Return the finite candidate grid G_h as (k_report, h, raw_ess).

    The paper-facing grid is the median-distance grid.  The original ESS grid is
    kept as a compatibility mode so old simulation reports remain reproducible.
    """
    Z = np.asarray(Z, dtype=float)
    x0 = np.asarray(x0, dtype=float).ravel()
    M = int(Z.shape[0])
    mode = str(cfg.h_grid_mode).lower()
    if mode in {"median", "median_grid", "h0"}:
        h0 = distance_median_anchor(Z, x0)
        rows = []
        seen = set()
        for factor in _parse_h_factors(cfg):
            h = float(max(h0 * factor, cfg.min_h))
            key = round(h, 14)
            if key in seen:
                continue
            seen.add(key)
            ess0 = raw_ess_for_h(Z, x0, h, cfg.kernel)
            rows.append((int(round(ess0)), h, ess0))
        return rows
    h_stub = TuningConfig(
        k_min_floor=cfg.k_min_floor,
        k_max_frac=cfg.k_max_frac,
        k_growth=cfg.k_growth,
        min_h=cfg.min_h,
        max_h_mult=cfg.max_h_mult,
        kernel=cfg.kernel,
        backend=cfg.backend,
    )
    rows = []
    for k in k_grid(M, h_stub):
        h = solve_h_for_ess(Z, x0, k, h_stub)
        ess0 = raw_ess_for_h(Z, x0, h, cfg.kernel)
        rows.append((int(k), h, ess0))
    return rows


def _gh_gamma_used(n: int, M: int, cfg: JointTuningConfig) -> float:
    if not cfg.gh_adaptive:
        return float(cfg.gh_gamma)
    ratio = max(float(2 * M) / max(float(n), 1.0), 1e-12)  # fold M ~= N/2, so 2M/n ~= N/n
    if str(cfg.gh_adaptive_rule).lower() in {"log_ratio", "log", "paper"}:
        return float(cfg.gh_c0 * math.sqrt(1.0 + max(math.log(ratio), 0.0)))
    boost = max(0.0, math.log(ratio) - math.log(max(cfg.gh_ref_ratio, 1e-12)))
    return float(cfg.gh_gamma0 * (1.0 + cfg.gh_rho * boost))


def _gh_gamma_eff(n: int, M: int, cfg: JointTuningConfig, edge_score: float) -> float:
    base = _gh_gamma_used(n, M, cfg)
    shrink = 1.0 + max(float(cfg.gh_edge_rho), 0.0) * max(float(edge_score), 0.0)
    return float(base / max(shrink, _EPS))


def _bias_budget(n: int, M: int, cfg: JointTuningConfig, edge_score: float) -> float:
    screen = str(cfg.bias_screen).lower()
    ratio = max(float(2 * M) / max(float(n), 1.0), 1e-12)
    if screen in {"p1", "p1_label", "label", "constant"}:
        return float(cfg.c_bias)
    if screen in {"p2", "p2_log", "log", "log_ratio"}:
        return float(cfg.c_bias * math.sqrt(1.0 + max(math.log(ratio), 0.0)))
    if screen in {"p3", "p3_full", "full"}:
        return float(cfg.c_bias)
    return _gh_gamma_eff(n, M, cfg, edge_score)


def _row_bias_score(row: dict, n: int, M: int, cfg: JointTuningConfig) -> float:
    screen = str(cfg.bias_screen).lower()
    if screen in {"p3", "p3_full", "full"}:
        return float(row["bias_score_full"])
    if screen in {"legacy"}:
        return float(row["R_GH"])
    return float(row["bias_score_label"])


def _raw_edge_score(Z: np.ndarray, x0: np.ndarray, h: float, kernel: str, ridge: float) -> float:
    kernel_fn = get_kernel(kernel)
    Z = np.asarray(Z, dtype=float)
    x0 = np.asarray(x0, dtype=float).reshape(1, -1)
    a = np.asarray(kernel_fn(Z, x0, float(h)).ravel(), dtype=float)
    denom = float(np.sum(a))
    if denom <= _EPS or not np.isfinite(denom):
        return float("inf")
    delta = Z - x0
    mu = np.sum(a[:, None] * delta, axis=0) / denom
    S = (delta.T @ (a[:, None] * delta)) / denom
    S = 0.5 * (S + S.T)
    d = max(int(S.shape[0]), 1)
    tr = float(np.trace(S))
    jitter = max(float(ridge) * max(tr / d, 0.0), _EPS)
    S_reg = S + jitter * np.eye(d)
    try:
        val = float(mu @ np.linalg.solve(S_reg, mu))
    except Exception:
        val = float(mu @ (np.linalg.pinv(S_reg) @ mu))
    return float(math.sqrt(max(val, 0.0)))


class _SpectralAtH:
    def __init__(self, Z: np.ndarray, x0: np.ndarray, h: float, kernel: str, backend: str):
        self.Z_cpu = np.asarray(Z, dtype=float)
        self.x0_cpu = np.asarray(x0, dtype=float).reshape(1, -1)
        self.h = float(h)
        self.kernel = str(kernel)
        self.M = int(self.Z_cpu.shape[0])
        self.edge_score = _raw_edge_score(self.Z_cpu, self.x0_cpu, self.h, self.kernel, ridge=1e-6)
        self.backend = "cpu"
        self._edge_score_cache: dict[float, float] = {}
        requested = str(backend).lower()
        if requested in {"torch", "gpu", "cuda"}:
            torch = _torch_or_none()
            if torch is not None and torch.cuda.is_available():
                self.backend = "torch"
        if self.backend == "torch":
            torch = _torch_or_none()
            self.torch = torch
            self.device = torch.device("cuda")
            with torch.no_grad():
                Zt = torch.as_tensor(self.Z_cpu, dtype=torch.float64, device=self.device)
                x0t = torch.as_tensor(self.x0_cpu, dtype=torch.float64, device=self.device)
                Sigma = _torch_kernel(Zt, Zt, self.h, self.kernel)
                Sigma = 0.5 * (Sigma + Sigma.T)
                k0 = _torch_kernel(Zt, x0t, self.h, self.kernel).reshape(-1)
                eigvals, eigvecs = torch.linalg.eigh(Sigma)
                eigvals = torch.clamp(eigvals, min=0.0)
                tmp = eigvecs.T @ k0
                k00 = _torch_kernel(x0t, x0t, self.h, self.kernel)[0, 0]
                self.Zt = Zt
                self.x0t = x0t
                self.eigvals_t = eigvals
                self.eigvecs_t = eigvecs
                self.tmp_t = tmp
                self.k00_t = k00
                # CPU copies for scalars/moments if desired.
                self.eigvals = eigvals.detach().cpu().numpy()
                self.tmp = tmp.detach().cpu().numpy()
                self.k00 = float(k00.detach().cpu().item())
        else:
            kernel_fn = get_kernel(self.kernel)
            Sigma = kernel_fn(self.Z_cpu, self.Z_cpu, self.h)
            Sigma = 0.5 * (Sigma + Sigma.T)
            k0 = kernel_fn(self.Z_cpu, self.x0_cpu, self.h).ravel()
            eigvals, eigvecs = eigh(Sigma, check_finite=False)
            eigvals = np.maximum(eigvals, 0.0)
            tmp = eigvecs.T @ k0
            self.eigvals = eigvals
            self.eigvecs = eigvecs
            self.tmp = tmp
            self.k00 = float(kernel_fn(self.x0_cpu, self.x0_cpu, self.h)[0, 0])

    def candidate(self, lam: float, n: int, cfg: JointTuningConfig) -> dict:
        M = self.M
        lam = float(lam)
        if self.backend == "torch":
            torch = self.torch
            with torch.no_grad():
                denom = self.eigvals_t + M * lam
                eff = torch.sum(self.eigvals_t / denom)
                power2 = torch.clamp(self.k00_t - torch.sum((self.tmp_t * self.tmp_t) / denom), min=0.0)
                point = power2 / lam
                xi = self.eigvecs_t @ (self.tmp_t / denom)
                w = M * xi
                w_np = w.detach().cpu().numpy()
                eff_f = float(eff.detach().cpu().item())
                point_f = float(point.detach().cpu().item())
                power_f = float(torch.sqrt(power2).detach().cpu().item())
        else:
            denom = self.eigvals + M * lam
            eff_f = float(np.sum(self.eigvals / denom))
            power2 = max(self.k00 - float(np.sum((self.tmp * self.tmp) / denom)), 0.0)
            power_f = float(math.sqrt(power2))
            point_f = float(power2 / lam)
            xi = self.eigvecs @ (self.tmp / denom)
            w_np = M * xi
        op_score = eff_f * math.sqrt(math.log(max(M, 3)) / max(M, 1))
        loc_score = point_f * math.log(max(n + M, 3)) / max(min(n, M), 1)
        stable = bool((op_score <= cfg.tau_op) and (loc_score <= cfg.tau_loc))
        J = float(np.mean(w_np))
        Q_h = float(np.mean(w_np * w_np))
        sw = float(math.sqrt(max(Q_h, 0.0)) / max(abs(J), cfg.min_abs_j))
        delta = self.Z_cpu - self.x0_cpu.reshape(1, -1)
        M2_lambda_min = np.nan
        A_score = np.inf
        if abs(J) < cfg.min_abs_j or not np.isfinite(J) or not np.isfinite(Q_h):
            m1_norm = np.inf
            M2_nuc = np.inf
            B_geo = np.inf
            R_MB = np.inf
        else:
            m1 = np.mean(w_np[:, None] * delta, axis=0) / J
            M2 = (delta.T @ (w_np[:, None] * delta)) / float(M) / J
            m1_norm = float(np.linalg.norm(m1, ord=2))
            try:
                M2_nuc = float(np.linalg.norm(M2, ord="nuc"))
            except Exception:
                M2_nuc = float(np.sum(np.linalg.svd(M2, compute_uv=False)))
            try:
                M2_sym = 0.5 * (M2 + M2.T)
                M2_lambda_min = float(np.min(np.linalg.eigvalsh(M2_sym)))
                A_score = float(m1_norm / math.sqrt(max(M2_lambda_min, _EPS)))
            except Exception:
                M2_lambda_min = np.nan
                A_score = np.inf
            B_geo = float(m1_norm + 0.5 * M2_nuc)
            R_MB = float(math.sqrt(max(n, 1)) * B_geo / max(sw, _EPS))
        ridge_key = float(cfg.gh_edge_ridge)
        if ridge_key not in self._edge_score_cache:
            self._edge_score_cache[ridge_key] = _raw_edge_score(
                self.Z_cpu, self.x0_cpu, self.h, self.kernel, ridge=ridge_key
            )
        edge_score = self._edge_score_cache[ridge_key]
        bias_numer = float(lam * max(point_f - Q_h, 0.0))
        N_eff = max(2 * M, 1)
        bias_score_label = float(math.sqrt(max(0.0, n * bias_numer / max(Q_h, _EPS))))
        bias_score_full = float(math.sqrt(max(0.0, bias_numer / max(Q_h * (1.0 / max(n, 1) + 1.0 / N_eff), _EPS))))
        bias_budget = _bias_budget(n, M, cfg, edge_score)
        bias_score = bias_score_full if str(cfg.bias_screen).lower() in {"p3", "p3_full", "full"} else bias_score_label
        R_GH = bias_score_label
        return {
            "h": self.h,
            "lambda": lam,
            "lambda_factor": lambda_factor_report(lam, n, cfg),
            "eff_dim": eff_f,
            "point_leverage": point_f,
            "op_score": op_score,
            "loc_score": loc_score,
            "stable": stable,
            "J_w": J,
            "V_w": Q_h,
            "Q_h": Q_h,
            "D_h_point": point_f,
            "D_h_op": eff_f,
            "sw_proxy": sw,
            "power": power_f,
            "m1_norm": m1_norm,
            "M2_nuc": M2_nuc,
            "B_geo": B_geo,
            "R_MB": R_MB,
            "R_GH": R_GH,
            "bias_score": bias_score,
            "bias_score_label": bias_score_label,
            "bias_score_full": bias_score_full,
            "bias_budget": bias_budget,
            "edge_score": edge_score,
            "A_score": A_score,
            "M2_lambda_min": M2_lambda_min,
            "w_train": w_np,
        }


class CachedRKHSLocalizationWeight:
    """RKHS localization weight built from an already-computed tuning spectrum."""

    def __init__(self, spec: _SpectralAtH, lam: float):
        self.spec = spec
        self.Z_cpu = spec.Z_cpu
        self.x0_cpu = spec.x0_cpu
        self.h = float(spec.h)
        self.lam = float(lam)
        self.kernel_name = spec.kernel
        self.M = int(spec.M)
        self.backend = spec.backend
        if self.backend == "torch":
            torch = spec.torch
            with torch.no_grad():
                denom = spec.eigvals_t + self.M * self.lam
                self.xi = spec.eigvecs_t @ (spec.tmp_t / denom)
        else:
            denom = spec.eigvals + self.M * self.lam
            self.xi = spec.eigvecs @ (spec.tmp / denom)
            self.kernel = get_kernel(spec.kernel)

    def __call__(self, X_query: np.ndarray) -> np.ndarray:
        X_query = np.asarray(X_query, dtype=float)
        if X_query.ndim == 1:
            X_query = X_query.reshape(1, -1)
        if self.backend == "torch":
            torch = self.spec.torch
            with torch.no_grad():
                Xg = torch.as_tensor(X_query, dtype=torch.float64, device=self.spec.device)
                k_x0 = _torch_kernel(Xg, self.spec.x0t, self.h, self.kernel_name).reshape(-1)
                K_xS = _torch_kernel(Xg, self.spec.Zt, self.h, self.kernel_name)
                out = (k_x0 - K_xS @ self.xi) / self.lam
                return out.detach().cpu().numpy()
        k_x0 = self.kernel(X_query, self.x0_cpu, self.h).ravel()
        K_xS = self.kernel(X_query, self.Z_cpu, self.h)
        return (k_x0 - K_xS @ self.xi) / self.lam

    def training_values(self) -> np.ndarray:
        if self.backend == "torch":
            return (self.M * self.xi).detach().cpu().numpy()
        return self.M * self.xi


def weight_from_joint_cache(cache: dict, tr: JointTuningResult, cfg: Optional[JointTuningConfig] = None):
    """Build a weight using the spectral decomposition already computed for tuning."""
    specs = cache.get("spec_by_h", {})
    h = float(tr.h)
    spec = specs.get(h)
    if spec is None and specs:
        spec = min(specs.values(), key=lambda s: abs(float(s.h) - h))
    if spec is not None and abs(float(spec.h) - h) <= max(1e-10, 1e-8 * max(abs(h), 1.0)):
        return CachedRKHSLocalizationWeight(spec, float(tr.lam))

    from .weights import RKHSLocalizationWeight

    cfg = cfg or JointTuningConfig()
    return RKHSLocalizationWeight(cache["Z"], cache["x0"], tr.h, tr.lam, cfg.kernel, backend=cfg.backend)


def _least_bad(rows: list[dict], cfg: JointTuningConfig) -> dict:
    return min(rows, key=lambda r: max(r["op_score"] / cfg.tau_op, r["loc_score"] / cfg.tau_loc, 0.0))


def _result_from_row(method: str, row: dict, status: str, M: int, n: int, k: int, ess0: float,
                     med: float, n_stable_this_h: int, n_feasible_total: int, cfg: JointTuningConfig,
                     power_ref: float = np.nan, pc_ratio: float = np.nan) -> JointTuningResult:
    return JointTuningResult(
        method=method,
        h=float(row["h"]),
        lam=float(row["lambda"]),
        status=status,
        M=int(M),
        n_label=int(n),
        k_target=int(k),
        ess0=float(ess0),
        lambda_factor=float(row["lambda_factor"]),
        lambda_grid_mode=str(cfg.lambda_grid_mode),
        eff_dim=float(row["eff_dim"]),
        point_leverage=float(row["point_leverage"]),
        op_score=float(row["op_score"]),
        loc_score=float(row["loc_score"]),
        stable=bool(row["stable"]),
        n_stable_this_h=int(n_stable_this_h),
        n_feasible_total=int(n_feasible_total),
        h_grid_mode=str(cfg.h_grid_mode),
        h_anchor_median=float(med),
        h_factor_vs_median=float(row["h"] / med),
        J_w=float(row["J_w"]),
        V_w=float(row["V_w"]),
        Q_h=float(row["Q_h"]),
        D_h_point=float(row["D_h_point"]),
        D_h_op=float(row["D_h_op"]),
        sw_proxy=float(row["sw_proxy"]),
        power=float(row["power"]),
        power_ref=float(power_ref),
        pc_ratio=float(pc_ratio),
        m1_norm=float(row["m1_norm"]),
        M2_nuc=float(row["M2_nuc"]),
        B_geo=float(row["B_geo"]),
        R_MB=float(row["R_MB"]),
        R_GH=float(row["R_GH"]),
        bias_screen=str(cfg.bias_screen),
        c_bias=float(cfg.c_bias),
        bias_score=float(_row_bias_score(row, n, M, cfg)),
        bias_score_label=float(row["bias_score_label"]),
        bias_score_full=float(row["bias_score_full"]),
        bias_budget=float(_bias_budget(n, M, cfg, row.get("edge_score", 0.0))),
        gh_gamma_used=float(_gh_gamma_used(n, M, cfg)),
        gh_gamma_eff=float(_gh_gamma_eff(n, M, cfg, row.get("edge_score", 0.0))),
        edge_score=float(row.get("edge_score", np.nan)),
        A_score=float(row.get("A_score", np.nan)),
        M2_lambda_min=float(row.get("M2_lambda_min", np.nan)),
    )


def _constraint_violation(method: str, row: dict, pc_ratio: float, n: int, M: int, cfg: JointTuningConfig) -> float:
    method_u = str(method).upper()
    if method_u == "PC":
        return float(pc_ratio / max(cfg.pc_r, _EPS))
    if method_u == "MB":
        return float(row["R_MB"] / max(cfg.mb_gamma, _EPS))
    if method_u == "GH":
        budget = _bias_budget(n, M, cfg, row.get("edge_score", 0.0))
        vals = [float(_row_bias_score(row, n, M, cfg) / max(budget, _EPS))]
        if np.isfinite(cfg.gh_a_tau):
            vals.append(float(row.get("A_score", np.inf) / max(cfg.gh_a_tau, _EPS)))
        if np.isfinite(cfg.gh_pc_r):
            vals.append(float(pc_ratio / max(cfg.gh_pc_r, _EPS)))
        return max(vals)
    return 1.0


def _normalized_selection_violation(method: str, row: dict, pc_ratio: float, n: int, M: int, cfg: JointTuningConfig) -> float:
    stability = max(
        float(row["op_score"] / max(cfg.tau_op, _EPS)),
        float(row["loc_score"] / max(cfg.tau_loc, _EPS)),
    )
    if str(method).upper() in {"PC", "MB", "GH"}:
        return float(max(stability, _constraint_violation(method, row, pc_ratio, n, M, cfg)))
    return float(stability)


def collect_joint_candidate_cache(
    Z: np.ndarray,
    x0: np.ndarray,
    n: int,
    cfg: Optional[JointTuningConfig] = None,
) -> dict:
    """Compute the shared h/lambda diagnostic grid once for a fold.

    PC, MB, and GH differ only in their admissibility budget after the common
    stability screen.  This cache lets server-scale races reuse the expensive
    spectral diagnostics across all budget values.
    """
    cfg = cfg or JointTuningConfig()
    Z = np.asarray(Z, dtype=float)
    x0 = np.asarray(x0, dtype=float).ravel()
    M = int(Z.shape[0])
    if M < 4:
        raise ValueError("Need at least four covariates for joint tuning.")

    lam_grid = lambda_grid_joint(n, cfg)
    med = distance_median_anchor(Z, x0)
    stable_pool: list[tuple[dict, int, float, int, float, float]] = []
    candidate_pool: list[tuple[dict, int, float, int, float, float]] = []
    spec_by_h: dict[float, _SpectralAtH] = {}
    first_stable: tuple[dict, int, float, int, float, float] | None = None
    fallback: tuple[dict, int, float, int, float, float] | None = None

    G_h = h_grid_joint(Z, x0, cfg)
    for k, h, ess0 in G_h:
        spec = _SpectralAtH(Z, x0, h, cfg.kernel, cfg.backend)
        spec_by_h[float(h)] = spec
        rows = [spec.candidate(lam, n, cfg) for lam in lam_grid]
        stable = [r for r in rows if r["stable"]]
        ref_row = sorted(stable if stable else rows, key=lambda r: r["lambda"])[0]
        power_ref_all = max(float(ref_row["power"]), _EPS)
        for r in rows:
            candidate_pool.append((r, k, ess0, len(stable), power_ref_all, float(r["power"] / power_ref_all)))

        bad = _least_bad(rows, cfg)
        if fallback is None:
            fallback = (bad, k, ess0, len(stable), np.nan, np.nan)
        else:
            curr_badness = max(bad["op_score"] / cfg.tau_op, bad["loc_score"] / cfg.tau_loc)
            old_badness = max(fallback[0]["op_score"] / cfg.tau_op, fallback[0]["loc_score"] / cfg.tau_loc)
            if curr_badness < old_badness:
                fallback = (bad, k, ess0, len(stable), np.nan, np.nan)

        if not stable:
            continue

        stable_sorted = sorted(stable, key=lambda r: r["lambda"])
        lambda_min_row = stable_sorted[0]
        power_ref = max(float(lambda_min_row["power"]), _EPS)
        if first_stable is None:
            first_stable = (lambda_min_row, k, ess0, len(stable), power_ref, 1.0)
        for r in stable:
            stable_pool.append((r, k, ess0, len(stable), power_ref, float(r["power"] / power_ref)))

    return {
        "M": M,
        "n": int(n),
        "Z": Z,
        "x0": x0,
        "med": med,
        "spec_by_h": spec_by_h,
        "stable_pool": stable_pool,
        "candidate_pool": candidate_pool,
        "first_stable": first_stable,
        "fallback": fallback,
    }


def select_joint_from_cache(
    cache: dict,
    method: str,
    cfg: Optional[JointTuningConfig] = None,
) -> JointTuningResult:
    """Select one tuning result from a shared fold-level candidate cache."""
    cfg = cfg or JointTuningConfig()
    method_u = str(method).upper()
    if method_u not in {"INC", "PC", "MB", "GH"}:
        raise ValueError("method must be one of INC, PC, MB, GH")

    M = int(cache["M"])
    n = int(cache["n"])
    med = float(cache["med"])
    stable_pool = cache["stable_pool"]
    candidate_pool = cache.get("candidate_pool", [])

    if method_u == "INC":
        if cache["first_stable"] is not None:
            row, k, ess0, n_stable, power_ref, pc_ratio = cache["first_stable"]
            return _result_from_row(
                method_u, row, "stable", M, n, k, ess0, med,
                n_stable, 1, cfg, power_ref=power_ref, pc_ratio=pc_ratio,
            )
    else:
        feasible: list[tuple[dict, int, float, int, float, float]] = []
        for r, k, ess0, n_stable, power_ref, pc_ratio in stable_pool:
            if method_u == "PC":
                ok = bool(pc_ratio <= cfg.pc_r)
            elif method_u == "MB":
                ok = bool(r["R_MB"] <= cfg.mb_gamma)
            else:
                ok = bool(_row_bias_score(r, n, M, cfg) <= _bias_budget(n, M, cfg, r.get("edge_score", 0.0)))
                if ok and np.isfinite(cfg.gh_a_tau):
                    ok = bool(r.get("A_score", np.inf) <= cfg.gh_a_tau)
                if ok and np.isfinite(cfg.gh_pc_r):
                    ok = bool(pc_ratio <= cfg.gh_pc_r)
            if ok:
                feasible.append((r, k, ess0, n_stable, power_ref, pc_ratio))
        if feasible:
            row, k, ess0, n_stable, power_ref, pc_ratio = min(feasible, key=lambda t: t[0]["sw_proxy"])
            return _result_from_row(
                method_u, row, "stable_feasible", M, n, k, ess0, med,
                n_stable, len(feasible), cfg, power_ref=power_ref, pc_ratio=pc_ratio,
            )

    if stable_pool or candidate_pool:
        if str(cfg.constraint_fallback).lower() in {"least_violation", "violation"} and method_u in {"PC", "MB", "GH"}:
            pool = candidate_pool if candidate_pool else stable_pool
            row, k, ess0, n_stable, power_ref, pc_ratio = min(
                pool,
                key=lambda t: (_normalized_selection_violation(method_u, t[0], t[5], n, M, cfg), t[0]["sw_proxy"]),
            )
            status = "fallback_least_normalized_violation"
        elif stable_pool:
            row, k, ess0, n_stable, power_ref, pc_ratio = min(stable_pool, key=lambda t: t[0]["sw_proxy"])
            status = "fallback_constraint_min_sw"
        else:
            row, k, ess0, n_stable, power_ref, pc_ratio = cache["fallback"]
            status = "fallback_unstable"
        return _result_from_row(
            method_u, row, status, M, n, k, ess0, med,
            n_stable, 0, cfg, power_ref=power_ref, pc_ratio=pc_ratio,
        )

    row, k, ess0, n_stable, power_ref, pc_ratio = cache["fallback"]
    return _result_from_row(
        method_u, row, "fallback_unstable", M, n, k, ess0, med,
        n_stable, 0, cfg, power_ref=power_ref, pc_ratio=pc_ratio,
    )


def tune_joint_from_covariates(
    Z: np.ndarray,
    x0: np.ndarray,
    n: int,
    method: str,
    cfg: Optional[JointTuningConfig] = None,
) -> JointTuningResult:
    """Tune h/lambda for one fold using only covariates.

    Parameters
    ----------
    Z : array, shape (M, d)
        Covariate fold used to construct the RKHS localization weight.
    x0 : array, shape (d,)
        Target point in the same scaled coordinate system as Z.
    n : int
        Labeled sample size used for lambda scaling and diagnostics.
    method : {"INC", "PC", "MB", "GH"}
        INC = incumbent; PC = joint power-constrained; MB = moment budget; GH = gamma-H.
    cfg : JointTuningConfig
        Tuning settings.
    """
    cfg = cfg or JointTuningConfig()
    method_u = str(method).upper()
    if method_u not in {"INC", "PC", "MB", "GH"}:
        raise ValueError("method must be one of INC, PC, MB, GH")
    Z = np.asarray(Z, dtype=float)
    x0 = np.asarray(x0, dtype=float).ravel()
    M = int(Z.shape[0])
    if M < 4:
        raise ValueError("Need at least four covariates for joint tuning.")
    lam_grid = lambda_grid_joint(n, cfg)
    med = distance_median_anchor(Z, x0)
    all_feasible: list[tuple[dict, int, float, int, float, float]] = []
    first_stable_result = None
    fallback = None

    G_h = h_grid_joint(Z, x0, cfg)
    for k, h, ess0 in G_h:
        spec = _SpectralAtH(Z, x0, h, cfg.kernel, cfg.backend)
        rows = [spec.candidate(lam, n, cfg) for lam in lam_grid]
        stable = [r for r in rows if r["stable"]]
        if fallback is None:
            bad = _least_bad(rows, cfg)
            fallback = (bad, k, ess0, len(stable), np.nan, np.nan)
        else:
            bad = _least_bad(rows, cfg)
            curr_badness = max(bad["op_score"] / cfg.tau_op, bad["loc_score"] / cfg.tau_loc)
            old_badness = max(fallback[0]["op_score"] / cfg.tau_op, fallback[0]["loc_score"] / cfg.tau_loc)
            if curr_badness < old_badness:
                fallback = (bad, k, ess0, len(stable), np.nan, np.nan)
        if not stable:
            continue
        stable_sorted = sorted(stable, key=lambda r: r["lambda"])
        lambda_min_row = stable_sorted[0]
        power_ref = max(float(lambda_min_row["power"]), _EPS)
        if method_u == "INC" and first_stable_result is None:
            return _result_from_row(
                method_u, lambda_min_row, "stable", M, n, k, ess0, med,
                len(stable), 1, cfg, power_ref=power_ref, pc_ratio=1.0,
            )
        for r in stable:
            pc_ratio = float(r["power"] / power_ref)
            feasible = False
            if method_u == "PC":
                feasible = bool(pc_ratio <= cfg.pc_r)
            elif method_u == "MB":
                feasible = bool(r["R_MB"] <= cfg.mb_gamma)
            elif method_u == "GH":
                feasible = bool(_row_bias_score(r, n, M, cfg) <= _bias_budget(n, M, cfg, r.get("edge_score", 0.0)))
                if feasible and np.isfinite(cfg.gh_a_tau):
                    feasible = bool(r.get("A_score", np.inf) <= cfg.gh_a_tau)
                if feasible and np.isfinite(cfg.gh_pc_r):
                    feasible = bool(pc_ratio <= cfg.gh_pc_r)
            if feasible:
                all_feasible.append((r, k, ess0, len(stable), power_ref, pc_ratio))

    if all_feasible:
        # Minimize covariate-only SE proxy, not labeled SE/outcomes.
        chosen, k, ess0, n_stable, power_ref, pc_ratio = min(all_feasible, key=lambda t: t[0]["sw_proxy"])
        return _result_from_row(
            method_u, chosen, "stable_feasible", M, n, k, ess0, med,
            n_stable, len(all_feasible), cfg, power_ref=power_ref, pc_ratio=pc_ratio,
        )

    # Constraint fallback: if stable pairs exist but none satisfy the family constraint, choose the
    # stable pair with the smallest covariate SE proxy and mark it clearly. If no stable pairs exist,
    # choose the least-bad stability row.
    stable_pool: list[tuple[dict, int, float, int, float, float]] = []
    candidate_pool: list[tuple[dict, int, float, int, float, float]] = []
    for k, h, ess0 in G_h:
        spec = _SpectralAtH(Z, x0, h, cfg.kernel, cfg.backend)
        rows = [spec.candidate(lam, n, cfg) for lam in lam_grid]
        stable = [r for r in rows if r["stable"]]
        ref_row = sorted(stable if stable else rows, key=lambda r: r["lambda"])[0]
        power_ref_all = max(float(ref_row["power"]), _EPS)
        for r in rows:
            candidate_pool.append((r, k, ess0, len(stable), power_ref_all, float(r["power"] / power_ref_all)))
        if stable:
            stable_sorted = sorted(stable, key=lambda r: r["lambda"])
            power_ref = max(float(stable_sorted[0]["power"]), _EPS)
            for r in stable:
                stable_pool.append((r, k, ess0, len(stable), power_ref, float(r["power"] / power_ref)))
    if stable_pool or candidate_pool:
        if str(cfg.constraint_fallback).lower() in {"least_violation", "violation"} and method_u in {"PC", "MB", "GH"}:
            pool = candidate_pool if candidate_pool else stable_pool
            chosen, k, ess0, n_stable, power_ref, pc_ratio = min(
                pool,
                key=lambda t: (_normalized_selection_violation(method_u, t[0], t[5], n, M, cfg), t[0]["sw_proxy"]),
            )
            status = "fallback_least_normalized_violation"
        elif stable_pool:
            chosen, k, ess0, n_stable, power_ref, pc_ratio = min(stable_pool, key=lambda t: t[0]["sw_proxy"])
            status = "fallback_constraint_min_sw"
        else:
            chosen, k, ess0, n_stable, power_ref, pc_ratio = fallback
            status = "fallback_unstable"
        return _result_from_row(
            method_u, chosen, status, M, n, k, ess0, med,
            n_stable, 0, cfg, power_ref=power_ref, pc_ratio=pc_ratio,
        )
    chosen, k, ess0, n_stable, power_ref, pc_ratio = fallback
    return _result_from_row(
        method_u, chosen, "fallback_unstable", M, n, k, ess0, med,
        n_stable, 0, cfg, power_ref=power_ref, pc_ratio=pc_ratio,
    )
