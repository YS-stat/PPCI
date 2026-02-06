# ============================================================
# conditional_quantile_functions.py
#   - Matérn 5/2 kernel (replace Gaussian)
#   - LOGO CV over log-ell grid (grouped by age bins) for ell tuning
#   - WeightAtX0 with L-curve selection:
#         x-axis: log Residual
#         y-axis: log E[f^2]   (REPLACED from log RKHS norm)
#   - PPCI conditional quantile (with sigma2 pieces)
#   - PPCI label-only conditional quantile (with sigma2)
# ============================================================

import numpy as np
import cupy as cp

xp = cp


# ============================================================
# 1) Matérn 5/2 kernel matrix
# ============================================================

def K_matern52_matrix(X, Z, ell):
    """
    Matérn 5/2 kernel:
      k(r) = (1 + t + t^2/3) exp(-t),  t = sqrt(5) r / ell
    X: (n,d), Z:(m,d) -> (n,m) on GPU
    """
    X = xp.asarray(X)
    Z = xp.asarray(Z)
    ell = float(ell)
    ell = max(ell, 1e-12)

    X2 = xp.sum(X**2, axis=1, keepdims=True)
    Z2 = xp.sum(Z**2, axis=1, keepdims=True).T
    d2 = X2 + Z2 - 2.0 * (X @ Z.T)
    d2 = xp.maximum(d2, 0.0)
    r = xp.sqrt(d2)

    t = (xp.sqrt(5.0) * r) / ell
    K = (1.0 + t + (t**2) / 3.0) * xp.exp(-t)
    return K


# ============================================================
# 2) LOGO CV helpers (grouped by age bin)
# ============================================================

def make_age_groups(age_raw, bin_width=1):
    """
    age_raw: 1D numpy array (raw age)
    returns group labels (ints) by binning ages.
    """
    age_raw = np.asarray(age_raw, dtype=float).ravel()
    bw = int(bin_width)
    if bw <= 0:
        bw = 1
    # group id by integer bins
    grp = np.floor(age_raw / bw).astype(int)
    return grp


def select_logell_by_pilot_logo_cv(
    LOGELL_grid,
    X_pilot,
    Y_pilot,
    group_labels,
):
    """
    LOGO-CV (leave-one-group-out) on pilot sample, minimizing MSE of NW mean predictor,
    using Matérn 5/2 kernel with ell = exp(logell).

    Inputs:
      - X_pilot: (n_pilot,d) cupy
      - Y_pilot: (n_pilot,) cupy
      - group_labels: (n_pilot,) numpy int labels

    Returns:
      - ell_star (float)
      - pilot_table: list of dicts with cv_mse per logell
    """
    LOGELL_grid = np.asarray(LOGELL_grid, dtype=float).ravel()
    if LOGELL_grid.size == 0:
        raise ValueError("LOGELL_grid is empty.")

    Xp = xp.asarray(X_pilot)
    Yp = xp.asarray(Y_pilot).ravel()
    n = int(Xp.shape[0])

    group_labels = np.asarray(group_labels)
    if group_labels.shape[0] != n:
        raise ValueError("group_labels length mismatch with X_pilot.")

    # unique groups
    groups = np.unique(group_labels)

    pilot_table = []
    best_mse = np.inf
    best_logell = float(LOGELL_grid[0])

    for logell in LOGELL_grid:
        ell = float(np.exp(logell))

        # full kernel (n,n)
        K = K_matern52_matrix(Xp, Xp, ell)
        # avoid asymmetry numeric
        K = 0.5 * (K + K.T)

        row_sum_total = xp.sum(K, axis=1)              # (n,)
        row_sum_total = xp.maximum(row_sum_total, 1e-12)
        num_total = K @ Yp                             # (n,)

        sse = 0.0
        # LOGO: for each group, exclude all points in that group
        for g in groups:
            idx_g = np.where(group_labels == g)[0]
            if idx_g.size == 0:
                continue
            idx_g_gpu = xp.asarray(idx_g, dtype=xp.int64)

            # contributions from excluded columns
            col_sum_g = xp.sum(K[:, idx_g_gpu], axis=1)                   # (n,)
            num_g = K[:, idx_g_gpu] @ Yp[idx_g_gpu]                       # (n,)

            row_sum_excl = row_sum_total - col_sum_g
            row_sum_excl = xp.maximum(row_sum_excl, 1e-12)
            num_excl = num_total - num_g

            pred_g = num_excl[idx_g_gpu] / row_sum_excl[idx_g_gpu]
            resid_g = Yp[idx_g_gpu] - pred_g
            sse += float(xp.sum(resid_g**2))

        cv_mse = sse / max(n, 1)
        pilot_table.append({"logell": float(logell), "ell": float(ell), "cv_mse": float(cv_mse)})

        if cv_mse < best_mse:
            best_mse = cv_mse
            best_logell = float(logell)

    ell_star = float(np.exp(best_logell))
    return ell_star, pilot_table


# ============================================================
# 3) WeightAtX0 (Matérn 5/2) with L-curve
#    y-axis changed from log RKHS norm to log E[f^2]
# ============================================================

class WeightAtX0:
    """
    Empirical localization weight function constructed from auxiliary covariates.

    For a given x0, work in span {K(·, X_aux[v])} and use:
      f_hat(x) = sum_v alpha_v K(x, X_aux[v])

    We precompute eigendecomposition of K_t = K(X_aux, X_aux).
    """

    def __init__(self, X_aux, x0, ell):
        self.X_aux = xp.asarray(X_aux)
        self.ell = float(ell)
        self.N_t = int(self.X_aux.shape[0])

        K_t = K_matern52_matrix(self.X_aux, self.X_aux, self.ell)
        K_t = 0.5 * (K_t + K_t.T)

        eigvals, eigvecs = xp.linalg.eigh(K_t)
        self.eigvals = xp.maximum(eigvals, 1e-12)
        self.eigvecs = eigvecs

        x0_gpu = xp.asarray(x0).reshape(1, -1)
        k_x0 = K_matern52_matrix(self.X_aux, x0_gpu, self.ell)[:, 0]
        self.tmp_x0 = self.eigvecs.T @ k_x0

    def alpha(self, lam: float):
        denom = self.eigvals + self.N_t * float(lam)
        return self.eigvecs @ (self.tmp_x0 / denom)

    def select_lambda_lcurve(self, lam_grid, normalize=False, make_plot=False):
        """
        Triangle rule on L-curve:
          x = log Residual^2
          y = log E[f^2]   (REPLACED)

        Residual^2 in eigenbasis (same as before):
          r_j(λ) = (N_t λ / (Λ_j + N_t λ)) * tmp_j
          resid_sq = sum_j r_j(λ)^2

        E[f^2] on aux empirical measure:
          f_vals = K_t alpha = Q (Λ * tmp / (Λ + N_t λ))
          Ef2 = (1/N_t) ||f_vals||^2 = (1/N_t) sum_j (Λ_j * tmp_j / (Λ_j + N_t λ))^2
        """
        lam_grid = np.asarray(lam_grid, dtype=float).ravel()
        lam_grid = np.sort(lam_grid)
        L = lam_grid.size
        if L == 0:
            raise ValueError("lam_grid is empty.")

        tmp = self.tmp_x0
        eigvals = self.eigvals

        ef2_log_raw = []
        resid_log_raw = []

        for lam in lam_grid:
            lam_val = float(lam)
            denom = eigvals + self.N_t * lam_val

            # residual^2
            ratio = (self.N_t * lam_val) / denom
            resid_sq = float(xp.sum((ratio * tmp) ** 2))

            # E[f^2]
            fcoef = (eigvals * tmp) / denom
            ef2 = float(xp.sum(fcoef**2)) / max(self.N_t, 1)

            ef2_log_raw.append(np.log(ef2 + 1e-30))
            resid_log_raw.append(np.log(resid_sq + 1e-30))

        ef2_log_raw = np.asarray(ef2_log_raw)
        resid_log_raw = np.asarray(resid_log_raw)

        x = resid_log_raw.copy()
        y = ef2_log_raw.copy()

        if normalize:
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            x = (x - x_min) / (x_max - x_min + 1e-16)
            y = (y - y_min) / (y_max - y_min + 1e-16)

        p_start = np.array([x[0], y[0]])
        p_end = np.array([x[-1], y[-1]])
        vec_line = p_end - p_start
        line_norm = np.linalg.norm(vec_line) + 1e-12

        dists = []
        for xi, yi in zip(x, y):
            p = np.array([xi, yi])
            vec_p = p - p_start
            cross = vec_p[0] * vec_line[1] - vec_p[1] * vec_line[0]
            dists.append(abs(cross) / line_norm)

        dist_curve = np.asarray(dists, dtype=float)
        best_idx = int(np.argmax(dist_curve))
        best_idx = int(np.clip(best_idx, 0, L - 1))
        best_lam = float(lam_grid[best_idx])

        # keep plotting switch, but default off (unchanged behavior)
        if make_plot:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(15, 5))

            ax1 = fig.add_subplot(131)
            ax1.plot(resid_log_raw, ef2_log_raw, "b-", lw=2, alpha=0.7, label="L-curve")
            ax1.scatter(resid_log_raw[best_idx], ef2_log_raw[best_idx], s=150, c="r", marker="*", zorder=10)
            ax1.plot([resid_log_raw[0], resid_log_raw[-1]], [ef2_log_raw[0], ef2_log_raw[-1]],
                     "k--", alpha=0.3, label="Baseline")
            ax1.set_xlabel("Log Residual")
            ax1.set_ylabel(r"Log $E[f^2]$")
            ax1.set_title("L-curve")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            ax2 = fig.add_subplot(132)
            ax2.plot(np.log10(lam_grid), dist_curve, "g-", lw=2)
            ax2.axvline(np.log10(best_lam), color="r", ls="--")
            ax2.set_xlabel(r"$\log_{10}(\lambda)$")
            ax2.set_ylabel("Geometric Distance")
            ax2.set_title("Selection Criterion")
            ax2.grid(True, alpha=0.3)

            ax3 = fig.add_subplot(133)
            ax3_t = ax3.twinx()
            ax3.plot(np.log10(lam_grid), resid_log_raw, "b--", label="Residual")
            ax3_t.plot(np.log10(lam_grid), ef2_log_raw, "orange", ls="--", label=r"$E[f^2]$")
            ax3.axvline(np.log10(best_lam), color="r", ls="-", lw=1.5)
            ax3.set_title("Trade-off")
            ax3.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        alpha_hat = self.alpha(best_lam)
        return best_lam, alpha_hat, lam_grid

    def make_f_hat(self, alpha_hat):
        X_aux = self.X_aux
        ell = self.ell
        alpha_hat = xp.asarray(alpha_hat)

        def f_hat(X):
            X = xp.asarray(X)
            K_x = K_matern52_matrix(X, X_aux, ell)
            return K_x @ alpha_hat

        return f_hat


# ============================================================
# 4) Logistic smoothing helpers (unchanged)
# ============================================================

def sigmoid_xp(z):
    z = xp.asarray(z, dtype=xp.float64)
    out = xp.empty_like(z)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + xp.exp(-z[pos]))
    ez = xp.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out

def smooth_ind_xp(Y, theta, b):
    z = (theta - Y) / max(float(b), 1e-12)
    return sigmoid_xp(z)

def smooth_pdf_xp(Y, theta, b):
    z = (theta - Y) / max(float(b), 1e-12)
    s = sigmoid_xp(z)
    return (s * (1.0 - s)) / max(float(b), 1e-12)

def choose_b_smooth_xp(Y, n, c_smooth):
    Y = xp.asarray(Y)
    if n > 1:
        sd = float(xp.std(Y, ddof=1))
    else:
        sd = 1.0
    return float(c_smooth) * 1.06 * sd * (n ** (-1 / 5))


def bisection_or_newton(Mfun, Hfun, lo, hi, theta_init=None, tol=1e-6, max_iter=120):
    f_lo = float(Mfun(lo))
    f_hi = float(Mfun(hi))
    if f_lo * f_hi <= 0.0:
        a, b = float(lo), float(hi)
        fa, fb = f_lo, f_hi
        for _ in range(max_iter):
            c = 0.5 * (a + b)
            fc = float(Mfun(c))
            if abs(fc) < tol or (b - a) / 2 < tol:
                return c
            if fa * fc <= 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        return 0.5 * (a + b)

    if theta_init is None:
        theta = 0.5 * (lo + hi)
    else:
        theta = float(theta_init)

    for _ in range(max_iter):
        Mv = float(Mfun(theta))
        Hv = float(Hfun(theta))
        if abs(Hv) < 1e-8:
            step = -0.1 * np.sign(Mv)
        else:
            step = -Mv / Hv
        step = float(np.clip(step, -0.25 * (hi - lo), 0.25 * (hi - lo)))
        theta_new = float(min(max(theta + step, lo), hi))
        if abs(theta_new - theta) < tol and abs(Mv) < 5 * tol:
            return theta_new
        theta = theta_new
    return theta


# ============================================================
# 5) NW conditional quantile at x0 (CPU) using Matérn weights
#     (used for theta0_smooth)
# ============================================================

def local_weighted_quantile(Y, w, tau):
    Y = np.asarray(Y).ravel()
    w = np.asarray(w).ravel()
    if np.sum(w) <= 0:
        raise ValueError("Sum of weights must be positive.")
    w = w / np.sum(w)

    order = np.argsort(Y)
    Y_sorted = Y[order]
    w_sorted = w[order]
    cw = np.cumsum(w_sorted)
    idx = int(np.searchsorted(cw, tau, side="left"))
    idx = min(idx, len(Y_sorted) - 1)
    return float(Y_sorted[idx])

def local_density_weighted(Y, w, theta):
    if isinstance(Y, cp.ndarray):
        Y = cp.asnumpy(Y)
    Y = np.asarray(Y, float).ravel()

    if isinstance(w, cp.ndarray):
        w = cp.asnumpy(w)
    w = np.asarray(w, float).ravel()

    if len(Y) <= 1 or np.sum(w) <= 0:
        return 1.0

    w = w / np.sum(w)
    n_eff = 1.0 / np.sum(w**2)
    n_eff = max(n_eff, 1.0)

    mu = np.sum(w * Y)
    var = np.sum(w * (Y - mu) ** 2)
    sd = float(np.sqrt(max(var, 1e-12)))

    h = 1.06 * sd * (n_eff ** (-1.0 / 5.0))
    h = max(h, 1e-3)

    z = (theta - Y) / h
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    f_hat = np.sum(w * phi) / h
    return float(max(f_hat, 1e-8))

def nw_quantile_point_ci_matern52(X_l, Y_l, x0, ell, tau, z_alpha=1.96):
    """
    NW local τ-quantile at x0 using Matérn 5/2 weights.
    Compatible with X_l/Y_l possibly cupy; computed on CPU.

    Returns: theta_hat, se, (lo,up)
    """
    if isinstance(X_l, cp.ndarray):
        X_l_np = cp.asnumpy(X_l)
    else:
        X_l_np = np.asarray(X_l, float)

    if isinstance(Y_l, cp.ndarray):
        Y_l_np = cp.asnumpy(Y_l)
    else:
        Y_l_np = np.asarray(Y_l, float).ravel()

    x0_np = np.asarray(cp.asnumpy(x0) if isinstance(x0, cp.ndarray) else x0, float).ravel()

    diff = X_l_np - x0_np.reshape(1, -1)
    d2 = np.sum(diff**2, axis=1)
    r = np.sqrt(np.maximum(d2, 0.0))

    ell = float(ell)
    ell = max(ell, 1e-12)
    t = (np.sqrt(5.0) * r) / ell
    k = (1.0 + t + (t**2) / 3.0) * np.exp(-t)

    ksum = float(np.sum(k))
    if ksum <= 1e-16:
        theta_hat = float(np.quantile(Y_l_np, tau))
        return theta_hat, np.nan, (np.nan, np.nan)

    w = k / ksum
    theta_hat = local_weighted_quantile(Y_l_np, w, tau)

    n_eff = 1.0 / np.sum(w**2)
    n_eff = max(n_eff, 1.0)

    f_hat = local_density_weighted(Y_l_np, w, theta_hat)
    var_hat = tau * (1 - tau) / (n_eff * (f_hat**2))
    var_hat = max(var_hat, 0.0)
    se = float(np.sqrt(var_hat))

    return theta_hat, se, (theta_hat - z_alpha * se, theta_hat + z_alpha * se)


# ============================================================
# 6) PPCI quantile core + Label-only core (with sigma2 pieces)
# ============================================================

def ppci_conditional_quantile(
    X_l, Y_l, X_u, f_hat, A_l, A_u,
    tau, z_alpha=1.96, c_smooth=0.3, smooth_var=False,
    return_extras=False,
):
    """
    PPCI conditional τ-quantile at x0 with localization weight f_hat.

    Returns:
      theta_hat, se, (lo,up) [, extras]
    extras contains:
      - sigma2_Y_minus_A : Var( f_l * ((tau-I{Y<=θ}) - (tau-I{A<=θ})) )
      - sigma2_A         : Var( f_u * (tau-I{A<=θ}) )
      - H_hat            : derivative estimate
    """
    X_l = xp.asarray(X_l)
    Y_l = xp.asarray(Y_l).ravel()
    X_u = xp.asarray(X_u)
    A_l = xp.asarray(A_l).ravel()
    A_u = xp.asarray(A_u).ravel()

    n = int(X_l.shape[0])
    N = int(X_u.shape[0])
    N_prime = n + N
    p_R = n / max(N_prime, 1)
    p_U = N / max(N_prime, 1)

    f_l = f_hat(X_l).ravel()
    f_u = f_hat(X_u).ravel()

    b = choose_b_smooth_xp(Y_l, n, c_smooth=c_smooth)

    def M_s(theta):
        loss_l_s = tau - smooth_ind_xp(Y_l, theta, b)
        loss_Al_s = tau - smooth_ind_xp(A_l, theta, b)
        loss_Au_s = tau - smooth_ind_xp(A_u, theta, b)
        term_R = xp.mean(f_l * (loss_l_s - loss_Al_s))
        term_U = xp.mean(f_u * loss_Au_s)
        return float(term_R + term_U)

    def H_s(theta):
        d_loss_l = -smooth_pdf_xp(Y_l, theta, b)
        d_loss_Al = -smooth_pdf_xp(A_l, theta, b)
        d_loss_Au = -smooth_pdf_xp(A_u, theta, b)
        term_R = xp.mean(f_l * (d_loss_l - d_loss_Al))
        term_U = xp.mean(f_u * d_loss_Au)
        return float(term_R + term_U)

    all_vals = xp.concatenate([Y_l, A_l, A_u])
    y_min = float(xp.min(all_vals))
    y_max = float(xp.max(all_vals))
    pad = 4.0 * float(xp.std(all_vals, ddof=1)) if (n + N) > 1 else 2.0
    lo = y_min - pad
    hi = y_max + pad

    # init guess: weighted quantile using positive part of f_l
    w_pos = xp.maximum(f_l, 0.0)
    if float(xp.sum(w_pos)) <= 1e-12:
        theta_init = float(cp.asnumpy(xp.median(Y_l)))
    else:
        Y_cpu = cp.asnumpy(Y_l)
        w_cpu = cp.asnumpy(w_pos)
        order = np.argsort(Y_cpu)
        Y_sorted = Y_cpu[order]
        W_sorted = w_cpu[order]
        cdf = np.cumsum(W_sorted) / np.sum(W_sorted)
        k_idx = int(np.searchsorted(cdf, tau, side="left"))
        k_idx = int(np.clip(k_idx, 0, len(Y_sorted) - 1))
        theta_init = float(Y_sorted[k_idx])

    theta_hat = bisection_or_newton(M_s, H_s, lo, hi, theta_init=theta_init, tol=1e-6, max_iter=120)
    H_hat = float(H_s(theta_hat))
    if abs(H_hat) < 1e-10:
        if return_extras:
            return float(theta_hat), np.nan, (np.nan, np.nan), {"sigma2_Y_minus_A": np.nan, "sigma2_A": np.nan, "H_hat": H_hat}
        return float(theta_hat), np.nan, (np.nan, np.nan)

    if smooth_var:
        loss_l = tau - smooth_ind_xp(Y_l, theta_hat, b)
        loss_Al = tau - smooth_ind_xp(A_l, theta_hat, b)
        loss_Au = tau - smooth_ind_xp(A_u, theta_hat, b)
        psi_R = f_l * (loss_l - loss_Al)
        psi_U = f_u * loss_Au
    else:
        loss_l = tau - (Y_l <= theta_hat).astype(xp.float64)
        loss_Al = tau - (A_l <= theta_hat).astype(xp.float64)
        loss_Au = tau - (A_u <= theta_hat).astype(xp.float64)
        psi_R = f_l * (loss_l - loss_Al)
        psi_U = f_u * loss_Au

    sigma2_YmA = float(xp.var(psi_R, ddof=1)) if n > 1 else 0.0
    sigma2_A = float(xp.var(psi_U, ddof=1)) if N > 1 else 0.0

    Sigma_hat = (sigma2_YmA / max(p_R, 1e-12)) + (sigma2_A / max(p_U, 1e-12))
    V_hat = Sigma_hat / (H_hat ** 2)
    se = float(np.sqrt(max(V_hat, 0.0) / max(N_prime, 1)))

    theta_hat = float(theta_hat)
    ci = (theta_hat - z_alpha * se, theta_hat + z_alpha * se)

    if return_extras:
        ex = {"sigma2_Y_minus_A": sigma2_YmA, "sigma2_A": sigma2_A, "H_hat": H_hat}
        return theta_hat, se, ci, ex
    return theta_hat, se, ci


def ppci_conditional_quantile_label_only(
    X_l, Y_l, f_hat,
    tau, z_alpha=1.96, c_smooth=0.3, smooth_var=False,
    return_extras=False,
):
    """
    PPCI label-only conditional quantile:
      solve E[ f(X) (tau - I{Y<=θ}) ] = 0  (via logistic smoothing)
    Variance uses psi = f(X)(tau-I{Y<=θ}) (or smoothed if smooth_var=True).

    extras contains:
      - sigma2_Y : Var(psi)
      - H_hat
    """
    X_l = xp.asarray(X_l)
    Y_l = xp.asarray(Y_l).ravel()
    n = int(X_l.shape[0])

    f_l = f_hat(X_l).ravel()
    b = choose_b_smooth_xp(Y_l, n, c_smooth=c_smooth)

    def M_s(theta):
        loss_l_s = tau - smooth_ind_xp(Y_l, theta, b)
        return float(xp.mean(f_l * loss_l_s))

    def H_s(theta):
        d_loss_l = -smooth_pdf_xp(Y_l, theta, b)
        return float(xp.mean(f_l * d_loss_l))

    y_min = float(xp.min(Y_l))
    y_max = float(xp.max(Y_l))
    pad = 4.0 * float(xp.std(Y_l, ddof=1)) if n > 1 else 2.0
    lo = y_min - pad
    hi = y_max + pad

    w_pos = xp.maximum(f_l, 0.0)
    if float(xp.sum(w_pos)) <= 1e-12:
        theta_init = float(cp.asnumpy(xp.median(Y_l)))
    else:
        Y_cpu = cp.asnumpy(Y_l)
        w_cpu = cp.asnumpy(w_pos)
        order = np.argsort(Y_cpu)
        Y_sorted = Y_cpu[order]
        W_sorted = w_cpu[order]
        cdf = np.cumsum(W_sorted) / np.sum(W_sorted)
        k_idx = int(np.searchsorted(cdf, tau, side="left"))
        k_idx = int(np.clip(k_idx, 0, len(Y_sorted) - 1))
        theta_init = float(Y_sorted[k_idx])

    theta_hat = bisection_or_newton(M_s, H_s, lo, hi, theta_init=theta_init, tol=1e-6, max_iter=120)
    H_hat = float(H_s(theta_hat))
    if abs(H_hat) < 1e-10:
        if return_extras:
            return float(theta_hat), np.nan, (np.nan, np.nan), {"sigma2_Y": np.nan, "H_hat": H_hat}
        return float(theta_hat), np.nan, (np.nan, np.nan)

    if smooth_var:
        loss_l = tau - smooth_ind_xp(Y_l, theta_hat, b)
        psi = f_l * loss_l
    else:
        loss_l = tau - (Y_l <= theta_hat).astype(xp.float64)
        psi = f_l * loss_l

    sigma2_Y = float(xp.var(psi, ddof=1)) if n > 1 else 0.0
    V_hat = sigma2_Y / (H_hat ** 2)
    se = float(np.sqrt(max(V_hat, 0.0) / max(n, 1)))

    theta_hat = float(theta_hat)
    ci = (theta_hat - z_alpha * se, theta_hat + z_alpha * se)

    if return_extras:
        ex = {"sigma2_Y": sigma2_Y, "H_hat": H_hat}
        return theta_hat, se, ci, ex
    return theta_hat, se, ci
