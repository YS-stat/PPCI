# ============================================================
#  conditional_mean_functions.py  (CLEAN sigma^2 implementation)
#
#  What is fixed / cleaned:
#   - sigma^2 terms are computed EXACTLY as you defined (theta-aware):
#       sigma2_Y_minus_A(theta) = Var( f(X) * (ell(Y;theta)-ell(A;theta)) )
#       sigma2_A(theta)         = Var( f(X) * ell(A;theta) )
#       sigma2_Y(theta)         = Var( f(X) * ell(Y;theta) )
#   - Avoid pointless duplication:
#       For mean-ell, ell(Y;theta)-ell(A;theta) = Y-A (theta cancels),
#       so we compute g_YmA = f*(Y-A) directly, but still store theta_for_var.
#   - CI construction stays exactly as your previous influence-style CI.
#     (psi_R, psi_U used for Sigma_hat; we just don't recompute same vectors twice.)
# ============================================================

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

xp = cp


# ============================================================
# 0) X preprocessing: standardize (CPU)
# ============================================================
def preprocess_X(X_total):
    X_total = np.asarray(X_total, dtype=float)
    mean = X_total.mean(axis=0)
    std = X_total.std(axis=0)
    std[std == 0.0] = 1.0
    X_scaled = (X_total - mean) / std
    return X_scaled, mean, std


# ============================================================
# 1) Matérn kernel ν=2.5 (Matérn 5/2) matrix
# ============================================================
def K_matern52_matrix(X, Z, ell):
    X = xp.asarray(X)
    Z = xp.asarray(Z)
    ell = float(ell)

    X2 = xp.sum(X**2, axis=1, keepdims=True)
    Z2 = xp.sum(Z**2, axis=1, keepdims=True).T
    d2 = X2 + Z2 - 2.0 * (X @ Z.T)
    d2 = xp.maximum(d2, 0.0)
    r = xp.sqrt(d2 + 1e-18)

    sqrt5 = np.sqrt(5.0)
    t = (sqrt5 * r) / (ell + 1e-18)
    poly = 1.0 + t + (t**2) / 3.0
    return poly * xp.exp(-t)


# ============================================================
# 2) Pilot ell selection: Leave-one-age-group-out CV (LOGO-CV)
# ============================================================
def make_age_groups(age_raw, bin_width=1):
    age_raw = np.asarray(age_raw, dtype=float)
    if bin_width <= 1:
        return age_raw.astype(int)
    return (np.floor(age_raw / bin_width) * bin_width).astype(int)


def nw_logo_cv_mse_matern52(X, Y, group_labels, ell, min_train=5):
    X = xp.asarray(X)
    Y = xp.asarray(Y).ravel()
    group_labels = np.asarray(group_labels)

    n = int(Y.size)
    if n <= 2:
        return np.inf

    K = K_matern52_matrix(X, X, ell)  # (n, n)

    uniq = np.unique(group_labels)
    se_sum_total = 0.0
    n_total = 0

    for g in uniq:
        te = np.where(group_labels == g)[0]
        if te.size == 0:
            continue
        tr = np.where(group_labels != g)[0]
        if tr.size < min_train:
            continue

        te_gpu = cp.asarray(te, dtype=cp.int32)
        tr_gpu = cp.asarray(tr, dtype=cp.int32)

        K_te_tr = K[te_gpu][:, tr_gpu]
        den = xp.sum(K_te_tr, axis=1)
        den = xp.maximum(den, 1e-12)

        Y_tr = Y[tr_gpu]
        pred = (K_te_tr @ Y_tr) / den

        resid = Y[te_gpu] - pred
        se_sum_total += float(xp.sum(resid**2))
        n_total += int(te.size)

    if n_total == 0:
        return np.inf
    return se_sum_total / n_total


def select_logell_by_pilot_logo_cv(LOGELL_grid, X_pilot, Y_pilot, group_labels):
    records = []
    for logell in LOGELL_grid:
        logell_val = float(logell)
        ell_val = float(np.exp(logell_val))
        cv_mse = nw_logo_cv_mse_matern52(X=X_pilot, Y=Y_pilot, group_labels=group_labels, ell=ell_val)
        records.append({"logell": logell_val, "ell": ell_val, "cv_mse": cv_mse})

    scores = np.array([r["cv_mse"] for r in records], dtype=float)
    best_idx = int(np.argmin(scores))
    ell_star = float(records[best_idx]["ell"])
    return ell_star, records


# ============================================================
# 3) NW point estimator (only to define theta0_smooth)
# ============================================================
def nw_point_mean_ci_matern52(X_l, Y_l, x0, ell, z_alpha=1.96):
    X_l = xp.asarray(X_l)
    Y_l = xp.asarray(Y_l).ravel()
    x0_gpu = xp.asarray(x0).ravel()

    k = K_matern52_matrix(x0_gpu.reshape(1, -1), X_l, ell).ravel()
    ksum = float(xp.sum(k))

    if abs(ksum) <= 1e-16:
        d2 = xp.sum((X_l - x0_gpu) ** 2, axis=1)
        idx = int(xp.argmin(d2))
        theta_hat = float(Y_l[idx])
        return theta_hat, 0.0, (theta_hat, theta_hat)

    w = k / (ksum + 1e-16)
    theta_hat = float(w @ Y_l)

    var_hat = float(xp.sum((w**2) * (Y_l - theta_hat) ** 2))
    var_hat = max(var_hat, 0.0)
    se = float(np.sqrt(var_hat))
    return theta_hat, se, (theta_hat - z_alpha * se, theta_hat + z_alpha * se)


# ============================================================
# 4) WeightAtX0 + L-curve lambda selection
# ============================================================
class WeightAtX0:
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
        self.tmp_x0 = self.eigvecs.T @ k_x0  # Q^T k_x0

    def alpha(self, lam: float):
        denom = self.eigvals + self.N_t * float(lam)
        return self.eigvecs @ (self.tmp_x0 / denom)

    def select_lambda_lcurve(self, lam_grid, normalize=False, make_plot=False):
        lam_grid = np.asarray(lam_grid, dtype=float)
        lam_grid = np.sort(lam_grid)
        if lam_grid.size == 0:
            raise ValueError("lam_grid is empty.")

        tmp = self.tmp_x0
        eigvals = self.eigvals
        Nt = self.N_t

        bias_log_raw = []
        ef2_log_raw = []

        for lam in lam_grid:
            lam_val = float(lam)
            denom = eigvals + Nt * lam_val

            ratio = (Nt * lam_val) / denom
            resid_sq = float(xp.sum((ratio * tmp) ** 2))

            k_alpha = eigvals * (tmp / denom)
            ef2 = float(xp.sum(k_alpha**2)) / max(Nt, 1)

            bias_log_raw.append(np.log(resid_sq + 1e-30))
            ef2_log_raw.append(np.log(ef2 + 1e-30))

        bias_log_raw = np.asarray(bias_log_raw)
        ef2_log_raw = np.asarray(ef2_log_raw)

        x = bias_log_raw.copy()
        y = ef2_log_raw.copy()

        if normalize:
            x = (x - x.min()) / (x.max() - x.min() + 1e-16)
            y = (y - y.min()) / (y.max() - y.min() + 1e-16)

        p_start = np.array([x[0], y[0]])
        p_end = np.array([x[-1], y[-1]])
        vec_line = p_end - p_start
        line_norm = np.linalg.norm(vec_line) + 1e-12

        dists = []
        for xi, yi in zip(x, y):
            vec_p = np.array([xi, yi]) - p_start
            cross = vec_p[0] * vec_line[1] - vec_p[1] * vec_line[0]
            dists.append(abs(cross) / line_norm)

        dist_curve = np.asarray(dists, dtype=float)
        best_idx = int(np.argmax(dist_curve))
        best_lam = float(lam_grid[best_idx])

        if make_plot:
            fig = plt.figure(figsize=(15, 5))
            ax1 = fig.add_subplot(131)
            ax1.plot(bias_log_raw, ef2_log_raw, "b-", lw=2, alpha=0.7)
            ax1.scatter(bias_log_raw[best_idx], ef2_log_raw[best_idx], s=150, c="r", marker="*", zorder=10)
            ax1.set_xlabel("Log Bias Proxy")
            ax1.set_ylabel(r"Log Empirical $E[\hat f^2]$")
            ax1.grid(True, alpha=0.3)

            ax2 = fig.add_subplot(132)
            ax2.plot(np.log10(lam_grid), dist_curve, "g-", lw=2)
            ax2.axvline(np.log10(best_lam), color="r", ls="--")
            ax2.set_xlabel(r"$\log_{10}(\lambda)$")
            ax2.set_ylabel("Geometric Distance")
            ax2.grid(True, alpha=0.3)

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
            return K_matern52_matrix(X, X_aux, ell) @ alpha_hat

        return f_hat


# ============================================================
# 5) Estimating function (conditional mean)
# ============================================================
def ell_mean(v, theta):
    v = xp.asarray(v).ravel()
    return v - float(theta)


# ============================================================
# 6) PPCI conditional mean + CI + sigma^2 items (clean)
# ============================================================
def ppci_conditional_mean(
    X_l, Y_l, X_u, f_hat, A_l, A_u,
    z_alpha=1.96, return_extras=True, theta_for_var=None
):
    """
    theta_hat = ( E[f_l (Y-A)] + E[f_u A_u] ) / E[f_u]

    Save (theta-aware) sigma^2:
      sigma2_Y_minus_A(theta) = Var( f_l * ( ell(Y;theta) - ell(A;theta) ) )
      sigma2_A(theta)         = Var( f_u * ell(A;theta) )

    NOTE (mean case): ell(Y;theta)-ell(A;theta) = Y-A, theta cancels.
    We compute it directly (no redundant calls), but still store theta_for_var.
    """
    X_l = xp.asarray(X_l)
    Y_l = xp.asarray(Y_l).ravel()
    X_u = xp.asarray(X_u)
    A_l = xp.asarray(A_l).ravel()
    A_u = xp.asarray(A_u).ravel()

    n = int(X_l.shape[0])
    N = int(X_u.shape[0])
    Np = n + N

    f_l = f_hat(X_l).ravel()
    f_u = f_hat(X_u).ravel()

    den = xp.mean(f_u)
    num = xp.mean(f_l * (Y_l - A_l)) + xp.mean(f_u * A_u)
    theta_hat = float(num / (den + 1e-16))

    theta_var = theta_hat if (theta_for_var is None) else float(theta_for_var)

    # ---- sigma^2 items ----
    g_YmA = f_l * (Y_l - A_l)                  # = f*(ell(Y;θ)-ell(A;θ)) for mean
    sigma2_Y_minus_A = float(xp.var(g_YmA, ddof=1)) if n > 1 else 0.0

    g_A = f_u * ell_mean(A_u, theta_var)       # = f*ell(A;θ)
    sigma2_A = float(xp.var(g_A, ddof=1)) if N > 1 else 0.0

    # ---- CI (same as your original influence-form) ----
    psi_R = g_YmA                               # reuse exactly
    psi_U = f_u * (A_u - theta_hat)             # must use theta_hat for linearization

    var_R = float(xp.var(psi_R, ddof=1)) if n > 1 else 0.0
    var_U = float(xp.var(psi_U, ddof=1)) if N > 1 else 0.0

    p_R = n / max(Np, 1)
    p_U = N / max(Np, 1)
    Sigma_hat = (var_R / max(p_R, 1e-12)) + (var_U / max(p_U, 1e-12))

    H_hat = float(-den)
    V_hat = Sigma_hat / ((H_hat + 1e-16) ** 2)
    se = float(np.sqrt(max(V_hat, 0.0) / max(Np, 1)))

    ci = (theta_hat - z_alpha * se, theta_hat + z_alpha * se)

    if not return_extras:
        return theta_hat, se, ci

    extras = {
        "n": n,
        "N": N,
        "N_prime": Np,
        "den": float(den),
        "H_hat": float(H_hat),
        "Sigma_hat": float(Sigma_hat),
        "V_hat": float(V_hat),
        "theta_for_var": float(theta_var),
        "sigma2_Y_minus_A": float(sigma2_Y_minus_A),
        "sigma2_A": float(sigma2_A),
        # optional debug (often helpful)
        "var_R": float(var_R),
        "var_U": float(var_U),
    }
    return theta_hat, se, ci, extras


# ============================================================
# 7) PPCI label-only + CI + sigma^2_Y(theta) (clean)
# ============================================================
def ppci_conditional_mean_label_only(
    X_l, Y_l, f_hat,
    z_alpha=1.96, return_extras=True, theta_for_var=None
):
    """
    Label-only estimator (same as before):
      w_i ∝ f_l(X_i), theta_hat = sum w_i Y_i

    Save (theta-aware):
      sigma2_Y(theta) = Var( f_l * ell(Y;theta) )
    """
    X_l = xp.asarray(X_l)
    Y_l = xp.asarray(Y_l).ravel()
    n = int(Y_l.size)

    f_l = f_hat(X_l).ravel()

    denom = float(xp.sum(f_l))
    if abs(denom) <= 1e-16:
        theta_hat = float(xp.mean(Y_l))
        var_hat = float(xp.var(Y_l, ddof=1)) / max(n, 1) if n > 1 else 0.0
        var_hat = max(var_hat, 0.0)
        se = float(np.sqrt(var_hat))
        ci = (theta_hat - z_alpha * se, theta_hat + z_alpha * se)

        theta_var = theta_hat if (theta_for_var is None) else float(theta_for_var)
        gY = f_l * ell_mean(Y_l, theta_var)
        sigma2_Y = float(xp.var(gY, ddof=1)) if n > 1 else 0.0

        if not return_extras:
            return theta_hat, se, ci
        return theta_hat, se, ci, {
            "n": n,
            "denom_sumf": float(denom),
            "theta_for_var": float(theta_var),
            "sigma2_Y": float(sigma2_Y),
            "var_weighted": float(var_hat),
        }

    w = f_l / (denom + 1e-16)
    theta_hat = float(w @ Y_l)

    var_hat = float(xp.sum((w**2) * (Y_l - theta_hat) ** 2))
    var_hat = max(var_hat, 0.0)
    se = float(np.sqrt(var_hat))
    ci = (theta_hat - z_alpha * se, theta_hat + z_alpha * se)

    theta_var = theta_hat if (theta_for_var is None) else float(theta_for_var)
    gY = f_l * ell_mean(Y_l, theta_var)
    sigma2_Y = float(xp.var(gY, ddof=1)) if n > 1 else 0.0

    if not return_extras:
        return theta_hat, se, ci

    extras = {
        "n": n,
        "denom_sumf": float(denom),
        "theta_for_var": float(theta_var),
        "sigma2_Y": float(sigma2_Y),
        "var_weighted": float(var_hat),
    }
    return theta_hat, se, ci, extras
