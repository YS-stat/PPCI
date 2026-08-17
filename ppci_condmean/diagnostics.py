from __future__ import annotations

import numpy as np

from .kernels import get_kernel


def nw_closeness_diagnostics(weight) -> dict[str, float]:
    """Compare a fitted RKHS localization training weight with its NW kernel vector.

    These quantities are descriptive only and must not be used to select tuning
    parameters. The relative difference removes the best scalar multiple of the
    kernel vector, matching the fact that both estimators self-normalize weights.
    """
    w = np.asarray(weight.training_values(), dtype=float).ravel()
    z = np.asarray(weight.Z_cpu, dtype=float)
    x0 = np.asarray(weight.x0_cpu, dtype=float).reshape(1, -1)
    kernel = get_kernel(weight.kernel_name)
    k = np.asarray(kernel(z, x0, float(weight.h)), dtype=float).ravel()
    if w.size < 2 or np.std(w) <= 0.0 or np.std(k) <= 0.0:
        corr = np.nan
    else:
        corr = float(np.corrcoef(w, k)[0, 1])
    alpha = float(np.dot(w, k) / max(float(np.dot(k, k)), 1e-16))
    rel_diff = float(np.linalg.norm(w - alpha * k) / max(float(np.linalg.norm(w)), 1e-16))
    if hasattr(weight, "spec"):
        eigvals = np.asarray(weight.spec.eigvals, dtype=float)
    else:
        eigvals = np.asarray(weight.eigvals, dtype=float)
    eigmax = float(np.max(eigvals)) if eigvals.size else np.nan
    spectral_ratio = float(weight.M * weight.lam / eigmax) if eigmax > 0.0 else np.inf
    return {
        "nw_corr": corr,
        "nw_relative_difference": rel_diff,
        "negative_weight_fraction": float(np.mean(w < 0.0)),
        "M_lambda_over_eigmax": spectral_ratio,
        "lambda_trace_lower_bound": float(weight.lam),
    }


def average_nw_closeness(*weights) -> dict[str, float]:
    diagnostics = [nw_closeness_diagnostics(weight) for weight in weights]
    return {
        key: float(np.nanmean([item[key] for item in diagnostics]))
        for key in diagnostics[0]
    }
