from __future__ import annotations
import numpy as np


def pairwise_sq_dists(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    X2 = np.sum(X * X, axis=1, keepdims=True)
    Z2 = np.sum(Z * Z, axis=1, keepdims=True).T
    d2 = X2 + Z2 - 2.0 * X @ Z.T
    return np.maximum(d2, 0.0)


def matern52_kernel(X: np.ndarray, Z: np.ndarray, h: float) -> np.ndarray:
    """Matérn 5/2 kernel, K_h(x,z)=(1+t+t^2/3) exp(-t), t=sqrt(5)||x-z||/h."""
    h = float(max(h, 1e-12))
    r = np.sqrt(pairwise_sq_dists(X, Z) + 1e-18)
    t = np.sqrt(5.0) * r / h
    return (1.0 + t + (t * t) / 3.0) * np.exp(-t)


def gaussian_kernel(X: np.ndarray, Z: np.ndarray, h: float) -> np.ndarray:
    h = float(max(h, 1e-12))
    return np.exp(-0.5 * pairwise_sq_dists(X, Z) / (h * h))


def get_kernel(name: str):
    name = str(name).lower()
    if name in {"matern52", "matern5/2", "matern_52"}:
        return matern52_kernel
    if name in {"gaussian", "rbf"}:
        return gaussian_kernel
    raise ValueError(f"Unknown kernel '{name}'. Use 'matern52' or 'gaussian'.")
