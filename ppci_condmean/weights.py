from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh
from .kernels import get_kernel


def _cupy_or_none():
    try:
        import cupy as cp
        return cp
    except Exception:
        return None


def _torch_or_none():
    try:
        import torch
        return torch
    except Exception:
        return None


@dataclass
class WeightDiagnostics:
    h: float
    lam: float
    M: int
    eff_dim: float
    point_leverage: float
    op_score: float
    loc_score: float
    status: str = ""


def _cupy_kernel(X, Z, h: float, kernel: str):
    cp = _cupy_or_none()
    if cp is None:
        raise RuntimeError("CuPy is not available")
    X = cp.asarray(X, dtype=cp.float64)
    Z = cp.asarray(Z, dtype=cp.float64)
    h = float(max(h, 1e-12))
    X2 = cp.sum(X * X, axis=1, keepdims=True)
    Z2 = cp.sum(Z * Z, axis=1, keepdims=True).T
    d2 = cp.maximum(X2 + Z2 - 2.0 * (X @ Z.T), 0.0)
    name = str(kernel).lower()
    if name in {"gaussian", "rbf"}:
        return cp.exp(-0.5 * d2 / (h * h))
    r = cp.sqrt(d2 + 1e-18)
    t = np.sqrt(5.0) * r / h
    return (1.0 + t + (t * t) / 3.0) * cp.exp(-t)


def _torch_kernel(X, Z, h: float, kernel: str):
    torch = _torch_or_none()
    if torch is None:
        raise RuntimeError("PyTorch is not available")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.is_tensor(X):
        X = torch.as_tensor(X, dtype=torch.float64, device=device)
    else:
        X = X.to(device=device, dtype=torch.float64)
    if not torch.is_tensor(Z):
        Z = torch.as_tensor(Z, dtype=torch.float64, device=device)
    else:
        Z = Z.to(device=device, dtype=torch.float64)
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


class RKHSLocalizationWeight:
    """Empirical RKHS localization weight.

    w_hat(x) = {K_h(x,x0) - K_h(x,S)^T (Sigma + M lambda I)^{-1} k0} / lambda.
    On the training covariates, w_hat(S) = M * xi.
    """

    def __init__(self, Z: np.ndarray, x0: np.ndarray, h: float, lam: float, kernel: str = "matern52", backend: str = "cpu"):
        requested = str(backend).lower()
        self.Z_cpu = np.asarray(Z, dtype=float)
        self.x0_cpu = np.asarray(x0, dtype=float).reshape(1, -1)
        self.h = float(h)
        self.lam = float(lam)
        self.kernel_name = kernel
        self.M = int(self.Z_cpu.shape[0])
        if self.M < 1:
            raise ValueError("Need at least one covariate to build a weight.")

        # backend can be 'cpu', 'cupy', or 'torch'.  For backward compatibility, 'gpu'
        # means torch if available, then cupy, otherwise cpu.  configure_backend() should
        # already have done this choice, but the class is robust on its own.
        self.backend = "cpu"
        if requested in {"torch", "gpu", "cuda"}:
            torch = _torch_or_none()
            if torch is not None and torch.cuda.is_available():
                self.backend = "torch"
        if self.backend == "cpu" and requested in {"cupy", "gpu", "cuda"}:
            cp = _cupy_or_none()
            if cp is not None:
                self.backend = "cupy"

        if self.backend == "torch":
            torch = _torch_or_none()
            self.device = torch.device("cuda")
            with torch.no_grad():
                self.Z = torch.as_tensor(self.Z_cpu, dtype=torch.float64, device=self.device)
                self.x0 = torch.as_tensor(self.x0_cpu, dtype=torch.float64, device=self.device)
                Sigma = _torch_kernel(self.Z, self.Z, self.h, kernel)
                Sigma = 0.5 * (Sigma + Sigma.T)
                k0 = _torch_kernel(self.Z, self.x0, self.h, kernel).reshape(-1)
                eigvals, eigvecs = torch.linalg.eigh(Sigma)
                eigvals = torch.clamp(eigvals, min=0.0)
                tmp = eigvecs.T @ k0
                self.xi = eigvecs @ (tmp / (eigvals + self.M * self.lam))
                self.eigvals = eigvals
                self.eigvecs = eigvecs
                self._tmp = tmp
                self.k0 = k0
        elif self.backend == "cupy":
            cp = _cupy_or_none()
            self.Z = cp.asarray(self.Z_cpu)
            self.x0 = cp.asarray(self.x0_cpu)
            Sigma = _cupy_kernel(self.Z, self.Z, self.h, kernel)
            Sigma = 0.5 * (Sigma + Sigma.T)
            k0 = _cupy_kernel(self.Z, self.x0, self.h, kernel).ravel()
            eigvals, eigvecs = cp.linalg.eigh(Sigma)
            eigvals = cp.maximum(eigvals, 0.0)
            tmp = eigvecs.T @ k0
            self.xi = eigvecs @ (tmp / (eigvals + self.M * self.lam))
            self.eigvals = eigvals
            self.eigvecs = eigvecs
            self._tmp = tmp
            self.k0 = k0
        else:
            self.Z = self.Z_cpu
            self.x0 = self.x0_cpu
            self.kernel = get_kernel(kernel)
            Sigma = self.kernel(self.Z, self.Z, self.h)
            Sigma = 0.5 * (Sigma + Sigma.T)
            k0 = self.kernel(self.Z, self.x0, self.h).ravel()
            eigvals, eigvecs = eigh(Sigma, check_finite=False)
            eigvals = np.maximum(eigvals, 0.0)
            tmp = eigvecs.T @ k0
            self.xi = eigvecs @ (tmp / (eigvals + self.M * self.lam))
            self.eigvals = eigvals
            self.eigvecs = eigvecs
            self._tmp = tmp
            self.k0 = k0

    def __call__(self, X_query: np.ndarray) -> np.ndarray:
        X_query = np.asarray(X_query, dtype=float)
        if X_query.ndim == 1:
            X_query = X_query.reshape(1, -1)
        if self.backend == "torch":
            torch = _torch_or_none()
            with torch.no_grad():
                Xg = torch.as_tensor(X_query, dtype=torch.float64, device=self.device)
                k_x0 = _torch_kernel(Xg, self.x0, self.h, self.kernel_name).reshape(-1)
                K_xS = _torch_kernel(Xg, self.Z, self.h, self.kernel_name)
                out = (k_x0 - K_xS @ self.xi) / self.lam
                return out.detach().cpu().numpy()
        if self.backend == "cupy":
            cp = _cupy_or_none()
            Xg = cp.asarray(X_query)
            k_x0 = _cupy_kernel(Xg, self.x0, self.h, self.kernel_name).ravel()
            K_xS = _cupy_kernel(Xg, self.Z, self.h, self.kernel_name)
            out = (k_x0 - K_xS @ self.xi) / self.lam
            return cp.asnumpy(out)
        k_x0 = self.kernel(X_query, self.x0, self.h).ravel()
        K_xS = self.kernel(X_query, self.Z, self.h)
        return (k_x0 - K_xS @ self.xi) / self.lam

    def training_values(self) -> np.ndarray:
        if self.backend == "torch":
            return (self.M * self.xi).detach().cpu().numpy()
        if self.backend == "cupy":
            cp = _cupy_or_none()
            return cp.asnumpy(self.M * self.xi)
        return self.M * self.xi

    def effective_dimension(self, lam: float | None = None) -> float:
        lam = self.lam if lam is None else float(lam)
        if self.backend == "torch":
            import torch
            with torch.no_grad():
                return float(torch.sum(self.eigvals / (self.eigvals + self.M * lam)).detach().cpu().item())
        if self.backend == "cupy":
            cp = _cupy_or_none()
            return float(cp.asnumpy(cp.sum(self.eigvals / (self.eigvals + self.M * lam))))
        return float(np.sum(self.eigvals / (self.eigvals + self.M * lam)))

    def point_leverage(self, lam: float | None = None) -> float:
        lam = self.lam if lam is None else float(lam)
        if self.backend == "torch":
            import torch
            with torch.no_grad():
                k00 = float(_torch_kernel(self.x0, self.x0, self.h, self.kernel_name)[0, 0].detach().cpu().item())
                val = (k00 - float(torch.sum((self._tmp * self._tmp) / (self.eigvals + self.M * lam)).detach().cpu().item())) / lam
        elif self.backend == "cupy":
            cp = _cupy_or_none()
            k00 = float(cp.asnumpy(_cupy_kernel(self.x0, self.x0, self.h, self.kernel_name)[0, 0]))
            val = (k00 - float(cp.asnumpy(cp.sum((self._tmp * self._tmp) / (self.eigvals + self.M * lam))))) / lam
        else:
            k00 = float(self.kernel(self.x0, self.x0, self.h)[0, 0])
            val = (k00 - float(np.sum((self._tmp * self._tmp) / (self.eigvals + self.M * lam)))) / lam
        return float(max(val, 0.0))


def spectral_diagnostics(Z: np.ndarray, x0: np.ndarray, h: float, kernel: str = "matern52", backend: str = "cpu"):
    requested = str(backend).lower()
    if requested in {"torch", "gpu", "cuda"}:
        torch = _torch_or_none()
        if torch is not None and torch.cuda.is_available():
            with torch.no_grad():
                device = torch.device("cuda")
                Zg = torch.as_tensor(np.asarray(Z, dtype=float), dtype=torch.float64, device=device)
                x0g = torch.as_tensor(np.asarray(x0, dtype=float).reshape(1, -1), dtype=torch.float64, device=device)
                Sigma = _torch_kernel(Zg, Zg, h, kernel)
                Sigma = 0.5 * (Sigma + Sigma.T)
                k0 = _torch_kernel(Zg, x0g, h, kernel).reshape(-1)
                eigvals, eigvecs = torch.linalg.eigh(Sigma)
                eigvals = torch.clamp(eigvals, min=0.0)
                tmp = eigvecs.T @ k0
                k00 = float(_torch_kernel(x0g, x0g, h, kernel)[0, 0].detach().cpu().item())
                return eigvals.detach().cpu().numpy(), tmp.detach().cpu().numpy(), k00
    if requested in {"cupy", "gpu", "cuda"}:
        cp = _cupy_or_none()
        if cp is not None:
            Zg = cp.asarray(np.asarray(Z, dtype=float))
            x0g = cp.asarray(np.asarray(x0, dtype=float).reshape(1, -1))
            Sigma = _cupy_kernel(Zg, Zg, h, kernel)
            Sigma = 0.5 * (Sigma + Sigma.T)
            k0 = _cupy_kernel(Zg, x0g, h, kernel).ravel()
            eigvals, eigvecs = cp.linalg.eigh(Sigma)
            eigvals = cp.maximum(eigvals, 0.0)
            tmp = eigvecs.T @ k0
            k00 = float(cp.asnumpy(_cupy_kernel(x0g, x0g, h, kernel)[0, 0]))
            return cp.asnumpy(eigvals), cp.asnumpy(tmp), k00

    kernel_fn = get_kernel(kernel)
    Z = np.asarray(Z, dtype=float)
    x0 = np.asarray(x0, dtype=float).reshape(1, -1)
    Sigma = kernel_fn(Z, Z, h)
    Sigma = 0.5 * (Sigma + Sigma.T)
    k0 = kernel_fn(Z, x0, h).ravel()
    eigvals, eigvecs = eigh(Sigma, check_finite=False)
    eigvals = np.maximum(eigvals, 0.0)
    tmp = eigvecs.T @ k0
    k00 = float(kernel_fn(x0, x0, h)[0, 0])
    return eigvals, tmp, k00


def effective_dimension_from_spectrum(eigvals: np.ndarray, M: int, lam: float) -> float:
    eigvals = np.asarray(eigvals, dtype=float)
    return float(np.sum(eigvals / (eigvals + M * float(lam))))


def point_leverage_from_spectrum(eigvals: np.ndarray, tmp: np.ndarray, k00: float, M: int, lam: float) -> float:
    eigvals = np.asarray(eigvals, dtype=float)
    tmp = np.asarray(tmp, dtype=float)
    lam = float(lam)
    val = (float(k00) - float(np.sum((tmp * tmp) / (eigvals + M * lam)))) / lam
    return float(max(val, 0.0))
