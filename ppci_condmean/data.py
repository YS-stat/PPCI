from __future__ import annotations
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from .utils import standardize_fit, standardize_apply


def m_true_simulation(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return (
        np.sin(2 * np.pi * x1)
        + 0.6 * np.cos(2 * np.pi * x2)
        + 0.4 * np.sin(2 * np.pi * x3)
        + 0.25 * np.sin(2 * np.pi * (x1 + x2))
    )


def stress_signal_simulation(X: np.ndarray) -> np.ndarray:
    """A mean-zero signal orthogonal to ``m_true_simulation`` under Unif([0,1]^3)."""
    X = np.asarray(X, dtype=float)
    return np.sqrt(1.5825) * np.sin(6.0 * np.pi * X[:, 0])


def simulation_predictor(X: np.ndarray, quality: float = 0.9) -> np.ndarray:
    """Fixed deployable predictor family used in the synthetic experiments.

    ``quality=1`` gives the oracle regression function. As quality decreases, the
    predictor moves toward an orthogonal signal with the same marginal variance.
    The returned value is a deterministic function of ``X`` and never uses ``Y``.
    """
    quality = float(quality)
    if not 0.0 <= quality <= 1.0:
        raise ValueError("quality must lie in [0, 1].")
    X = np.asarray(X, dtype=float)
    return quality * m_true_simulation(X) + (1.0 - quality) * stress_signal_simulation(X)


def standardize_unif01(X: np.ndarray) -> np.ndarray:
    return (np.asarray(X, dtype=float) - 0.5) / (1.0 / np.sqrt(12.0))


def generate_simulation_labeled(
    rng: np.random.Generator,
    n: int,
    sigma_eps: float = 1.0,
    predictor_quality: float = 0.9,
):
    X_raw = rng.uniform(0.0, 1.0, size=(n, 3))
    m = m_true_simulation(X_raw)
    eps = rng.normal(0.0, sigma_eps, size=n)
    Y = m + eps
    f = simulation_predictor(X_raw, predictor_quality)
    return X_raw, standardize_unif01(X_raw), Y, f


def generate_simulation_unlabeled(
    rng: np.random.Generator,
    N: int,
    predictor_quality: float = 0.9,
):
    X_raw = rng.uniform(0.0, 1.0, size=(N, 3))
    f = simulation_predictor(X_raw, predictor_quality)
    return X_raw, standardize_unif01(X_raw), f


def load_census_npz(path: str | Path):
    z = np.load(path, allow_pickle=True)
    X = np.asarray(z["X"], dtype=float)
    Y = np.asarray(z["Y"], dtype=float) / 10000.0
    f = np.asarray(z["Yhat"], dtype=float) / 10000.0
    return X, Y, f


def census_sex_subset(X: np.ndarray, Y: np.ndarray, f: np.ndarray, sex: int):
    mask = X[:, 1] == float(sex)
    Xs = X[mask]
    Ys = Y[mask]
    fs = f[mask]
    mean, std = standardize_fit(Xs)
    Xstd = standardize_apply(Xs, mean, std)
    return Xs, Xstd, Ys, fs, mean, std


def nw_oracle_mean(X: np.ndarray, Y: np.ndarray, x0: np.ndarray, h: float | None = None, kernel: str = "matern52") -> float:
    from .kernels import get_kernel
    X = np.asarray(X, dtype=float)
    x0 = np.asarray(x0, dtype=float).reshape(1, -1)
    if h is None:
        d = np.linalg.norm(X - x0, axis=1)
        h = max(float(np.median(d)), 1e-8)
    k = get_kernel(kernel)(X, x0, h).ravel()
    den = float(np.sum(k))
    if den <= 1e-14:
        idx = int(np.argmin(np.linalg.norm(X - x0, axis=1)))
        return float(Y[idx])
    return float(np.sum(k * Y) / den)


def load_blogfeedback_raw(zip_path: str | Path, csv_name: str = "blogData_train.csv", nrows: int | None = None):
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(csv_name) as f:
            df_raw = pd.read_csv(f, header=None, nrows=nrows)
    df = df_raw.drop_duplicates()
    X_all = df.iloc[:, :-1]
    mask_unique = ~X_all.duplicated(keep=False)
    df = df[mask_unique].reset_index(drop=True)
    X_raw = df.iloc[:, :-1].to_numpy(dtype=float)
    Y_raw = df.iloc[:, -1].to_numpy(dtype=float)
    Y = np.log1p(Y_raw)
    mean, std = standardize_fit(X_raw)
    X = standardize_apply(X_raw, mean, std)
    return X, Y, mean, std


def prepare_blogfeedback_ppci(
    zip_path: str | Path,
    seed: int = 2025,
    n_x0: int = 10,
    ppci_fraction: float = 0.3,
    max_train: int = 0,
    model: str = "lightgbm",
    model_n_jobs: int = 1,
    include_x0_in_model_train: bool = False,
    max_raw_rows: int | None = None,
):
    """Prepare BlogFeedback data with predictor/target/PPCI outcome separation."""
    X, Y, mean, std = load_blogfeedback_raw(zip_path, nrows=max_raw_rows)
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(Y))
    idx_x0 = rng.choice(idx_all, size=min(n_x0, len(Y) // 10), replace=False)
    mask = np.ones(len(Y), dtype=bool)
    mask[idx_x0] = False
    idx_rem = idx_all[mask]
    X_rem, Y_rem = X[idx_rem], Y[idx_rem]
    X_train, X_ppci, Y_train, Y_ppci = train_test_split(
        X_rem,
        Y_rem,
        test_size=ppci_fraction,
        random_state=seed,
    )
    if include_x0_in_model_train:
        X_model = np.vstack([X[idx_x0], X_train])
        Y_model = np.concatenate([Y[idx_x0], Y_train])
    else:
        X_model, Y_model = X_train, Y_train
    if max_train and max_train > 0 and X_model.shape[0] > max_train:
        sub = rng.choice(np.arange(X_model.shape[0]), size=max_train, replace=False)
        X_train_fit, Y_train_fit = X_model[sub], Y_model[sub]
    else:
        X_train_fit, Y_train_fit = X_model, Y_model
    model_key = str(model).lower()
    if model_key in {"lightgbm", "lgbm"}:
        try:
            import lightgbm as lgb
        except Exception as exc:
            raise RuntimeError("model='lightgbm' requires the lightgbm package") from exc
        reg = lgb.LGBMRegressor(
            objective="regression",
            boosting_type="gbdt",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=int(model_n_jobs),
            verbosity=-1,
        )
    elif model_key == "extratrees":
        reg = ExtraTreesRegressor(
            n_estimators=100,
            max_features="sqrt",
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=int(model_n_jobs),
        )
    else:
        reg = Ridge(alpha=1.0)
    reg.fit(X_train_fit, Y_train_fit)
    f_ppci = reg.predict(X_ppci)
    x0 = X[idx_x0]
    # The repeated labelled/unlabelled samples are drawn from the held-out
    # inference population, so its complete outcomes define the empirical
    # conditional target.  The predictor-training outcomes remain disjoint.
    theta0 = np.array([nw_oracle_mean(X_ppci, Y_ppci, z, kernel="matern52") for z in x0])
    theta0_full = np.array([nw_oracle_mean(X, Y, z, kernel="matern52") for z in x0])
    return {
        "X_ppci": X_ppci,
        "Y_ppci": Y_ppci,
        "f_ppci": f_ppci,
        "x0": x0,
        "theta0": theta0,
        "theta0_full_data_sensitivity": theta0_full,
        "idx_x0": idx_x0,
        "X_all": X,
        "Y_all": Y,
        "model": reg,
        "mean": mean,
        "std": std,
        "n_model_train": int(X_train_fit.shape[0]),
        "n_ppci_pool": int(X_ppci.shape[0]),
        "targets_excluded_from_model": bool(not include_x0_in_model_train),
        "reference_population": "heldout_ppci_pool",
    }
