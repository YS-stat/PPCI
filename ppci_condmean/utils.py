from __future__ import annotations
import json
import hashlib
import platform
import sys
from datetime import datetime, timezone
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd


def source_sha256(root: str | Path | None = None) -> tuple[str, int]:
    """Fingerprint the release's executable and environment-specification files."""
    release_root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    files = sorted(release_root.rglob("*.py"))
    files.extend(
        path
        for name in ("pyproject.toml", "requirements.txt")
        if (path := release_root / name).is_file()
    )
    digest = hashlib.sha256()
    for path in sorted(files):
        relative = path.relative_to(release_root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), len(files)


def standardize_fit(X: np.ndarray):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return mean, std


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(X, dtype=float) - mean) / std


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: str | Path):
    def default(o):
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=default)


def write_run_manifest(path: str | Path, args, extra: dict | None = None) -> None:
    """Write a compact, deterministic record of a command-line experiment run."""
    arg_dict = dict(vars(args)) if hasattr(args, "__dict__") else dict(args)
    canonical = json.dumps(arg_dict, sort_keys=True, default=str, separators=(",", ":"))
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except Exception:
        pass
    source_hash, source_file_count = source_sha256()
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
        "arguments": arg_dict,
        "config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "source_sha256": source_hash,
        "source_file_count": source_file_count,
        "runtime": versions,
    }
    if extra:
        payload["extra"] = extra
    save_json(payload, path)


def write_csv(rows: list[dict], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    pd.DataFrame(rows).to_csv(path, index=False)


def summarize_replicates(df: pd.DataFrame, group_cols: list[str], theta0_col: str = "theta0") -> pd.DataFrame:
    methods = sorted(df["method"].unique())
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))
        theta0 = float(g[theta0_col].iloc[0])
        base["theta0"] = theta0
        for m in methods:
            gm = g[g["method"] == m]
            if gm.empty:
                continue
            err = gm["theta_hat"].to_numpy() - theta0
            theta = gm["theta_hat"].to_numpy()
            emp_sd = float(np.std(theta, ddof=1)) if theta.size > 1 else 0.0
            se_mean = float(gm["se"].mean())
            base[f"{m}_theta_mean"] = float(gm["theta_hat"].mean())
            base[f"{m}_bias"] = float(err.mean())
            base[f"{m}_rmse"] = float(np.sqrt(np.mean(err * err)))
            base[f"{m}_se_mean"] = se_mean
            base[f"{m}_emp_sd"] = emp_sd
            base[f"{m}_emp_sd_over_se_mean"] = float(emp_sd / se_mean) if se_mean > 0 else np.nan
            base[f"{m}_coverage"] = float(((gm["ci_low"] <= theta0) & (theta0 <= gm["ci_high"])).mean())
            base[f"{m}_width"] = float((gm["ci_high"] - gm["ci_low"]).mean())
            for col in ["h", "lambda", "lambda_value", "h_factor", "lambda_factor", "op_score", "loc_score", "h_mode", "lambda_selection", "h_mean", "lambda_mean", "lambda_1", "lambda_2", "h_1", "h_2", "h_factor_1", "h_factor_2", "lambda_factor_1", "lambda_factor_2", "ess0_1", "ess0_2", "op_score_1", "op_score_2", "loc_score_1", "loc_score_2", "tuning_status"]:
                if col in gm.columns:
                    vals = gm[col]
                    if vals.dtype.kind in "fiu":
                        base[f"{m}_{col}_mean"] = float(vals.mean())
                    else:
                        base[f"{m}_{col}_mode"] = str(vals.mode().iloc[0]) if not vals.mode().empty else ""
        rows.append(base)
    return pd.DataFrame(rows)
