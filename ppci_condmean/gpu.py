from __future__ import annotations
import os
import shutil
import subprocess


def cupy_available() -> bool:
    """Return True only if CuPy can actually execute a tiny GPU kernel."""
    try:
        import cupy as cp
        x = cp.array([1.0, 2.0])
        _ = (x * x).get()
        return True
    except Exception:
        return False


def torch_cuda_available() -> bool:
    """Return True only if PyTorch CUDA is available and can execute a tiny op."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        x = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float64)
        _ = (x * x).cpu().numpy()
        return True
    except Exception:
        return False


def select_idle_gpu() -> int | None:
    """Return the GPU id with the largest free memory, or None if nvidia-smi is unavailable."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        best = None
        best_free = -1
        for line in out.strip().splitlines():
            if not line.strip():
                continue
            idx_s, free_s = [x.strip() for x in line.split(",")[:2]]
            free = int(float(free_s))
            idx = int(idx_s)
            if free > best_free:
                best_free = free
                best = idx
        return best
    except Exception:
        return None


def configure_backend(backend: str = "auto", gpu_id: str | int | None = "auto") -> str:
    """Configure CUDA_VISIBLE_DEVICES if requested and return 'cpu', 'cupy', or 'torch'.

    The previous code only checked whether CuPy could be imported.  Some environments
    can import CuPy but cannot run kernels because NVRTC is missing.  Here we run a
    tiny GPU operation.  If CuPy is unavailable but PyTorch CUDA works, we use the
    PyTorch GPU backend.  This is useful on clusters where PyTorch ships with its own
    CUDA runtime.
    """
    backend = str(backend).lower()
    if backend == "cpu":
        return "cpu"
    if backend not in {"auto", "gpu", "cuda", "cupy", "torch"}:
        raise ValueError("backend must be 'auto', 'cpu', 'gpu', 'cupy', or 'torch'.")

    if gpu_id == "auto":
        gid = select_idle_gpu()
    elif gpu_id is None:
        gid = None
    else:
        gid = int(gpu_id)
    if gid is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gid))

    if backend == "cupy":
        if cupy_available():
            return "cupy"
        print("[warning] CuPy backend requested but CuPy/CUDA runtime test failed; falling back to CPU.")
        return "cpu"

    if backend == "torch":
        if torch_cuda_available():
            return "torch"
        print("[warning] Torch backend requested but PyTorch CUDA test failed; falling back to CPU.")
        return "cpu"

    # auto/gpu/cuda: prefer torch if available because many ML environments already
    # have a working PyTorch CUDA install; otherwise use CuPy; otherwise CPU.
    if torch_cuda_available():
        return "torch"
    if cupy_available():
        return "cupy"
    if backend in {"gpu", "cuda"}:
        print("[warning] GPU backend requested but neither PyTorch CUDA nor CuPy is usable; falling back to CPU.")
    return "cpu"
