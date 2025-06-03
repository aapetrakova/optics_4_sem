
"""
compute.py — быстрый расчёт «ковра Талбота» (FFT + Numba)

CPU: NumPy-FFT, ускоряется Numba.
GPU: torch.fft (ROCm/NVIDIA) включается чекбоксом GPU (torch).
"""

from __future__ import annotations
import numpy as np

# optional numba
try:
    import numba as nb
    NUMBA = True
except ImportError:
    NUMBA = False

# optional torch
try:
    import torch
    TORCH_OK = torch.cuda.is_available()
except ImportError:
    TORCH_OK = False


def _mask_1d(x: np.ndarray, a: float, duty: float, nslits: int):
    half = duty * a / 2
    m = np.zeros_like(x, dtype=bool)
    for i in range(nslits):
        c = (i - nslits // 2) * a
        m |= (x >= c - half) & (x <= c + half)
    return m.astype(np.float32)


def _cpu_fft(a, wl, duty, nslits, xmin, xmax, res, z_rel):
    nx = int((xmax - xmin) * res) + 1
    dx = (xmax - xmin) / (nx - 1)
    x = np.linspace(xmin, xmax, nx)
    mask = _mask_1d(x, a, duty, nslits)

    A_k = np.fft.fft(mask)
    k = np.fft.fftfreq(nx, d=dx)
    k2 = k * k

    z_T = 2 * a * a / wl
    out = np.empty((z_rel.size, nx), dtype=np.float32)
    for i, alpha in enumerate(z_rel):
        z = alpha * z_T
        H = np.exp(-1j * np.pi * wl * z * k2)
        E = np.fft.ifft(A_k * H)
        out[i] = np.abs(E) ** 2
    return out

if NUMBA:
    _cpu_fft = nb.njit(_cpu_fft, fastmath=True, cache=True)


def _gpu_fft(a, wl, duty, nslits, xmin, xmax, res, z_rel):
    import torch
    nx = int((xmax - xmin) * res) + 1
    dx = (xmax - xmin) / (nx - 1)
    dev = torch.device("cuda")
    x = torch.linspace(xmin, xmax, nx, device=dev)
    half = duty * a / 2
    m = torch.zeros_like(x, dtype=torch.float32)
    for i in range(nslits):
        c = (i - nslits // 2) * a
        m |= ((x >= c - half) & (x <= c + half)).float()

    A_k = torch.fft.fft(m)
    k = torch.fft.fftfreq(nx, d=dx, device=dev)
    k2 = k * k
    z_T = 2 * a * a / wl
    z = torch.tensor(z_rel, device=dev).view(-1, 1) * z_T
    H = torch.exp(-1j * np.pi * wl * z * k2)
    E = torch.fft.ifft(A_k * H)
    return torch.abs(E) ** 2.0.cpu().numpy()


def talbot_carpet(*, a, wavelength, duty, nslits, x_min, x_max, z_max, res, use_gpu):
    z_rel = np.linspace(0.01, z_max, int(z_max * res) + 1)
    if use_gpu and TORCH_OK:
        return _gpu_fft(a, wavelength, duty, nslits, x_min, x_max, res, z_rel)
    return _cpu_fft(a, wavelength, duty, nslits, x_min, x_max, res, z_rel)
