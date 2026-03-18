from __future__ import annotations
import torch
from .grid import EdgeField, Grid3D


def _pad_replicate_3d(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty((a.shape[0] + 2, a.shape[1] + 2, a.shape[2] + 2), device=a.device, dtype=a.dtype)
    out[1:-1, 1:-1, 1:-1] = a
    out[0, 1:-1, 1:-1] = a[0]
    out[-1, 1:-1, 1:-1] = a[-1]
    out[1:-1, 0, 1:-1] = a[:, 0]
    out[1:-1, -1, 1:-1] = a[:, -1]
    out[1:-1, 1:-1, 0] = a[:, :, 0]
    out[1:-1, 1:-1, -1] = a[:, :, -1]
    out[0, 0, 1:-1] = a[0, 0]
    out[0, -1, 1:-1] = a[0, -1]
    out[-1, 0, 1:-1] = a[-1, 0]
    out[-1, -1, 1:-1] = a[-1, -1]
    out[0, 1:-1, 0] = a[0, :, 0]
    out[0, 1:-1, -1] = a[0, :, -1]
    out[-1, 1:-1, 0] = a[-1, :, 0]
    out[-1, 1:-1, -1] = a[-1, :, -1]
    out[1:-1, 0, 0] = a[:, 0, 0]
    out[1:-1, 0, -1] = a[:, 0, -1]
    out[1:-1, -1, 0] = a[:, -1, 0]
    out[1:-1, -1, -1] = a[:, -1, -1]
    out[0, 0, 0] = a[0, 0, 0]
    out[0, 0, -1] = a[0, 0, -1]
    out[0, -1, 0] = a[0, -1, 0]
    out[0, -1, -1] = a[0, -1, -1]
    out[-1, 0, 0] = a[-1, 0, 0]
    out[-1, 0, -1] = a[-1, 0, -1]
    out[-1, -1, 0] = a[-1, -1, 0]
    out[-1, -1, -1] = a[-1, -1, -1]
    return out


def sigma_to_edges(sigma: torch.Tensor, grid: Grid3D) -> EdgeField:
    s = _pad_replicate_3d(sigma)
    ex = 0.25 * (s[1:-1, 1:, 1:] + s[1:-1, :-1, 1:] + s[1:-1, 1:, :-1] + s[1:-1, :-1, :-1])
    ey = 0.25 * (s[1:, 1:-1, 1:] + s[:-1, 1:-1, 1:] + s[1:, 1:-1, :-1] + s[:-1, 1:-1, :-1])
    ez = 0.25 * (s[1:, 1:, 1:-1] + s[:-1, 1:, 1:-1] + s[1:, :-1, 1:-1] + s[:-1, :-1, 1:-1])
    return EdgeField(ex.to(grid.dtype), ey.to(grid.dtype), ez.to(grid.dtype))


def edge_energy_to_cells(ex_term: torch.Tensor, ey_term: torch.Tensor, ez_term: torch.Tensor) -> torch.Tensor:
    out = torch.zeros((ex_term.shape[0], ey_term.shape[1], ez_term.shape[2]), device=ex_term.device, dtype=ex_term.dtype)
    out += 0.25 * (ex_term[:, :-1, :-1] + ex_term[:, 1:, :-1] + ex_term[:, :-1, 1:] + ex_term[:, 1:, 1:])
    out += 0.25 * (ey_term[:-1, :, :-1] + ey_term[1:, :, :-1] + ey_term[:-1, :, 1:] + ey_term[1:, :, 1:])
    out += 0.25 * (ez_term[:-1, :-1, :] + ez_term[1:, :-1, :] + ez_term[:-1, 1:, :] + ez_term[1:, 1:, :])
    return out
