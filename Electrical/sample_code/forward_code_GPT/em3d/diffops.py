from __future__ import annotations
import torch
from .grid import EdgeField, Grid3D, StretchProfile


def _fwd_diff(a: torch.Tensor, axis: int, h: float) -> torch.Tensor:
    return torch.diff(a, dim=axis) / h


def _adj_diff(v: torch.Tensor, axis: int, h: float) -> torch.Tensor:
    out_shape = list(v.shape)
    out_shape[axis] += 1
    out = torch.zeros(out_shape, dtype=v.dtype, device=v.device)
    sl_out = [slice(None)] * out.ndim
    sl_v = [slice(None)] * v.ndim
    sl_out[axis] = 0; sl_v[axis] = 0
    out[tuple(sl_out)] = -v[tuple(sl_v)] / h
    if v.shape[axis] > 1:
        sl_out[axis] = slice(1, -1)
        sl_v_prev = [slice(None)] * v.ndim; sl_v_prev[axis] = slice(0, -1)
        sl_v_curr = [slice(None)] * v.ndim; sl_v_curr[axis] = slice(1, None)
        out[tuple(sl_out)] = (v[tuple(sl_v_prev)] - v[tuple(sl_v_curr)]) / h
    sl_out[axis] = -1; sl_v[axis] = -1
    out[tuple(sl_out)] = v[tuple(sl_v)] / h
    return out


def curl_e(e: EdgeField, grid: Grid3D, stretch: StretchProfile | None = None):
    d_ez_dy = _fwd_diff(e.ez, 1, grid.dy)
    d_ey_dz = _fwd_diff(e.ey, 2, grid.dz)
    d_ex_dz = _fwd_diff(e.ex, 2, grid.dz)
    d_ez_dx = _fwd_diff(e.ez, 0, grid.dx)
    d_ey_dx = _fwd_diff(e.ey, 0, grid.dx)
    d_ex_dy = _fwd_diff(e.ex, 1, grid.dy)
    if stretch is None:
        hx = d_ez_dy - d_ey_dz
        hy = d_ex_dz - d_ez_dx
        hz = d_ey_dx - d_ex_dy
    else:
        wy = stretch.wy[None, :, None]
        wz = stretch.wz[None, None, :]
        wx = stretch.wx[:, None, None]
        hx = wy * d_ez_dy - wz * d_ey_dz
        hy = wz * d_ex_dz - wx * d_ez_dx
        hz = wx * d_ey_dx - wy * d_ex_dy
    return hx, hy, hz


def curl_h(hx: torch.Tensor, hy: torch.Tensor, hz: torch.Tensor, grid: Grid3D, stretch: StretchProfile | None = None) -> EdgeField:
    if stretch is None:
        ex = _adj_diff(hz, 1, grid.dy) - _adj_diff(hy, 2, grid.dz)
        ey = _adj_diff(hx, 2, grid.dz) - _adj_diff(hz, 0, grid.dx)
        ez = _adj_diff(hy, 0, grid.dx) - _adj_diff(hx, 1, grid.dy)
    else:
        wy = stretch.wy[None, :, None].conj()
        wz = stretch.wz[None, None, :].conj()
        wx = stretch.wx[:, None, None].conj()
        ex = _adj_diff(wy * hz, 1, grid.dy) - _adj_diff(wz * hy, 2, grid.dz)
        ey = _adj_diff(wz * hx, 2, grid.dz) - _adj_diff(wx * hz, 0, grid.dx)
        ez = _adj_diff(wx * hy, 0, grid.dx) - _adj_diff(wy * hx, 1, grid.dy)
    return EdgeField(ex, ey, ez)
