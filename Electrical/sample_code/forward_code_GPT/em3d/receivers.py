from __future__ import annotations
import torch
from .grid import Grid3D, EdgeField, Receiver


def _interp_trilinear(values: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                      dx: float, dy: float, dz: float,
                      x_origin: float = 0.0, y_origin: float = 0.0, z_origin: float = 0.0):
    sx = (x - x_origin) / dx
    sy = (y - y_origin) / dy
    sz = (z - z_origin) / dz
    i0 = torch.clamp(torch.floor(sx).long(), 0, values.shape[0] - 2)
    j0 = torch.clamp(torch.floor(sy).long(), 0, values.shape[1] - 2)
    k0 = torch.clamp(torch.floor(sz).long(), 0, values.shape[2] - 2)
    tx = (sx - i0).to(values.real.dtype)
    ty = (sy - j0).to(values.real.dtype)
    tz = (sz - k0).to(values.real.dtype)

    c000 = values[i0, j0, k0]
    c100 = values[i0 + 1, j0, k0]
    c010 = values[i0, j0 + 1, k0]
    c110 = values[i0 + 1, j0 + 1, k0]
    c001 = values[i0, j0, k0 + 1]
    c101 = values[i0 + 1, j0, k0 + 1]
    c011 = values[i0, j0 + 1, k0 + 1]
    c111 = values[i0 + 1, j0 + 1, k0 + 1]

    c00 = c000 * (1 - tx) + c100 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c11 = c011 * (1 - tx) + c111 * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    return c0 * (1 - tz) + c1 * tz


def sample_e_components(field: EdgeField, grid: Grid3D, receivers: list[Receiver]):
    xs = torch.tensor([r.x for r in receivers], device=grid.device, dtype=grid.rdtype)
    ys = torch.tensor([r.y for r in receivers], device=grid.device, dtype=grid.rdtype)
    zs = torch.tensor([r.z for r in receivers], device=grid.device, dtype=grid.rdtype)
    ex = _interp_trilinear(field.ex, xs, ys, zs, grid.dx, grid.dy, grid.dz, x_origin=0.5 * grid.dx, y_origin=0.0, z_origin=0.0)
    ey = _interp_trilinear(field.ey, xs, ys, zs, grid.dx, grid.dy, grid.dz, x_origin=0.0, y_origin=0.5 * grid.dy, z_origin=0.0)
    ez = _interp_trilinear(field.ez, xs, ys, zs, grid.dx, grid.dy, grid.dz, x_origin=0.0, y_origin=0.0, z_origin=0.5 * grid.dz)
    return ex, ey, ez


def sample_face_components(hx: torch.Tensor, hy: torch.Tensor, hz: torch.Tensor, grid: Grid3D, receivers: list[Receiver]):
    xs = torch.tensor([r.x for r in receivers], device=grid.device, dtype=grid.rdtype)
    ys = torch.tensor([r.y for r in receivers], device=grid.device, dtype=grid.rdtype)
    zs = torch.tensor([r.z for r in receivers], device=grid.device, dtype=grid.rdtype)
    sx = _interp_trilinear(hx, xs, ys, zs, grid.dx, grid.dy, grid.dz, x_origin=0.0, y_origin=0.5 * grid.dy, z_origin=0.5 * grid.dz)
    sy = _interp_trilinear(hy, xs, ys, zs, grid.dx, grid.dy, grid.dz, x_origin=0.5 * grid.dx, y_origin=0.0, z_origin=0.5 * grid.dz)
    sz = _interp_trilinear(hz, xs, ys, zs, grid.dx, grid.dy, grid.dz, x_origin=0.5 * grid.dx, y_origin=0.5 * grid.dy, z_origin=0.0)
    return sx, sy, sz
