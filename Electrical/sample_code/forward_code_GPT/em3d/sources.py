
from __future__ import annotations
import torch
from .grid import EdgeField, Grid3D


def point_electric_dipole(grid: Grid3D, x: float, y: float, z: float, component: str = 'x', amplitude: complex = 1.0 + 0j):
    s = grid.zeros_edge()
    amp = torch.tensor(amplitude, device=grid.device, dtype=grid.dtype)
    if component == 'x':
        ix = min(max(int(x / grid.dx), 0), grid.nx - 1)
        iy = min(max(int(round(y / grid.dy)), 0), grid.ny)
        iz = min(max(int(round(z / grid.dz)), 0), grid.nz)
        s.ex[ix, iy, iz] = amp
    elif component == 'y':
        ix = min(max(int(round(x / grid.dx)), 0), grid.nx)
        iy = min(max(int(y / grid.dy), 0), grid.ny - 1)
        iz = min(max(int(round(z / grid.dz)), 0), grid.nz)
        s.ey[ix, iy, iz] = amp
    elif component == 'z':
        ix = min(max(int(round(x / grid.dx)), 0), grid.nx)
        iy = min(max(int(round(y / grid.dy)), 0), grid.ny)
        iz = min(max(int(z / grid.dz), 0), grid.nz - 1)
        s.ez[ix, iy, iz] = amp
    else:
        raise ValueError('component must be x/y/z')
    return s


def plane_wave_tangential_bc_source(grid: Grid3D, polarization: str = 'x', amplitude: complex = 1.0 + 0j,
                                    z_layer: int = 1, taper: bool = True):
    """Approximate incident plane wave by prescribing tangential E on a top interior layer.
    This is still a source-term approximation, but when combined with primary-secondary it is
    significantly more stable and interpretable than exciting the full model directly.
    """
    s = grid.zeros_edge()
    amp = torch.tensor(amplitude, device=grid.device, dtype=grid.dtype)
    z_layer = max(1, min(int(z_layer), grid.nz - 1))
    if taper:
        wx = torch.hann_window(max(grid.nx, 2), periodic=False, device=grid.device, dtype=grid.rdtype)
        wy = torch.hann_window(max(grid.ny, 2), periodic=False, device=grid.device, dtype=grid.rdtype)
        wx = torch.clamp(wx, min=1e-3)
        wy = torch.clamp(wy, min=1e-3)
    if polarization == 'x':
        if taper:
            s.ex[:, 1:-1, z_layer] = amp * wx[:, None]
        else:
            s.ex[:, 1:-1, z_layer] = amp
    elif polarization == 'y':
        if taper:
            s.ey[1:-1, :, z_layer] = amp * wy[None, :]
        else:
            s.ey[1:-1, :, z_layer] = amp
    else:
        raise ValueError('polarization must be x or y')
    return s


def plane_wave_source_pair(grid: Grid3D, amplitude: complex = 1.0 + 0j, z_layer: int = 1):
    return [
        plane_wave_tangential_bc_source(grid, 'x', amplitude=amplitude, z_layer=z_layer),
        plane_wave_tangential_bc_source(grid, 'y', amplitude=amplitude, z_layer=z_layer),
    ]
