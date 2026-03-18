from __future__ import annotations
import math
from .grid import EdgeField, Grid3D
from .diffops import curl_e, curl_h
from .averaging import sigma_to_edges

MU0 = 4e-7 * math.pi


def apply_operator(e: EdgeField, sigma, omega: float, grid: Grid3D, mu_r: float = 1.0) -> EdgeField:
    sigma_edge = sigma_to_edges(sigma, grid)
    ei = EdgeField(e.ex * grid.mask.ex, e.ey * grid.mask.ey, e.ez * grid.mask.ez)
    stretch = grid.make_stretch(omega)
    hx, hy, hz = curl_e(ei, grid, stretch=stretch)
    cc = curl_h(hx / mu_r, hy / mu_r, hz / mu_r, grid, stretch=stretch)
    sigma_bg = ((sigma_edge.ex.real.mean() + sigma_edge.ey.real.mean() + sigma_edge.ez.real.mean()) / 3.0).clamp_min(1e-8)
    damp = EdgeField(sigma_edge.ex + grid.sponge.ex * sigma_bg,
                     sigma_edge.ey + grid.sponge.ey * sigma_bg,
                     sigma_edge.ez + grid.sponge.ez * sigma_bg)
    mass = EdgeField((-1j * omega * MU0) * damp.ex * ei.ex,
                     (-1j * omega * MU0) * damp.ey * ei.ey,
                     (-1j * omega * MU0) * damp.ez * ei.ez)
    out = cc + mass
    out.ex = out.ex * grid.mask.ex + (1.0 - grid.mask.ex) * e.ex
    out.ey = out.ey * grid.mask.ey + (1.0 - grid.mask.ey) * e.ey
    out.ez = out.ez * grid.mask.ez + (1.0 - grid.mask.ez) * e.ez
    return out


def magnetic_field_from_e(e: EdgeField, omega: float, grid: Grid3D, mu_r: float = 1.0):
    stretch = grid.make_stretch(omega)
    hx, hy, hz = curl_e(e, grid, stretch=stretch)
    factor = 1.0 / (1j * omega * MU0 * mu_r)
    return hx * factor, hy * factor, hz * factor
