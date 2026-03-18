from __future__ import annotations
from dataclasses import dataclass
import torch

EPS0 = 8.854187817e-12

@dataclass
class Receiver:
    x: float
    y: float
    z: float
    name: str = ""


@dataclass
class EdgeField:
    ex: torch.Tensor
    ey: torch.Tensor
    ez: torch.Tensor

    def clone(self) -> "EdgeField":
        return EdgeField(self.ex.clone(), self.ey.clone(), self.ez.clone())

    def zeros_like(self) -> "EdgeField":
        return EdgeField(torch.zeros_like(self.ex), torch.zeros_like(self.ey), torch.zeros_like(self.ez))

    def conj(self) -> "EdgeField":
        return EdgeField(self.ex.conj(), self.ey.conj(), self.ez.conj())

    def __add__(self, other: "EdgeField") -> "EdgeField":
        return EdgeField(self.ex + other.ex, self.ey + other.ey, self.ez + other.ez)

    def __sub__(self, other: "EdgeField") -> "EdgeField":
        return EdgeField(self.ex - other.ex, self.ey - other.ey, self.ez - other.ez)

    def __mul__(self, val):
        return EdgeField(self.ex * val, self.ey * val, self.ez * val)

    __rmul__ = __mul__

    def flatten(self) -> torch.Tensor:
        return torch.cat([self.ex.reshape(-1), self.ey.reshape(-1), self.ez.reshape(-1)])

    @staticmethod
    def unflatten(vec: torch.Tensor, grid: "Grid3D") -> "EdgeField":
        n1 = grid.n_ex
        n2 = n1 + grid.n_ey
        ex = vec[:n1].reshape(grid.shape_ex)
        ey = vec[n1:n2].reshape(grid.shape_ey)
        ez = vec[n2:].reshape(grid.shape_ez)
        return EdgeField(ex, ey, ez)


@dataclass
class StretchProfile:
    wx: torch.Tensor  # 1/sx on x-centered locations, shape (nx,)
    wy: torch.Tensor  # 1/sy on y-centered locations, shape (ny,)
    wz: torch.Tensor  # 1/sz on z-centered locations, shape (nz,)


class Grid3D:
    def __init__(self, nx: int, ny: int, nz: int, dx: float, dy: float, dz: float,
                 device: str = "cpu", dtype: torch.dtype = torch.complex128,
                 pml_n: int = 8, pml_strength: float = 3.0,
                 pml_kappa_max: float = 5.0, pml_alpha_max: float = 2.0 * 3.141592653589793,
                 pml_order: int = 2):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = float(dx), float(dy), float(dz)
        self.device = torch.device(device)
        self.rdtype = torch.float64 if dtype == torch.complex128 else torch.float32
        self.dtype = dtype
        self.pml_n = pml_n
        self.pml_strength = pml_strength
        self.pml_kappa_max = pml_kappa_max
        self.pml_alpha_max = pml_alpha_max
        self.pml_order = pml_order

        self.shape_cells = (nx, ny, nz)
        self.shape_ex = (nx, ny + 1, nz + 1)
        self.shape_ey = (nx + 1, ny, nz + 1)
        self.shape_ez = (nx + 1, ny + 1, nz)

        self.n_ex = nx * (ny + 1) * (nz + 1)
        self.n_ey = (nx + 1) * ny * (nz + 1)
        self.n_ez = (nx + 1) * (ny + 1) * nz
        self.n_total = self.n_ex + self.n_ey + self.n_ez

        self.xn = torch.arange(nx + 1, device=self.device, dtype=self.rdtype) * self.dx
        self.yn = torch.arange(ny + 1, device=self.device, dtype=self.rdtype) * self.dy
        self.zn = torch.arange(nz + 1, device=self.device, dtype=self.rdtype) * self.dz
        self.xc = (torch.arange(nx, device=self.device, dtype=self.rdtype) + 0.5) * self.dx
        self.yc = (torch.arange(ny, device=self.device, dtype=self.rdtype) + 0.5) * self.dy
        self.zc = (torch.arange(nz, device=self.device, dtype=self.rdtype) + 0.5) * self.dz

        self.mask = self._build_boundary_mask()
        self.sponge = self._build_sponge_profile()

    def zeros_edge(self) -> EdgeField:
        z = lambda shape: torch.zeros(shape, device=self.device, dtype=self.dtype)
        return EdgeField(z(self.shape_ex), z(self.shape_ey), z(self.shape_ez))

    def _build_boundary_mask(self) -> EdgeField:
        mx = torch.ones(self.shape_ex, device=self.device, dtype=self.rdtype)
        my = torch.ones(self.shape_ey, device=self.device, dtype=self.rdtype)
        mz = torch.ones(self.shape_ez, device=self.device, dtype=self.rdtype)
        # outer PEC shell only; interior PML handles absorption
        mx[:, 0, :] = 0; mx[:, -1, :] = 0; mx[:, :, 0] = 0; mx[:, :, -1] = 0
        mx[0, :, :] = 0; mx[-1, :, :] = 0
        my[0, :, :] = 0; my[-1, :, :] = 0; my[:, :, 0] = 0; my[:, :, -1] = 0
        my[:, 0, :] = 0; my[:, -1, :] = 0
        mz[0, :, :] = 0; mz[-1, :, :] = 0; mz[:, 0, :] = 0; mz[:, -1, :] = 0
        mz[:, :, 0] = 0; mz[:, :, -1] = 0
        return EdgeField(mx.to(self.dtype), my.to(self.dtype), mz.to(self.dtype))

    def _axis_sponge(self, n_edges: int) -> torch.Tensor:
        n = max(1, self.pml_n)
        idx = torch.arange(n_edges, device=self.device, dtype=self.rdtype)
        left = torch.clamp((n - idx) / n, min=0.0)
        right = torch.clamp((idx - (n_edges - 1 - n)) / n, min=0.0)
        prof = (left ** 2 + right ** 2) * self.pml_strength
        return prof

    def _build_sponge_profile(self) -> EdgeField:
        px = self._axis_sponge(self.nx)
        py_n = self._axis_sponge(self.ny + 1)
        pz_n = self._axis_sponge(self.nz + 1)
        ex = px[:, None, None] + py_n[None, :, None] + pz_n[None, None, :]

        px_n = self._axis_sponge(self.nx + 1)
        py = self._axis_sponge(self.ny)
        ey = px_n[:, None, None] + py[None, :, None] + pz_n[None, None, :]

        pz = self._axis_sponge(self.nz)
        ez = px_n[:, None, None] + py_n[None, :, None] + pz[None, None, :]
        return EdgeField(ex.to(self.dtype), ey.to(self.dtype), ez.to(self.dtype))

    def _pml_axis_profile(self, n: int, h: float, omega: float):
        idx = torch.arange(n, device=self.device, dtype=self.rdtype)
        n_pml = max(1, self.pml_n)
        dist_left = torch.clamp((n_pml - (idx + 0.5)) / n_pml, min=0.0)
        dist_right = torch.clamp(((idx + 0.5) - (n - n_pml - 0.5)) / n_pml, min=0.0)
        t = torch.maximum(dist_left, dist_right)
        sigma = self.pml_strength * (t ** self.pml_order) / max(h, 1e-12)
        kappa = 1.0 + (self.pml_kappa_max - 1.0) * (t ** self.pml_order)
        alpha = self.pml_alpha_max * (1.0 - t)
        denom = alpha + 1j * omega * EPS0
        s = kappa + sigma / denom
        return (1.0 / s).to(self.dtype)

    def make_stretch(self, omega: float) -> StretchProfile:
        return StretchProfile(
            wx=self._pml_axis_profile(self.nx, self.dx, omega),
            wy=self._pml_axis_profile(self.ny, self.dy, omega),
            wz=self._pml_axis_profile(self.nz, self.dz, omega),
        )
