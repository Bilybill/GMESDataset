from __future__ import annotations
import math
import torch
from .grid import EdgeField, Grid3D
from .operators import MU0


def layered_background_from_sigma(sigma: torch.Tensor, mode: str = 'mean') -> torch.Tensor:
    """Build a 1D layered background from a 3D conductivity model.

    Parameters
    ----------
    sigma : (nx, ny, nz) real tensor
        3D conductivity model.
    mode : {'mean', 'median'}
        How to collapse x/y variability into a 1D vertical profile.
    """
    if mode == 'mean':
        s1d = sigma.mean(dim=(0, 1), keepdim=True)
    elif mode == 'median':
        s1d = sigma.median(dim=0).values.median(dim=0).values.reshape(1, 1, -1)
    else:
        raise ValueError("mode must be 'mean' or 'median'")
    return s1d.expand_as(sigma).clone()


def mt1d_propagation_constant(sigma_z: torch.Tensor, freq_hz: float) -> torch.Tensor:
    """Propagation constant for our e^{+iωt} / operator sign convention.

    The 3D code uses curl curl E - iωμσ E = s, so the 1D vertical equation becomes
        E'' - k^2 E = 0,   k^2 = -i ω μ σ.
    We therefore use k = sqrt(-i ω μ σ).
    """
    rdtype = sigma_z.dtype
    cdtype = torch.complex128 if rdtype == torch.float64 else torch.complex64
    omega = 2.0 * math.pi * float(freq_hz)
    return torch.sqrt((-1j * omega * MU0 * sigma_z).to(cdtype))


def mt1d_intrinsic_impedance(k: torch.Tensor, freq_hz: float) -> torch.Tensor:
    """Layer intrinsic impedance Z = E/H for each layer.

    With our sign convention and downward-positive z,
        H = (1/(iωμ)) dE/dz,
    so for a downgoing wave exp(-kz),
        Z = E/H = - i ω μ / k.
    """
    omega = 2.0 * math.pi * float(freq_hz)
    return (-1j * omega * MU0) / k


def mt1d_surface_impedance_recursion(sigma_z: torch.Tensor, dz: float, freq_hz: float):
    """Compute top-of-layer input impedance recursively for a layered earth.

    Each cell in sigma_z is treated as one homogeneous layer of thickness dz,
    except the deepest cell which is treated as a half-space.

    Returns
    -------
    Zin_top : complex scalar tensor
        Surface impedance at the top of the stack.
    Zin_layers : (nz,) complex tensor
        Input impedance seen at the top of each layer.
    k : (nz,) complex tensor
        Propagation constant in each layer.
    Z : (nz,) complex tensor
        Intrinsic impedance of each layer.
    """
    k = mt1d_propagation_constant(sigma_z, freq_hz)
    Z = mt1d_intrinsic_impedance(k, freq_hz)
    nz = sigma_z.numel()
    Zin = torch.empty_like(Z)
    Zin[-1] = Z[-1]  # bottom half-space
    if nz > 1:
        h = torch.as_tensor(float(dz), dtype=Z.real.dtype, device=Z.device)
        for j in range(nz - 2, -1, -1):
            th = torch.tanh(k[j] * h)
            Zin[j] = Z[j] * (Zin[j + 1] + Z[j] * th) / (Z[j] + Zin[j + 1] * th)
    return Zin[0], Zin, k, Z


def solve_1d_plane_wave_profile_recursive(
    sigma_z: torch.Tensor,
    dz: float,
    freq_hz: float,
    amplitude: complex = 1.0 + 0j,
):
    """Exact within-layer plane-wave primary profile for a 1D layered earth.

    This is more standard than solving one global 1D finite-difference system. We:
    1) use impedance recursion from the bottom half-space upward to get the input
       impedance at the top of each layer;
    2) propagate E and H through each layer analytically using downgoing/upgoing
       exponentials.

    Parameters
    ----------
    sigma_z : (nz,) real tensor
        Layer conductivities.
    dz : float
        Uniform layer thickness.
    freq_hz : float
        Frequency.
    amplitude : complex
        Surface electric field amplitude E(0).

    Returns
    -------
    E_nodes : (nz+1,) complex tensor
        Electric field sampled at layer interfaces / grid nodes.
    H_nodes : (nz+1,) complex tensor
        Tangential magnetic field at the same z locations.
    meta : dict
        Includes surface impedance and per-layer recursion quantities.
    """
    rdtype = sigma_z.dtype
    cdtype = torch.complex128 if rdtype == torch.float64 else torch.complex64
    device = sigma_z.device
    nz = sigma_z.numel()
    Zin_top, Zin_layers, k, Z = mt1d_surface_impedance_recursion(sigma_z, dz, freq_hz)

    E_nodes = torch.empty(nz + 1, dtype=cdtype, device=device)
    H_nodes = torch.empty(nz + 1, dtype=cdtype, device=device)
    E_nodes[0] = torch.as_tensor(amplitude, dtype=cdtype, device=device)
    H_nodes[0] = E_nodes[0] / Zin_top

    h = torch.as_tensor(float(dz), dtype=rdtype, device=device)
    for j in range(nz):
        E_top = E_nodes[j]
        H_top = H_nodes[j]
        A = 0.5 * (E_top + Z[j] * H_top)  # downgoing amplitude at layer top
        B = 0.5 * (E_top - Z[j] * H_top)  # upgoing amplitude at layer top
        em = torch.exp(-k[j] * h)
        ep = torch.exp(k[j] * h)
        E_bot = A * em + B * ep
        H_bot = (A * em - B * ep) / Z[j]
        E_nodes[j + 1] = E_bot
        H_nodes[j + 1] = H_bot

    meta = {
        'surface_impedance': Zin_top,
        'input_impedance_layers': Zin_layers,
        'k': k,
        'Z_layer': Z,
    }
    return E_nodes, H_nodes, meta


# Keep a compatibility wrapper with the old name used elsewhere in the package.
def solve_1d_plane_wave_profile(sigma_z: torch.Tensor, dz: float, freq_hz: float, amplitude: complex = 1.0 + 0j):
    E_nodes, _, _ = solve_1d_plane_wave_profile_recursive(sigma_z, dz, freq_hz, amplitude=amplitude)
    return E_nodes


def primary_edge_field_from_profile(grid: Grid3D, E_nodes: torch.Tensor, polarization: str = 'x') -> EdgeField:
    f = grid.zeros_edge()
    if polarization == 'x':
        prof = E_nodes.to(grid.dtype).reshape(1, 1, -1)
        f.ex = prof.expand(grid.nx, grid.ny + 1, grid.nz + 1).clone()
    elif polarization == 'y':
        prof = E_nodes.to(grid.dtype).reshape(1, 1, -1)
        f.ey = prof.expand(grid.nx + 1, grid.ny, grid.nz + 1).clone()
    else:
        raise ValueError("polarization must be 'x' or 'y'")
    return f


def solve_mt_primary_1d_fields(grid: Grid3D, sigma_bg: torch.Tensor, freq_hz: float, amplitude: complex = 1.0 + 0j):
    """Return x/y polarized 1D layered primary electric fields embedded on the 3D grid.

    Why a 1D primary solver in a 3D MT code?
    ----------------------------------------
    In 3D MT primary-secondary formulations, the *full* 3D field is split into:
        total = primary(background response) + secondary(3D anomaly response).
    The primary field is most often computed from a 1D layered background because
    the natural-source plane wave over a layered earth has an efficient semi-analytic
    solution. The 3D solver then only needs to solve for the secondary field caused
    by lateral conductivity contrasts.
    """
    sigma_bg_real = sigma_bg.real if torch.is_complex(sigma_bg) else sigma_bg
    sigma_z = sigma_bg_real.mean(dim=(0, 1)).to(grid.rdtype)
    E_nodes, H_nodes, meta = solve_1d_plane_wave_profile_recursive(sigma_z, grid.dz, freq_hz, amplitude=amplitude)
    fx = primary_edge_field_from_profile(grid, E_nodes, 'x')
    fy = primary_edge_field_from_profile(grid, E_nodes, 'y')
    meta = dict(meta)
    meta['E_nodes'] = E_nodes
    meta['H_nodes'] = H_nodes
    meta['sigma_z'] = sigma_z
    return [fx, fy], meta
