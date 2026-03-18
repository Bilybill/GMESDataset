from __future__ import annotations
from typing import Iterable
import math
import torch
from .grid import Grid3D, EdgeField, Receiver
from .operators import apply_operator, magnetic_field_from_e, MU0
from .solver import gmres, block_gmres, jacobi_precond_from_diag, vertical_line_precond, make_flexible_combo_precond
from .receivers import sample_e_components, sample_face_components
from .observations import csem_data_dict, impedance_and_tipper
from .averaging import sigma_to_edges
from .mt1d import solve_mt_primary_1d_fields, layered_background_from_sigma


def _diag_proxy(grid: Grid3D, sigma: torch.Tensor, omega: float):
    se = sigma_to_edges(sigma, grid)
    d = torch.cat([se.ex.reshape(-1), se.ey.reshape(-1), se.ez.reshape(-1)]).abs() + 1.0 / max(grid.dx, 1e-12) ** 2
    d = d + abs(omega * MU0)
    return d.to(torch.complex128 if sigma.dtype == torch.float64 else torch.complex64)


def _make_precond(grid: Grid3D, sigma: torch.Tensor, omega: float, mode: str = 'flex'):
    """Create preconditioner for flexible/block Krylov solvers.

    Modes
    -----
    - 'flex' / 'combo' / 'line+jacobi': vertical line preconditioner followed by a Jacobi sweep
    - 'line': vertical line only
    - 'jacobi': Jacobi only
    - 'none': no preconditioner
    """
    if mode in (None, 'none'):
        return None
    if mode == 'line':
        return vertical_line_precond(grid, sigma, omega)
    diag = jacobi_precond_from_diag(_diag_proxy(grid, sigma, omega))
    if mode == 'jacobi':
        return diag
    if mode in ('flex', 'combo', 'line+jacobi'):
        return make_flexible_combo_precond(vertical_line_precond(grid, sigma, omega), diag)
    raise ValueError(f'Unknown preconditioner mode: {mode}')


def solve_frequency(grid: Grid3D, sigma: torch.Tensor, freq_hz: float, source: EdgeField,
                    tol: float = 1e-6, maxiter: int = 200, restart: int = 40,
                    x0: EdgeField | None = None, precond: str = 'flex'):
    omega = 2.0 * math.pi * float(freq_hz)
    b = source.flatten()
    def A_mv(vec: torch.Tensor):
        ef = EdgeField.unflatten(vec, grid)
        af = apply_operator(ef, sigma, omega, grid)
        return af.flatten()
    M_inv = _make_precond(grid, sigma, omega, mode=precond)
    x_init = grid.zeros_edge().flatten() if x0 is None else x0.flatten()
    x, info = gmres(A_mv, b, x0=x_init, tol=tol, maxiter=maxiter, restart=restart, M_inv=M_inv)
    info['precond'] = precond
    return EdgeField.unflatten(x, grid), info


def solve_frequency_block(grid: Grid3D, sigma: torch.Tensor, freq_hz: float, sources: list[EdgeField],
                          tol: float = 1e-6, maxiter: int = 200, restart: int = 20,
                          x0: list[EdgeField] | None = None, precond: str = 'flex'):
    omega = 2.0 * math.pi * float(freq_hz)
    B = torch.stack([src.flatten() for src in sources], dim=1)
    def A_mv_block(X: torch.Tensor):
        cols = []
        for i in range(X.shape[1]):
            ef = EdgeField.unflatten(X[:, i], grid)
            af = apply_operator(ef, sigma, omega, grid)
            cols.append(af.flatten())
        return torch.stack(cols, dim=1)
    M_inv = _make_precond(grid, sigma, omega, mode=precond)
    X0 = torch.zeros_like(B) if x0 is None else torch.stack([f.flatten() for f in x0], dim=1)
    X, info = block_gmres(A_mv_block, B, X0=X0, tol=tol, maxiter=maxiter, restart=restart, M_inv=M_inv)
    info['precond'] = precond
    fields = [EdgeField.unflatten(X[:, i], grid) for i in range(X.shape[1])]
    return fields, info


def solve_batch_block(grid: Grid3D, sigma: torch.Tensor, frequencies_hz: Iterable[float],
                      sources: list[EdgeField], tol: float = 1e-6, maxiter: int = 200, restart: int = 20,
                      precond: str = 'flex'):
    freqs = list(frequencies_hz)
    all_fields = []
    all_infos = []
    x0 = None
    for f in freqs:
        fields, info = solve_frequency_block(grid, sigma, f, sources, tol=tol, maxiter=maxiter, restart=restart, x0=x0, precond=precond)
        all_fields.append(fields)
        all_infos.append(info)
        x0 = fields
    return all_fields, all_infos


def secondary_rhs_from_primary(grid: Grid3D, sigma: torch.Tensor, sigma_bg: torch.Tensor, freq_hz: float, primary_fields: list[EdgeField]):
    omega = 2.0 * math.pi * float(freq_hz)
    ds = sigma_to_edges(sigma - sigma_bg, grid)
    rhs = []
    fac = 1j * omega * MU0
    for ep in primary_fields:
        rhs.append(EdgeField(fac * ds.ex * ep.ex, fac * ds.ey * ep.ey, fac * ds.ez * ep.ez))
    return rhs


def solve_mt_primary_secondary_frequency(grid: Grid3D, sigma: torch.Tensor, sigma_bg: torch.Tensor | None, freq_hz: float,
                                         tol: float = 1e-6, maxiter: int = 240, restart: int = 20,
                                         use_block_secondary: bool = True, precond: str = 'flex',
                                         layered_bg_mode: str = 'mean'):
    if sigma_bg is None:
        sigma_bg = layered_background_from_sigma(sigma, mode=layered_bg_mode)
    primary_fields, primary_meta = solve_mt_primary_1d_fields(grid, sigma_bg, freq_hz)
    rhs_sec = secondary_rhs_from_primary(grid, sigma, sigma_bg, freq_hz, primary_fields)
    if use_block_secondary:
        sec_fields, info_s = solve_frequency_block(grid, sigma, freq_hz, rhs_sec, tol=tol, maxiter=maxiter, restart=restart, precond=precond)
    else:
        sec_fields, scalar_infos = [], []
        for rhs in rhs_sec:
            sf, inf = solve_frequency(grid, sigma, freq_hz, rhs, tol=tol, maxiter=maxiter, restart=restart, precond=precond)
            sec_fields.append(sf); scalar_infos.append(inf)
        info_s = {'mode': 'scalar_fallback', 'per_rhs': scalar_infos, 'precond': precond}
    total_fields = [primary_fields[i] + sec_fields[i] for i in range(2)]
    info = {'primary': primary_meta, 'secondary': info_s, 'block_size': 2, 'background': sigma_bg.mean(dim=(0, 1))}
    return {'primary': primary_fields, 'secondary': sec_fields, 'total': total_fields}, info


def simulate_mt_primary_secondary_batch(grid: Grid3D, sigma: torch.Tensor, sigma_bg: torch.Tensor | None,
                                        frequencies_hz: Iterable[float], receivers: list[Receiver],
                                        tol: float = 1e-6, maxiter: int = 240, restart: int = 20,
                                        use_block_secondary: bool = True, precond: str = 'flex',
                                        layered_bg_mode: str = 'mean'):
    out, infos = [], []
    for f in frequencies_hz:
        fields_pack, info = solve_mt_primary_secondary_frequency(grid, sigma, sigma_bg, f, tol=tol, maxiter=maxiter,
                                                                 restart=restart, use_block_secondary=use_block_secondary,
                                                                 precond=precond, layered_bg_mode=layered_bg_mode)
        fld_x, fld_y = fields_pack['total']
        omega = 2.0 * math.pi * float(f)
        ex_x, ey_x, _ = sample_e_components(fld_x, grid, receivers)
        ex_y, ey_y, _ = sample_e_components(fld_y, grid, receivers)
        hx, hy, hz = magnetic_field_from_e(fld_x, omega, grid)
        hx2, hy2, hz2 = magnetic_field_from_e(fld_y, omega, grid)
        hx_x, hy_x, hz_x = sample_face_components(hx, hy, hz, grid, receivers)
        hx_y, hy_y, hz_y = sample_face_components(hx2, hy2, hz2, grid, receivers)
        out.append(impedance_and_tipper(ex_x, ey_x, hz_x, hx_x, hy_x, ex_y, ey_y, hz_y, hx_y, hy_y, omega))
        infos.append(info)
    return out, infos


def simulate_csem_block(grid: Grid3D, sigma: torch.Tensor, frequencies_hz: Iterable[float],
                        sources: list[EdgeField], receivers: list[Receiver],
                        tol: float = 1e-6, maxiter: int = 200, restart: int = 20,
                        precond: str = 'flex'):
    fields, infos = solve_batch_block(grid, sigma, frequencies_hz, sources, tol=tol, maxiter=maxiter, restart=restart, precond=precond)
    out = []
    for fi, f in enumerate(frequencies_hz):
        omega = 2.0 * math.pi * float(f)
        freq_list = []
        for fld in fields[fi]:
            ex, ey, ez = sample_e_components(fld, grid, receivers)
            hx, hy, hz = magnetic_field_from_e(fld, omega, grid)
            hx_r, hy_r, hz_r = sample_face_components(hx, hy, hz, grid, receivers)
            freq_list.append(csem_data_dict(ex, ey, ez, hx_r, hy_r, hz_r))
        out.append(freq_list)
    return out, infos
