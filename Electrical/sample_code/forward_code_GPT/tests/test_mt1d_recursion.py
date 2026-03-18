import torch
from em3d.mt1d import mt1d_surface_impedance_recursion, solve_1d_plane_wave_profile_recursive
from em3d.operators import MU0
import math


def test_homogeneous_halfspace_impedance_matches_closed_form():
    sigma = torch.full((8,), 0.01, dtype=torch.float64)
    f = 1.0
    Zin_top, Zin_layers, k, Z = mt1d_surface_impedance_recursion(sigma, dz=50.0, freq_hz=f)
    omega = 2.0 * math.pi * f
    k0 = torch.sqrt(torch.tensor(-1j * omega * MU0 * 0.01, dtype=torch.complex128))
    Z0 = (-1j * omega * MU0) / k0
    rel = abs((Zin_top - Z0) / Z0)
    assert rel < 1e-10, rel.item()
    assert torch.max(torch.abs((Zin_layers - Z0) / Z0)) < 1e-10


def test_profile_boundary_relation():
    sigma = torch.tensor([0.01, 0.1, 0.01], dtype=torch.float64)
    E, H, meta = solve_1d_plane_wave_profile_recursive(sigma, dz=100.0, freq_hz=0.5)
    Zin = meta['surface_impedance']
    rel = abs((E[0] / H[0] - Zin) / Zin)
    assert rel < 1e-12, rel.item()
    assert E.numel() == sigma.numel() + 1
    assert H.numel() == sigma.numel() + 1
