from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from em3d.grid import Grid3D, Receiver
from em3d.forward import simulate_mt_primary_secondary_batch
from em3d.mt1d import layered_background_from_sigma, solve_mt_primary_1d_fields


def main():
    grid = Grid3D(nx=4, ny=4, nz=6, dx=50.0, dy=50.0, dz=50.0, device="cpu", dtype=torch.complex128)
    sigma = torch.full(grid.shape_cells, 1e-2, dtype=grid.rdtype, device=grid.device)
    sigma[:, :, 3:] = 5e-2
    sigma_bg = layered_background_from_sigma(sigma)
    prim, meta = solve_mt_primary_1d_fields(grid, sigma_bg, 1.0)
    assert prim[0].ex.shape == grid.shape_ex
    assert prim[1].ey.shape == grid.shape_ey
    receivers = [Receiver(x=100.0, y=100.0, z=50.0), Receiver(x=150.0, y=100.0, z=50.0)]
    data, infos = simulate_mt_primary_secondary_batch(grid, sigma, sigma_bg, [1.0], receivers,
                                                      tol=1e-5, maxiter=40, restart=8,
                                                      use_block_secondary=True, precond='jacobi')
    d = data[0]
    assert d['Zxy'].shape[0] == len(receivers)
    assert d['rho_xy'].shape == d['phi_xy_deg'].shape
    assert d['Tzx'].shape == d['Tzy'].shape
    print('MT layered primary shapes PASS')

if __name__ == '__main__':
    main()
