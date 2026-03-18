from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from em3d.grid import Grid3D, Receiver
from em3d.forward import simulate_mt_primary_secondary_batch


def main():
    grid = Grid3D(nx=4, ny=4, nz=4, dx=50.0, dy=50.0, dz=50.0, device="cpu", dtype=torch.complex128)
    sigma = torch.full(grid.shape_cells, 1e-2, dtype=grid.rdtype, device=grid.device)
    sigma_bg = torch.full_like(sigma, 1e-2)
    sigma[1:3, 1:3, 1:3] = 5e-2
    receivers = [Receiver(x=100.0, y=100.0, z=75.0), Receiver(x=150.0, y=100.0, z=75.0)]
    data, infos = simulate_mt_primary_secondary_batch(grid, sigma, sigma_bg, [1.0], receivers, tol=1e-6, maxiter=80, restart=10)
    d = data[0]
    assert d['Zxy'].shape[0] == len(receivers)
    assert d['rho_xy'].shape == d['phi_xy_deg'].shape
    assert d['Tzx'].shape == d['Tzy'].shape
    print('MT shapes PASS')

if __name__ == '__main__':
    main()
