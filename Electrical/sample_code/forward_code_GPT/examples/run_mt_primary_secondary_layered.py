from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from em3d.grid import Grid3D, Receiver
from em3d.forward import simulate_mt_primary_secondary_batch
from em3d.mt1d import layered_background_from_sigma


def main():
    grid = Grid3D(nx=3, ny=3, nz=5, dx=100.0, dy=100.0, dz=100.0, device="cpu", dtype=torch.complex128)
    sigma = torch.full(grid.shape_cells, 1e-2, dtype=grid.rdtype, device=grid.device)
    sigma[:, :, 4:] = 5e-2
    sigma[1:2, 1:2, 2:4] = 2e-1
    sigma_bg = layered_background_from_sigma(sigma)
    receivers = [Receiver(x=150.0, y=150.0, z=100.0, name="r0"), Receiver(x=200.0, y=150.0, z=100.0, name="r1")]
    data, infos = simulate_mt_primary_secondary_batch(grid, sigma, sigma_bg, [0.1, 1.0], receivers,
                                                      tol=1e-5, maxiter=60, restart=8,
                                                      use_block_secondary=False, precond='jacobi')
    for i, f in enumerate([0.1, 1.0]):
        print("freq:", f)
        print("rho_xy:", data[i]['rho_xy'])
        print("phi_xy_deg:", data[i]['phi_xy_deg'])
        print("tipper_amp_x:", data[i]['tipper_amp_x'])
        print("secondary info:", infos[i]['secondary'])

if __name__ == '__main__':
    main()
