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
    receivers = [Receiver(x=100.0, y=100.0, z=75.0, name="r0"), Receiver(x=150.0, y=100.0, z=75.0, name="r1")]
    data, infos = simulate_mt_primary_secondary_batch(grid, sigma, sigma_bg, [1.0], receivers, tol=1e-6, maxiter=80, restart=10)
    print("MT infos:", infos)
    print("rho_xy:", data[0]["rho_xy"])
    print("phi_xy_deg:", data[0]["phi_xy_deg"])
    print("|Tzx|:", data[0]["tipper_amp_x"])

if __name__ == "__main__":
    main()
