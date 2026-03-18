from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from em3d.grid import Grid3D, Receiver
from em3d.sources import point_electric_dipole
from em3d.forward import simulate_csem_block


def main():
    grid = Grid3D(nx=4, ny=4, nz=4, dx=50.0, dy=50.0, dz=50.0, device="cpu", dtype=torch.complex128)
    sigma = torch.full(grid.shape_cells, 1e-2, dtype=grid.rdtype, device=grid.device)
    sigma[1:3, 1:3, 1:3] = 5e-2
    receivers = [Receiver(x=100.0, y=100.0, z=75.0, name="r0"), Receiver(x=150.0, y=100.0, z=75.0, name="r1")]
    sources = [
        point_electric_dipole(grid, 100.0, 100.0, 100.0, component="x", amplitude=1.0 + 0j),
        point_electric_dipole(grid, 100.0, 100.0, 100.0, component="y", amplitude=1.0 + 0j),
    ]
    data, infos = simulate_csem_block(grid, sigma, [1.0], sources, receivers, tol=1e-6, maxiter=80, restart=10)
    print("CSEM block infos:", infos)
    print("Example |Ex|:", data[0][0]["amp_Ex"][0].item())

if __name__ == "__main__":
    main()
