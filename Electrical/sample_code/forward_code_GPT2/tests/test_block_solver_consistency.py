
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from em3d.grid import Grid3D
from em3d.sources import point_electric_dipole
from em3d.forward import solve_frequency, solve_frequency_block


def main():
    grid = Grid3D(nx=4, ny=4, nz=4, dx=50.0, dy=50.0, dz=50.0, device='cpu', dtype=torch.complex128)
    sigma = torch.full(grid.shape_cells, 1e-2, dtype=grid.rdtype, device=grid.device)
    freq = 2.0
    sources = [
        point_electric_dipole(grid, 100.0, 100.0, 100.0, component='x', amplitude=1.0 + 0j),
        point_electric_dipole(grid, 100.0, 100.0, 100.0, component='y', amplitude=1.0 + 0j),
    ]
    fields_block, info = solve_frequency_block(grid, sigma, freq, sources, tol=1e-6, maxiter=80, restart=10)
    fields_scalar = [solve_frequency(grid, sigma, freq, s, tol=1e-6, maxiter=80, restart=10)[0] for s in sources]

    errs = []
    norms = []
    for fb, fs in zip(fields_block, fields_scalar):
        vb = fb.flatten()
        vs = fs.flatten()
        errs.append(torch.linalg.norm(vb - vs))
        norms.append(torch.linalg.norm(vs))
    rel = sum(errs) / sum(norms)
    print('block-vs-scalar relative mismatch =', float(rel))
    assert rel < 1e-6, 'block solver deviates too much from scalar solves'
    print('PASS')


if __name__ == '__main__':
    main()
