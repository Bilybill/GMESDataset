from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from em3d.grid import Grid3D
from em3d.solver import vertical_line_precond


def main():
    grid = Grid3D(nx=4, ny=4, nz=5, dx=50.0, dy=50.0, dz=50.0, device='cpu', dtype=torch.complex128)
    sigma = torch.full(grid.shape_cells, 1e-2, dtype=grid.rdtype)
    M = vertical_line_precond(grid, sigma, omega=2.0 * 3.1415926535)
    x = torch.randn(grid.n_total, dtype=grid.dtype)
    y = M(x)
    assert y.shape == x.shape
    X = torch.randn(grid.n_total, 2, dtype=grid.dtype)
    Y = M(X)
    assert Y.shape == X.shape
    print('preconditioner shapes PASS')

if __name__ == '__main__':
    main()
