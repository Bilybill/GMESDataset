import torch
from em3d.grid import Grid3D, Receiver
from em3d.sources import point_electric_dipole
from em3d.forward import simulate_csem_batch, simulate_mt_batch


def main():
    grid = Grid3D(6, 6, 5, 50.0, 50.0, 50.0, device='cpu', dtype=torch.complex128, pml_n=2)
    sigma = torch.full(grid.shape_cells, 1e-2, device=grid.device, dtype=grid.rdtype)
    freqs = [1.0, 3.0]
    srcs = [
        point_electric_dipole(grid, 100.0, 100.0, 50.0, 'x'),
        point_electric_dipole(grid, 150.0, 150.0, 50.0, 'y'),
    ]
    recs = [Receiver(100.0, 100.0, 50.0), Receiver(150.0, 150.0, 50.0)]
    csem, infos = simulate_csem_batch(grid, sigma, freqs, srcs, recs, tol=1e-4, maxiter=50, restart=10)
    assert len(csem) == 2 and len(csem[0]) == 2
    assert csem[0][0]['Ex'].shape[0] == 2
    mt, _ = simulate_mt_batch(grid, sigma, freqs, recs, tol=1e-4, maxiter=50, restart=10)
    assert len(mt) == 2
    assert mt[0]['Zxy'].shape[0] == 2
    print('batch shape test passed')


if __name__ == '__main__':
    main()
