import torch
from em3d.grid import Grid3D, Receiver
from em3d.sources import point_electric_dipole
from em3d.forward import simulate_csem_batch, simulate_mt_batch


def main():
    device = 'cpu'
    grid = Grid3D(nx=10, ny=10, nz=8, dx=50.0, dy=50.0, dz=50.0, device=device, dtype=torch.complex128, pml_n=2)
    sigma = torch.full(grid.shape_cells, 1e-2, device=grid.device, dtype=grid.rdtype)
    sigma[4:7, 4:7, 3:5] = 5e-2

    freqs = [0.5, 2.0]
    sources = [
        point_electric_dipole(grid, 250.0, 250.0, 100.0, 'x', 1.0 + 0j),
        point_electric_dipole(grid, 300.0, 200.0, 100.0, 'y', 1.0 + 0j),
    ]
    recs = [
        Receiver(200.0, 200.0, 100.0, 'r1'),
        Receiver(300.0, 300.0, 100.0, 'r2'),
    ]

    csem, infos = simulate_csem_batch(grid, sigma, freqs, sources, recs, tol=1e-5, maxiter=100, restart=20)
    print('=== CSEM batch ===')
    for i, f in enumerate(freqs):
        for j in range(len(sources)):
            print(f'freq={f}Hz src={j} converged={infos[i][j]["converged"]} iters={infos[i][j]["iterations"]} residual={infos[i][j]["residual"]:.3e}')
            print('  amp_Ex:', csem[i][j]['amp_Ex'])
            print('  phase_Ex_deg:', csem[i][j]['phase_Ex_deg'])

    mt, mt_infos = simulate_mt_batch(grid, sigma, freqs, recs, tol=1e-5, maxiter=100, restart=20)
    print('\n=== MT-like batch ===')
    for i, f in enumerate(freqs):
        print(f'freq={f}Hz src-pol-x converged={mt_infos[i][0]["converged"]}, src-pol-y converged={mt_infos[i][1]["converged"]}')
        print('  rho_xy:', mt[i]['rho_xy'])
        print('  phi_xy_deg:', mt[i]['phi_xy_deg'])
        print('  rho_yx:', mt[i]['rho_yx'])
        print('  phi_yx_deg:', mt[i]['phi_yx_deg'])


if __name__ == '__main__':
    main()
