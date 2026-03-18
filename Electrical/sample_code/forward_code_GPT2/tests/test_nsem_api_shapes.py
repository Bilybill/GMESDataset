import numpy as np
import torch
from em3d import Grid3D, NSEM


def test_nsem_api_shapes():
    grid = Grid3D(8, 8, 8, 100.0, 100.0, 50.0, device='cpu', dtype=torch.complex128, pml_n=2)
    sig = torch.full(grid.shape_cells, 1e-2, dtype=torch.complex128)
    sig[:, :, :1] = 1e-8
    sigBG = sig.clone()
    rx_loc = np.array([[300.0, 300.0, grid.dz], [500.0, 500.0, grid.dz]])
    rxs = [
        NSEM.Rx.Impedance(rx_loc, 'xy', 'real'),
        NSEM.Rx.Impedance(rx_loc, 'xy', 'imag'),
        NSEM.Rx.Tipper(rx_loc, 'zx', 'real'),
        NSEM.Rx.Tipper(rx_loc, 'zx', 'imag'),
    ]
    srcs = [NSEM.Src.PlanewaveXYPrimary(rxs, f) for f in [10.0, 1.0]]
    survey = NSEM.Survey(srcs)
    sim = NSEM.Simulation3DPrimarySecondary(grid, survey, sig, sigBG, maxiter=20, restart=8, use_block_secondary=False, precond='jacobi')
    d = sim.dpred().detach().cpu()
    assert d.numel() == survey.nD
    data = NSEM.Data(survey, d)
    assert data.numpy().shape[0] == survey.nD
