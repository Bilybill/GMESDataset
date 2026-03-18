import numpy as np
import torch
import matplotlib.pyplot as plt

from em3d import Grid3D, NSEM


def run(plotIt=True):
    grid = Grid3D(16, 16, 16, 100.0, 100.0, 50.0, device='cpu', dtype=torch.complex128, pml_n=4)

    sig = torch.full(grid.shape_cells, 1e-2, dtype=grid.rdtype, device=grid.device)
    sig[:, :, :2] = 1e-8  # air-ish
    sig[6:10, 6:10, 6:10] = 1.0
    sig[:, :, -4:] = 1e-1

    sigBG = torch.full_like(sig, 1e-2)
    sigBG[:, :, :2] = 1e-8

    xs, ys = np.meshgrid(np.arange(300.0, 1301.0, 200.0), np.arange(300.0, 1301.0, 200.0))
    rx_loc = np.c_[xs.reshape(-1), ys.reshape(-1), np.full(xs.size, grid.dz)]

    receiver_list = []
    for ori in ['xx', 'xy', 'yx', 'yy']:
        receiver_list.append(NSEM.Rx.Impedance(rx_loc, orientation=ori, component='real'))
        receiver_list.append(NSEM.Rx.Impedance(rx_loc, orientation=ori, component='imag'))
    for ori in ['zx', 'zy']:
        receiver_list.append(NSEM.Rx.Tipper(rx_loc, orientation=ori, component='real'))
        receiver_list.append(NSEM.Rx.Tipper(rx_loc, orientation=ori, component='imag'))

    freqs = np.logspace(2, -1, 5)
    source_list = [NSEM.Src.PlanewaveXYPrimary(receiver_list, float(f)) for f in freqs]
    survey = NSEM.Survey(source_list)

    sim = NSEM.Simulation3DPrimarySecondary(
        grid, survey=survey, sigma=sig.to(torch.complex128), sigmaPrimary=sigBG.to(torch.complex128),
        forward_only=True, maxiter=60, restart=12, use_block_secondary=False, precond='jacobi'
    )

    data = NSEM.Data(survey=survey, dobs=sim.dpred().detach().cpu())
    data.relative_error = 0.1
    data.noise_floor = 0.0

    print('nD =', survey.nD)
    print('first 10 dobs =', data.numpy()[:10])

    if plotIt:
        fig, axes = plt.subplots(2, 1, figsize=(7, 5))
        plt.subplots_adjust(right=0.8)
        [(ax.invert_xaxis(), ax.set_xscale('log')) for ax in axes]
        ax_r, ax_p = axes
        ax_r.set_yscale('log')
        ax_r.set_ylabel('Apparent resistivity [xy-yx]')
        ax_p.set_ylabel('Apparent phase')
        ax_p.set_xlabel('Frequency [Hz]')
        loc = np.array([700.0, 700.0])
        data.plot_app_res(loc, components=['xy', 'yx'], ax=ax_r, errorbars=False)
        data.plot_app_phs(loc, components=['xy', 'yx'], ax=ax_p, errorbars=False)
        ax_r.legend()
        ax_p.legend()
        plt.tight_layout()
    return data


if __name__ == '__main__':
    d = run(plotIt=True)
    plt.show()
