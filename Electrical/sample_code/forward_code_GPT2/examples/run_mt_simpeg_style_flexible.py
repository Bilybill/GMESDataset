import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from em3d import Grid3D, NSEM, plot_survey_layout, plot_mt_curves


def run(plotIt=True, save_outputs=True, outdir=None):
    if outdir is None:
        outdir = ROOT / 'examples' / 'outputs_mt_flexible_demo'
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    grid = Grid3D(6, 6, 6, 100.0, 100.0, 50.0, device='cpu', dtype=torch.complex128, pml_n=2)

    sig = torch.full(grid.shape_cells, 1e-2, dtype=grid.rdtype, device=grid.device)
    sig[:, :, :2] = 1e-8
    sig[2:4, 2:4, 2:4] = 1.0
    sig[:, :, -2:] = 1e-1

    sig_bg = torch.full_like(sig, 1e-2)
    sig_bg[:, :, :2] = 1e-8

    xs, ys = np.meshgrid(np.arange(200.0, 401.0, 200.0), np.arange(200.0, 401.0, 200.0))
    rx_loc = np.c_[xs.reshape(-1), ys.reshape(-1), np.full(xs.size, grid.dz)]

    receiver_list = []
    for ori in ['xy', 'yx']:
        receiver_list.append(NSEM.Rx.Impedance(rx_loc, orientation=ori, component='real'))
        receiver_list.append(NSEM.Rx.Impedance(rx_loc, orientation=ori, component='imag'))
    for ori in ['zx', 'zy']:
        receiver_list.append(NSEM.Rx.Tipper(rx_loc, orientation=ori, component='real'))
        receiver_list.append(NSEM.Rx.Tipper(rx_loc, orientation=ori, component='imag'))

    freqs = np.array([0.1, 1.0, 10.0])
    source_list = [NSEM.Src.PlanewaveXYPrimary(receiver_list, float(f)) for f in freqs]
    survey = NSEM.Survey(source_list)

    sim = NSEM.Simulation3DPrimarySecondary(
        grid, survey=survey, sigma=sig, sigmaPrimary=sig_bg,
        forward_only=True, maxiter=20, restart=8,
        use_block_secondary=False, precond='flex'
    )

    data = NSEM.Data(survey=survey, dobs=sim.dpred().detach().cpu())
    data.relative_error = 0.05
    data.noise_floor = 0.0

    print('nD =', survey.nD)
    print('first 10 dobs =', data.numpy()[:10])
    if sim._last_infos is not None:
        print('first frequency secondary solver info =', sim._last_infos[0]['secondary'])

    center_loc = np.array([400.0, 400.0])

    # Save structured numeric outputs
    metadata = {
        'frequencies_hz': freqs.tolist(),
        'receiver_locations_m': rx_loc.tolist(),
        'nD': int(survey.nD),
        'precond': 'flex',
    }
    xy_freqs, xy_z = data.get_complex_series(center_loc, orientation='xy', rx_type='impedance')
    yx_freqs, yx_z = data.get_complex_series(center_loc, orientation='yx', rx_type='impedance')
    zx_freqs, zx_t = data.get_complex_series(center_loc, orientation='zx', rx_type='tipper')
    zy_freqs, zy_t = data.get_complex_series(center_loc, orientation='zy', rx_type='tipper')

    if save_outputs:
        np.save(outdir / 'predicted_data.npy', data.numpy())
        np.save(outdir / 'receiver_locations.npy', rx_loc)
        np.savez(
            outdir / 'central_station_mt_curves.npz',
            xy_freqs=xy_freqs, xy_z=xy_z,
            yx_freqs=yx_freqs, yx_z=yx_z,
            zx_freqs=zx_freqs, zx_t=zx_t,
            zy_freqs=zy_freqs, zy_t=zy_t,
        )
        with open(outdir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    figs = {}
    if save_outputs or plotIt:
        fig1, _ = plot_survey_layout(rx_loc, out_path=(outdir / 'survey_layout.png') if save_outputs else None,
                                     title='MT survey layout')
        figs['survey_layout'] = fig1
        fig2, _ = plot_mt_curves(data, center_loc, out_path=(outdir / 'central_station_rho_phase.png') if save_outputs else None,
                                 components=('xy', 'yx'), title='Central station MT curves')
        figs['rho_phase'] = fig2

    if plotIt:
        plt.show()
    else:
        for fig in figs.values():
            plt.close(fig)
    return data, outdir


if __name__ == '__main__':
    run(plotIt=True, save_outputs=True)
