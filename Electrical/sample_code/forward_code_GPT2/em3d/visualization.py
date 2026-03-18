from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_survey_layout(rx_locations, out_path=None, title='MT survey layout'):
    rx = np.asarray(rx_locations, dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.0))
    ax.scatter(rx[:,0], rx[:,1], marker='^')
    for i, (x, y, _) in enumerate(rx):
        ax.text(x, y, f'{i}', fontsize=8, ha='left', va='bottom')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches='tight')
    return fig, ax


def plot_mt_curves(data, location_xy, out_path=None, components=('xy','yx'), title='MT curves'):
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True)
    ax_r, ax_p = axes
    ax_r.set_xscale('log'); ax_r.set_yscale('log')
    ax_p.set_xscale('log')
    ax_r.invert_xaxis(); ax_p.invert_xaxis()
    data.plot_app_res(location_xy, components=components, ax=ax_r, errorbars=False)
    data.plot_app_phs(location_xy, components=components, ax=ax_p, errorbars=False)
    ax_r.set_ylabel('Apparent resistivity [Ohm m]')
    ax_p.set_ylabel('Phase [deg]')
    ax_p.set_xlabel('Frequency [Hz]')
    ax_r.legend(); ax_p.legend()
    fig.suptitle(title)
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches='tight')
    return fig, axes
