import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

FORWARD_DIR = Path(__file__).resolve().parents[1]
if str(FORWARD_DIR) not in sys.path:
    sys.path.insert(0, str(FORWARD_DIR))

import mt_forward


class HalfspaceSmokeBackend:
    __name__ = "mt_forward_mfem"

    def compute_mt_3d(
        self,
        rho,
        dx,
        dy,
        dz,
        freqs,
        npad_xy=10,
        npad_z=10,
        alpha=1.4,
        use_partial_assembly=False,
        rel_tol=1e-6,
        max_iter=2000,
        verbose=False,
        device="cpu",
    ):
        nx, ny, _ = rho.shape
        app_res = np.empty((len(freqs), nx, ny, 2), dtype=np.float64)
        phase = np.empty_like(app_res)
        app_res[...] = float(rho.mean())
        phase[..., 0] = 45.0
        phase[..., 1] = -45.0
        return app_res, phase


def test_halfspace_smoke_keeps_public_contract(monkeypatch):
    monkeypatch.setattr(mt_forward, "mt_backend", HalfspaceSmokeBackend())

    rho = torch.full((4, 5, 3), 100.0)
    operator = mt_forward.MTForward3D(freqs=[8.0, 1.0], dx=100.0, dy=100.0, dz=100.0)
    app_res, phase = operator(rho)

    assert tuple(app_res.shape) == (2, 4, 5, 2)
    assert tuple(phase.shape) == (2, 4, 5, 2)
    assert torch.allclose(app_res, torch.full_like(app_res, 100.0))
    assert torch.all(phase[..., 0] == 45.0)
    assert torch.all(phase[..., 1] == -45.0)
