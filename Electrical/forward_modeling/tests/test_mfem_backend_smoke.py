import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

FORWARD_DIR = Path(__file__).resolve().parents[1]
if str(FORWARD_DIR) not in sys.path:
    sys.path.insert(0, str(FORWARD_DIR))

import mt_forward


def test_real_mfem_backend_runs_tiny_halfspace():
    if mt_forward.mt_backend is None:
        pytest.skip("No compiled MT backend is available.")
    if mt_forward.mt_backend.__name__.split(".")[-1] != "mt_forward_mfem":
        pytest.skip("MFEM backend is not the active backend.")

    rho = torch.full((2, 2, 2), 100.0, dtype=torch.float64)
    operator = mt_forward.MTForward3D(freqs=[1.0], dx=100.0, dy=100.0, dz=100.0)
    app_res, phase = operator(rho)

    assert app_res.shape == (1, 2, 2, 2)
    assert phase.shape == (1, 2, 2, 2)
    assert torch.isfinite(app_res).all()
    assert torch.isfinite(phase).all()
    assert app_res.mean().item() > 0.0
