import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

FORWARD_DIR = Path(__file__).resolve().parents[1]
if str(FORWARD_DIR) not in sys.path:
    sys.path.insert(0, str(FORWARD_DIR))

import mt_forward


class FakeMfemBackend:
    __name__ = "mt_forward_mfem"

    def compute_mt_3d(self, rho, dx, dy, dz, freqs):
        assert rho.dtype == np.float64
        assert rho.flags["C_CONTIGUOUS"]
        nx, ny, _ = rho.shape
        app_res = np.full((len(freqs), nx, ny, 2), 123.0, dtype=np.float64)
        phase = np.full_like(app_res, 45.0)
        return app_res, phase


def test_forward_api_shape_dtype_and_device(monkeypatch):
    monkeypatch.setattr(mt_forward, "mt_backend", FakeMfemBackend())

    rho = torch.ones((3, 4, 2), dtype=torch.float32)
    operator = mt_forward.MTForward3D(freqs=[2.0, 1.0], dx=50.0, dy=50.0, dz=25.0)
    app_res, phase = operator(rho)

    assert app_res.shape == (2, 3, 4, 2)
    assert phase.shape == (2, 3, 4, 2)
    assert app_res.dtype == torch.float64
    assert phase.dtype == torch.float64
    assert app_res.device == rho.device
    assert phase.device == rho.device
    assert torch.all(app_res == 123.0)
    assert torch.all(phase == 45.0)
