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

    def __init__(self):
        self.last_call = None

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
        assert rho.dtype == np.float64
        assert rho.flags["C_CONTIGUOUS"]
        self.last_call = {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "freqs": tuple(freqs),
            "npad_xy": npad_xy,
            "npad_z": npad_z,
            "alpha": alpha,
            "use_partial_assembly": use_partial_assembly,
            "rel_tol": rel_tol,
            "max_iter": max_iter,
            "verbose": verbose,
            "device": device,
        }
        nx, ny, _ = rho.shape
        app_res = np.full((len(freqs), nx, ny, 2), 123.0, dtype=np.float64)
        phase = np.full_like(app_res, 45.0)
        return app_res, phase


def test_forward_api_shape_dtype_and_device(monkeypatch):
    backend = FakeMfemBackend()
    monkeypatch.setattr(mt_forward, "mt_backend", backend)

    rho = torch.ones((3, 4, 2), dtype=torch.float32)
    operator = mt_forward.MTForward3D(
        freqs=[2.0, 1.0],
        dx=50.0,
        dy=50.0,
        dz=25.0,
        device="cpu",
        use_partial_assembly=True,
        npad_xy=6,
        npad_z=4,
        alpha=1.2,
        rel_tol=1e-5,
        max_iter=321,
        verbose=True,
    )
    app_res, phase = operator(rho)

    assert app_res.shape == (2, 3, 4, 2)
    assert phase.shape == (2, 3, 4, 2)
    assert app_res.dtype == torch.float64
    assert phase.dtype == torch.float64
    assert app_res.device == rho.device
    assert phase.device == rho.device
    assert torch.all(app_res == 123.0)
    assert torch.all(phase == 45.0)
    assert backend.last_call == {
        "dx": 50.0,
        "dy": 50.0,
        "dz": 25.0,
        "freqs": (2.0, 1.0),
        "npad_xy": 6,
        "npad_z": 4,
        "alpha": 1.2,
        "use_partial_assembly": True,
        "rel_tol": 1e-5,
        "max_iter": 321,
        "verbose": True,
        "device": "cpu",
    }


def test_ceed_device_requires_partial_assembly():
    with pytest.raises(ValueError, match="use_partial_assembly=True"):
        mt_forward.MTForward3D(
            freqs=[1.0],
            device="ceed-cuda:/gpu/cuda/shared",
            use_partial_assembly=False,
        )
