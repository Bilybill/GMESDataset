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
        self.freqs = None

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
        self.freqs = tuple(freqs)
        nx, ny, _ = rho.shape
        app_res = np.zeros((len(freqs), nx, ny, 2), dtype=np.float64)
        phase = np.zeros_like(app_res)
        return app_res, phase


def test_manual_frequencies_are_sorted_high_to_low(monkeypatch):
    backend = FakeMfemBackend()
    monkeypatch.setattr(mt_forward, "mt_backend", backend)

    operator = mt_forward.MTForward3D(freqs=[0.1, 10.0, 1.0])
    rho = torch.ones((2, 3, 4), dtype=torch.float32)
    operator(rho)

    assert backend.freqs == (10.0, 1.0, 0.1)
    assert operator.last_freqs == (10.0, 1.0, 0.1)
