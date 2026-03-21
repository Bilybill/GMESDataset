import torch
from .base import BaseForwardSolver
from Electrical.forward_modeling.mt_forward import MTForward3D

class ElectricalForwardSolver(BaseForwardSolver):
    """
    Electrical Forward Modeling Solver (MT 3D).
    """
    def __init__(self, freqs, dx, dy, dz):
        super().__init__()
        self.solver = MTForward3D(freqs, dx, dy, dz)

    def forward(self, rho_tensor: torch.Tensor):
        """
        Forward MT 3D.
        Args:
            rho_tensor: 3D contiguous tensor of shape (NX, NY, NZ) with resistivity values (Ohm-m).
        Returns:
            app_res: Apparent resistivity tensor of shape (n_freqs, NY, NX, 2)
            phase: Phase tensor of shape (n_freqs, NY, NX, 2)
        """
        return self.solver(rho_tensor)
