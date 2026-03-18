import torch
import torch.nn as nn
import mt_forward_cuda

class MTForward3D(nn.Module):
    def __init__(self, freqs, dx, dy, dz):
        """
        PyTorch wrapper for CUDA MT 3D Forward Modeling.
        Args:
            freqs: List of frequencies (Hz)
            dx: Cell width in x-direction (m)
            dy: Cell width in y-direction (m)
            dz: Cell width in z-direction (m)
        """
        super(MTForward3D, self).__init__()
        self.freqs = freqs
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def forward(self, rho_tensor):
        """
        Forward MT 3D.
        Args:
            rho_tensor: 3D contiguous tensor of shape (NX, NY, NZ) with resistivity values (Ohm-m).
        Returns:
            app_res: Apparent resistivity tensor of shape (n_freqs, NY, NX, 2)
            phase: Phase tensor of shape (n_freqs, NY, NX, 2)
        """
        # Ensure tensor is CPU, double, and contiguous
        rho_cpu = rho_tensor.detach().cpu().to(torch.float64).contiguous()
        app_res, phase = mt_forward_cuda.compute_mt_3d(rho_cpu, self.dx, self.dy, self.dz, self.freqs)
        return app_res, phase
