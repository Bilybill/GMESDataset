import torch
from .base import BaseForwardSolver

class GravityForwardSolver(BaseForwardSolver):
    """
    Gravity Forward Modeling Solver (vertical component gz).
    """
    def __init__(self, dx, dy, dz, heights_m, obs_conf, G=6.67430e-11, output_unit="mgal", pad_factor=2, density_unit="kg/m3"):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.heights_m = heights_m
        self.obs_conf = obs_conf
        self.G = G
        self.output_unit = output_unit
        self.pad_factor = pad_factor
        self.density_unit = density_unit

    def forward(self, density: torch.Tensor):
        """
        Forward gravity gz.
        Args:
            density: 3D contiguous tensor of shape (nx, ny, nz)
        """
        from Gravity.forward_modeling.gra_forward import forward_gravity_gz
        # The gra_forward expects density, dx, dy, dz
        data, meta = forward_gravity_gz(
            density,
            self.dx, self.dy, self.dz,
            self.heights_m,
            self.obs_conf,
            G=self.G,
            output_unit=self.output_unit,
            pad_factor=self.pad_factor,
            density_unit=self.density_unit
        )
        return data, meta
