import numpy as np
from .anomalies.base import Anomaly

class DatasetBuilder:
    def __init__(self, dx, dy, dz):
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def make_grid(self, shape_xyz):
        nx, ny, nz = shape_xyz
        # Create coordinates arrays
        # Assuming origin at (0,0,0) for simplicity, can be adjustable
        x = np.arange(nx) * self.dx
        y = np.arange(ny) * self.dy
        z = np.arange(nz) * self.dz
        # Meshgrid: indexing='ij' gives (nx, ny, nz) order if we pass x, y, z
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        return X, Y, Z

    def inject_anomalies(self, vp_bg: np.ndarray, anomalies: list[Anomaly]):
        """
        Inject anomalies into the background Vp model.
        Returns:
            vp: The perturbed velocity model.
            label: The anomaly label mask (0 for background).
            X, Y, Z: The coordinate grids used.
        """
        X, Y, Z = self.make_grid(vp_bg.shape)
        vp = vp_bg.copy()
        label = np.zeros_like(vp_bg, dtype=np.int16)  # 0=background
        
        for k, anom in enumerate(anomalies, start=1):
            m = anom.soft_mask(X, Y, Z)
            # Apply perturbation
            # vp = vp * (1.0 + anom.strength * m)
            vp = anom.apply_to_vp(vp, X, Y, Z)
            
            # Update label
            # If overlap, this simple logic overwrites with the latest anomaly
            # Use 0.5 threshold for soft mask to define label boundary
            label[m > 0.5] = k
            
        return vp, label, X, Y, Z
