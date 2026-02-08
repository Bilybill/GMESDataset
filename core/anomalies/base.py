from dataclasses import dataclass
import numpy as np

@dataclass
class Anomaly:
    type: str
    strength: float          # Relative perturbation coefficient alpha (positive or negative)
    edge_width_m: float      # Soft boundary width (meters)

    def mask(self, X, Y, Z) -> np.ndarray:
        """Return hard mask {0,1} in grid coordinates."""
        raise NotImplementedError

    def soft_mask(self, X, Y, Z) -> np.ndarray:
        # Simple version: use hard mask first, can be upgraded to SDF+sigmoid
        m = self.mask(X, Y, Z).astype(np.float32)
        return m

    def apply_to_vp(self, vp: np.ndarray, X, Y, Z) -> np.ndarray:
        m = self.soft_mask(X, Y, Z)
        return vp * (1.0 + self.strength * m)
