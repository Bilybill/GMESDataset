#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Igneous Intrusion (Dykes/Sills) Anomaly implementation.
Models sheet-like intrusions with varying strike, dip, rough surfaces, and fading edges.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import math
from typing import Tuple, Optional
from .base import Anomaly

# ------------------------------------------------------------------
# Internal Helpers
# ------------------------------------------------------------------

def _rotation_matrix(strike_deg: float, dip_deg: float) -> np.ndarray:
    """
    Construct rotation matrix to transform global (dx, dy, dz) to local (u, v, w).
    Local coords:
      u: Along strike (horizontal)
      v: Up-dip/Down-dip direction in the plane
      w: Normal to the plane
    
    Convention (Z positive down):
    1. Strike (azimuth): Rotate around Z axis (Right-hand rule).
       0 deg strike -> aligned with X? 
       Let's assume strike is angle from North (Y) clockwise or similar. 
       We'll stick to mathematical rotation:
       Rz(alpha): x' = x cos a + y sin a...
    
    Let's align local U with Strike vector.
    Let's align local W with Normal vector.
    Dip is angle from horizontal plane.
    """
    # Convert to radians
    # Adjust conventions as needed for the project. 
    # Here: Strike 0 = aligned with X axis. Dip 90 = Vertical.
    s = np.radians(strike_deg)
    d = np.radians(dip_deg)
    
    # 1. Rotate around Z by strike (align U with strike)
    cz, sz = np.cos(s), np.sin(s)
    Rz = np.array([
        [cz,  sz, 0],
        [-sz, cz, 0],
        [0,   0,  1]
    ])
    
    # 2. Rotate around X (the new U axis) by dip
    # If dip=0, plane is horizontal (normal is Z? No, normal should be Z for sill)
    # If we want w to be normal.
    # Initially (before dip rotation), local coords (u,v,w) map to (x,y,z).
    # Plane is u-v plane. Normal is w (z-axis).
    # So dip=0 means horizontal sheet.
    # Dip=90 means vertical sheet.
    
    cd, sd = np.cos(d), np.sin(d)
    Rx = np.array([
        [1, 0,   0],
        [0, cd, sd],
        [0, -sd, cd] # This rotates Y towards Z
    ])
    
    # Combined rotation: First Rz, then Rx
    # Global to Local: P_local = R * P_global
    return Rx @ Rz

def _fractal_noise_2d(u: np.ndarray, v: np.ndarray, seed: int = 42, octaves: int = 3, persistence: float = 0.5, scale: float = 0.1) -> np.ndarray:
    """
    Simple 2D fractal noise using superposition of random sines/cosines.
    Not strictly Perlin, but sufficient for rocky textures.
    """
    rng = np.random.default_rng(seed)
    height = np.zeros_like(u)
    
    freq_u = scale
    freq_v = scale
    amp = 1.0
    
    # Generate random phases/directions for a few components per octave
    for _ in range(octaves):
        # Add 3 random sine waves per octave to reduce grid artifacts
        for _ in range(3):
            theta = rng.uniform(0, 2*np.pi)
            phase = rng.uniform(0, 2*np.pi)
            
            # Direction vector
            kx = np.cos(theta) * freq_u
            ky = np.sin(theta) * freq_v
            
            height += amp * np.sin(u * kx + v * ky + phase)
            
        amp *= persistence
        freq_u *= 2.0
        freq_v *= 2.0
        
    return height

def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(x, dtype=np.float32)
    x = x.astype(np.float32, copy=False)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out

# ------------------------------------------------------------------
# Igneous Intrusion Class
# ------------------------------------------------------------------

@dataclass
class IgneousIntrusion(Anomaly):
    """
    Models Dykes (vertical/steep) and Sills (horizontal/gentle) or generic sheets.
    High density, High velocity, High magnetism.
    """
    # Geometry
    center: Tuple[float, float, float] = (0,0,0)  # Center point (x, y, z)
    strike: float = 0.0          # Azimuth degrees (0-360)
    dip: float = 90.0            # Dip degrees (0=horizontal/sill, 90=vertical/dyke)
    
    # Dimensions (Local coordinates)
    length_m: float = 1000.0     # Extent along strike (u)
    width_m: float = 1000.0      # Extent down dip (v)
    thickness_m: float = 50.0    # Thickness (w)
    
    # Texture/Roughness
    roughness_amp_m: float = 5.0 # Amplitude of surface variations
    roughness_scale: float = 0.02 # Frequency scale of variations
    
    # Physics Properties (Optional for now, but good for record)
    density_val: float = 2.9     # g/cm3 (Basalt/Gabbro)
    susceptibility: float = 0.05 # SI units
    
    # Additional
    taper_edge_m: float = 20.0   # Tapering at the edges of the sheet to void sharp cuts
    
    # Horizon control (Optional Override)
    # If provided, the sill will follow this surface instead of the planar strike/dip center.
    # The 'thickness_m' will be applied around this surface.
    horizon_depths: Optional[np.ndarray] = field(default=None, repr=False)
    horizon_geotransform: Optional[Tuple[float, float, float, float]] = None # (ox, oy, dx, dy)
    
    def mask(self, X, Y, Z) -> np.ndarray:
        return self.soft_mask(X, Y, Z) > 0.5

    def soft_mask(self, X, Y, Z) -> np.ndarray:
        # Check if horizon control is active
        if self.horizon_depths is not None:
             return self._soft_mask_horizon(X, Y, Z)
             
        # 1. Coordinate Transform Global -> Local
        cx, cy, cz = self.center
        dx = X - cx
        dy = Y - cy
        dz = Z - cz
        
        # Flatten for matrix operation
        orig_shape = X.shape
        coords = np.stack([dx.flatten(), dy.flatten(), dz.flatten()], axis=0) # (3, N)
        
        # Rotation
        R = _rotation_matrix(self.strike, self.dip)
        local_coords = R @ coords # (3, N)
        
        u = local_coords[0].reshape(orig_shape)
        v = local_coords[1].reshape(orig_shape)
        w = local_coords[2].reshape(orig_shape)
        
        # 2. Compute Roughness (perturb w coordinate or thickness)
        # We perturb the reference "middle" plane. 
        # Ideally, we want detailed rock face.
        # Check performance: if grid is huge, this might be slow.
        # But for demo 300x300x200 it's 18M points, numpy vectorized is fine.
        
        if self.roughness_amp_m > 0:
            # Generate deterministic noise based on u, v coordinates
            # To make it fast, we can use u, v directly.
            # Using a simplified noise here to avoid huge overhead
            # We'll use a few sin waves
            noise = _fractal_noise_2d(u, v, seed=int(self.center[0]), 
                                      scale=self.roughness_scale, 
                                      octaves=2, persistence=0.5)
            w_surf = w - noise * self.roughness_amp_m
        else:
            w_surf = w

        # 3. SDF for Box/Sheet
        # Box limits in u, v
        half_l = self.length_m / 2.0
        half_w = self.width_m / 2.0
        half_t = self.thickness_m / 2.0
        
        # Signed distance to the box boundaries
        # d_u = max(|u| - hl, 0)
        # We want a smooth body.
        # Let's use a "Round Box" SDF or just intersected planes approach.
        
        # Distance from thickness center (thickness is the primary constraint)
        # Positive inside, Negative outside
        
        # Tapering: We want the thickness to decrease as we get close to length/width limits
        # Normalized distance to edges
        dist_u = np.abs(u)
        dist_v = np.abs(v)
        
        # Elliptical footprint mask or Box footprint? 
        # Rocks are irregular. Let's use a soft box.
        
        # Factor 0..1 indicating how close we are to the UV center
        # f_u = smoothstep( (half_l - dist_u) / taper )
        
        def smooth_edge(dist, limit, fade):
            return np.clip((limit - dist) / max(fade, 1e-3), 0.0, 1.0)
            
        edge_f = smooth_edge(dist_u, half_l, self.taper_edge_m) * \
                 smooth_edge(dist_v, half_w, self.taper_edge_m)
        
        # Modulate thickness by edge factor (lens shape)
        local_thickness = half_t * np.power(edge_f, 0.5) # Square root for rounded edges
        
        # SDF in W direction
        # sdf > 0 inside
        sdf = local_thickness - np.abs(w_surf)
        
        # Also clip by uv bounds strictly if needed, but the thickness modulation handles it naturally
        # (thickness becomes 0 outside bounds)
        
        # 4. Soft Mask Sigmoid
        wd = max(self.edge_width_m, 1e-4)
        
        # Using simple sigmoid
        # Store as float32
        val = sdf / wd
        mask = np.zeros_like(val, dtype=np.float32)
        
        # Stable sigmoid
        pos = val >= 0
        neg = ~pos
        mask[pos] = 1.0 / (1.0 + np.exp(-val[pos]))
        e_neg = np.exp(val[neg])
        mask[neg] = e_neg / (1.0 + e_neg)
        
        return mask

    def _soft_mask_horizon(self, X, Y, Z) -> np.ndarray:
        horizon = self.horizon_depths
        nx, ny, nz = X.shape
        h_shape = horizon.shape
        
        # Match horizon to grid
        # X varies along axis 0, Y along axis 1 in (nx, ny, nz)
        if h_shape == (nx, ny):
             Z_surf = horizon[:, :, np.newaxis] # (nx, ny, 1)
        elif h_shape == (ny, nx): 
             Z_surf = horizon.T[:, :, np.newaxis]
        else:
             # Just map to whatever shape fits or fail.
             # fallback to broadcasting if possible
             # If horizon is 2D and matches X,Y projection?
             pass 
             # Let's assume user provides correct shape for now or we resize?
             # For a robust implementation we might interp, but for now:
             raise ValueError(f"Horizon shape {h_shape} doesn't match grid X/Y shape {(nx, ny)}")
        
        # Handle NaN values in horizon (e.g. pinch-outs or missing data)
        # Create a validity mask
        valid_horizon = ~np.isnan(Z_surf)
        # Fill NaNs with a safe value (e.g. 0) to avoid runtime warnings computation, 
        # but we will mask them out at the end.
        Z_surf_safe = np.where(valid_horizon, Z_surf, 0.0)

        # Add roughness
        w_surf = Z_surf_safe
        if self.roughness_amp_m > 0:
            noise = _fractal_noise_2d(X, Y, seed=int(self.center[0]), 
                                      scale=self.roughness_scale, 
                                      octaves=2)
            w_surf = w_surf + noise * self.roughness_amp_m
            
        dist_w = Z - w_surf
        
        # SDF for thickness (vertical SDF)
        half_t = self.thickness_m / 2.0
        
        # Lateral tapering (Radial distance from center)
        # Limit the sill extent if length/width are reasonable
        limit_r = min(self.length_m, self.width_m) / 2.0
        
        if limit_r < 1e6: # If finite
             cx, cy, _ = self.center
             dist_r = np.sqrt((X - cx)**2 + (Y - cy)**2)
             fade = max(self.taper_edge_m, 1e-3)
             edge_f = np.clip((limit_r - dist_r) / fade, 0.0, 1.0)
             # Modulate thickness (lens profile)
             # Using sqrt for rounded tip
             effective_half_t = half_t * np.sqrt(edge_f)
             sdf = effective_half_t - np.abs(dist_w)
        else:
             sdf = half_t - np.abs(dist_w)
        
        # Soft mask
        wd = max(self.edge_width_m, 1e-4)
        val = sdf / wd
        mask = _sigmoid_stable(val)
        
        # Apply validity mask - where horizon was NaN, mask should be 0
        # Determine valid mask for the whole 3D volume based on 2D horizon validity
        # valid_horizon is (nx, ny, 1) or (ny, nx, 1)
        # Broadcast to full Z
        if valid_horizon.ndim == 3:
             # It broadcasts automatically against `mask` (nx, ny, nz)
             mask = np.where(valid_horizon, mask, 0.0)
        
        return mask

    def apply_to_vp(self, vp: np.ndarray, X, Y, Z) -> np.ndarray:
        m = self.soft_mask(X, Y, Z)
        
        # If density/mag are needed, they would be used similarly in other processors.
        # For Vp:
        # Check if type involves relative or absolute.
        # Base implementation uses: vp * (1 + strength * m)
        # Basalt intrusions are typically much faster than sediments. 
        # If existing VP is ~2000-3000, and Basalt is 5000+.
        # A relative update is okay, but if background varies, the intrusion might vary too much.
        # Usually intrusions are homogeneous.
        # So we might prefer mixing:
        # vp_new = (1-m)*vp_bg + m * vp_intrusion
        
        # Let's decide based on whether explicit vp is provided?
        # For now, let's stick to the Base class pattern: Use `strength` or override.
        # But generic `Anomaly` doesn't have `vp_val`.
        # I will use a mix approach if strength is very large (> 1.0) often implies significant change.
        # Better: let's look at `strength`. If it's a relative perturbation (e.g. 0.5 = +50%), we use that.
        # But if we want fixed velocity (like SaltDome), we could add a `velocity` param.
        
        # Let's add a fixed velocity override if it's set in this class method
        # If user passes strength, we use perturbation.
        
        # Actually, let's use the standard base method but maybe allow an absolute mode?
        # For consistency with the requested "Intrusion" nature,
        # I'll implement a hybrid: If `strength` is > 10, treat as absolute velocity? No, that's hacky.
        
        # I'll stick to relative for now, as that's what `Ellipsoid` does.
        # Unless I add `constant_vp` to the dataclass.
        # Let's assumes intrusions are constant Vp ~4500-6000 m/s.
        # I'll add a constant VP param.
        
        target_vp = 5500.0 # Default fallback
        # If we have a constant property, we use it. 
        # But dataclass fields must probably be declared in __init__.
        # I will just use the mix logic here with a hardcoded high Vp or a param if I add it.
        # I'll define a default high Vp for intrusions if not relative.
        
        # Let's use a "velocity" param if we want. 
        # For now, I'll rely on the base class behavior unless I want to enforce constant Vp.
        # Given "High Vp" description, constant is better.
        
        # Let's act like SaltDome:
        intrusion_vp = 5500.0 # Typical basalt
        
        # We can reuse the mix logic:
        return vp * (1.0 - m) + intrusion_vp * m

