#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Igneous Intrusion (Dykes/Sills) Anomaly implementation.
Refactored to support advanced stratigraphic sills with layer control.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import math
from typing import Tuple, Optional, List, Dict, Any
from .base import Anomaly

# ------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------

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

def _rotation_matrix(strike_deg: float, dip_deg: float) -> np.ndarray:
    """Rotation matrix for planar features."""
    s = np.radians(strike_deg)
    d = np.radians(dip_deg)
    # Rotate around Z (Strike)
    cz, sz = np.cos(s), np.sin(s)
    Rz = np.array([[cz, sz, 0], [-sz, cz, 0], [0, 0, 1]])
    # Rotate around X (Dip)
    cd, sd = np.cos(d), np.sin(d)
    Rx = np.array([[1, 0, 0], [0, cd, sd], [0, -sd, cd]])
    return Rx @ Rz

def _fractal_noise_2d(u: np.ndarray, v: np.ndarray, seed: int = 42, octaves: int = 3, scale: float = 0.1) -> np.ndarray:
    """Simple 2D fractal noise."""
    rng = np.random.default_rng(seed)
    height = np.zeros_like(u)
    freq_u, freq_v = scale, scale
    amp, persistence = 1.0, 0.5
    for _ in range(octaves):
        for _ in range(3):
            theta = rng.uniform(0, 2*np.pi)
            phase = rng.uniform(0, 2*np.pi)
            kx = np.cos(theta) * freq_u
            ky = np.sin(theta) * freq_v
            height += amp * np.sin(u * kx + v * ky + phase)
        amp *= persistence
        freq_u *= 2.0
        freq_v *= 2.0
    return height

def _extract_longest_segment_top_bot(
    layer_labels: np.ndarray,
    layer_id: int,
    z_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each (x,y), find the longest contiguous segment where labels==layer_id (handles faults / repeats).
    Return z_top(x,y), z_bot(x,y), thickness(x,y), valid(x,y).
    """
    nx, ny, nz = layer_labels.shape
    # Ensure z_arr is flat
    if z_arr.ndim > 1:
        z_arr = z_arr.flatten()
    z_arr = z_arr.astype(np.float32, copy=False)

    # Boolean mask for the layer
    flat = (layer_labels.reshape(-1, nz) == int(layer_id))
    anyv = flat.any(axis=1)

    z_top = np.full((nx * ny,), np.nan, dtype=np.float32)
    z_bot = np.full((nx * ny,), np.nan, dtype=np.float32)

    cols = np.where(anyv)[0]
    for c in cols:
        m = flat[c].astype(np.int8, copy=False)
        # segment starts/ends by diff
        dm = np.diff(m)
        # starts are where 0->1 (index+1)
        starts = list(np.where(dm == 1)[0] + 1)
        # ends are where 1->0 (index)
        ends = list(np.where(dm == -1)[0])

        if m[0] == 1:
            starts = [0] + starts
        if m[-1] == 1:
            ends = ends + [nz - 1]

        if len(starts) == 0 or len(ends) == 0:
            continue

        # pair starts and ends in order; choose longest
        best_len = -1
        best_s, best_e = starts[0], ends[0]
        
        # Simple pairing assuming well-formed intervals
        # Safe logic: iterate limits
        i_s, i_e = 0, 0
        while i_s < len(starts) and i_e < len(ends):
            s = starts[i_s]
            # Find first end >= s
            while i_e < len(ends) and ends[i_e] < s:
                i_e += 1
            if i_e >= len(ends):
                break
            e = ends[i_e]
            
            L = (e - s + 1)
            if L > best_len:
                best_len = L
                best_s, best_e = s, e
            
            i_s += 1
            i_e += 1

        z_top[c] = z_arr[best_s]
        z_bot[c] = z_arr[best_e]

    z_top = z_top.reshape(nx, ny)
    z_bot = z_bot.reshape(nx, ny)
    thickness = (z_bot - z_top).astype(np.float32)
    valid = np.isfinite(z_top) & np.isfinite(z_bot) & (thickness > 0)
    return z_top, z_bot, thickness, valid

def _pick_sill_layer_id_from_labels(
    layer_labels: np.ndarray,
    dz: float,
    rng: np.random.Generator,
    min_coverage_frac: float = 0.05,
    n_try: int = 24,
) -> int:
    """
    Pick a layer id that has enough areal coverage and thickness.
    layer_labels: (nx,ny,nz) int, layer index 0..N.
    We score candidates by: coverage * median_thickness.
    """
    vals, counts = np.unique(layer_labels.astype(np.int32), return_counts=True)
    if vals.size == 0:
        return -1

    vmin, vmax = int(vals.min()), int(vals.max())
    candidates = vals[(vals > vmin) & (vals < vmax)]
    if candidates.size == 0:
        candidates = vals

    # random subset for speed
    if candidates.size > n_try:
        candidates = rng.choice(candidates, size=n_try, replace=False)

    best_k = int(candidates[0])
    best_score = -1.0

    for k in candidates:
        # Check quick coverage on a downsampled grid if huge?
        # For now, simplistic check
        mask_slice = (layer_labels == int(k))
        # 2D projection
        proj = mask_slice.any(axis=2)
        cover = proj.mean()
        
        if cover < min_coverage_frac:
            continue
            
        # Estimator of thickness: count voxels
        thick_est = (mask_slice.sum(axis=2).astype(np.float32) * float(dz))
        med = float(np.median(thick_est[proj])) if np.any(proj) else 0.0
        
        score = float(cover * (med + 1e-6))
        if score > best_score:
            best_score = score
            best_k = int(k)

    return best_k


# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------

@dataclass
class IgneousIntrusionParams:
    # Common
    kind: str = "sill" # "sill" or "dyke"
    
    # Geometry center (fallback for sill if labels not used)
    cx_m: float = 1500.0
    cy_m: float = 1500.0
    cz_m: float = 1000.0
    
    # Generic Dimensions
    length_m: float = 1000.0
    width_m: float = 1000.0
    max_thickness_m: float = 50.0
    
    # Planar (Dyke/Simple Sill)
    strike_deg: float = 0.0
    dip_deg: float = 90.0
    
    # ---- SILL (with optional stratigraphic labels) ----
    sill_layer_id: int = -1                 # >=0 means use that layer id; -1 means auto-pick
    sill_alpha: float = 0.5                 # within-layer relative position, 0..1
    sill_thick_max_frac_of_layer: float = 0.35  # t(x,y) <= frac * layer_thickness(x,y)
    sill_min_layer_thickness_m: float = 60.0    # if layer thinner than this, no sill
    
    sill_undulation_amp_m: float = 10.0
    sill_undulation_kmax: int = 3
    sill_thickness_var_frac: float = 0.2
    
    # Roughness (Fractal)
    roughness_amp_m: float = 5.0
    roughness_scale: float = 0.02
    
    edge_width_m: float = 20.0
    taper_edge_m: float = 50.0

# ------------------------------------------------------------------
# Main Class
# ------------------------------------------------------------------

@dataclass
class IgneousIntrusion(Anomaly):
    """
    Igneous Intrusion Anomaly (Dykes & Sills).
    Supports advanced stratigraphic logic if layer_labels are provided.
    """
    params: IgneousIntrusionParams = None
    layer_labels: Optional[np.ndarray] = field(default=None, repr=False)
    rng_seed: int = 42
    
    _rng: np.random.Generator = None
    
    def __post_init__(self):
        self._rng = np.random.default_rng(self.rng_seed)
        if self.params is None:
            # Create default params
            self.params = IgneousIntrusionParams()

    @staticmethod
    def create_random_params(
        grid_shape: tuple, 
        grid_spacing: tuple, 
        seed: int = 42,
        kind: str = "sill",
        layer_labels: Optional[np.ndarray] = None
    ) -> IgneousIntrusionParams:
        
        nx, ny, nz = grid_shape
        dx, dy, dz = grid_spacing
        Lx, Ly, Lz = nx*dx, ny*dy, nz*dz
        
        rng = np.random.default_rng(seed)
        p = IgneousIntrusionParams(kind=kind)
        
        # Random dimensions
        p.length_m = float(math.exp(rng.uniform(math.log(500), math.log(Lx*0.8))))
        p.width_m = float(math.exp(rng.uniform(math.log(500), math.log(Ly*0.8))))
        p.max_thickness_m = float(rng.uniform(20.0, 100.0))
        
        # Center defaults
        p.cx_m = float(rng.uniform(Lx*0.2, Lx*0.8))
        p.cy_m = float(rng.uniform(Ly*0.2, Ly*0.8))
        p.cz_m = float(rng.uniform(Lz*0.3, Lz*0.8))

        if kind == "sill":
            p.strike_deg = float(rng.uniform(0, 360))
            p.dip_deg = float(rng.uniform(0, 10)) # Gentle dip for simple sills
            
            # --------------------------
            # NEW: if layer_labels provided, pick a reasonable sill layer & center
            # --------------------------
            if layer_labels is not None:
                # choose layer id if not specified
                p.sill_layer_id = _pick_sill_layer_id_from_labels(
                    layer_labels=layer_labels,
                    dz=float(dz),
                    rng=rng,
                    min_coverage_frac=0.05
                )

                p.sill_alpha = float(rng.uniform(0.2, 0.8))
                p.sill_thick_max_frac_of_layer = float(rng.uniform(0.2, 0.45))
                p.sill_min_layer_thickness_m = float(rng.uniform(40.0, 120.0))

                # choose xc/yc from valid columns (layer exists and thick enough)
                k = int(p.sill_layer_id)
                if k >= 0:
                    mask = (layer_labels == k)
                    cover = mask.any(axis=2)  # (nx,ny)
                    thick_est = mask.sum(axis=2).astype(np.float32) * float(dz) 
                    valid_xy = cover & (thick_est >= (p.sill_min_layer_thickness_m + 2.0 * p.edge_width_m))
                    coords = np.argwhere(valid_xy)
                    
                    if coords.size > 0:
                        ii, jj = coords[rng.integers(0, coords.shape[0])]
                        p.cx_m = float(ii * dx)
                        p.cy_m = float(jj * dy)
                        # Z will be determined by layer logic
        
        elif kind == "dyke":
            p.strike_deg = float(rng.uniform(0, 360))
            p.dip_deg = float(rng.uniform(70, 90)) # Steep dip
            
        return p

    def mask(self, X, Y, Z) -> np.ndarray:
        return self.soft_mask(X, Y, Z) > 0.5
        
    def soft_mask(self, X, Y, Z) -> np.ndarray:
        # We need z_arr for stratigraphic logic
        # Assuming Z is meshgrid Z[0,0,:]
        z_arr = Z[0, 0, :].astype(np.float64)
        
        # Precompute fields (2D maps etc)
        # X and Y are 3D, we need 2D slice for precomputation
        X2d = X[:, :, 0]
        Y2d = Y[:, :, 0]
        
        pre = self._precompute_fields(self.params, X2d, Y2d, z_arr, self._rng)
        
        # Compute 3D mask
        mask = np.zeros_like(Z, dtype=np.float32)
        nz = Z.shape[2] # (nx, ny, nz)
        
        # Iterate Z slices for memory efficiency
        # Z varies along axis 2
        for iz in range(nz):
            z_val = z_arr[iz]
            sdf_slice = self._compute_sdf_slice(self.params, X2d, Y2d, z_val, iz, pre)
            
            # Soft mask sigmoid
            wd = max(self.params.edge_width_m, 1e-4)
            # Clip SDF to avoid overflow in exp, though _sigmoid_stable handles most
            sdf_slice = np.clip(sdf_slice, -1000, 1000)
            mask[:, :, iz] = _sigmoid_stable(sdf_slice / wd)
            
        return mask

    def apply_to_vp(self, vp: np.ndarray, X, Y, Z) -> np.ndarray:
        m = self.soft_mask(X, Y, Z)
        # Assuming High Velocity Intrusion
        vp_intrusion = 5500.0 
        return vp * (1.0 - m) + vp_intrusion * m

    # ----------------------------------------------------
    # Internal Logic
    # ----------------------------------------------------

    def _precompute_fields(self, p: IgneousIntrusionParams, X2d: np.ndarray, Y2d: np.ndarray, z_arr: np.ndarray, rng: np.random.Generator) -> Dict[str, Any]:
        """Precompute 2D height maps, thickness maps, etc."""
        nx, ny = X2d.shape
        pre = {}
        
        # --- Stratigraphic Sill Logic ---
        use_labels = (self.layer_labels is not None) and (p.kind == "sill") and (p.sill_layer_id >= 0)
        
        if use_labels:
            k = int(p.sill_layer_id)
            z_top, z_bot, thick, valid = _extract_longest_segment_top_bot(self.layer_labels, k, z_arr)
            
            # Reuse Fourier noise for undulation
            kmax = max(int(p.sill_undulation_kmax), 1)
            noise = np.zeros((nx, ny), dtype=np.float32)
            # Simple noise gen
            for kx in range(1, kmax + 1):
                for ky in range(1, kmax + 1):
                   a = rng.normal(0.0, 1.0)
                   phi = rng.uniform(0.0, 2*np.pi)
                   xx = X2d / max(p.length_m*2, 1e-6) # use length as scale ref
                   yy = Y2d / max(p.width_m*2, 1e-6)
                   noise += (a * np.cos(2*np.pi*(kx*xx + ky*yy) + phi)).astype(np.float32)
            noise = noise / (np.max(np.abs(noise)) + 1e-9)
            
            alpha = np.clip(p.sill_alpha, 0.0, 1.0)
            z_h = (z_top + alpha * thick + p.sill_undulation_amp_m * noise).astype(np.float32)
            
            # Thickness map
            tv = rng.normal(0.0, 1.0, size=(nx, ny)).astype(np.float32)
            t0 = p.max_thickness_m
            t_map = t0 * (1.0 + p.sill_thickness_var_frac * tv)
            t_map = np.clip(t_map, 0.25*t0, 3.0*t0)
            
            # Constraint: fraction of layer
            frac = np.clip(p.sill_thick_max_frac_of_layer, 0.05, 0.95)
            t_cap = frac * thick
            t_map = np.minimum(t_map, t_cap)
            
            # Min thickness threshold
            dz = z_arr[1] - z_arr[0] if len(z_arr) > 1 else 10.0
            margin = max(dz, 2.0 * p.edge_width_m)
            
            ok = valid & (thick >= (p.sill_min_layer_thickness_m + 2.0*margin)) & (t_map >= 2.0*dz)
            
            # Clamp z_h
            z_low = z_top + 0.5*t_map + margin
            z_high = z_bot - 0.5*t_map - margin
            
            # If constraint fails (z_low > z_high), just clip safely
            # Note: maximum(z_low, z_high) handles the inversion case
            z_h = np.clip(z_h, z_low, np.maximum(z_low, z_high))
            
            t_map[~ok] = 0.0
            
            pre["use_labels"] = True
            pre["layer_id"] = k
            pre["z_h"] = z_h
            pre["t_map"] = t_map
            pre["valid"] = ok
            
            # Planar Footprint SDF
            # Ellipse mask
            dx = X2d - p.cx_m
            dy = Y2d - p.cy_m
            # Simple rotation by strike (though sills usually irregular)
            # Let's just use simple ellipse
            r2 = (dx / (p.length_m/2))**2 + (dy / (p.width_m/2))**2
            # SDF for unit circle is 1 - r
            # Scaled SDF ~ 
            # We just need a 2D mask.
            sdf_win = (1.0 - np.sqrt(r2)) * min(p.length_m, p.width_m)/2.0
            pre["sdf_win"] = sdf_win
            
        else:
            # --- Planar Logic (Dyke or Simple Sill) ---
            pre["use_labels"] = False
            # Compute Rotation Matrix once
            pre["R"] = _rotation_matrix(p.strike_deg, p.dip_deg)
            
            # Precompute roughness on UV plane?
            # Roughness for dykes depends on U, V.
            # Since V depends on Z, we can't fully precompute 2D roughness unless we project.
            # For simplicity, we'll do roughness in the loop or use a 3D noise approximation?
            # Or just use X/Y/Z noise.
            pass
            
        return pre

    def _compute_sdf_slice(self, p: IgneousIntrusionParams, X2d: np.ndarray, Y2d: np.ndarray, z_val: float, iz: int, pre: dict) -> np.ndarray:
        
        if pre["use_labels"]:
            # --- Stratigraphic Sill ---
            z_h = pre["z_h"]
            t_map = pre["t_map"] # full 2D map
            sdf_win = pre["sdf_win"]
            
            # Vertical SDF: t/2 - |z - z_center|
            # (inside positive)
            sdf_thick = 0.5 * t_map - np.abs(z_val - z_h)
            
            # Combine with window
            sdf = np.minimum(sdf_thick, sdf_win)
            
            # Strict Gating
            if self.layer_labels is not None:
                k = pre["layer_id"]
                # layer_labels is (nx, ny, nz)
                # Check strict equality at this Z slice
                in_layer = (self.layer_labels[:, :, iz] == k)
                # Where not in layer, force -1e6 (very far outside)
                sdf = np.where(in_layer, sdf, -1e6)
                
            return sdf
            
        else:
            # --- Planar (Dyke/Simple) ---
            # Center relative
            dx = X2d - p.cx_m
            dy = Y2d - p.cy_m
            dz = z_val - p.cz_m
            
            # We need to rotate (dx, dy, dz)
            # Rot is 3x3. 
            # (u, v, w) = R @ (dx, dy, dz)
            # R is (3,3)
            # dx, dy are (nx, ny)
            R = pre["R"]
            
            u = R[0,0]*dx + R[0,1]*dy + R[0,2]*dz
            v = R[1,0]*dx + R[1,1]*dy + R[1,2]*dz
            w = R[2,0]*dx + R[2,1]*dy + R[2,2]*dz
            
            # Roughness (perturb w surface)
            if p.roughness_amp_m > 0:
                # Use global coordinates for noise stability? Or local u,v?
                # Local u,v makes texture follow the plane
                noise = _fractal_noise_2d(u, v, seed=self.rng_seed, scale=p.roughness_scale)
                w_surf = w - noise * p.roughness_amp_m
            else:
                w_surf = w
                
            # Box/Sheet SDF
            half_l = p.length_m / 2.0
            half_w = p.width_m / 2.0
            half_t = p.max_thickness_m / 2.0
            
            # Tapering
            dist_u = np.abs(u)
            dist_v = np.abs(v)
            
            def smooth_edge(d, lim, fade):
                return np.clip((lim - d)/max(fade, 1e-3), 0.0, 1.0)
            
            edge_f = smooth_edge(dist_u, half_l, p.taper_edge_m) * \
                     smooth_edge(dist_v, half_w, p.taper_edge_m)
            
            local_t = half_t * np.sqrt(edge_f)
            
            # SDF
            sdf = local_t - np.abs(w_surf)
            
            return sdf
