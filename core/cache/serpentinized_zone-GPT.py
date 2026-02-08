#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serpentinized Zone Anomaly (Serpentinization / Alteration Corridor)

Geologic idea (why it's "non-traditional" coupling):
- Serpentinization hydrates ultramafic rocks (peridotite) via water-rock reactions.
- Density and Vp typically decrease (more porous / altered mineralogy, lower elastic moduli),
  while magnetic susceptibility can increase due to formation of secondary magnetite.
- This breaks the common "higher Vp ↔ higher density ↔ higher chi" intuition, which is
  excellent for testing joint inversion robustness.

This anomaly can be generated in two ways:
1) Layer-anchored corridor (recommended if you have layer_labels): attach the altered band
   to a target layer_id (or auto-pick), inheriting faults implicitly from labels.
2) Geometric corridor: a curvilinear ribbon/sheet controlled by a 2D centerline spline
   and a depth function.

Outputs:
- soft_mask(): continuous [0,1] mask (core)
- mask(): binary mask (>0.5)
- apply_to_vp(): modifies Vp
- build_property_models(): creates density (g/cc) and susceptibility (SI) volumes plus subtype labels
  (you can integrate these into your multi-physics pipeline like SBI did).

Conventions:
- Coordinate arrays X,Y,Z are meters on a regular grid shaped (nx,ny,nz).
- layer_labels is optional, shape (nx,ny,nz), int layer ids 0..N (faults cause offsets).

Author: GMESDataset synthetic anomaly module
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np

# If your project has a common Anomaly base, keep this import.
# Otherwise, the class still works as a plain python object for masking/applying.
try:
    from .base import Anomaly
except Exception:  # pragma: no cover
    class Anomaly:  # minimal fallback
        def mask(self, X, Y, Z): raise NotImplementedError
        def soft_mask(self, X, Y, Z): raise NotImplementedError
        def apply_to_vp(self, vp, X, Y, Z): raise NotImplementedError


# -------------------------
# Utilities
# -------------------------

def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out

def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def _catmull_rom_1d(xk: np.ndarray, yk: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Catmull-Rom spline without SciPy. xk must be sorted ascending."""
    xk = np.asarray(xk, dtype=np.float64)
    yk = np.asarray(yk, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = len(xk)
    if n < 4:
        return np.interp(x, xk, yk).astype(np.float32)

    idx = np.searchsorted(xk, x) - 1
    idx = np.clip(idx, 0, n - 2)

    i1 = idx
    i2 = idx + 1
    i0 = np.clip(i1 - 1, 0, n - 1)
    i3 = np.clip(i2 + 1, 0, n - 1)

    x1 = xk[i1]
    x2 = xk[i2]
    denom = np.maximum(x2 - x1, 1e-9)
    t = (x - x1) / denom
    t = np.clip(t, 0.0, 1.0)

    p0 = yk[i0]
    p1 = yk[i1]
    p2 = yk[i2]
    p3 = yk[i3]

    t2 = t * t
    t3 = t2 * t
    y = 0.5 * (
        (2.0 * p1) +
        (-p0 + p2) * t +
        (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
        (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )
    return y.astype(np.float32)

def _extract_longest_segment_top_bot(layer_labels: np.ndarray, layer_id: int, z_arr: np.ndarray):
    """
    For each (x,y), find top/bottom of the longest contiguous segment where labels==layer_id.
    Returns z_top, z_bot, thickness, valid (all shape nx,ny).
    """
    nx, ny, nz = layer_labels.shape
    z_arr = z_arr.astype(np.float32, copy=False)

    flat = (layer_labels.reshape(-1, nz) == int(layer_id))
    anyv = flat.any(axis=1)

    z_top = np.full((nx * ny,), np.nan, dtype=np.float32)
    z_bot = np.full((nx * ny,), np.nan, dtype=np.float32)

    cols = np.where(anyv)[0]
    for c in cols:
        m = flat[c].astype(np.int8, copy=False)
        dm = np.diff(m)
        starts = list(np.where(dm == 1)[0] + 1)
        ends = list(np.where(dm == -1)[0])

        if m[0] == 1:
            starts = [0] + starts
        if m[-1] == 1:
            ends = ends + [nz - 1]

        if len(starts) == 0 or len(ends) == 0:
            continue

        best_len = -1
        best_s, best_e = starts[0], ends[0]
        i_s, i_e = 0, 0
        while i_s < len(starts) and i_e < len(ends):
            s = starts[i_s]
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

def _auto_pick_layer_id(layer_labels: np.ndarray, rng: np.random.Generator, prefer_mid: bool = True) -> int:
    vals = np.unique(layer_labels.astype(np.int32))
    if vals.size == 0:
        return -1
    vmin, vmax = int(vals.min()), int(vals.max())
    if prefer_mid and vmax - vmin >= 3:
        # avoid extreme top/bottom
        candidates = np.arange(vmin + 1, vmax, dtype=np.int32)
        return int(rng.choice(candidates))
    return int(rng.choice(vals))


# -------------------------
# Params
# -------------------------

@dataclass
class SerpentinizedZoneParams:
    # Geometry / mode
    mode: str = "corridor"  # "corridor" (ribbon/sheet) | "patchy" (blobs along corridor)
    use_layer_labels: bool = True
    layer_id: int = -1          # >=0 fixed; -1 auto-pick (if labels); otherwise ignored

    # Corridor footprint (XY)
    center_x_m: float = 1300.0
    center_y_m: float = 1300.0
    corridor_length_m: float = 1800.0
    corridor_halfwidth_m: float = 250.0  # lateral half-width (meters)
    corridor_irregularity: float = 0.25  # 0..0.6 (extra lobes)

    # Depth control
    z_center_m: float = 3500.0      # used when not label-anchored
    thickness_m: float = 350.0      # vertical thickness of altered band
    undulation_amp_m: float = 60.0
    undulation_kmax: int = 3

    # Patchiness (for mode="patchy")
    patch_count: int = 6
    patch_radius_m: float = 220.0
    patch_radius_var: float = 0.35

    # Soft edges
    edge_width_m: float = 60.0
    halo_thickness_m: float = 220.0  # alteration halo around core (labels and properties)

    # Property deltas (core)
    # Typical: Vp drop from ~8 km/s to ~5 km/s in mantle contexts
    vp_delta_frac: float = -0.25      # multiply background Vp by (1 + delta) inside core
    rho_delta_frac: float = -0.12     # density decrease
    chi_add_SI: float = 0.02          # susceptibility increase (absolute add)

    # Background property defaults for build_property_models (if not provided)
    rho_bg_gcc: float = 2.70
    chi_bg_SI: float = 0.001

    # Subtype codes (match your demo style; pick unused range)
    subtype_core: int = 40
    subtype_halo: int = 41

    rng_seed: int = 123


# -------------------------
# Main anomaly
# -------------------------

@dataclass
class SerpentinizedZone(Anomaly):
    type: str = "serpentinized"
    params: SerpentinizedZoneParams = field(default_factory=SerpentinizedZoneParams)
    layer_labels: Optional[np.ndarray] = field(default=None, repr=False)

    _rng: np.random.Generator = field(default=None, init=False, repr=False)
    _pre: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(int(self.params.rng_seed))

    # ---- public ----
    def mask(self, X, Y, Z) -> np.ndarray:
        return self.soft_mask(X, Y, Z) > 0.5

    def soft_mask(self, X, Y, Z) -> np.ndarray:
        _, m, _ = self._compute_full(vp_bg=None, X=X, Y=Y, Z=Z, only_mask=True)
        return m

    def apply_to_vp(self, vp: np.ndarray, X, Y, Z) -> np.ndarray:
        vp_new, _, _ = self._compute_full(vp_bg=vp, X=X, Y=Y, Z=Z, only_mask=False)
        return vp_new

    def build_property_models(
        self,
        X, Y, Z,
        vp_bg: Optional[np.ndarray] = None,
        rho_bg_gcc: Optional[np.ndarray] = None,
        chi_bg_SI: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Create density and susceptibility volumes and a subtype label volume.
        If vp_bg is provided, the same core mask can also be used to output a modified vp (optional).
        """
        p = self.params
        nx, ny, nz = Z.shape

        if rho_bg_gcc is None:
            rho_bg = np.full((nx, ny, nz), float(p.rho_bg_gcc), dtype=np.float32)
        else:
            rho_bg = rho_bg_gcc.astype(np.float32, copy=False)

        if chi_bg_SI is None:
            chi_bg = np.full((nx, ny, nz), float(p.chi_bg_SI), dtype=np.float32)
        else:
            chi_bg = chi_bg_SI.astype(np.float32, copy=False)

        # Get core mask + halo mask
        _, m_core, m_halo = self._compute_full(vp_bg=None, X=X, Y=Y, Z=Z, only_mask=True, need_halo=True)

        rho = rho_bg * (1.0 + float(p.rho_delta_frac) * m_core)
        # halo: weaker density reduction
        if m_halo is not None and float(p.halo_thickness_m) > 0:
            rho = rho * (1.0 + 0.35 * float(p.rho_delta_frac) * m_halo)

        chi = chi_bg + float(p.chi_add_SI) * m_core
        if m_halo is not None and float(p.halo_thickness_m) > 0:
            chi = chi + 0.35 * float(p.chi_add_SI) * m_halo

        sub = np.zeros((nx, ny, nz), dtype=np.int32)
        sub[m_halo > 0.5] = int(p.subtype_halo)
        sub[m_core > 0.5] = int(p.subtype_core)

        out = {
            "rho_gcc": rho.astype(np.float32),
            "chi_SI": chi.astype(np.float32),
            "subtype": sub,
            "mask_core": m_core.astype(np.float32),
            "mask_halo": m_halo.astype(np.float32),
        }
        if vp_bg is not None:
            vp_mod = vp_bg.astype(np.float32, copy=False) * (1.0 + float(p.vp_delta_frac) * m_core)
            if m_halo is not None and float(p.halo_thickness_m) > 0:
                vp_mod = vp_mod * (1.0 + 0.35 * float(p.vp_delta_frac) * m_halo)
            out["vp_mps"] = vp_mod.astype(np.float32)
        return out

    # ---- internal ----
    def _compute_full(self, vp_bg, X, Y, Z, only_mask: bool, need_halo: bool = False):
        p = self.params
        rng = self._rng
        nx, ny, nz = Z.shape

        # Regular grid assumption: X[:,0,0], Y[0,:,0], Z[0,0,:]
        x_arr = X[:, 0, 0].astype(np.float32, copy=False)
        y_arr = Y[0, :, 0].astype(np.float32, copy=False)
        z_arr = Z[0, 0, :].astype(np.float32, copy=False)
        X2d = x_arr[:, None]
        Y2d = y_arr[None, :]

        # Precompute once
        if not self._pre:
            self._pre = self._precompute(p, X2d, Y2d, z_arr, rng)

        wd = max(float(p.edge_width_m), 1e-3)

        vp_new = None if (only_mask or vp_bg is None) else np.empty_like(vp_bg, dtype=np.float32)
        m_core_vol = np.empty((nx, ny, nz), dtype=np.float32)
        m_halo_vol = np.zeros((nx, ny, nz), dtype=np.float32) if need_halo else None

        for iz in range(nz):
            z_m = float(z_arr[iz])
            sdf_core = self._sdf_core_slice(p, X2d, Y2d, z_m, iz, self._pre)
            sdf_core = np.clip(sdf_core, -1e6, 1e6)
            m_core = _sigmoid_stable(sdf_core / wd)
            m_core_vol[..., iz] = m_core

            if need_halo and float(p.halo_thickness_m) > 0:
                t = float(p.halo_thickness_m)
                # halo ring outside core: 0 < distance < t
                # using sdf: inside core sdf>0, outside sdf<0. We want a band outside: -t < sdf < 0.
                m_out = _sigmoid_stable(-sdf_core / wd)         # outside selector
                m_near = _sigmoid_stable((sdf_core + t) / wd)   # within t selector
                m_halo = m_out * m_near
                m_halo_vol[..., iz] = m_halo

            if vp_new is not None:
                vp_slice = vp_bg[..., iz].astype(np.float32, copy=False)
                vp_slice = vp_slice * (1.0 + float(p.vp_delta_frac) * m_core)
                if need_halo and float(p.halo_thickness_m) > 0:
                    vp_slice = vp_slice * (1.0 + 0.35 * float(p.vp_delta_frac) * m_halo_vol[..., iz])
                vp_new[..., iz] = vp_slice

        return vp_new, m_core_vol, m_halo_vol

    def _precompute(self, p, X2d, Y2d, z_arr, rng):
        # Decide anchor layer if requested and available
        use_labels = bool(p.use_layer_labels) and (self.layer_labels is not None)
        layer_id = int(p.layer_id)

        if use_labels and layer_id < 0:
            layer_id = _auto_pick_layer_id(self.layer_labels, rng, prefer_mid=True)

        pre: Dict[str, Any] = {"use_labels": use_labels, "layer_id": layer_id}

        # 1) Build a curvilinear centerline in XY (polyline + Catmull-Rom)
        # Parameter s in [0,1]
        nkn = 6
        L = float(p.corridor_length_m)
        xc, yc = float(p.center_x_m), float(p.center_y_m)
        # Create control points with gentle meander
        s_kn = np.linspace(0.0, 1.0, nkn)
        # Base direction: random azimuth
        theta = rng.uniform(0.0, 2*np.pi)
        dx_dir, dy_dir = np.cos(theta), np.sin(theta)
        # Perp
        px, py = -dy_dir, dx_dir
        # Control points
        xk = xc + (s_kn - 0.5) * L * dx_dir
        yk = yc + (s_kn - 0.5) * L * dy_dir
        # Add lateral meander
        meander = (0.18 * L) * rng.normal(0.0, 1.0, size=nkn)
        xk = xk + meander * px
        yk = yk + meander * py

        # Sample polyline dense for distance computation
        ns = 160
        s = np.linspace(0.0, 1.0, ns).astype(np.float32)
        xs = _catmull_rom_1d(s_kn, xk, s).astype(np.float32)
        ys = _catmull_rom_1d(s_kn, yk, s).astype(np.float32)

        pre["xs"] = xs
        pre["ys"] = ys

        # 2) Depth function zc(x,y)
        nx, ny = X2d.shape[0], Y2d.shape[1]
        kmax = max(int(p.undulation_kmax), 1)
        noise = np.zeros((nx, ny), dtype=np.float32)
        # Low-frequency cosine noise
        for kx in range(1, kmax + 1):
            for ky in range(1, kmax + 1):
                a = rng.normal(0.0, 1.0)
                phi = rng.uniform(0.0, 2*np.pi)
                xx = X2d / max(float(p.corridor_length_m), 1e-6)
                yy = Y2d / max(float(p.corridor_length_m), 1e-6)
                noise += (a * np.cos(2*np.pi*(kx*xx + ky*yy) + phi)).astype(np.float32)
        noise = noise / (np.max(np.abs(noise)) + 1e-9)

        if use_labels and layer_id >= 0:
            z_top, z_bot, thick, valid = _extract_longest_segment_top_bot(self.layer_labels, layer_id, z_arr)
            # put serpentinization near middle of that layer, plus undulation
            zc = (0.5 * (z_top + z_bot) + float(p.undulation_amp_m) * noise).astype(np.float32)
            # if invalid, fall back to geometric depth
            zc = np.where(valid, zc, float(p.z_center_m)).astype(np.float32)
            pre["zc"] = zc
            pre["valid_layer"] = valid
        else:
            pre["zc"] = (float(p.z_center_m) + float(p.undulation_amp_m) * noise).astype(np.float32)
            pre["valid_layer"] = None

        # 3) Build an XY window SDF around the centerline:
        #    sdf_xy = halfwidth - distance_to_polyline
        dist_xy = self._distance_to_polyline_xy(X2d, Y2d, xs, ys)
        base_hw = float(p.corridor_halfwidth_m)

        # add window irregularity: union with a few ellipses along the line
        sdf_xy = (base_hw - dist_xy).astype(np.float32)
        n_extra = int(round(float(p.corridor_irregularity) * 6.0))
        if n_extra > 0:
            for _ in range(n_extra):
                j = int(rng.integers(0, len(xs)))
                ex, ey = float(xs[j]), float(ys[j])
                a = base_hw * rng.uniform(0.8, 1.6)
                b = base_hw * rng.uniform(0.6, 1.3)
                sdf_e = self._sdf_ellipse_pos(X2d, Y2d, ex, ey, a, b)
                sdf_xy = np.maximum(sdf_xy, sdf_e)

        pre["sdf_xy"] = sdf_xy

        # 4) Patchiness seeds (optional)
        if p.mode == "patchy":
            pc = int(p.patch_count)
            jidx = rng.integers(0, len(xs), size=pc)
            centers = [(float(xs[j]), float(ys[j])) for j in jidx]
            pre["patch_centers"] = centers

        return pre

    def _sdf_core_slice(self, p, X2d, Y2d, z_m, iz, pre):
        # Vertical band sdf_z
        zc = pre["zc"]
        half_t = 0.5 * float(p.thickness_m)
        sdf_z = (half_t - np.abs(z_m - zc)).astype(np.float32)

        # Lateral corridor sdf_xy
        sdf_xy = pre["sdf_xy"]

        sdf = np.minimum(sdf_z, sdf_xy)

        # If label-anchored: optionally gate to layer id (softly)
        if pre["use_labels"] and pre["layer_id"] >= 0 and (self.layer_labels is not None):
            in_layer = (self.layer_labels[:, :, iz] == int(pre["layer_id"]))
            # Do NOT hard-gate to avoid zeroing everywhere due to faults; instead, soften:
            # inside layer: keep sdf, outside: penalize but allow slight spill (realistic alteration leaks)
            sdf = np.where(in_layer, sdf, sdf - 2.5 * float(p.edge_width_m))

        # Patchiness: keep only blobs along corridor
        if p.mode == "patchy":
            sdf = np.minimum(sdf, self._sdf_patchiness(X2d, Y2d, pre, p))

        return sdf

    def _sdf_patchiness(self, X2d, Y2d, pre, p):
        centers = pre.get("patch_centers", [])
        if not centers:
            return np.full((X2d.shape[0], Y2d.shape[1]), -1e6, dtype=np.float32)
        rng = self._rng
        r0 = float(p.patch_radius_m)
        rv = float(p.patch_radius_var)
        sdf_u = np.full((X2d.shape[0], Y2d.shape[1]), -1e6, dtype=np.float32)
        for (cx, cy) in centers:
            rr = r0 * rng.uniform(1.0 - rv, 1.0 + rv)
            dx = (X2d - cx).astype(np.float32)
            dy = (Y2d - cy).astype(np.float32)
            r = np.sqrt(dx*dx + dy*dy)
            sdf = (rr - r).astype(np.float32)
            sdf_u = np.maximum(sdf_u, sdf)
        return sdf_u

    @staticmethod
    def _distance_to_polyline_xy(X2d, Y2d, xs, ys):
        """
        Approx distance from each (x,y) to a polyline defined by points (xs,ys).
        Uses nearest point among dense samples (fast enough for nx~256,ny~256, ns~160).
        """
        # Shape: (nx,ny,ns)
        dx = X2d[..., None] - xs[None, None, :]
        dy = Y2d[..., None] - ys[None, None, :]
        d2 = dx*dx + dy*dy
        return np.sqrt(np.min(d2, axis=2)).astype(np.float32)

    @staticmethod
    def _sdf_ellipse_pos(X2d, Y2d, xc, yc, a, b):
        dx = (X2d - float(xc)).astype(np.float32)
        dy = (Y2d - float(yc)).astype(np.float32)
        rr = np.sqrt((dx / max(float(a), 1e-6)) ** 2 + (dy / max(float(b), 1e-6)) ** 2)
        q = (1.0 - rr)
        return (q * float(min(a, b))).astype(np.float32)
