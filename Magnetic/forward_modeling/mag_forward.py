import torch
import math
from typing import Dict, List, Tuple, Optional

def _unit_scale(output_unit: str) -> float:
    """
    Convert SI (Tesla) to desired unit.
    - "si": 1.0 (Tesla)
    - "nt": 1e9 (NanoTesla)
    """
    u = (output_unit or "nt").lower()
    if u == "si":
        return 1.0
    if u in ["nt", "nanotesla"]:
        return 1e9
    raise ValueError(f"Unknown output_unit: {output_unit}")

def _make_obs_indices(nx: int, ny: int, obs_conf: Dict) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Build observation indices in x/y (integer grid indices).
    Supports:
    - layout: "grid" with optional {n_x, n_y, first_x, first_y, d_x, d_y}
    - layout: "points" with {points: [[ix,iy], ...]}
    Returns (x_idx, y_idx, layout)
    """
    layout = (obs_conf.get("layout", "grid") or "grid").lower()
    if layout == "grid":
        n_x = int(obs_conf.get("n_x", nx))
        n_y = int(obs_conf.get("n_y", ny))
        first_x = int(obs_conf.get("first_x", 0))
        first_y = int(obs_conf.get("first_y", 0))
        d_x = int(obs_conf.get("d_x", 1))
        d_y = int(obs_conf.get("d_y", 1))

        # Create ranges
        # Note: In the gravity code, this produces a grid of indices to select from the full field.
        # If we compute the full field at (nx, ny), we select specific points.
        
        # Ranges for indexing
        x_idx = torch.arange(first_x, first_x + n_x * d_x, d_x, dtype=torch.long)
        y_idx = torch.arange(first_y, first_y + n_y * d_y, d_y, dtype=torch.long)
        
        # Clamp to domain in case specifications go outside
        x_idx = x_idx.clamp(0, nx - 1)
        y_idx = y_idx.clamp(0, ny - 1)
        
        return x_idx, y_idx, "grid"

    if layout == "points":
        pts = obs_conf.get("points", [])
        if not pts:
            raise ValueError("observation.layout=points requires observation.points=[[ix,iy],...]")
        pts_t = torch.tensor(pts, dtype=torch.long)
        x_idx = pts_t[:, 0].clamp(0, nx - 1)
        y_idx = pts_t[:, 1].clamp(0, ny - 1)
        return x_idx, y_idx, "points"

    raise ValueError(f"Unknown observation.layout: {layout}")

def calculate_h_kernel_fft(
    nx: int, ny: int, nz: int,
    dx: float, dy: float, dz: float,
    I: float, A: float,
    I0: float, A0: float,
    height: float = 0.0,
    pad_factor: int = 2,
    device: torch.device = torch.device('cpu')
):
    """Legacy placeholder.

    Earlier versions considered explicitly building a spatial-domain prism kernel and FFT'ing it.
    The current code path uses a standard wavenumber-domain earth-filter operator directly in
    :func:`forward_mag_tmi`, so this helper is intentionally not implemented.
    """
    raise NotImplementedError(
        "calculate_h_kernel_fft() is intentionally not implemented. "
        "Use forward_mag_tmi(), which applies a standard earth-filter formulation directly in the "
        "wavenumber domain."
    )


def forward_mag_tmi(
    susceptibility: torch.Tensor,
    dx: float,
    dy: float,
    dz: float,
    heights_m: List[float],
    obs_conf: Dict,
    input_type: str = "susceptibility",
    B0: float = 50000.0, # Inducing field magnitude in nT
    I_deg: float = 90.0, # Inclination of inducing field (degrees)
    A_deg: float = 0.0,  # Declination of inducing field (degrees)
    M_I_deg: Optional[float] = None, # Inclination of magnetization (if different, e.g. remanent)
    M_A_deg: Optional[float] = None, # Declination of magnetization
    output_unit: str = "nt",
    pad_factor: int = 2,
) -> Tuple[torch.Tensor, Dict]:
    """ 
    Magnetic forward modeling (TMI: total magnetic intensity anomaly) for a 3D **susceptibility** model.

    Parameters
    ----------
    input_type:
        - "susceptibility" (default): input is dimensionless SI susceptibility χ.
          The forward operator assumes induced magnetization magnitude J = χ · H0 with H0 = B0/μ0.
        - "magnetization": input is magnetization intensity J (A/m). In this mode, the result does NOT
          depend on B0 for amplitude scaling (only for direction via I_deg/A_deg).

    Notes
    -----
    For the default (susceptibility) mode, induced magnetization magnitude is proportional to the
    inducing field strength:

        H0 = B0 / μ0   (A/m),   B0 is the inducing field magnitude (Tesla), μ0 = 4π×10⁻⁷.
        J = χ · H0     (A/m)

    The output is the total-field anomaly (projection of the anomalous field onto the inducing field direction).

    susceptibility: (nx, ny, nz)
        - if input_type="susceptibility": dimensionless SI susceptibility χ
        - if input_type="magnetization": magnetization intensity J in A/m
    dx, dy, dz: cell sizes in meters.
    heights_m: list of observation heights (positive upwards, z=0 is surface).
    obs_conf: observation layout config.
    B0: Geomagnetic field magnitude in nT.
    I_deg: Geomagnetic Inclination in degrees.
    A_deg: Geomagnetic Declination in degrees.
    M_I_deg: Magnetization Inclination (default to I_deg i.e., induced only).
    M_A_deg: Magnetization Declination (default to A_deg).
    
    Returns:
       data: (1, n_heights, ...) TMI anomaly in output_unit ("nt" or "si")
       meta: dict
    """
    
    if susceptibility.ndim != 3:
        raise ValueError(f"susceptibility must be 3D, got {susceptibility.shape}")
        
    device = susceptibility.device
    nx, ny, nz = susceptibility.shape
    
    if M_I_deg is None: M_I_deg = I_deg
    if M_A_deg is None: M_A_deg = A_deg
    
    # Convert angles to radians
    d2r = math.pi / 180.0
    i0 = I_deg * d2r
    a0 = A_deg * d2r
    mi = M_I_deg * d2r
    ma = M_A_deg * d2r
    
    # Direction cosines of inducing field (F) and magnetization (M)
    # Coordinate system: X (North), Y (East), Z (Down)
    # Standard geophysical convention:
    # m_x = cos(I) cos(A)
    # m_y = cos(I) sin(A)
    # m_z = sin(I)
    
    # Inducing field direction hat{F}
    fx = math.cos(i0) * math.cos(a0)
    fy = math.cos(i0) * math.sin(a0)
    fz = math.sin(i0)
    
    # Magnetization direction hat{M}
    mx = math.cos(mi) * math.cos(ma)
    my = math.cos(mi) * math.sin(ma)
    mz = math.sin(mi)
    
    # ---- Physical constants (SI) ----
    mu0 = 4.0 * math.pi * 1e-7  # N/A^2 = T·m/A
    # Convert inducing field magnitude from nT to Tesla
    B0_T = float(B0) * 1e-9
    
    scale = _unit_scale(output_unit)
    
    # Observation indices
    x_idx, y_idx, layout = _make_obs_indices(nx, ny, obs_conf)
    x_idx = x_idx.to(device)
    y_idx = y_idx.to(device)

    heights = [float(h) for h in (heights_m or [0.0])]

    # Padded dimensions for FFT
    nx_pad = int(pad_factor) * nx
    ny_pad = int(pad_factor) * ny
    
    # Frequency coordinates
    # u corresponds to x (North), v corresponds to y (East)
    # fftfreq returns cycles/unit_sample. multiply by 2pi/L?
    # PyTorch FFT:
    # rfft2 returns frequencies [0, 1, ..., N/2]
    
    u = torch.fft.fftfreq(nx_pad, d=dx, device=device) * 2 * math.pi # Standard fftfreq is 1/T, so * 2pi for angular
    v = torch.fft.rfftfreq(ny_pad, d=dy, device=device) * 2 * math.pi
    
    # Meshgrid of wavenumbers
    # dim=0 is x (rows), dim=1 is y (columns)
    U, V = torch.meshgrid(u, v, indexing='ij')
    
    # Wavenumber K = sqrt(u^2 + v^2)
    K_sq = U**2 + V**2
    K_abs = torch.sqrt(K_sq)
    
    # Handle singularity at K=0 (DC component)
    # For TMI, DC component is 0 if no net monopole ?
    # Usually we set K=0 to a small value or handle explicitly. 
    # But let's look at the multiplier.
    K_abs[0, 0] = 1.0 # Avoid div by zero, we will zero out the term later if needed
    
    # ---- Earth filter (wavenumber-domain operator for TMI) ----
    # A common (Blakely-style) expression for total-field anomaly in 2D Fourier domain.
    # The potential phi(k) is proportional to M(k) * ( i*mx*u + i*my*v + mz*k ) * exp(-k|z|) / k
    # The field T(k) is - grad(phi) dot F = phi(k) * ( i*fx*u + i*fy*v + fz*k )
    # Note: vertical derivative in wavenumber domain is |k| (for sources below observation).
    #
    # With z positive down:
    # Horizontal derivatives: d/dx -> i*u, d/dy -> i*v
    # Vertical derivative:    d/dz -> k  (because potential decays as exp(-k|z|), and we look at z<0 region relative to source?)
    # Actually, standard reformulation:
    # Theta_m = i*mx*u + i*my*v + mz*K_abs
    # Theta_f = i*fx*u + i*fy*v + fz*K_abs
    # Filter = (Theta_m * Theta_f) / K_abs  (Not K^2, one K cancels with potential's 1/K)
    #
    # But we perform integration over dz (1/k factor from integration of exp(-kz)) later?
    # No, the layer integration (exp(-kz1) - exp(-kz2)) already brings a 1/K factor implicitly? 
    # Let's check the layer factor: exp_top - exp_bot.
    # Integral of e^{-kz} is -1/k e^{-kz}. So (exp_top - exp_bot) corresponds to integral * k ? No.
    #
    # Let's stick to the "layer density" approach.
    # Gravity/Mag potential of a layer: G = 2*pi*G * density * (exp(-k z_top) - exp(-k z_bot))/k
    # Here we sum (exp - exp). This lacks the 1/k factor if we consider it density.
    #
    # Revised formula:
    # factor = 1/K_abs
    # term_m = 1j * (mx * U + my * V) + mz * K_abs
    # term_f = 1j * (fx * U + fy * V) + fz * K_abs
    # earth_filter = factor * term_m * term_f
    
    # Updated formula:
    # Scale K term
    # earth_filter: order K (Field)
    # The term (i*mx*u + i*my*v + mz*k) is (i*k_vec dot M).
    # The term (i*fx*u + i*fy*v + fz*k) is (i*k_vec dot F).
    # We divide by k because the Green's function for potential is 2*pi/k * exp(-kz).
    # So Field ~ (k dot M)(k dot F) * (1/k) * exp(-kz).
    
    # K_abs handles the 1/k factor.
    # We apply a mask to avoid division by zero at k=0.
    K_inv = torch.zeros_like(K_abs)
    msk = K_abs > 1e-10
    K_inv[msk] = 1.0 / K_abs[msk]

    # Note signs:
    # u,v are wavenumbers. Spatial derivative d/dx -> i*u.
    # Vertical derivative d/dz -> k (for z positive down, source below obs).
    # Potential V ~ -div(M/r) -> - M dot grad(1/r).
    # Fourier transform of 1/r is 2*pi/k * exp(-k|z|).
    # grad(1/r) -> (iu, iv, k) * (2*pi/k) exp(-kz) ?
    # Actually, usually d/dz -> |k|.
    # So Term_M = i*mx*U + i*my*V + mz*K_abs.
    # And Term_F = i*fx*U + i*fy*V + fz*K_abs.
    # Wait, B = -grad V.
    # So B_fixed_comp = - F dot grad V.
    # So another (iu, iv, k) dot F.
    # Total: - (Term_M) * (Term_F) / K_abs ? 
    # Or + ?
    # Let's check a standard reference (e.g. Blakely).
    # Blakely uses z positive down.
    # B = Cm * sum [ theta_m * theta_f * exp(-k*z0) * (1 - exp(-k*t)) / k ] * sum ...
    # theta_m = m_z*k + i*(m_x*u + m_y*v).
    # theta_f = f_z*k + i*(f_x*u + f_y*v).
    # This matches my Term_M and Term_F.
    # The factor is 1/k.
    # The overall constant is Cm = mu0 / 4pi = 10^-7.
    # My constant is 0.5 * mu0 = 2*pi * 10^-7.
    # Blakely has 2*pi from Fourier definition? Yes, 2*pi * Cm = 0.5 * mu0. (Correct).
    # Signs?
    # B = 2*pi*Cm * ...
    # It appears positive.
    
    term_m = 1j * (mx * U + my * V) + mz * K_abs
    term_f = 1j * (fx * U + fy * V) + fz * K_abs
    
    earth_filter = term_m * term_f * K_inv
    earth_filter[0, 0] = 0.0
    
    # Precompute RFFT of layers?
    # We have a 3D volume of susceptibility.
    # Usually we treat each layer nz independently.
    # chi is (nx, ny, nz)
    
    # Move to frequency domain
    # chi_pad: (nx_pad, ny_pad, nz)
    sus_t = susceptibility.to(torch.float32)
    # Batch FFT over the last dimension (nz)? No, RFFT2 is over dim 0,1.
    # We can permute to put z in batch dim or iteration.
    # Let's iterate z to save memory or batch if memory allows. 
    # (nx=100, ny=100, nz=100) -> 1M floats. Tiny.
    
    chi_hat = torch.fft.rfft2(sus_t, s=(nx_pad, ny_pad), dim=(0, 1))  # (nx_pad, ny_pad//2 + 1, nz)

    input_type_l = str(input_type).strip().lower()
    if input_type_l in {"susceptibility", "chi", "kappa"}:
        # Overall scale (Tesla) for susceptibility input
        X_term = (0.5 * B0_T) * earth_filter
    elif input_type_l in {"magnetization", "j", "m"}:
        # Overall scale (Tesla) for magnetization intensity input J (A/m)
        X_term = (0.5 * mu0) * earth_filter
    else:
        raise ValueError(f"Unknown input_type={input_type!r}. Use 'susceptibility' or 'magnetization'.")
    
    # Depth term
    # We are integrating over prisms of thickness dz.
    # Integration of e^{-kz} from z1 to z2 is (e^{-kz1} - e^{-kz2})/k
    # z1 = depth_top, z2 = depth_bottom.
    # depth is positive downwards.
    
    # Depth boundaries for each voxel layer (z positive downward from surface)
    z_top = torch.arange(nz, device=device, dtype=torch.float32) * float(dz)
    z_bot = (torch.arange(nz, device=device, dtype=torch.float32) + 1.0) * float(dz)
    
    if layout == "grid":
        out = torch.zeros((1, len(heights), x_idx.numel(), y_idx.numel()), device=device, dtype=torch.float32)
    else:
        out = torch.zeros((1, len(heights), x_idx.numel()), device=device, dtype=torch.float32)

    # Scale factor from standard FFT definition on discrete grids.
    # The continuous Fourier transform F(k) is related to discrete F[k] by dx*dy factor.
    # i.e. Integral f(x) e^{-ikx} dx ~ Sum f(n) e^{-ikn} * dx
    # So F_continuous ~ F_discrete * dx * dy.
    # We need to multiply by dx * dy when moving to frequency domain to match physical units.
    # Or multiply by dx*dy after IFFT?
    # Actually, if we view it as convolution:
    # f * g = IFFT( FFT(f) * FFT(g) ) ?
    # In discrete: (f * g)[n] = sum f[k] g[n-k].
    # In continuous: (f * g)(x) = int f(u) g(x-u) du ~ sum f(k) g(n-k) * dx.
    # So we need one factor of cell area (dx*dy) for the convolution integral.
    # Since our earth_filter is the continuous transfer function evaluated at discrete k,
    # we effectively are doing: Result = IFFT( FFT(Model) * TransferFunction ).
    # This corresponds to discrete convolution of Model with (IFFT(TransferFunction)).
    # But the physical convolution integral needs a `dV` factor.
    # Since we integrate over z analytically (giving the 1/k and exp factors), we are left with x,y integration.
    # So we need a factor of (dx * dy).
    
    # Let's apply this scaling.
    scale_xy = dx * dy
    
    # Combined scaling
    scale *= scale_xy

    for hi, h in enumerate(heights):
        # Observation height h (meters above surface). 
        # Surface is z=0. Obs z = -h.
        # Distance to source at z (positive) is z - (-h) = z + h.
        
        # Vectorized depth factor for voxel slabs.
        # Each voxel layer contributes a factor (exp(-K(z_top+h)) - exp(-K(z_bot+h))).
        K_ext = K_abs.unsqueeze(-1)  # (nx_pad, ny_pad_half, 1)
        z_tops = z_top + float(h)
        z_bots = z_bot + float(h)
        
        # e_{-k z} terms
        # (nx_pad, ny_pad_half, nz)
        # We need to handle k=0 case separately.
        # limit (e^{-k z1} - e^{-k z2}) as k->0 is -(z1 - z2) * k? No.
        # It is just thickness? No. DC component of mag field.
        # For k=0, TMI is usually 0 unless we have monopole term.
        
        # exp_factors: (nx_pad, ny_pad_half, nz)
        exp_top = torch.exp(-K_ext * z_tops)
        exp_bot = torch.exp(-K_ext * z_bots)
        
        layer_factors = exp_top - exp_bot
        
        # Apply mask for K=0 to avoid issues if any, though exp(0)-exp(0)=0. Good.
        
        # Sum over layers: sum(chi_hat * layer_factor, dim=2)
        # accumulated: (nx_pad, ny_pad_half)
        acc_hat = torch.sum(chi_hat * layer_factors, dim=2)
        
        # Multiply by common factors
        total_hat = acc_hat * X_term
        
        # Inverse FFT
        tmi_full = torch.fft.irfft2(total_hat, s=(nx_pad, ny_pad))
        
        # Crop
        tmi_full = tmi_full[:nx, :ny] * scale
        
        if layout == "grid":
            # sample by indices
            val = tmi_full.index_select(0, x_idx).index_select(1, y_idx)  # (n_x, n_y)
            out[0, hi] = val
        else:
            val = tmi_full[x_idx, y_idx]  # (n_points,)
            out[0, hi] = val

    meta = {
        "layout": layout,
        "obs_x_idx": x_idx.detach().cpu(),
        "obs_y_idx": y_idx.detach().cpu(),
        "heights_m": torch.tensor(heights, dtype=torch.float32),
        "dx": torch.tensor([dx, dy, dz], dtype=torch.float32),
        "mu0": mu0,
        "B0": B0,
        "I": I_deg,
        "A": A_deg,
        "input_type": input_type_l,
        "output_unit": output_unit,
        "pad_factor": int(pad_factor),
        "model_shape": (nx, ny, nz),
    }

    return out, meta

if __name__ == "__main__":
    # Test code
    nx, ny, nz = 20, 20, 10
    dx, dy, dz = 100., 100., 100.
    
    # Create model: a prism in the middle
    # Corresponding to MATLAB: M(9:12,9:12,3:6)=1; (1-based indices)
    # Python 0-based: 8:12 (since 12 is exclusive), etc.
    model = torch.zeros((nx, ny, nz))
    model[8:12, 8:12, 2:6] = 0.05  # example SI susceptibility (dimensionless)
    
    # Example run with a typical inducing field.
    
    obs_conf = {"layout": "grid", "n_x": nx, "n_y": ny}
    
    tmi, meta = forward_mag_tmi(
        model, dx, dy, dz, 
        heights_m=[0], 
        obs_conf=obs_conf, 
        B0=50000, 
        I_deg=90, 
        A_deg=0
    )
    
    print("Output shape:", tmi.shape)
    print("Max anomaly:", tmi.max().item(), "nT")
    print("Min anomaly:", tmi.min().item(), "nT")
