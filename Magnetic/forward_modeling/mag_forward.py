import torch
import math
from typing import Dict, List, Tuple

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
    """
    Calculate the kernel for magnetic forward modeling in the frequency domain.
    Based on Bhattacharyya (1964) or similar prism integration methods adapted for convolution.
    
    Here we implement the formulation where the TMI anomaly T(x,y) is the convolution of 
    Magnetization J(x,y,z) with a kernel G(x,y,z).
    
    For a prism with constant magnetization, the response can be formulated in the wavenumber domain.
    However, often the structure `M` provided is susceptibility $\chi$.
    The inductive magnetization is $J = \chi F_0$.
    The total field anomaly $\Delta T$ is the projection of the anomalous field vector $\mathbf{B}_a$ onto the direction of the inducing field $\hat{\mathbf{F}}_0$.
    
    Ref: "Potential Theory in Gravity and Magnetic Applications", Blakely.
    The Fourier transform of the TMI anomaly caused by a layer of dipoles (or prisms) with magnetization $M(x,y)$ 
    can be expressed using the earth filter.
    
    Let's assume the susceptibility model $\chi(x,y,z)$ is given.
    The inducing field has magnitude $B_0$ (or $F_0$), inclination $I_0$, declination $A_0$.
    The magnetization vector has inclination $I$, declination $A$. 
    (Usually for induced magnetization, $I=I_0, A=A_0$).
    
    In Fourier domain (u, v):
    $\mathcal{F}[\Delta T] = \mathcal{F}[\chi] \cdot \Theta(u,v) \cdot \text{depth\_term}$
    
    where $\Theta(u,v)$ depends on directions.
    
    Using the formulation for a single prism centered at origin is complex in space domain effectively used in MATLAB code?
    The MATLAB code `fun_forward_mag` computes `uijk * atan(...)` which looks like the exact solution for a prism.
    
    To make it efficient on GPU for 3D grids, we use the property that gravity/magnetic fields satisfy convolution.
    $\Delta T(x, y) = \sum_z \chi(x,y,z) * K_z(x,y)$
    
    We need to construct the Kernel $K_z(x,y)$ which is the response of a unit susceptibility prism at depth $z$.
    
    However, `gra_forward.py` implies we can use FFT.
    The kernel for magnetic potential of a dipole is $\frac{\mathbf{m} \cdot \mathbf{r}}{r^3}$.
    The field is $-\nabla V$.
    
    Let's stick to the frequency domain formulation which is standard for constant voxel layers.
    
    Variables:
    - B0: magnitude of inducing field (nT). We assume the input `susceptibility` is just that (dimensionless). 
      The output needs to be scaled by B0. Or maybe the input is Magnetization intensity?
      The MATLAB code uses `G_T = 10^2 * Ta`. And `M` is 1. `Ta` is calculated geometrically.
      This suggests `Ta` is the geometric factor.
    
    Let's implement the specific kernel for TMI from a prism in Fourier domain.
    
    Kernel in Fourier Domain (u, v) for a layer of prisms at depth $z_c$ with thickness $dz$:
    $K(u,v) = 2\pi \cdot C_m \cdot e^{-|k|z_c} \cdot (\text{direction\_factors}) \cdot (\text{prism\_shape\_factor})$
    
    Actually, let's look at the MATLAB `fun_forward_mag`. It computes `atan` terms. This is the spatial domain prism integration.
    Computing this for every pair of (obs, source) is $O(N^2)$. Python will suffer.
    We should use FFT convolution.
    
    We need to compute the TMI response of a single prism of unit M, located at depth z, at all grid points (x,y).
    Then use this as a kernel for convolution with the M layer at that depth.
    
    Kernel Generation (Spatial Domain):
    We can compute the exact response of a prism at (0,0,z) on a grid (x,y) using the exact formula (like in the MATLAB code), 
    and then FFT it.
    
    Inputs:
    I, A: Inclination and Declination of Magnetization vector (radians).
    I0, D0: Inclination and Declination of Geomagnetic field (radians).
    
    The MATLAB code implements specific components. 
    Let's replicate the structure of `gra_forward.py` but with the Magnetic Kernel.
    
    The variable `Ta` in MATLAB `fun_forward_mag` seems to calculate a generic potential or simpler component? 
    Wait, `fun_forward_mag` computes `T` using `atan(xi*yj/(zk*rijk))`.
    The formula $\tan^{-1}(\frac{xy}{zr})$ appears in the potential of a prism or vertical component?
    Actually, for a prism with bounding box $[x_1, x_2] \times [y_1, y_2] \times [z_1, z_2]$, the potential involves logs and atans.
    
    Let's implement the standard Blakely (1995) or similar trusted algorithm for a prism TMI, 
    vectorized on PyTorch.
    
    TMI Anomaly due to a prism with magnetization $M$ (magnitude):
    $\Delta T \approx \hat{\mathbf{F}} \cdot \mathbf{B}$
    
    Calculations:
    We construct the kernel $K(x,y)$ for a prism at depth $z$ with unit magnetization, with direction $(I, A)$, 
    projected onto inducing field $(I_0, A_0)$.
    
    """
    pass


def forward_mag_tmi(
    susceptibility: torch.Tensor,
    dx: float,
    dy: float,
    dz: float,
    heights_m: List[float],
    obs_conf: Dict,
    B0: float = 50000.0, # Inducing field magnitude in nT
    I_deg: float = 90.0, # Inclination of inducing field (degrees)
    A_deg: float = 0.0,  # Declination of inducing field (degrees)
    M_I_deg: Optional[float] = None, # Inclination of magnetization (if different, e.g. remanent)
    M_A_deg: Optional[float] = None, # Declination of magnetization
    output_unit: str = "nt",
    pad_factor: int = 2,
) -> Tuple[torch.Tensor, Dict]:
    """
    Magnetic forward modeling (TMI) for a 3D susceptibility model.
    
    susceptibility: (nx, ny, nz) SI units (dimensionless). 
                    Or Magnetization if B0=1.
    dx, dy, dz: cell sizes in meters.
    heights_m: list of observation heights (positive upwards, z=0 is surface).
    obs_conf: observation layout config.
    B0: Geomagnetic field magnitude in nT.
    I_deg: Geomagnetic Inclination in degrees.
    A_deg: Geomagnetic Declination in degrees.
    M_I_deg: Magnetization Inclination (default to I_deg i.e., induced only).
    M_A_deg: Magnetization Declination (default to A_deg).
    
    Returns:
       data: (1, n_heights, ...) TMI anomaly in nT (if output_unit='nt')
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
    
    # Factor to convert susceptibility to magnetization: M = chi * H = chi * B0 / mu0 ? 
    # Usually in potential field gravity/mag methods, we work with B directly.
    # Anomaly B = Cm * grad(grad(V) . m) ...
    # Let's use the standard "Relation between Gravity and Magnetic Fields" (Poisson's Relation)
    # or simply the frequency domain geometric factor.
    
    # Constants
    # Cm = mu0 / 4pi = 10^-7 H/m (SI)
    Cm = 1e-7 
    
    # We will compute the field in Tesla, then convert to nT.
    # If susceptibility is given, Magnetization J = chi * F. 
    # F is the inducing B-field vector magnitude B0 (Tesla).
    # Note: B0 input is often nT. Let's standardize B0 to Tesla for internal calc using 1e-9.
    
    B0_T = B0 * 1e-9
    
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
    
    # Frequencies u, v in rad/meter
    # dk = 2pi / (N * dx)
    du = 2 * math.pi / (nx_pad * dx)
    dv = 2 * math.pi / (ny_pad * dy)
    
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
    
    # Frequency domain TMI Operator (Blakely 1995)
    # F[Delta T] = F[Phi] * Theta_m * Theta_f
    # where Phi is Potential? No, simplified:
    # F[Delta T] = 2 * pi * Cm * M(u,v) * E(z) * D(u,v)
    # D(u,v) = (m_z + i*(m_x*u + m_y*v)/k) * (f_z + i*(f_x*u + f_y*v)/k)
    # Wait, sign convention of FFT matters. 
    # If DFT is sum f(x) e^{-ikx}, then derivative calc involves (ik).
    
    # Let's form the direction factors.
    # alpha_m = m_x * cos(theta) + m_y * sin(theta) ... NO
    # Let's use (u, v) directly.
    # Note: K_abs > 0 required.
    
    # Direction factor for Magnetization
    # Theta_m = m_z + 1j * (m_x * U + m_y * V) / K_abs
    Theta_m = mz + 1j * (mx * U + my * V) / K_abs
    
    # Direction factor for Field
    # Theta_f = f_z + 1j * (fx * U + fy * V) / K_abs
    Theta_f = fz + 1j * (fx * U + fy * V) / K_abs
    
    # Earth filter
    # Factor = 2 * pi * Cm * Theta_m * Theta_f
    # But wait, Magnetization J is in M(u,v).
    # If using susceptibility model chi, J = chi * B0_T / mu0 ? 
    # Actually B = mu0 * (H + M). B_anomaly is due to M. 
    # Anomaly field is roughly Cm * ...
    # Specifically: F[T] = 2*pi * Cm * |K| * F[J] * ... ? No.
    
    # Standard formula for layer of thickness dz at depth z0:
    # F[T](u,v) = 2*pi * Cm * F[chi](u,v) * B0_T * Theta_m * Theta_f * e^{-|K|*z0} * (1 - e^{-|K|*dz})
    # Wait, the term (1 - e^{-|K|*dz}) accounts for thickness dz. 
    # We can model the block as a layer z_top to z_bottom.
    
    # Let's verify constants.
    # Cm = 1e-7.
    # F[T] (Tesla) = result of inverse FFT.
    
    # Fix singularity at K=0
    Theta_m[0,0] = 0.0 # or appropriate limit. If mz != 0, it's mz.
    Theta_f[0,0] = 0.0
    if K_abs[0,0] == 1.0: # Reset if we changed it
         pass # Actually at k=0, formula is singular or 0.
         # The DC component of potential field is usually zero.
    
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
    
    chi_hat = torch.fft.rfft2(sus_t, s=(nx_pad, ny_pad), dim=(0, 1)) # (nx_pad, ny_pad//2 + 1, nz)
    
    X_term = 2 * math.pi * Cm * B0_T * Theta_m * Theta_f
    
    # Depth term
    # We are integrating over prisms of thickness dz.
    # Integration of e^{-kz} from z1 to z2 is (e^{-kz1} - e^{-kz2})/k
    # z1 = depth_top, z2 = depth_bottom.
    # depth is positive downwards.
    
    z_centers = (torch.arange(nz, device=device, dtype=torch.float32) + 0.5) * float(dz)
    
    if layout == "grid":
        out = torch.zeros((1, len(heights), x_idx.numel(), y_idx.numel()), device=device, dtype=torch.float32)
    else:
        out = torch.zeros((1, len(heights), x_idx.numel()), device=device, dtype=torch.float32)

    for hi, h in enumerate(heights):
        # Observation height h (meters above surface). 
        # Surface is z=0. Obs z = -h.
        # Distance to source at z (positive) is z - (-h) = z + h.
        
        # Accumulate frequency sum
        # We can sum (chi_hat_z * depth_factor_z)
        
        # Vectorized depth factor calculation
        # z_centers is (nz,)
        # K_abs is (nx_pad, ny_pad//2+1)
        
        # Expand K_abs for broadcasting with z
        # K_ext: (nx_pad, ny_pad_half, 1)
        K_ext = K_abs.unsqueeze(-1)
        
        # For each layer j, depth z_j.
        # We can use the center approximation: e^{-|k| * (z_j + h)} * dz * |k|?
        # Exact slab formula: (e^{-k(z_j - dz/2 + h)} - e^{-k(z_j + dz/2 + h)}) / k * ...?
        # Actually for TMI convolution:
        # Integrated kernel over z is best.
        # But here susceptibility varies with z.
        # So we sum layers.
        # Layer integral: int_{z_top}^{z_bot} e^{-k z} dz = (e^{-k z_top} - e^{-k z_bot}) / k
        # Multiply by remaining factor from potential derivatives which might supply a 'k' factor?
        # The TMI operator Theta_m * Theta_f is dimensionless?
        # Let's check potential V -> T  Relation.
        # V ~ 1/r. F[V] ~ 1/k * e^{-kz}.
        # T ~ grad grad V. F[T] ~ k^2 * F[V] ~ k * e^{-kz}.
        # So yes, we need a factor of K_abs? No, Theta terms have 1/k? 
        # Theta terms are dimensionless (order 1).
        # So we need a factor of K_abs?
        # Standard: F[T] = 2 pi Cm M (Theta Theta) * e^{-kz} * k.
        # But wait, M is magnetization intensity (A/m).
        
        # Let's use the explicit "prism" layer formula for convolution.
        # Parker's formula or similarity.
        # Term = (e^{-k(z_top+h)} - e^{-k(z_bot+h)}) * (Theta_m * Theta_f) * 2 * pi * Cm * B0_T
        # Note: No 'k' in the denominator because deriving potential -> field brings 'k', 
        # and integrating dz removes 'k'. So they cancel?
        # Let's check dimensions.
        # M (A/m). Cm (H/m = T m / A). product is Tesla. Correct.
        # If we had k in numerator, we'd get T/m?
        # So it seems just the exponential difference.
        
        # z_tops = z_centers - dz/2 + h
        # z_bots = z_centers + dz/2 + h
        
        z_tops = z_centers - 0.5 * dz + h
        z_bots = z_centers + 0.5 * dz + h
        
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
        "B0": B0,
        "I": I_deg,
        "A": A_deg,
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
    model[8:12, 8:12, 2:6] = 1.0 # SI susceptibility
    
    # MATLAB params
    # I0 = 90 deg (pi/2) -> Vertical induction (at pole)
    # G_T = 10^2 * Ta. 
    # M=1.
    
    # In my code:
    # If I_deg = 90.
    # B0. If the MATLAB code produces specific nT, we need to know what 'Ta' is scaled by.
    # "G_T=10^2*Ta" and "M=1".
    # This might mean B0 is related to 100? Or Cm?
    # Let's just run with standard Earth field 50000 nT and see shape.
    
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
