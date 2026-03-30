import torch
import torch.nn as nn
import mt_forward_cuda


PHOENIX_COEFFS = [8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0]


def _estimate_boundary_background_rho(rho_tensor: torch.Tensor) -> float:
    nx, ny, nz = rho_tensor.shape
    boundary_mask = torch.zeros((nx, ny, nz), dtype=torch.bool, device=rho_tensor.device)
    boundary_mask[0, :, :] = True
    boundary_mask[-1, :, :] = True
    boundary_mask[:, 0, :] = True
    boundary_mask[:, -1, :] = True
    boundary_mask[:, :, 0] = True
    boundary_mask[:, :, -1] = True
    return float(rho_tensor[boundary_mask].mean().item())


def _generate_phoenix_frequencies(f_min: float, f_max: float) -> list[float]:
    if f_min > f_max:
        f_min, f_max = f_max, f_min

    max_power = int(torch.ceil(torch.log10(torch.tensor(f_max, dtype=torch.float64))).item())
    min_power = int(torch.floor(torch.log10(torch.tensor(f_min, dtype=torch.float64))).item())

    freqs = []
    for p in range(max_power, min_power - 1, -1):
        power_of_10 = 10.0 ** p
        for coeff in PHOENIX_COEFFS:
            current_f = coeff * power_of_10
            if current_f <= f_max and current_f >= f_min:
                freqs.append(float(current_f))

    if not freqs:
        freqs = [float(f_max), float(f_min)]
    return freqs


def resolve_auto_mt_frequencies(rho_tensor: torch.Tensor, dz: float):
    bg_rho = _estimate_boundary_background_rho(rho_tensor)
    skin_depth_min = dz * 2.0
    total_depth = rho_tensor.shape[2] * dz
    skin_depth_max = total_depth / 3.0

    f_max = bg_rho * (503.0 / skin_depth_min) ** 2
    f_min = bg_rho * (503.0 / skin_depth_max) ** 2
    freqs = _generate_phoenix_frequencies(f_min, f_max)
    return freqs, bg_rho, f_min, f_max


class MTForward3D(nn.Module):
    def __init__(self, freqs=None, dx=1.0, dy=1.0, dz=1.0):
        """
        PyTorch wrapper for CUDA MT 3D Forward Modeling.
        Args:
            freqs: Optional list of frequencies (Hz). If omitted or empty,
                the frequency range is auto-generated from the model extent and
                boundary background resistivity following MTForward3D.
            dx: Cell width in x-direction (m)
            dy: Cell width in y-direction (m)
            dz: Cell width in z-direction (m)
        """
        super(MTForward3D, self).__init__()
        self.freqs = None if freqs is None else list(freqs)
        self.last_freqs = None
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def forward(self, rho_tensor):
        """
        Forward MT 3D.
        Args:
            rho_tensor: 3D contiguous tensor of shape (NX, NY, NZ) with resistivity values (Ohm-m).
        Returns:
            app_res: Apparent resistivity tensor of shape (n_freqs, NX, NY, 2)
            phase: Phase tensor of shape (n_freqs, NX, NY, 2)
        """
        # Ensure tensor is CPU, double, and contiguous
        rho_cpu = rho_tensor.detach().cpu().to(torch.float64).contiguous()
        if self.freqs:
            freqs = list(self.freqs)
        else:
            freqs, bg_rho, f_min, f_max = resolve_auto_mt_frequencies(rho_cpu, self.dz)
            print(f"-> Auto-calculated Background Rho: {bg_rho:.6g} Ohm-m")
            print("-> Auto-calculated Frequency Range:")
            print(f"   f_max: {f_max:.6g} Hz, f_min: {f_min:.6g} Hz")
            print(f"   Range: [{freqs[0]:.6g} Hz  ...  {freqs[-1]:.6g} Hz], Points = {len(freqs)}")

        self.last_freqs = tuple(freqs)
        app_res, phase = mt_forward_cuda.compute_mt_3d(rho_cpu, self.dx, self.dy, self.dz, freqs)
        return app_res, phase
