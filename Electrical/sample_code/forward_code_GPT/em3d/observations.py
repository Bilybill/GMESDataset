
from __future__ import annotations
import math
import torch
from .operators import MU0


def csem_data_dict(ex, ey, ez, hx, hy, hz):
    return {
        'Ex': ex, 'Ey': ey, 'Ez': ez,
        'Hx': hx, 'Hy': hy, 'Hz': hz,
        'amp_Ex': ex.abs(), 'phase_Ex_deg': torch.angle(ex) * 180.0 / math.pi,
        'amp_Ey': ey.abs(), 'phase_Ey_deg': torch.angle(ey) * 180.0 / math.pi,
        'amp_Ez': ez.abs(), 'phase_Ez_deg': torch.angle(ez) * 180.0 / math.pi,
    }


def impedance_and_tipper(ex_x, ey_x, hz_x, hx_x, hy_x,
                         ex_y, ey_y, hz_y, hx_y, hy_y,
                         omega: float):
    n = ex_x.numel()
    Zxx = torch.zeros(n, dtype=ex_x.dtype, device=ex_x.device)
    Zxy = torch.zeros_like(Zxx)
    Zyx = torch.zeros_like(Zxx)
    Zyy = torch.zeros_like(Zxx)
    Tzx = torch.zeros_like(Zxx)
    Tzy = torch.zeros_like(Zxx)
    eps = 1e-10
    I2 = torch.eye(2, dtype=ex_x.dtype, device=ex_x.device)
    for i in range(n):
        E = torch.stack([torch.stack([ex_x[i], ex_y[i]]),
                         torch.stack([ey_x[i], ey_y[i]])])
        H = torch.stack([torch.stack([hx_x[i], hx_y[i]]),
                         torch.stack([hy_x[i], hy_y[i]])])
        H = H + eps * I2
        Hinv = torch.linalg.inv(H)
        Z = E @ Hinv
        Hzrow = torch.stack([hz_x[i], hz_y[i]]).reshape(1, 2)
        T = Hzrow @ Hinv
        Zxx[i], Zxy[i], Zyx[i], Zyy[i] = Z[0, 0], Z[0, 1], Z[1, 0], Z[1, 1]
        Tzx[i], Tzy[i] = T[0, 0], T[0, 1]

    def rho_phi(z):
        rho = (z.abs() ** 2) / (MU0 * omega)
        phi = torch.angle(z) * 180.0 / math.pi
        return rho, phi

    rho_xy, phi_xy = rho_phi(Zxy)
    rho_yx, phi_yx = rho_phi(Zyx)
    return {
        'Zxx': Zxx, 'Zxy': Zxy, 'Zyx': Zyx, 'Zyy': Zyy,
        'rho_xy': rho_xy, 'phi_xy_deg': phi_xy,
        'rho_yx': rho_yx, 'phi_yx_deg': phi_yx,
        'Tzx': Tzx, 'Tzy': Tzy,
        'tipper_amp_x': Tzx.abs(), 'tipper_phase_x_deg': torch.angle(Tzx) * 180.0 / math.pi,
        'tipper_amp_y': Tzy.abs(), 'tipper_phase_y_deg': torch.angle(Tzy) * 180.0 / math.pi,
    }
