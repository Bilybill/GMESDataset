from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from em3d.solver import fgmres, block_fgmres, jacobi_precond_from_diag, make_flexible_combo_precond


def main():
    torch.manual_seed(0)
    n = 30
    dtype = torch.complex128
    # easy-to-check diagonal system
    diag = torch.linspace(1.0, 50.0, n, dtype=torch.float64).to(dtype) * (1.0 + 0.3j)

    def A_mv(x):
        return diag.unsqueeze(-1) * x if x.ndim == 2 else diag * x

    b = torch.randn(n, dtype=dtype) + 1j * torch.randn(n, dtype=dtype)
    x_true = b / diag

    # dynamic/flexible preconditioner: alternate two dampings
    base = jacobi_precond_from_diag(diag)
    def flexible_precond(v, iteration=0, **kwargs):
        y = base(v)
        alpha = 1.0 if (iteration % 2 == 0) else 0.85
        return alpha * y

    x, info = fgmres(A_mv, b, tol=1e-10, maxiter=60, restart=10, M_inv=flexible_precond)
    rel = torch.linalg.norm(x - x_true) / torch.linalg.norm(x_true)
    print('fgmres rel error =', float(rel))
    print('info =', info)
    assert rel < 1e-8

    B = torch.stack([b, 2.0 * b + 0.3j * b], dim=1)
    X_true = B / diag.unsqueeze(-1)
    X, info_b = block_fgmres(A_mv, B, tol=1e-10, maxiter=80, restart=8, M_inv=flexible_precond)
    rel_b = torch.linalg.norm(X - X_true) / torch.linalg.norm(X_true)
    print('block_fgmres rel error =', float(rel_b))
    print('info_b =', info_b)
    assert rel_b < 1e-8
    print('PASS')


if __name__ == '__main__':
    main()
