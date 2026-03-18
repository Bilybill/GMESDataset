from __future__ import annotations
import torch
from .grid import Grid3D, EdgeField
from .averaging import sigma_to_edges
from .operators import MU0


def dotc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.conj() * b).sum()


def _small_ls(H: torch.Tensor, G: torch.Tensor, lam: float = 1e-12):
    if G.ndim == 1:
        G = G.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False
    A = H.conj().T @ H
    n = A.shape[0]
    A = A + lam * torch.eye(n, dtype=A.dtype, device=A.device)
    B = H.conj().T @ G
    Y = torch.linalg.solve(A, B)
    return Y.reshape(-1) if squeeze else Y


def gmres(A_mv, b: torch.Tensor, x0: torch.Tensor | None = None, tol: float = 1e-6,
          maxiter: int = 200, restart: int = 40, M_inv=None):
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    bn = torch.linalg.norm(b)
    if bn == 0:
        return x, {"converged": True, "iterations": 0, "residual": 0.0}
    total_iter = 0
    r = b - A_mv(x)
    if M_inv is not None:
        r = M_inv(r)
    beta = torch.linalg.norm(r)
    if beta / bn < tol:
        return x, {"converged": True, "iterations": 0, "residual": float((beta / bn).real)}
    while total_iter < maxiter:
        V = []
        H = torch.zeros((restart + 1, restart), dtype=b.dtype, device=b.device)
        g = torch.zeros(restart + 1, dtype=b.dtype, device=b.device)
        g[0] = beta
        V.append(r / beta)
        for j in range(restart):
            w = A_mv(V[j])
            if M_inv is not None:
                w = M_inv(w)
            for i in range(j + 1):
                H[i, j] = dotc(V[i], w)
                w = w - H[i, j] * V[i]
            H[j + 1, j] = torch.linalg.norm(w)
            V.append(w / H[j + 1, j] if H[j + 1, j].abs() > 0 else torch.zeros_like(w))
            y = _small_ls(H[:j + 2, :j + 1], g[:j + 2])[:j + 1]
            x_candidate = x + sum(y[k] * V[k] for k in range(j + 1))
            res = torch.linalg.norm(b - A_mv(x_candidate)) / bn
            total_iter += 1
            if res < tol:
                return x_candidate, {"converged": True, "iterations": total_iter, "residual": float(res.real)}
            if total_iter >= maxiter:
                return x_candidate, {"converged": False, "iterations": total_iter, "residual": float(res.real)}
        y = _small_ls(H[:restart + 1, :restart], g[:restart + 1])[:restart]
        x = x + sum(y[k] * V[k] for k in range(restart))
        r = b - A_mv(x)
        if M_inv is not None:
            r = M_inv(r)
        beta = torch.linalg.norm(r)
        if beta / bn < tol:
            return x, {"converged": True, "iterations": total_iter, "residual": float((beta / bn).real)}
    return x, {"converged": False, "iterations": total_iter, "residual": float((beta / bn).real)}


def _qr_block(X: torch.Tensor, eps: float = 1e-12):
    Q, R = torch.linalg.qr(X, mode='reduced')
    diag = torch.abs(torch.diag(R))
    keep = diag > eps
    if keep.all():
        return Q, R, int(Q.shape[1])
    if keep.sum() == 0:
        n = X.shape[0]; p = X.shape[1]
        return torch.zeros((n, 0), dtype=X.dtype, device=X.device), torch.zeros((0, p), dtype=X.dtype, device=X.device), 0
    idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
    Q2 = Q[:, idx]
    R2 = Q2.conj().T @ X
    return Q2, R2, int(Q2.shape[1])


def block_gmres(A_mv, B: torch.Tensor, X0: torch.Tensor | None = None, tol: float = 1e-6,
                maxiter: int = 200, restart: int = 20, M_inv=None):
    if B.ndim != 2:
        raise ValueError('B must have shape (n, nrhs)')
    n, p = B.shape
    X = torch.zeros_like(B) if X0 is None else X0.clone()
    Bn = torch.linalg.norm(B)
    if Bn == 0:
        return X, {"converged": True, "iterations": 0, "residual": 0.0, "block_size": p}
    total_iter = 0
    R = B - A_mv(X)
    if M_inv is not None:
        R = M_inv(R)
    res0 = torch.linalg.norm(R) / Bn
    if res0 < tol:
        return X, {"converged": True, "iterations": 0, "residual": float(res0.real), "block_size": p}
    while total_iter < maxiter:
        V = []
        Q1, Beta, rdim = _qr_block(R)
        if rdim == 0:
            return X, {"converged": True, "iterations": total_iter, "residual": 0.0, "block_size": p}
        V.append(Q1)
        H = torch.zeros(((restart + 1) * rdim, restart * rdim), dtype=B.dtype, device=B.device)
        G = torch.zeros(((restart + 1) * rdim, p), dtype=B.dtype, device=B.device)
        G[:rdim, :] = Beta
        for j in range(restart):
            W = A_mv(V[j])
            if M_inv is not None:
                W = M_inv(W)
            for i in range(j + 1):
                Hij = V[i].conj().T @ W
                H[i * rdim:(i + 1) * rdim, j * rdim:(j + 1) * rdim] = Hij
                W = W - V[i] @ Hij
            Qn, Rn, newdim = _qr_block(W)
            if newdim < rdim:
                padQ = torch.zeros((n, rdim), dtype=B.dtype, device=B.device)
                padR = torch.zeros((rdim, rdim), dtype=B.dtype, device=B.device)
                if newdim > 0:
                    padQ[:, :newdim] = Qn
                    padR[:newdim, :] = Rn
                Qn, Rn = padQ, padR
            H[(j + 1) * rdim:(j + 2) * rdim, j * rdim:(j + 1) * rdim] = Rn
            V.append(Qn)
            Y = _small_ls(H[: (j + 2) * rdim, : (j + 1) * rdim], G[: (j + 2) * rdim, :])[: (j + 1) * rdim, :]
            Xcand = X.clone()
            for k in range(j + 1):
                Xcand = Xcand + V[k] @ Y[k * rdim:(k + 1) * rdim, :]
            res = torch.linalg.norm(B - A_mv(Xcand)) / Bn
            total_iter += 1
            if res < tol:
                return Xcand, {"converged": True, "iterations": total_iter, "residual": float(res.real), "block_size": p}
            if total_iter >= maxiter:
                return Xcand, {"converged": False, "iterations": total_iter, "residual": float(res.real), "block_size": p}
        Y = _small_ls(H[: (restart + 1) * rdim, : restart * rdim], G[: (restart + 1) * rdim, :])[: restart * rdim, :]
        for k in range(restart):
            X = X + V[k] @ Y[k * rdim:(k + 1) * rdim, :]
        R = B - A_mv(X)
        if M_inv is not None:
            R = M_inv(R)
        res = torch.linalg.norm(R) / Bn
        if res < tol:
            return X, {"converged": True, "iterations": total_iter, "residual": float(res.real), "block_size": p}
    return X, {"converged": False, "iterations": total_iter, "residual": float(res.real), "block_size": p}


def jacobi_precond_from_diag(diag: torch.Tensor, floor: float = 1e-8):
    scale = 1.0 / torch.clamp(diag.abs(), min=floor)
    def _apply(x: torch.Tensor):
        return scale.unsqueeze(-1) * x if x.ndim == 2 else scale * x
    return _apply


def _thomas_tridiag(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, rhs: torch.Tensor):
    n = b.numel()
    if rhs.ndim == 1:
        rhs = rhs.reshape(n, 1)
        squeeze = True
    else:
        squeeze = False
    cp = c.clone()
    dp = rhs.clone()
    bp = b.clone()
    for i in range(1, n):
        m = a[i - 1] / bp[i - 1]
        bp[i] = bp[i] - m * cp[i - 1]
        dp[i] = dp[i] - m * dp[i - 1]
    x = torch.zeros_like(dp)
    x[-1] = dp[-1] / bp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dp[i] - cp[i] * x[i + 1]) / bp[i]
    return x.reshape(-1) if squeeze else x


def vertical_line_precond(grid: Grid3D, sigma: torch.Tensor, omega: float):
    sigma_e = sigma_to_edges(sigma.to(grid.rdtype), grid)
    cdtype = torch.complex128 if sigma.dtype == torch.float64 else torch.complex64

    def _solve_lines(field: torch.Tensor, sigma_line_tensor: torch.Tensor, axis_len: int, dz: float):
        # field shape (..., axis_len); sigma_line_tensor same shape
        out = torch.empty_like(field)
        batch_shape = field.shape[:-1]
        for idx in torch.cartesian_prod(*[torch.arange(s, device=field.device) for s in batch_shape]) if len(batch_shape) > 0 else [torch.tensor([], device=field.device)]:
            key = tuple(idx.tolist()) if idx.numel() else ()
            vec = field[key]
            sline = sigma_line_tensor[key]
            n = axis_len
            if n == 1:
                out[key] = vec / (1.0 - 1j * omega * MU0 * sline)
                continue
            a = torch.full((n - 1,), -1.0 / dz ** 2, dtype=cdtype, device=field.device)
            c = a.clone()
            xy_shift = 2.0 / max(grid.dx, 1e-12) ** 2 + 2.0 / max(grid.dy, 1e-12) ** 2
            mass_shift = torch.abs(-1j * omega * MU0 * sline.to(cdtype))
            beta = (mass_shift + xy_shift + (2.0 / dz ** 2)).to(cdtype)
            b = beta.clone()
            b[0] = (mass_shift[0] + xy_shift + (1.0 / dz ** 2)).to(cdtype)
            b[-1] = (mass_shift[-1] + xy_shift + (1.0 / dz ** 2)).to(cdtype)
            out[key] = _thomas_tridiag(a, b, c, vec)
        return out

    def _apply(x: torch.Tensor):
        mat = x.ndim == 2
        if mat:
            cols = [_apply(x[:, i]) for i in range(x.shape[1])]
            return torch.stack(cols, dim=1)
        ef = EdgeField.unflatten(x, grid)
        ex = _solve_lines(ef.ex, sigma_e.ex.to(cdtype), grid.nz + 1, grid.dz)
        ey = _solve_lines(ef.ey, sigma_e.ey.to(cdtype), grid.nz + 1, grid.dz)
        ez = _solve_lines(ef.ez, sigma_e.ez.to(cdtype), grid.nz, grid.dz)
        return EdgeField(ex, ey, ez).flatten()
    return _apply
