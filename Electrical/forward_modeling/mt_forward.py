import ctypes
import os
import sys
from pathlib import Path


def _preload_conda_cxx_runtime():
    """Prefer the active conda C++ runtime before torch loads system libstdc++."""
    prefixes = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefixes.append(Path(conda_prefix))

    python_exe = Path(sys.executable).resolve()
    if python_exe.parent.name == "bin":
        prefixes.append(python_exe.parent.parent)

    seen = set()
    for prefix in prefixes:
        if prefix in seen:
            continue
        seen.add(prefix)
        lib_dir = prefix / "lib"
        loaded_any = False
        for lib_name in ("libstdc++.so.6", "libgcc_s.so.1"):
            lib_path = lib_dir / lib_name
            if not lib_path.exists():
                continue
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                loaded_any = True
            except OSError:
                pass
        if loaded_any:
            return


def _preload_local_mfem_stack():
    """Preload locally installed MFEM stack libraries when present."""
    root = Path(__file__).resolve().parent
    candidates = [
        root / ".local" / "libCEED-cuda" / "lib" / "libceed.so",
        root / ".local" / "hypre-cuda" / "lib" / "libHYPRE.so",
        root / ".local" / "mfem-cuda" / "lib" / "libmfem.so",
    ]
    for lib_path in candidates:
        if not lib_path.exists():
            continue
        try:
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


_preload_conda_cxx_runtime()
_preload_local_mfem_stack()

import torch
import torch.nn as nn
import importlib


def _load_backend():
    errors = []
    module_candidates = [
        (".mt_forward_mfem", __package__),
        ("mt_forward_mfem", None),
        (".legacy_cuda.mt_forward_cuda", __package__),
    ]

    for module_name, package in module_candidates:
        if module_name.startswith(".") and not package:
            continue
        try:
            return importlib.import_module(module_name, package=package), None
        except ImportError as exc:
            errors.append(f"{module_name}: {exc}")

    legacy_dir = Path(__file__).resolve().parent / "legacy_cuda"
    if legacy_dir.exists():
        sys.path.insert(0, str(legacy_dir))
        try:
            return importlib.import_module("mt_forward_cuda"), None
        except ImportError as exc:
            errors.append(f"legacy_cuda/mt_forward_cuda: {exc}")
        finally:
            try:
                sys.path.remove(str(legacy_dir))
            except ValueError:
                pass

    message = (
        "Failed to import an MT forward backend. Build the MFEM backend "
        "`mt_forward_mfem`, or keep the legacy CUDA extension available under "
        "`legacy_cuda/`. Import attempts:\n- " + "\n- ".join(errors)
    )
    return None, ImportError(message)


mt_backend, _BACKEND_IMPORT_ERROR = _load_backend()


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


def generate_mt_frequencies(f_min: float, f_max: float) -> list[float]:
    """
    Public helper for building the Phoenix-style MT frequency list used by
    the project from a requested frequency range.
    """
    return _generate_phoenix_frequencies(float(f_min), float(f_max))


def resolve_auto_mt_frequencies(rho_tensor: torch.Tensor, dz: float):
    bg_rho = _estimate_boundary_background_rho(rho_tensor)
    skin_depth_min = dz * 2.0
    total_depth = rho_tensor.shape[2] * dz
    skin_depth_max = total_depth / 1.5

    f_max = bg_rho * (503.0 / skin_depth_min) ** 2
    f_min = bg_rho * (503.0 / skin_depth_max) ** 2
    freqs = _generate_phoenix_frequencies(f_min, f_max)
    return freqs, bg_rho, f_min, f_max


def _sort_frequencies_desc(freqs) -> list[float]:
    return sorted((float(freq) for freq in freqs), reverse=True)


def _backend_is_mfem(backend) -> bool:
    return backend is not None and backend.__name__.split(".")[-1] == "mt_forward_mfem"


def mfem_cuda_enabled() -> bool:
    """Return whether the loaded MFEM backend was compiled with CUDA support."""
    return bool(
        _backend_is_mfem(mt_backend)
        and hasattr(mt_backend, "is_cuda_enabled")
        and mt_backend.is_cuda_enabled()
    )


def mfem_ceed_enabled() -> bool:
    """Return whether the loaded MFEM backend was compiled with libCEED."""
    return bool(
        _backend_is_mfem(mt_backend)
        and hasattr(mt_backend, "is_ceed_enabled")
        and mt_backend.is_ceed_enabled()
    )


def _device_requests_ceed(device: str) -> bool:
    return "ceed-" in str(device).lower()


def _default_mfem_device() -> str:
    return os.environ.get("GMES_MT_MFEM_DEVICE", "cpu")


class MTForward3D(nn.Module):
    def __init__(
        self,
        freqs=None,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        device=None,
        use_partial_assembly=False,
        npad_xy=10,
        npad_z=10,
        alpha=1.4,
        rel_tol=1e-6,
        max_iter=2000,
        verbose=False,
    ):
        """
        PyTorch wrapper for MFEM/CUDA MT 3D Forward Modeling.
        Args:
            freqs: Optional list of frequencies (Hz). If omitted or empty,
                the frequency range is auto-generated from the model extent and
                boundary background resistivity following MTForward3D.
            dx: Cell width in x-direction (m)
            dy: Cell width in y-direction (m)
            dz: Cell width in z-direction (m)
            device: MFEM device string, e.g. "cpu" or "cuda". If omitted,
                GMES_MT_MFEM_DEVICE is used when set, otherwise "cpu". For a
                libCEED-backed GPU path use strings such as
                "ceed-cuda:/gpu/cuda/shared".
            use_partial_assembly: Enable MFEM partial assembly. This is the
                route needed for device-oriented execution such as CUDA or
                ceed-cuda.
        """
        super(MTForward3D, self).__init__()
        self.freqs = None if freqs is None else list(freqs)
        self.last_freqs = None
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.device = _default_mfem_device() if device is None else str(device)
        self.use_partial_assembly = bool(use_partial_assembly)
        self.npad_xy = int(npad_xy)
        self.npad_z = int(npad_z)
        self.alpha = float(alpha)
        self.rel_tol = float(rel_tol)
        self.max_iter = int(max_iter)
        self.verbose = bool(verbose)
        if _device_requests_ceed(self.device) and not self.use_partial_assembly:
            raise ValueError(
                "MFEM CEED devices require use_partial_assembly=True."
            )

    def forward(self, rho_tensor):
        """
        Forward MT 3D.
        Args:
            rho_tensor: 3D contiguous tensor of shape (NX, NY, NZ) with resistivity values (Ohm-m).
        Returns:
            app_res: Apparent resistivity tensor of shape (n_freqs, NX, NY, 2),
                where the last dimension is (Zxy, Zyx). Returned on the same
                device as `rho_tensor`.
            phase: Phase tensor of shape (n_freqs, NX, NY, 2),
                where the last dimension is (Zxy, Zyx). Returned on the same
                device as `rho_tensor`.
        """
        if mt_backend is None:
            raise _BACKEND_IMPORT_ERROR

        input_device = rho_tensor.device
        rho_prepared = rho_tensor.detach().to(torch.float64)
        rho_solver_input = rho_prepared.cpu().contiguous()
        if self.freqs:
            freqs = _sort_frequencies_desc(self.freqs)
        else:
            freqs, _, _, _ = resolve_auto_mt_frequencies(rho_solver_input, self.dz)
            freqs = _sort_frequencies_desc(freqs)

        self.last_freqs = tuple(freqs)

        if _backend_is_mfem(mt_backend):
            app_res_raw, phase_raw = mt_backend.compute_mt_3d(
                rho_solver_input.numpy(),
                self.dx,
                self.dy,
                self.dz,
                freqs,
                self.npad_xy,
                self.npad_z,
                self.alpha,
                self.use_partial_assembly,
                self.rel_tol,
                self.max_iter,
                self.verbose,
                device=self.device,
            )
            app_res = torch.as_tensor(app_res_raw, dtype=torch.float64)
            phase = torch.as_tensor(phase_raw, dtype=torch.float64)
        else:
            app_res_raw, phase_raw = mt_backend.compute_mt_3d(
                rho_solver_input, self.dx, self.dy, self.dz, freqs
            )
            app_res = torch.as_tensor(app_res_raw, dtype=torch.float64)
            phase = torch.as_tensor(phase_raw, dtype=torch.float64)

        if input_device.type != "cpu":
            app_res = app_res.to(input_device)
            phase = phase.to(input_device)

        if not _backend_is_mfem(mt_backend) and torch.cuda.is_available():
            torch.cuda.synchronize()
        return app_res, phase
