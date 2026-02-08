import os
import yaml
import numpy as np
import torch
from loguru import logger


def load_config(config_path: str) -> dict:
    """Load YAML config (same style as Seismic/forward_modeling/utils.py)."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(path: str, config_path: str) -> str:
    """
    Resolve relative paths:
    - If `path` is relative, interpret it relative to the directory containing the config file.
    - If empty/None, return as-is.
    """
    if not path:
        return path
    if os.path.isabs(path):
        return path
    cfg_dir = os.path.dirname(os.path.abspath(config_path))
    return os.path.normpath(os.path.join(cfg_dir, path))


def load_density_model(config: dict, config_path: str) -> torch.Tensor:
    """
    Load density model as torch.Tensor with shape (nx, ny, nz), float32.

    Supports:
    - .npy: raw numpy array
    - .npz: specify key with model.npz_key (default tries 'density' then first key)
    - .bin: raw float32 binary, requires model.shape
    """
    mconf = config.get("model", {})
    fpath = _resolve_path(mconf.get("file_path", ""), config_path)
    shape = mconf.get("shape", None)
    npz_key = mconf.get("npz_key", None)

    if fpath and os.path.exists(fpath):
        ext = os.path.splitext(fpath)[1].lower()
        if ext == ".npy":
            arr = np.load(fpath)
        elif ext == ".npz":
            z = np.load(fpath)
            if npz_key and npz_key in z:
                arr = z[npz_key]
            elif "density" in z:
                arr = z["density"]
            else:
                # fallback: first key
                arr = z[z.files[0]]
        elif ext == ".bin":
            if shape is None:
                raise ValueError("model.shape is required for .bin density model.")
            arr = np.fromfile(fpath, dtype=np.float32).reshape(tuple(shape))
        else:
            raise ValueError(f"Unsupported density model format: {ext}")
        logger.info(f"Loaded density model: {fpath}, shape={arr.shape}, dtype={arr.dtype}")
    else:
        if shape is None:
            raise FileNotFoundError(f"Density model file not found: {fpath} and model.shape not provided.")
        default_rho = float(mconf.get("default_density", 2670.0))
        arr = np.full(tuple(shape), default_rho, dtype=np.float32)
        logger.warning(f"Density model not found, using constant density={default_rho} kg/m^3, shape={shape}")

    if arr.ndim != 3:
        raise ValueError(f"Density model must be 3D, got shape={arr.shape}")
    # Ensure (nx, ny, nz)
    if shape is not None and tuple(arr.shape) != tuple(shape):
        logger.warning(f"Density shape mismatch: file={arr.shape}, config.shape={shape}. Using file shape.")
    return torch.from_numpy(arr.astype(np.float32))
