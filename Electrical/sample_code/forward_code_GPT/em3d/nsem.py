from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence
import math
import numpy as np
import torch

from .grid import Grid3D, Receiver as PointReceiver
from .forward import simulate_mt_primary_secondary_batch


@dataclass
class BaseRx:
    locations: np.ndarray
    orientation: str
    component: str
    rx_type: str

    def __post_init__(self):
        arr = np.asarray(self.locations, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, 3)
        if arr.shape[1] != 3:
            raise ValueError('locations must have shape (N,3)')
        self.locations = arr
        self.orientation = self.orientation.lower()
        self.component = self.component.lower()

    @property
    def nD(self) -> int:
        return int(self.locations.shape[0])

    def point_receivers(self) -> list[PointReceiver]:
        return [PointReceiver(float(x), float(y), float(z), name=f'{self.rx_type}:{self.orientation}:{i}')
                for i, (x, y, z) in enumerate(self.locations)]

    def eval_from_mt_dict(self, d: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class Impedance(BaseRx):
    def __init__(self, locations, orientation='xy', component='real'):
        super().__init__(locations=locations, orientation=orientation, component=component, rx_type='impedance')

    def eval_from_mt_dict(self, d: dict[str, torch.Tensor]) -> torch.Tensor:
        key = f'Z{self.orientation}'
        if key not in d:
            raise KeyError(f'{key} not present in MT response dict')
        z = d[key]
        if self.component == 'real':
            return z.real
        if self.component == 'imag':
            return z.imag
        if self.component == 'complex':
            return z
        if self.component in {'apparent_resistivity', 'rho'}:
            return d[f'rho_{self.orientation}']
        if self.component == 'phase':
            return d[f'phi_{self.orientation}_deg']
        raise ValueError(f'Unsupported impedance component {self.component}')


class Tipper(BaseRx):
    def __init__(self, locations, orientation='zx', component='real'):
        super().__init__(locations=locations, orientation=orientation, component=component, rx_type='tipper')

    def eval_from_mt_dict(self, d: dict[str, torch.Tensor]) -> torch.Tensor:
        key = f'T{self.orientation}'
        if key not in d:
            raise KeyError(f'{key} not present in MT response dict')
        t = d[key]
        if self.component == 'real':
            return t.real
        if self.component == 'imag':
            return t.imag
        if self.component == 'complex':
            return t
        if self.component in {'amp', 'amplitude'}:
            return t.abs()
        if self.component == 'phase':
            return torch.angle(t) * 180.0 / math.pi
        raise ValueError(f'Unsupported tipper component {self.component}')


class Rx:
    Impedance = Impedance
    Tipper = Tipper


@dataclass
class PlanewaveXYPrimary:
    receiver_list: Sequence[BaseRx]
    frequency: float

    @property
    def receivers(self) -> Sequence[BaseRx]:
        return self.receiver_list


class Src:
    PlanewaveXYPrimary = PlanewaveXYPrimary


class Survey:
    def __init__(self, source_list: Sequence[PlanewaveXYPrimary]):
        self.source_list = list(source_list)
        self._build_index()

    def _build_index(self):
        self._slices = []
        start = 0
        for src in self.source_list:
            per_src = []
            for rx in src.receivers:
                stop = start + rx.nD
                per_src.append(slice(start, stop))
                start = stop
            self._slices.append(per_src)
        self.nD = start

    @property
    def frequencies(self) -> np.ndarray:
        return np.array([src.frequency for src in self.source_list], dtype=float)

    def flat_receivers(self) -> list[PointReceiver]:
        pts = []
        seen = set()
        for src in self.source_list:
            for rx in src.receivers:
                for p in rx.point_receivers():
                    key = (round(p.x, 9), round(p.y, 9), round(p.z, 9))
                    if key not in seen:
                        pts.append(p)
                        seen.add(key)
        return pts

    def lookup_indices(self, points: list[PointReceiver]) -> dict[tuple[float, float, float], int]:
        return {(round(p.x, 9), round(p.y, 9), round(p.z, 9)): i for i, p in enumerate(points)}


class Data:
    def __init__(self, survey: Survey, dobs: torch.Tensor | np.ndarray):
        self.survey = survey
        if isinstance(dobs, np.ndarray):
            dobs = torch.from_numpy(dobs)
        self.dobs = dobs.detach().cpu().clone().reshape(-1)
        if self.dobs.numel() != survey.nD:
            raise ValueError(f'dobs length {self.dobs.numel()} != survey.nD {survey.nD}')
        self.relative_error: float | torch.Tensor | None = None
        self.noise_floor: float | torch.Tensor | None = None

    def numpy(self) -> np.ndarray:
        return self.dobs.numpy()

    def _find_rx(self, freq: float, rx_type: str, orientation: str, component: str) -> tuple[int, int, BaseRx] | None:
        for isrc, src in enumerate(self.survey.source_list):
            if abs(src.frequency - freq) > 1e-12:
                continue
            for irx, rx in enumerate(src.receivers):
                if rx.rx_type == rx_type and rx.orientation == orientation and rx.component == component:
                    return isrc, irx, rx
        return None

    def _value_at_location(self, slice_: slice, rx: BaseRx, location_xy: np.ndarray) -> float:
        target = np.asarray(location_xy, dtype=float).reshape(1, 2)
        xy = rx.locations[:, :2]
        d2 = np.sum((xy - target) ** 2, axis=1)
        idx = int(np.argmin(d2))
        return float(self.dobs[slice_.start + idx].item())

    def _collect_complex_series(self, location_xy, orientation='xy', rx_type='impedance'):
        freqs = sorted(set(float(s.frequency) for s in self.survey.source_list), reverse=True)
        vals = []
        keep_freqs = []
        for f in freqs:
            fr = self._find_rx(f, rx_type, orientation, 'real')
            fi = self._find_rx(f, rx_type, orientation, 'imag')
            if fr is None or fi is None:
                continue
            isr, irx, rxr = fr
            _, iix, rxi = fi
            vr = self._value_at_location(self.survey._slices[isr][irx], rxr, location_xy)
            vi = self._value_at_location(self.survey._slices[isr][iix], rxi, location_xy)
            vals.append(vr + 1j * vi)
            keep_freqs.append(f)
        return np.array(keep_freqs), np.array(vals)

    def plot_app_res(self, location_xy, components=('xy', 'yx'), ax=None, errorbars=False):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(1, 1)
        for comp in components:
            freqs, z = self._collect_complex_series(location_xy, orientation=comp, rx_type='impedance')
            if freqs.size == 0:
                continue
            rho = (np.abs(z) ** 2) / (4e-7 * math.pi * 2.0 * math.pi * freqs)
            if errorbars and self.relative_error is not None:
                yerr = np.asarray(self.relative_error) * rho
                ax.errorbar(freqs, rho, yerr=yerr, marker='o', label=comp)
            else:
                ax.plot(freqs, rho, marker='o', label=comp)
        return ax

    def plot_app_phs(self, location_xy, components=('xy', 'yx'), ax=None, errorbars=False):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(1, 1)
        for comp in components:
            freqs, z = self._collect_complex_series(location_xy, orientation=comp, rx_type='impedance')
            if freqs.size == 0:
                continue
            phase = np.angle(z, deg=True)
            if errorbars and self.relative_error is not None:
                yerr = np.zeros_like(phase)
                ax.errorbar(freqs, phase, yerr=yerr, marker='o', label=comp)
            else:
                ax.plot(freqs, phase, marker='o', label=comp)
        return ax


class Simulation3DPrimarySecondary:
    def __init__(self, mesh: Grid3D, survey: Survey, sigma: torch.Tensor, sigmaPrimary: torch.Tensor | None = None,
                 forward_only: bool = True, tol: float = 1e-6, maxiter: int = 240, restart: int = 20,
                 use_block_secondary: bool = True, precond: str = 'line', layered_bg_mode: str = 'mean'):
        self.mesh = mesh
        self.survey = survey
        self.sigma = sigma
        self.sigmaPrimary = sigmaPrimary
        self.forward_only = forward_only
        self.tol = tol
        self.maxiter = maxiter
        self.restart = restart
        self.use_block_secondary = use_block_secondary
        self.precond = precond
        self.layered_bg_mode = layered_bg_mode
        self._last_mt = None
        self._last_infos = None

    def _compute_mt_dicts(self):
        # use union of receiver points across all sources, then index into each rx object
        all_points = self.survey.flat_receivers()
        lookup = self.survey.lookup_indices(all_points)
        freqs = [src.frequency for src in self.survey.source_list]
        mt_dicts, infos = simulate_mt_primary_secondary_batch(
            self.mesh, self.sigma, self.sigmaPrimary, freqs, all_points,
            tol=self.tol, maxiter=self.maxiter, restart=self.restart,
            use_block_secondary=self.use_block_secondary, precond=self.precond,
            layered_bg_mode=self.layered_bg_mode,
        )
        self._last_mt, self._last_infos = mt_dicts, infos
        return mt_dicts, infos, lookup

    def dpred(self) -> torch.Tensor:
        mt_dicts, _, lookup = self._compute_mt_dicts()
        vals = []
        for isrc, src in enumerate(self.survey.source_list):
            d = mt_dicts[isrc]
            for rx in src.receivers:
                full = rx.eval_from_mt_dict(d)
                idxs = [lookup[(round(float(x), 9), round(float(y), 9), round(float(z), 9))] for x, y, z in rx.locations]
                vals.append(full[torch.tensor(idxs, device=full.device, dtype=torch.long)].reshape(-1))
        if vals:
            out = torch.cat(vals)
        else:
            out = torch.empty(0, dtype=self.sigma.real.dtype, device=self.sigma.device)
        if out.is_complex():
            return out
        return out.real

    def make_synthetic_data(self, relative_error=0.05, noise_floor=0.0, add_noise=True, seed: int = 0) -> Data:
        pred = self.dpred().detach().cpu()
        if pred.is_complex():
            mag = pred.abs()
            std = relative_error * mag + noise_floor
            if add_noise:
                g = torch.Generator(device='cpu'); g.manual_seed(seed)
                noise = std / math.sqrt(2.0) * (torch.randn(pred.shape, generator=g) + 1j * torch.randn(pred.shape, generator=g))
                pred = pred + noise
        else:
            mag = pred.abs()
            std = relative_error * mag + noise_floor
            if add_noise:
                g = torch.Generator(device='cpu'); g.manual_seed(seed)
                pred = pred + std * torch.randn(pred.shape, generator=g)
        data = Data(self.survey, pred)
        data.relative_error = relative_error
        data.noise_floor = noise_floor
        return data


class NSEMNamespace:
    Rx = Rx
    Src = Src
    Survey = Survey
    Data = Data
    Simulation3DPrimarySecondary = Simulation3DPrimarySecondary


NSEM = NSEMNamespace()
