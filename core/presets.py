from dataclasses import dataclass
from typing import Callable
import numpy as np

try:
    import segyio
except ImportError:
    segyio = None

from .anomalies.brine_fault_zone import BrineFaultZone, BrineFaultZoneParams
from .anomalies.hydrocarbon_hydrate import HydrocarbonHydrate, HydrocarbonHydrateParams
from .anomalies.igneous_intrusion import IgneousIntrusion, IgneousIntrusionParams
from .anomalies.massive_sulfide import MassiveSulfide, MassiveSulfideParams
from .anomalies.salt_dome_anomaly import SaltDomeAnomaly
from .anomalies.sediment_basement_interface import SedimentBasementInterface, SedimentBasementParams
from .anomalies.serpentinized_zone import SerpentinizedZone, SerpentinizedZoneParams


DEFAULT_SEGY_SPACING = (10.0, 10.0, 25.0)


@dataclass(frozen=True)
class PresetBuildContext:
    vp_bg: object
    label_vol: object
    spacing: tuple[float, float, float]

    @property
    def shape(self):
        return self.vp_bg.shape


@dataclass(frozen=True)
class AnomalyPreset:
    key: str
    name_en: str
    name_zh: str
    anomaly: object


@dataclass(frozen=True)
class RegisteredAnomaly:
    key: str
    name_en: str
    name_zh: str
    factory: Callable[[PresetBuildContext], object]
    include_in_forward: bool = False
    include_in_default_viz: bool = True
    requires_explicit_include: bool = False


def read_segy_volume(path, spacing=DEFAULT_SEGY_SPACING, verbose=True):
    if segyio is None:
        raise ImportError("segyio is required to read SEGY volumes.")
    if verbose:
        print(f"Reading SEGY: {path}")
    with segyio.open(path, iline=segyio.TraceField.INLINE_3D, xline=segyio.TraceField.CROSSLINE_3D) as f:
        try:
            vol = segyio.tools.cube(f)
        except Exception:
            f.mmap()
            vol = segyio.tools.cube(f)
    dx, dy, dz = spacing
    return vol, (float(dx), float(dy), float(dz))


def _build_igneous_swarm(ctx: PresetBuildContext):
    nx, ny, nz = ctx.shape
    dx, dy, dz = ctx.spacing
    x_span_m = max((nx - 1) * dx, dx)
    y_span_m = max((ny - 1) * dy, dy)
    z_span_m = max((nz - 1) * dz, dz)
    center_x_m = 0.5 * x_span_m
    center_y_m = 0.5 * y_span_m
    lateral_span_m = max(x_span_m, y_span_m)
    swarm_count = int(np.clip(round(min(x_span_m, y_span_m) / 220.0), 7, 12))
    spacing_m = float(np.clip(min(x_span_m, y_span_m) / max(swarm_count + 3, 1), 80.0, 180.0))
    length_max_m = float(np.clip(lateral_span_m * 0.95, 1800.0, 10000.0))
    length_min_m = float(np.clip(length_max_m * 0.45, 1000.0, max(length_max_m - 200.0, 1000.0)))
    vertical_base_m = max(800.0, z_span_m * 0.78)
    vertical_base_m = float(min(vertical_base_m, z_span_m * 0.95))
    params = IgneousIntrusionParams(
        kind="swarm",
        dyke_x0_m=center_x_m,
        dyke_y0_m=center_y_m,
        dyke_z0_m=0.5 * vertical_base_m,
        dyke_thickness_m=24.0,
        dyke_length_m=length_max_m,
        dyke_width_m=vertical_base_m,
        dyke_strike_deg=35.0,
        dyke_dip_deg=88.0,
        swarm_count=swarm_count,
        swarm_spacing_m=spacing_m,
        swarm_fan_deg=4.0,
        swarm_spacing_jitter_frac=0.12,
        swarm_parallel_jitter_deg=0.8,
        swarm_dip_jitter_deg=0.6,
        swarm_thickness_min_m=20.0,
        swarm_thickness_max_m=30.0,
        swarm_length_min_m=length_min_m,
        swarm_length_max_m=length_max_m,
        swarm_echelon_zone_frac=0.5,
        swarm_echelon_step_m=spacing_m * 0.8,
        swarm_top_z_m=0.0,
        swarm_base_z_m=vertical_base_m,
        vp_intr_mps=5000.0,
        aureole_enable=True,
        aureole_thickness_m=30.0,
        edge_width_m=1.5,
    )
    return IgneousIntrusion(params=params, layer_labels=None, rng_seed=101)


def _build_igneous_stock(ctx: PresetBuildContext):
    params = IgneousIntrusionParams(
        kind="stock",
        stock_xc_m=2200.0,
        stock_yc_m=2200.0,
        stock_z_top_m=1500.0,
        stock_z_base_m=4500.0,
        stock_radius_m=400.0,
        vp_intr_mps=5800.0,
        aureole_enable=True,
    )
    return IgneousIntrusion(params=params, layer_labels=None, rng_seed=202)


def _build_gas(ctx: PresetBuildContext):
    params = HydrocarbonHydrateParams(
        kind="gas",
        layer_id=-1,
        center_x_m=1500.0,
        center_y_m=1500.0,
        lens_extent_x_m=1200.0,
        lens_extent_y_m=700.0,
        lens_thickness_m=120.0,
        vp_gas_mps=1800.0,
        gas_enable_chimney=True,
        chimney_height_m=1200.0,
        rng_seed=11,
    )
    return HydrocarbonHydrate(params=params, layer_labels=ctx.label_vol)


def _build_hydrate(ctx: PresetBuildContext):
    params = HydrocarbonHydrateParams(
        kind="hydrate",
        layer_id=-1,
        center_x_m=2000.0,
        center_y_m=2000.0,
        hydrate_offset_above_m=40.0,
        hydrate_thickness_m=70.0,
        vp_hydrate_mps=3800.0,
        hard_gate_to_layer=True,
        hydrate_enable_patchy=False,
        rng_seed=22,
    )
    return HydrocarbonHydrate(params=params, layer_labels=ctx.label_vol)


def _build_brine_fault(ctx: PresetBuildContext):
    params = BrineFaultZoneParams(
        top_k=2,
        fault_quantile=0.996,
        core_thickness_m=40.0,
    )
    return BrineFaultZone(params=params, vp_ref=ctx.vp_bg, rng_seed=999)


def _build_sediment_basement(ctx: PresetBuildContext):
    params = SedimentBasementParams(
        use_layer_labels=True,
        basement_layer_id=-2,
        vp_basement_mps=6200.0,
        rng_seed=777,
    )
    return SedimentBasementInterface(params=params, layer_labels=ctx.label_vol)


def _build_serpentinized(ctx: PresetBuildContext):
    params = SerpentinizedZoneParams(
        mode="corridor",
        use_layer_labels=True,
        corridor_length_m=2400.0,
        corridor_halfwidth_m=180.0,
        thickness_m=420.0,
        halo_thickness_m=180.0,
        fault_gate_radius_m=260.0,
        matrix_volume_frac=0.60,
        block_target_frac=0.32,
        resolved_block_count=28,
        block_diameter_min_m=5.0,
        block_diameter_max_m=30.0,
        vein_spacing_m=48.0,
        rng_seed=1212,
    )
    return SerpentinizedZone(params=params, layer_labels=ctx.label_vol)


def _build_massive_sulfide(ctx: PresetBuildContext):
    params = MassiveSulfideParams(
        layer_id=-1,
        center_x_m=1000.0,
        center_y_m=1000.0,
        lens_extent_x_m=600.0,
        lens_extent_y_m=500.0,
        lens_thickness_m=150.0,
    )
    return MassiveSulfide(params=params, layer_labels=ctx.label_vol)


def _build_salt_dome(ctx: PresetBuildContext):
    nx, ny, nz = ctx.shape
    dx, dy, dz = ctx.spacing
    params = SaltDomeAnomaly.create_random_params((nx, ny, nz), (dx, dy, dz), seed=333)
    return SaltDomeAnomaly(
        type="salt_dome",
        strength=0.0,
        edge_width_m=20.0,
        params=params,
        rng_seed=333,
    )


ANOMALY_REGISTRY = (
    RegisteredAnomaly(
        key="igneous_swarm",
        name_en="Igneous_Swarm",
        name_zh="岩墙群",
        factory=_build_igneous_swarm,
        include_in_forward=True,
    ),
    RegisteredAnomaly(
        key="igneous_stock",
        name_en="Igneous_Stock",
        name_zh="岩株",
        factory=_build_igneous_stock,
        requires_explicit_include=True,
    ),
    RegisteredAnomaly(
        key="gas",
        name_en="Hydrocarbon_Gas",
        name_zh="气藏",
        factory=_build_gas,
    ),
    RegisteredAnomaly(
        key="hydrate",
        name_en="Hydrocarbon_Hydrate",
        name_zh="天然气水合物",
        factory=_build_hydrate,
    ),
    RegisteredAnomaly(
        key="brine_fault",
        name_en="Brine_Fault",
        name_zh="含卤水断层",
        factory=_build_brine_fault,
        include_in_forward=True,
    ),
    RegisteredAnomaly(
        key="massive_sulfide",
        name_en="Massive_Sulfide",
        name_zh="块状硫化物",
        factory=_build_massive_sulfide,
        include_in_forward=True,
    ),
    RegisteredAnomaly(
        key="salt_dome",
        name_en="Salt_Dome",
        name_zh="盐丘",
        factory=_build_salt_dome,
        include_in_forward=True,
    ),
    RegisteredAnomaly(
        key="sediment_basement",
        name_en="Sediment_Basement",
        name_zh="沉积-基底界面",
        factory=_build_sediment_basement,
    ),
    RegisteredAnomaly(
        key="serpentinized",
        name_en="Serpentinized_Zone",
        name_zh="蛇纹岩化带",
        factory=_build_serpentinized,
        include_in_forward=True,
    ),
)

ANOMALY_REGISTRY_BY_KEY = {entry.key: entry for entry in ANOMALY_REGISTRY}
FORWARD_ANOMALY_TYPES = tuple(entry.key for entry in ANOMALY_REGISTRY if entry.include_in_forward)
REGISTERED_ANOMALY_TYPES = tuple(entry.key for entry in ANOMALY_REGISTRY)


def _build_preset_from_entry(entry: RegisteredAnomaly, ctx: PresetBuildContext):
    return AnomalyPreset(
        key=entry.key,
        name_en=entry.name_en,
        name_zh=entry.name_zh,
        anomaly=entry.factory(ctx),
    )


def build_registered_presets(vp_bg, label_vol, spacing, keys):
    ctx = PresetBuildContext(vp_bg=vp_bg, label_vol=label_vol, spacing=spacing)
    presets = []
    for key in keys:
        entry = ANOMALY_REGISTRY_BY_KEY.get(key)
        if entry is None:
            raise ValueError(f"Unsupported anomaly type: {key}")
        presets.append(_build_preset_from_entry(entry, ctx))
    return presets


def build_named_anomaly_preset(anomaly_type, vp_bg, label_vol, spacing):
    return build_registered_presets(vp_bg, label_vol, spacing, [anomaly_type])[0]


def build_default_viz_presets(vp_bg, label_vol, spacing, include_stock=False):
    keys = []
    for entry in ANOMALY_REGISTRY:
        if not entry.include_in_default_viz:
            continue
        if entry.requires_explicit_include and not include_stock:
            continue
        keys.append(entry.key)
    return build_registered_presets(vp_bg, label_vol, spacing, keys)


def build_all_anomalies(vp_bg, label_vol, spacing, include_stock=True):
    keys = []
    for entry in ANOMALY_REGISTRY:
        if entry.requires_explicit_include and not include_stock:
            continue
        keys.append(entry.key)
    return [preset.anomaly for preset in build_registered_presets(vp_bg, label_vol, spacing, keys)]
