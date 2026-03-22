import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import os
import sys
import numpy as np
import segyio
import cigvis

from core.builder import DatasetBuilder
from core.anomalies.igneous_intrusion import IgneousIntrusion, IgneousIntrusionParams
from core.anomalies.massive_sulfide import MassiveSulfide, MassiveSulfideParams
from core.anomalies.hydrocarbon_hydrate import HydrocarbonHydrate, HydrocarbonHydrateParams
from core.anomalies.brine_fault_zone import BrineFaultZone, BrineFaultZoneParams
from core.anomalies.sediment_basement_interface import SedimentBasementInterface, SedimentBasementParams
from core.anomalies.serpentinized_zone import SerpentinizedZone, SerpentinizedZoneParams
from core.anomalies.salt_dome_anomaly import SaltDomeAnomaly, SaltDomeParams

def read_segy_volume(path):
    print(f"Reading SEGY: {path}")
    with segyio.open(path, iline=segyio.TraceField.INLINE_3D, xline=segyio.TraceField.CROSSLINE_3D) as f:
        try:
            vol = segyio.tools.cube(f)
        except Exception:
            f.mmap()
            vol = segyio.tools.cube(f)
    print(f"Volume loaded with shape: {vol.shape}")
    dx, dy, dz = 10.0, 10.0, 25.0
    return vol, (dx, dy, dz)

def main():
    print("=== Small-Scale Visualization Verification with all anomalies ===")
    vp_segy_path = "/home/wangyh/DATAFOLDER/3DSeismic/AYLModel/3DExample/Velocity_choas/braided/AYL-00000.sgy"
    label_segy_path = "/home/wangyh/DATAFOLDER/3DSeismic/AYLModel/3DExample/Layer_choas/braided/AYL-00000.sgy"
    
    vp_bg, (dx, dy, dz) = read_segy_volume(vp_segy_path)
    label_vol, _ = read_segy_volume(label_segy_path)

    builder = DatasetBuilder(dx, dy, dz)
    anomalies = []

    print("--- 1. Adding Igneous Intrusion (Swarm & Stock) ---")
    swarm_params = IgneousIntrusionParams(
        kind="swarm", dyke_x0_m=1000.0, dyke_y0_m=1500.0, dyke_z0_m=2500.0,
        dyke_thickness_m=40.0, dyke_length_m=2500.0, dyke_width_m=3000.0,
        dyke_strike_deg=45.0, dyke_dip_deg=80.0, swarm_count=5, swarm_spacing_m=300.0,
        swarm_fan_deg=15.0, vp_intr_mps=5000.0, aureole_enable=True
    )
    anomalies.append(IgneousIntrusion(params=swarm_params, layer_labels=None, rng_seed=101))

    stock_params = IgneousIntrusionParams(
        kind="stock", stock_xc_m=2200.0, stock_yc_m=2200.0, stock_z_top_m=1500.0, stock_z_base_m=4500.0,
        stock_radius_m=400.0, vp_intr_mps=5800.0, aureole_enable=True
    )
    anomalies.append(IgneousIntrusion(params=stock_params, layer_labels=None, rng_seed=202))

    print("--- 2. Adding Hydrocarbon Gas & Hydrate ---")
    gas_params = HydrocarbonHydrateParams(
        kind="gas", layer_id=-1, center_x_m=1500.0, center_y_m=1500.0,
        lens_extent_x_m=1200.0, lens_extent_y_m=700.0, lens_thickness_m=120.0, vp_gas_mps=1800.0,
        gas_enable_chimney=True, chimney_height_m=1200.0, rng_seed=11
    )
    anomalies.append(HydrocarbonHydrate(params=gas_params, layer_labels=label_vol))

    hyd_params = HydrocarbonHydrateParams(
        kind="hydrate", layer_id=-1, center_x_m=2000.0, center_y_m=2000.0,
        hydrate_offset_above_m=40.0, hydrate_thickness_m=70.0, vp_hydrate_mps=3800.0,
        hard_gate_to_layer=True, hydrate_enable_patchy=False, rng_seed=22
    )
    anomalies.append(HydrocarbonHydrate(params=hyd_params, layer_labels=label_vol))

    print("--- 3. Adding Brine / Water-bearing Fault Zone ---")
    brine_params = BrineFaultZoneParams(top_k=2, fault_quantile=0.996, core_thickness_m=40.0)
    anomalies.append(BrineFaultZone(params=brine_params, vp_ref=vp_bg, rng_seed=999))

    print("--- 4. Adding Sediment-Basement Interface ---")
    sbi_params = SedimentBasementParams(
        use_layer_labels=True, basement_layer_id=-2, vp_basement_mps=6200.0, rng_seed=777
    )
    anomalies.append(SedimentBasementInterface(params=sbi_params, layer_labels=label_vol))

    print("--- 5. Adding Serpentinized Zone ---")
    serp_params = SerpentinizedZoneParams(
        mode="patchy", use_layer_labels=True, corridor_length_m=2200.0, rng_seed=1212
    )
    anomalies.append(SerpentinizedZone(params=serp_params, layer_labels=label_vol))

    print("--- 6. Adding Massive Sulfide ---")
    mass_sulfide_params = MassiveSulfideParams(
        layer_id=-1, center_x_m=1000.0, center_y_m=1000.0, lens_extent_x_m=600.0, lens_extent_y_m=500.0,
        lens_thickness_m=150.0
    )
    anomalies.append(MassiveSulfide(params=mass_sulfide_params, layer_labels=label_vol))

    print("--- 7. Adding Salt Dome ---")
    nx, ny, nz = vp_bg.shape
    salt_params = SaltDomeAnomaly.create_random_params((nx, ny, nz), (dx, dy, dz), seed=333)
    salt_dome = SaltDomeAnomaly(type="salt_dome", strength=0.0, edge_width_m=20.0, params=salt_params, rng_seed=333)
    anomalies.append(salt_dome)

    print("-> Injecting anomalies into background Vp...")
    vp_final, mask_final, _, _, _ = builder.inject_anomalies(vp_bg, anomalies)

    np.save("vp_all_anomalies.npy", vp_final)
    print("Saved vp_all_anomalies.npy successfully.")

    print("-> Start visualization using cigvis (Saving to png)...")
    try:
        # Transpose to Match CigVis X,Y,Z expected formats if needed, or directly pass.
        # usually (X, Y, Z) is what segyio.tools.cube returns and cigvis expects, 
        # but cigvis docs use segy shape. Let's see visulize.py logic:
        # (NZ, NY, NX) -> transposes (2,1,0). But if SEGY cube is already INLINE(x), CROSSLINE(y), TIME(z),
        # shape is (NX, NY, NZ), we can pass directly.
        # We will use center coordinates for slicing
        pos_vol = [vp_final.shape[0]//2, vp_final.shape[1]//2, vp_final.shape[2]//2]
        
        slices_bg = cigvis.create_slices(vp_bg, cmap='jet', pos=pos_vol, vmin=vp_bg.min(), vmax=vp_bg.max())
        slices_final = cigvis.create_slices(vp_final, cmap='jet', pos=pos_vol, vmin=vp_final.min(), vmax=vp_final.max())
        
        cbar_bg = cigvis.create_colorbar(cmap='jet', clim=[vp_bg.min(), vp_bg.max()], label_str='Background Vp (m/s)')
        cbar_final = cigvis.create_colorbar(cmap='jet', clim=[vp_final.min(), vp_final.max()], label_str='Final Anomaly Vp (m/s)')
        
        # 提取各个异常体的掩码 (用于 cigvis 高亮显示)
        # mask_final records the index of applied anomalies (1 for first, 2 for second, etc.)
        anomaly_bodies = []
        colors = ['red', 'yellow', 'cyan', 'magenta', 'green', 'orange', 'purple', 'blue']
        for i, anomaly in enumerate(anomalies):
            idx = i + 1
            # 取出当前异常体的布尔掩码
            sub_mask = (mask_final == idx).astype(np.float32)
            if sub_mask.sum() > 0:
                c = colors[i % len(colors)]
                # 使用 cigvis 创建该异常体的 3D 轮廓体
                anomaly_bodies += cigvis.create_bodys(sub_mask, level=0.5, color=c, alpha=0.3)
                print(f"Adding anomaly outline for: {anomaly.type} (cigvis body, color={c})")

        # Plot and save
        try:
            # We already have run_app=False, but cigvis/vispy might crash if no OpenGL Context exists in pure headless Linux.
            nodes1 = slices_bg + [cbar_bg]
            nodes2 = slices_final + anomaly_bodies + [cbar_final]
            cigvis.plot3D([nodes1, nodes2], grid=(1, 2), share=True, size=(1200, 600), savename='3D_Anomalies_Preview.png', run_app=False)
        except Exception as vis_e:
            print(f"cigvis plot3D directly failed ({vis_e}), falling back to matplotlib 3D slice plotting...")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(16, 8))
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')
            
            x_mid, y_mid, z_mid = pos_vol
            
            def plot_3d_slices(ax, vol, title, with_anomalies=False):
                nx, ny, nz = vol.shape
                x, y, z = np.arange(nx), np.arange(ny), np.arange(nz)
                cmap_m = plt.get_cmap('jet')
                norm = plt.Normalize(vmin=vol.min(), vmax=vol.max())
                
                stride = max(1, nx // 64)  # downsample for plotting speed so it does not hang

                # Z slice (horizontal) - Move to the bottom of the cube
                z_bottom = nz - 1
                yy, xx = np.meshgrid(y, x)
                zz = np.full_like(xx, z_bottom)
                ax.plot_surface(xx, yy, zz, facecolors=cmap_m(norm(vol[:, :, z_bottom])), rstride=stride, cstride=stride, shade=False, zorder=1)

                # X slice (vertical) - X slice lies on YZ plane
                zz_x, yy_x = np.meshgrid(z, y)
                xx_x = np.full_like(yy_x, x_mid)
                ax.plot_surface(xx_x, yy_x, zz_x, facecolors=cmap_m(norm(vol[x_mid, :, :])), rstride=stride, cstride=stride, shade=False, zorder=2)

                # Y slice (vertical) - Y slice lies on XZ plane
                zz_y, xx_y = np.meshgrid(z, x)
                yy_y = np.full_like(xx_y, y_mid)
                ax.plot_surface(xx_y, yy_y, zz_y, facecolors=cmap_m(norm(vol[:, y_mid, :])), rstride=stride, cstride=stride, shade=False, zorder=3)

                # 如果带有异常，我们在表面上方叠加半透明散点来勾勒出异常体形态
                if with_anomalies:
                    colors = ['red', 'yellow', 'cyan', 'magenta', 'green', 'orange', 'purple', 'blue']
                    for i in range(len(anomalies)):
                        idx = i + 1
                        pts = np.where(mask_final == idx)
                        if len(pts[0]) > 0:
                            # 降采样散点以防止卡死
                            skip = stride * 3
                            pts_ds = (pts[0][::skip], pts[1][::skip], pts[2][::skip])
                            c = colors[i % len(colors)]
                            ax.scatter(pts_ds[0], pts_ds[1], pts_ds[2], c=c, s=1, alpha=0.1, label=anomalies[i].type, zorder=4)
                    
                    # 取消注释以下代码将显示图例，但可能遮挡图形
                    # ax.legend(loc='upper right', fontsize='small')

                ax.set_title(title)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Depth (Z)')
                ax.invert_zaxis() # Invert Z axis so depth goes down
                
                # Add colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap_m, norm=norm)
                sm.set_array([])
                plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.1, label='Vp (m/s)')

            print("Rendering 3D slices with Matplotlib (this may take a few seconds)...")
            plot_3d_slices(ax1, vp_bg, "Background Vp Slices", with_anomalies=False)
            plot_3d_slices(ax2, vp_final, "Final Anomaly Vp Slices", with_anomalies=True)
            
            plt.tight_layout()
            plt.savefig('3D_Slices_Matplotlib.png', dpi=150)
            print("Done! Fallback 3D visualization saved to '3D_Slices_Matplotlib.png'")

        print("Done! Visualization saved to '3D_Anomalies_Preview.png'")
        
    except Exception as e:
        print(f"Failed to visualize: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
