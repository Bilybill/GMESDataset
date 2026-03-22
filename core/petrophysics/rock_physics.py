import numpy as np
from scipy.ndimage import gaussian_filter

class PetrophysicsConverter:
    """
    面向深度学习预训练的多物理场随机岩石物理转换引擎。
    采用空间相关的3D高斯随机场算法，确保生成的参数分布具有真实的“斑块状”地质非均质性，避免纯白噪声带来的非物理跳变。
    """
    def __init__(self):
        # 初始化时不固定参数，而是在调用时动态生成
        pass

    def _generate_correlated_noise(self, shape, sigma=3.0):
        """
        生成具有空间相关性的 3D 噪声，避免纯白噪声导致的非物理跳变。
        """
        noise = np.random.normal(0, 1, shape)
        filtered_noise = gaussian_filter(noise, sigma=sigma)
        # 归一化到近似 N(0, 1) 的正态分布，保证乘性或加性系数容易控制
        std_val = np.std(filtered_noise)
        if std_val > 1e-8:
            filtered_noise = filtered_noise / std_val
        return filtered_noise

    def generate_background(self, vp_model, label_vol=None):
        """
        根据 Vp 模型生成具有动态参数及空间相干分布特征的背景多物理场。
        如果提供 label_vol，将依据地层标签分配不同的岩性物理分布参数，从而加强不同物理场在层界面的结构一致性。
        """
        print("Generating diverse background multi-physics properties with correlated heterogeneity...")
        shape = vp_model.shape
        
        # 初始化输出矩阵
        rho_model = np.zeros_like(vp_model)
        res_model = np.zeros_like(vp_model)
        chi_model = np.zeros_like(vp_model)
        
        # 1. 噪声场（全局相干）
        noise_a = self._generate_correlated_noise(shape, sigma=8.0)
        noise_rw = self._generate_correlated_noise(shape, sigma=15.0)
        noise_chi = self._generate_correlated_noise(shape, sigma=5.0)
        
        # 如果没有层标签，默认当成单一沉积岩层处理
        if label_vol is None:
            labels = [0]
            label_vol = np.zeros_like(vp_model, dtype=int)
        else:
            labels = np.unique(label_vol)
            
        Vf = np.random.uniform(1450.0, 1600.0) # 全局水速
        
        for lbl in labels:
            mask = (label_vol == lbl)
            if not np.any(mask):
                continue
                
            # 为每一层随机指派一种"伪岩性"（砂岩/泥岩/碳酸盐岩等特征）
            litho_type = np.random.choice(['sandstone', 'shale', 'carbonate'])
            
            if litho_type == 'sandstone':
                # 砂岩: 骨架速度居中，电阻率较高，导电性由含水孔隙主导，孔隙度相对较高
                a_base = np.random.uniform(300.0, 320.0)
                b = np.random.uniform(0.24, 0.26)
                Vm = np.random.uniform(5000.0, 5600.0)
                Rw_base = np.random.uniform(0.5, 2.5) 
                m = np.random.uniform(1.8, 2.1)
                chi_mean = np.random.uniform(1e-5, 5e-5)
                
            elif litho_type == 'shale':
                # 泥岩: 密度系数小，骨架速度偏低，含泥质导电（即使Rw高，整体电阻率也相对较低）
                a_base = np.random.uniform(280.0, 305.0)
                b = np.random.uniform(0.26, 0.29)
                Vm = np.random.uniform(4000.0, 4800.0)
                Rw_base = np.random.uniform(0.1, 0.8) # 模拟泥岩低阻（高矿化度/阳离子交换）
                m = np.random.uniform(1.9, 2.3)
                chi_mean = np.random.uniform(5e-5, 1.5e-4) # 泥岩往往含更多顺磁性黏土矿物
                
            else: # carbonate
                # 碳酸盐: 高速高密，致密高阻，极低磁性
                a_base = np.random.uniform(310.0, 340.0)
                b = np.random.uniform(0.22, 0.24)
                Vm = np.random.uniform(6000.0, 7000.0)
                Rw_base = np.random.uniform(1.0, 4.0)
                m = np.random.uniform(2.0, 2.5)
                chi_mean = np.random.uniform(0.0, 2e-5)

            # --- A. 密度 (Gardner) ---
            a_field = np.clip(a_base + 10.0 * noise_a[mask], 250.0, 350.0)
            rho_model[mask] = a_field * np.power(vp_model[mask], b)
            
            # --- B. 电阻率 (Wyllie + Archie) ---
            vp_clipped = np.clip(vp_model[mask], Vf + 10, Vm - 10)
            phi = (1.0/vp_clipped - 1.0/Vm) / (1.0/Vf - 1.0/Vm)
            phi = np.clip(phi, 0.005, 0.5) # 极少数致密岩必须给个极小孔隙，防止除0
            
            Rw_field = np.clip(Rw_base + 0.2 * Rw_base * noise_rw[mask], 0.05, 10.0)
            
            # 增加地层微小非均质孔隙/饱和度扰动
            res_model[mask] = Rw_field / np.power(phi, m)
            
            # --- C. 磁化率 ---
            c_val = chi_mean + (chi_mean * 0.5) * noise_chi[mask]
            chi_model[mask] = np.clip(c_val, 0.0, None)
            
        return rho_model, res_model, chi_model

    def apply_anomaly(self, mask, anomaly_type, vp, rho, res, chi):
        """
        为地质异常体赋予区间随机采样的多物理场属性。
        使用3D过滤噪声模拟地质体内部的物性不均匀分布。
        """
        mask = mask.astype(bool)
        if not np.any(mask):
            return vp, rho, res, chi
            
        shape = vp.shape

        # 辅助函数：生成包含均值采样和全空间相干扰动数组
        def generate_heterogeneous_prop(low, high, noise_level=0.05, sigma=4.0):
            base_val = np.random.uniform(low, high)
            noise_3d = self._generate_correlated_noise(shape, sigma=sigma)
            # 在基准值上叠加 3D 空间内的相干波动
            vals = base_val + base_val * noise_level * noise_3d[mask]
            return np.clip(vals, low * 0.8, high * 1.2) # 限制极端值
            
        def generate_log_heterogeneous_prop(low, high, noise_level=0.1, sigma=4.0):
            """专门为跨度大的属性（如电阻率）对数域生成分布"""
            log_low, log_high = np.log10(low), np.log10(high)
            log_base = np.random.uniform(log_low, log_high)
            noise_3d = self._generate_correlated_noise(shape, sigma=sigma)
            # 在对数域上加减扰动 (noise_level为对数尺度的标准差)
            log_vals = log_base + noise_level * noise_3d[mask]
            log_vals = np.clip(log_vals, log_low - 0.2, log_high + 0.2)
            return np.power(10, log_vals)

        if anomaly_type == 'Igneous':
            vp[mask]  = generate_heterogeneous_prop(5000.0, 6500.0, noise_level=0.05)
            rho[mask] = generate_heterogeneous_prop(2700.0, 3100.0, noise_level=0.03)
            # 火成岩电阻率和磁性不均匀极其显著
            res[mask] = generate_log_heterogeneous_prop(2000.0, 10000.0, noise_level=0.2, sigma=6.0)
            chi[mask] = generate_heterogeneous_prop(0.01, 0.08, noise_level=0.20)
            
        elif anomaly_type == 'Sulfide':
            vp[mask]  = generate_heterogeneous_prop(5500.0, 6800.0, noise_level=0.04)
            rho[mask] = generate_heterogeneous_prop(3500.0, 4800.0, noise_level=0.05)
            res[mask] = generate_log_heterogeneous_prop(0.001, 0.1, noise_level=0.3, sigma=3.0) # 极低阻金属矿
            chi[mask] = generate_heterogeneous_prop(0.05, 0.2, noise_level=0.15)  # 高磁性
            
        elif anomaly_type == 'Gas':
            vp[mask]  = generate_heterogeneous_prop(1400.0, 2200.0, noise_level=0.05)
            rho[mask] = generate_heterogeneous_prop(1500.0, 2000.0, noise_level=0.03)
            # 气体由于饱和度的相干变化，导致电阻率放大倍数也应该是相干的
            base_multiplier_log = np.random.uniform(np.log10(5.0), np.log10(50.0))
            noise_res = self._generate_correlated_noise(shape, sigma=6.0)
            multiplier_log_field = base_multiplier_log + 0.2 * noise_res[mask]
            multiplier_field = np.power(10, np.clip(multiplier_log_field, np.log10(2.0), np.log10(100.0)))
            res[mask] = res[mask] * multiplier_field
            
            # 抗磁气体自身磁化率几乎没有空间非均质
            chi_gas = np.random.uniform(-1e-5, 0)
            chi[mask] = chi_gas 
            
        elif anomaly_type == 'Hydrate':
            vp[mask]  = generate_heterogeneous_prop(3000.0, 4200.0, noise_level=0.08)
            # 密度轻微下降，下降系数引入相干空间变动
            rho_decay_base = np.random.uniform(0.90, 0.98)
            noise_rho = self._generate_correlated_noise(shape, sigma=5.0)
            rho_decay_field = np.clip(rho_decay_base + 0.02 * noise_rho[mask], 0.85, 0.99)
            rho[mask] = rho[mask] * rho_decay_field
            
            # 水合物饱和度高度不均匀，对电阻率影响极大，采用对数相干扰动
            res_mult_base_log = np.random.uniform(np.log10(3.0), np.log10(15.0))
            noise_res = self._generate_correlated_noise(shape, sigma=4.0)
            res_mult_log_field = res_mult_base_log + 0.2 * noise_res[mask]
            res_mult_field = np.power(10, np.clip(res_mult_log_field, np.log10(1.5), np.log10(30.0)))
            res[mask] = res[mask] * res_mult_field
            
        elif anomaly_type == 'BrineFault':
            # 破碎带：孔隙度和裂缝的相干变化引起 Vp/密度下降
            vp_decay_base = np.random.uniform(0.85, 0.98)
            rho_decay_base = np.random.uniform(0.90, 0.98)
            noise_fault = self._generate_correlated_noise(shape, sigma=8.0)
            
            vp_decay_field = np.clip(vp_decay_base + 0.03 * noise_fault[mask], 0.70, 0.99)
            rho_decay_field = np.clip(rho_decay_base + 0.02 * noise_fault[mask], 0.80, 0.99)
            
            vp[mask]  = vp[mask] * vp_decay_field
            rho[mask] = rho[mask] * rho_decay_field
            
            # 卤水充填极低阻，由于裂缝通道不同产生相干分布
            res[mask] = generate_log_heterogeneous_prop(0.1, 2.0, noise_level=0.3, sigma=3.0)
            
        elif anomaly_type == 'Serpentinized':
            vp_decay_base = np.random.uniform(0.65, 0.85)
            rho_decay_base = np.random.uniform(0.80, 0.92)
            noise_serp = self._generate_correlated_noise(shape, sigma=10.0)
            
            vp[mask] = vp[mask] * np.clip(vp_decay_base + 0.05 * noise_serp[mask], 0.5, 0.95)
            rho[mask] = rho[mask] * np.clip(rho_decay_base + 0.03 * noise_serp[mask], 0.7, 0.98)
            
            # 生成磁铁矿，磁化率绝对增加，具有相干斑块特征
            chi_add_base = np.random.uniform(0.01, 0.05)
            chi_add_field = chi_add_base + chi_add_base * 0.2 * noise_serp[mask]
            chi[mask] = chi[mask] + np.clip(chi_add_field, 0.0, 0.1)
            
            res[mask] = generate_log_heterogeneous_prop(50.0, 500.0, noise_level=0.3, sigma=5.0)
            
        elif anomaly_type == 'SaltDome':
            vp[mask]  = generate_heterogeneous_prop(4200.0, 5500.0, noise_level=0.02, sigma=15.0)
            # 盐非常均质
            rho[mask] = generate_heterogeneous_prop(2100.0, 2250.0, noise_level=0.01, sigma=20.0)
            res[mask] = generate_log_heterogeneous_prop(10000.0, 50000.0, noise_level=0.1, sigma=10.0)
            chi[mask] = np.random.uniform(-1.5e-5, -0.5e-5) # 抗磁且高度均质
            
        else:
            print(f"Warning: Unknown anomaly type '{anomaly_type}'")
            
        return vp, rho, res, chi
