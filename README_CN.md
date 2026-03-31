# GMESDataset: 地质多物理场地球模拟数据集

[English Version](./README.md)

**GMESDataset** 是一个用于构建三维地质异常模型并生成重力、磁法、电法、地震四类正演响应的合成地球建模框架。项目现在围绕统一的异常体注册表、相控约束的多物理属性构建器以及四套正演求解器组织，因此同一个地质目标可以被稳定地可视化、正演和打包保存。

## 核心特性

### 1. 基于注册表的地质异常建模
当前代码在 [`core/presets.py`](./core/presets.py) 中维护了一套统一的异常体注册表，集中管理 preset 名称、构建函数以及可视化/正演暴露规则。

当前已注册的异常体包括：
- `igneous_swarm`：带热接触晕的岩墙群侵入体
- `igneous_stock`：岩株 / 岩栓侵入体
- `gas`：油气气藏 / 气烟囱系统
- `hydrate`：天然气水合物系统，可带下伏游离气
- `brine_fault`：含卤水断层核与破碎带
- `massive_sulfide`：透镜状块状硫化物 + 烟囱 + 网脉浸染
- `salt_dome`：盐丘构造
- `sediment_basement`：沉积-基底界面
- `serpentinized`：蛇纹岩化蚀变带

### 2. 相控一致的多物理属性生成
以背景速度体和层位标签体为输入，GMESDataset 会生成空间对齐的四类三维属性：
- `vp`：纵波速度
- `rho`：密度，单位 `g/cm^3`
- `res`：电阻率，单位 `Ohm-m`
- `chi`：磁化率，单位 SI

背景属性先由相控岩石物理关系生成，再叠加各类异常体的定制扰动，因此四种物性之间保持一致性。

### 3. 四类正演引擎
- `Seismic`：基于 `deepwave` 的三维声学波动方程正演
- `Gravity`：三维重力异常正演
- `Magnetic`：三维磁异常正演
- `Electrical`：通过 `Electrical/forward_modeling` 扩展实现的三维 MT 正演

### 4. 统一结果打包
`run_multiphysics_forward.py` 会将三维模型、采集几何以及各物理场正演结果统一保存到一个 `forward_bundle.npz` 中，方便后续成对读取、训练和可视化。

## 目录结构

```text
GMESDataset/
├── core/
│   ├── anomalies/                 # 地质异常体实现
│   ├── forward_modeling/          # 重 / 磁 / 电 / 震封装入口
│   ├── petrophysics/              # 岩石物理转换关系
│   ├── presets.py                 # 统一异常注册表与 SEGY 读取辅助
│   ├── multiphysics.py            # 多物理属性构建器
│   └── viz_utils.py               # 可视化共享辅助函数
├── Electrical/                    # MT 正演扩展与测试脚本
├── Gravity/                       # 重力正演代码
├── Magnetic/                      # 磁法正演代码
├── Seismic/                       # 基于 Deepwave 的地震正演代码
├── run_multiphysics_viz.py        # 单个异常体四属性可视化
├── run_all_anomalies_viz.py       # 多异常体对比可视化
├── run_separate_anomalies_viz.py  # 分异常体保存可视化结果
├── run_multiphysics_forward.py    # 重 / 磁 / 电 / 震联合正演入口
└── plot_saved_forward_data.py     # 读取已保存结果并二次绘图
```

## 快速开始

### 1. 环境依赖
核心依赖包括：
- `numpy`, `scipy`, `matplotlib`
- `torch`
- `segyio`
- `cigvis`，用于三维可视化
- `deepwave`，用于地震正演

推荐：
- 使用支持 CUDA 的 PyTorch 环境，以加速地震、重力、磁法以及 MT 的预处理/求解流程

### 2. 可视化一个多物理异常体
推荐使用基于注册表的可视化入口：

```bash
python run_multiphysics_viz.py
```

该脚本会根据配置的 SEGY 背景模型构建一个异常体，并联合显示 `vp / rho / res / chi` 四类属性。

### 3. 运行联合正演
联合正演主入口为：

```bash
python run_multiphysics_forward.py --device cuda
```

常用参数：
- `--anomaly-type {igneous_swarm,brine_fault,massive_sulfide,salt_dome}`
- `--seismic-preset {full,light}`
- `--seismic-batch-size N`
- `--skip_mt`
- `--skip_seismic`

示例：

```bash
python run_multiphysics_forward.py \
  --anomaly-type massive_sulfide \
  --device cuda \
  --seismic-preset light \
  --seismic-batch-size 8
```

### 4. 查看保存结果
正常运行结束后，三维模型和正演结果会被统一保存为：

```text
DATAFOLDER/Cache/ForwardOutput/forward_bundle.npz
```

当前 bundle 的键结构说明见 [`README_FORWARD_OUTPUTS.md`](./README_FORWARD_OUTPUTS.md)。

## 地球物理属性特征表

下表基于当前代码默认参数汇总了各类地质目标的典型物性趋势，以及它们在四法联合解释中的主要特征。若文档与代码不一致，请以实现为准；表中数值做了适度四舍五入，便于阅读。

| 目标 / Preset（中文） | 速度 `Vp` | 密度 `rho` | 电阻率 `res` | 磁化率 `chi` | 联合解释特征 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `igneous_swarm`, `igneous_stock`（岩浆侵入体：岩墙群、岩株） | 偏高。注册表默认约为 `5000-5800 m/s`，接触晕局部再增加约 `+3%`。 | 偏高，典型约 `3.0 g/cm^3`，接触晕约 `+2%`。 | 很高，约 `5000 Ohm-m`，接触晕相对侵入体核部略有下降。 | 偏高，约 `0.05 SI`，接触晕略有增强。 | 四种方法对比都较明显，是目前最均衡的联合建模 benchmark 目标之一。 |
| `massive_sulfide`（块状硫化物） | 核部很高，约 `5200-6200 m/s`；外围 halo 略低于背景。 | 偏高，约 `3.0-4.0 g/cm^3`。 | 很低，约 `0.5-10 Ohm-m`，是项目里最强导电目标之一。 | 高到很高，约 `1.8e-3` 到 `1.2e-2 SI`，并叠加弱磁性 halo。 | 典型矿体响应：高密度、高磁化率、低电阻率，异常范围较紧凑。 |
| `gas`（气藏 / 气烟囱系统） | 偏低，约 `1800 m/s`。 | 偏低，约 `2.0 g/cm^3`。 | 偏高，约 `100 Ohm-m`。 | 偏低或接近背景。 | 地震低速特征突出，磁法通常较弱，电法表现为相对高阻。 |
| `hydrate`（天然气水合物） | 水合物层偏高，约 `3700 m/s`；下伏游离气约 `2000 m/s`。 | 中等，水合物约 `2.3 g/cm^3`，游离气约 `2.1 g/cm^3`。 | 水合物偏高，约 `200 Ohm-m`；游离气约 `50 Ohm-m`。 | 偏低或接近背景。 | 更适合表现层状地震和电性差异，磁异常通常不明显。 |
| `brine_fault`（含卤水断层） | 当前默认 preset 基本中性；代码里断层核与破碎带速度扰动默认都是 `0%`。 | 默认接近背景。 | 导电性极强，断层核约 `0.5 Ohm-m`。 | 默认接近背景。 | 主要是 MT 靶体；若不额外改密度/磁化率，重磁异常通常较弱。 |
| `salt_dome`（盐丘） | 偏高，随机取值约 `4500-5500 m/s`。 | 偏低，约 `2.15 g/cm^3`。 | 很高，约 `3000 Ohm-m`。 | 很低到轻微负值，约 `-1e-5 SI`。 | 典型“高速、低密度、高阻、弱磁”的盐体目标。 |
| `sediment_basement`（沉积-基底界面） | 沉积层从约 `1700` 增长到 `4000 m/s`；基底在当前 preset 中约 `6200 m/s`。 | 沉积层约 `1.95-2.45 g/cm^3`；基底约 `2.75 g/cm^3`。 | 沉积层约 `5-80 Ohm-m`；基底约 `2000 Ohm-m`。 | 沉积层约 `5e-4 SI`；基底约 `0.02 SI`。 | 适合做大尺度结构基准，界面对四种物性的跨层对比都很明显。 |
| `serpentinized`（蛇纹岩化带） | 相对背景降低约 `25%`。 | 相对背景降低约 `12%`。 | 相对背景降低约 `30%`。 | 相对背景绝对增加约 `+0.02 SI`。 | 典型蚀变型目标：更慢、更轻，但磁化率显著升高。 |

## 说明

- `run_multiphysics_forward.py` 当前通过 `--anomaly-type` 对外暴露了四个开箱即用的联合正演靶体：`igneous_swarm`、`brine_fault`、`massive_sulfide`、`salt_dome`。
- 更完整的异常注册表已经在可视化和后续扩展中可用。
- 如果 README 与代码出现偏差，请以 [`core/`](./core) 下实现和 [`run_multiphysics_forward.py`](./run_multiphysics_forward.py) 为准。
