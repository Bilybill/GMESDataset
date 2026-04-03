# GMESDataset 大规模预训练数据集构建规划

## 1. 当前阶段目标

GMESDataset 第一阶段已经完成：

- 多种地质异常模型构建
- 四类物性模型构建：`Vp / rho / resist / chi`
- 四种正演代码打通：`Gravity / Magnetic / MT / Seismic`

接下来进入项目核心阶段：**大规模预训练数据集生产**。

该阶段的任务拆分为两个连续步骤：

1. 针对海量速度模型生成统一的四参数模型
2. 基于四参数模型开展重磁电震联合正演


## 2. 数据源与现状约束

### 2.1 速度数据源

当前速度模型目录：

- `/home/wangyh/DATAFOLDER/3DSeismic/AYLModel/ALLvelocity`

文件格式：

- 以 `.bin` 为主
- 读取方式与 `Seismic/forward_modeling/utils.py::load_velocity_volume` 保持一致

当前目录结构适合直接镜像为大规模数据集目录，例如：

- `train-river/.../*.bin`
- `train-choas/.../*.bin`
- `tests-river/.../*.bin`
- `tests-choas/.../*.bin`

### 2.1.1 label 生成数据源

当前层位/标签体并不是直接跟随 `ALLvelocity` 预先存成 `.bin`，而是通过镜像目录下的 sample `npz` 中的 `gtime` 体生成：

- `/home/wangyh/Project/GMESUni/GMESDataset/DATAFOLDER/samples`

生成逻辑参考：

- `Seismic/sample_code/read_npz_trans2layer.py`

当前正式采用的逻辑链为：

1. 读取背景速度体 `vp`
2. 查找镜像 sample `npz`
3. 从 `gtime` 体生成离散 `label volume`
4. 用 `vp + label volume` 生成背景 `rho / resist / chi`
5. 再注入目标地质异常并进行四法正演

### 2.2 当前约束

- `ALLvelocity` 当前主要提供背景速度体，并没有与之完全配套的层位标签体
- 因此第二阶段的第一版实现必须支持 **`label_vol=None` 的无标签背景模式**
- 幸好当前 `PetrophysicsConverter.generate_background()` 已支持无标签输入，且现有前向异常体在当前实现下也能在无标签模式中运行


## 3. 推荐的数据集生产策略

### 3.1 生产分两层进行

建议将大规模数据生产显式拆分成两层产物：

#### A. 模型层（Models Stage）

输入：

- 单个背景速度体 `vp`
- 指定异常类型 `anomaly_type`

输出：

- `model_bundle.npz`

内容至少包含：

- `vp_model`
- `rho_model`
- `res_model`
- `chi_model`
- `rho_bg_model`
- `chi_bg_model`
- `anomaly_label`
- `facies_bg`
- `spacing`
- `anomaly metadata`
- `source metadata`

作用：

- 作为后续四法正演的统一输入
- 便于先检查模型质量，再决定是否进入昂贵的正演阶段

#### B. 正演层（Full Forward Stage）

输入：

- `model_bundle.npz` 对应的四参数模型

输出：

- `forward_bundle.npz`

内容包含：

- 四参数模型
- 重力异常
- 磁异常
- MT 响应
- 3D 地震响应
- 各方法观测系统
- 频率、波子、采集参数等元数据

作用：

- 形成真正可用于预训练的大规模联合模拟样本


## 4. 目录组织建议

建议将输出目录设计成“源路径镜像 + 异常类型分层”的结构：

```text
PretrainDataset/
  train-river/
    splayed/
      AYL-00076/
        igneous_swarm/
          model_bundle.npz
          forward_bundle.npz
        massive_sulfide/
          model_bundle.npz
          forward_bundle.npz
  tests-choas/
    braided/
      AYL-00123/
        serpentinized/
          model_bundle.npz
          forward_bundle.npz
```

这样设计的好处：

- 保留原始数据分层
- 支持多异常体对同一背景模型的对比实验
- 支持 resume / skip / 并行切分
- 便于后续建立 train/val/test 清单


## 5. 批量生产流程设计

### 5.1 扫描阶段

- 遍历 `ALLvelocity`
- 收集全部 `.bin` 文件
- 可按子目录过滤，例如仅跑 `tests-river`
- 为每个背景模型扩展出多个 `anomaly_type` 任务

### 5.2 模型构建阶段

对每个任务：

1. 读取背景速度体
2. 查找镜像 sample `npz` 并由 `gtime` 生成 `label volume`
3. 在统一 spacing 下基于 `vp + label volume` 构建四参数背景模型
4. 注入指定地质异常
5. 保存 `model_bundle.npz`

### 5.3 正演阶段

对每个模型任务：

1. 运行重力正演
2. 运行磁法正演
3. 运行 MT 正演
4. 运行 3D 地震正演
5. 保存 `forward_bundle.npz`

### 5.4 断点续跑阶段

批量脚本必须支持：

- `resume`：已有结果自动跳过
- `max_samples`：先抽样一小批 smoke test
- `split_dirs`：按目录子集跑
- `stop_on_error`：调试时可快速中断


## 6. 工程实现建议

### 6.1 不再复制第二套逻辑

批量数据集脚本不应该重新手写四法逻辑，而应该复用：

- `run_multiphysics_forward.py` 中的模型构建函数
- `run_multiphysics_forward.py` 中的联合正演函数

因此代码层面要做两件事：

1. 把“模型构建”和“联合正演”拆成可复用函数
2. 单独增加一个批量入口脚本，例如：
   - `build_pretraining_dataset.py`

### 6.2 大规模生产时默认关闭预览图

在海量生产场景下：

- 不应该默认为每个样本生成 PNG
- 应该以 `.npz` 为主
- 可单独提供少量抽样样本生成预览图

### 6.3 记录 manifest

建议每次批量运行都写入 `manifest.jsonl`，记录：

- `source path`
- `relative path`
- `anomaly_type`
- `stage`
- `output_dir`
- `status`
- `duration_sec`
- `error`

这样后续：

- 可以快速筛查失败样本
- 可以重跑失败任务
- 可以统计不同异常体和不同数据源分布


## 7. 质量控制（QC）建议

批量数据生产至少要做三类 QC：

### 7.1 模型级 QC

- 检查 `vp / rho / res / chi` 是否存在 NaN/Inf
- 检查异常标签是否非空
- 检查电阻率和密度是否落入合理范围

### 7.2 正演级 QC

- 重力/磁法结果是否全零
- MT 是否收敛，频率列表是否正确保存
- 地震是否产生非空 shot gather

### 7.3 数据集级 QC

- 不同异常类型数量是否平衡
- 不同背景数据源目录数量是否平衡
- train/test 目录是否混淆


## 8. 第二阶段代码落地目标

本轮开发建议先完成以下最低可用版本：

1. 增加通用模型构建接口：
   - 支持直接从 `.bin` 背景速度体生成四参数模型
   - 支持从镜像 sample `npz` 自动生成 `label volume`
2. 增加模型-only 保存能力：
   - `model_bundle.npz`
3. 增加批量脚本：
   - 扫描 `ALLvelocity`
   - 自动解析 sample `npz` 生成标签体
   - 多异常类型循环
   - 支持 `models/full`
   - 支持 `resume`
   - 支持 `manifest`
   - 支持自动 QC 汇总
4. 保持 `run_multiphysics_forward.py` 单样本入口继续可用


## 9. 下一步优先事项

在最低可用版本稳定后，再推进：

- 并行/多卡批量生产
- 标签体自动匹配
- 数据集统计与抽样可视化
- 更标准的数据索引文件（CSV/Parquet）
- 模型级和正演级自动 QC 报告


## 10. 本轮实施结论

本轮的工程目标不是一次性把全部数据集平台做满，而是先搭出一个可持续演进的主干：

- 单样本入口继续保留
- 批量入口统一复用单样本逻辑
- 模型与正演分阶段保存
- 支持断点续跑与错误追踪

这是后续开展大规模预训练数据构建最稳妥的基础。
