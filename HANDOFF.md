# FedNyq 项目交接文档

更新时间：2026-09-06

## 1. 项目目标

本仓库用于 ICASSP 实验，比较相同 `NyqNet` 架构下的三种方法：

1. `FedAvg`：标准样本数加权全模型平均；
2. `FedNyq-Agg`：仅使用 Nyquist-support-normalized aggregation，命令行为
   `--method fednyq --lambda_cons 0`；
3. `Full FedNyq`：FedNyq aggregation 加 cross-rate consistency，默认
   `--lambda_cons 0.2`。

联邦学习完全使用原生 PyTorch 顺序模拟，没有使用 Flower、FedML、Lightning
或其他联邦学习框架。

## 2. 仓库位置与主要文件

本地路径：

```text
C:\Users\dyan0\Documents\ChatGPT\New project 2
```

- `train.py`：训练入口、逐轮日志、最终汇总和 sweep 汇总；
- `config.py`：dataclass 配置与 argparse；
- `data.py`：数据读取、固定 train/test split、IID/Dirichlet 划分、客户端
  sampling-rate 分配、固定 schedule、FIR 抗混叠降采样；
- `model.py`：NyqNet、三个独立 band encoder/head、物理频率路由；
- `federated.py`：客户端训练、symmetric KL、FedAvg/FedNyq delta 聚合和诊断；
- `metrics.py`：总体/分 rate 指标、统计汇总和绘图；
- `utils.py`：随机种子、设备选择、哈希、模型大小；
- `make_synthetic.py`：simple/complex 两种可复现合成数据；
- `scripts/`：FedAvg、FedNyq、多 seed、rate sweep 脚本；
- `README.md`：方法、公式、安装与使用说明。

## 3. 已实现的关键技术行为

- 输入为 `[N,C,L]` 的最高原始采样率窗口；所有 rate 保持相同物理时长。
- `lowpass_resample()` 使用缓存的 Hann-windowed sinc FIR、grouped `conv1d`
  和整数倍率 decimation，不使用直接切片。
- 模型对传感器时序执行 `rFFT`，由 `sample_rate` 和
  `torch.fft.rfftfreq()` 获得真实物理频率。
- 25/50/100 Hz 分别激活 band1、band1+2、band1+2+3；inactive band 不执行，
  因而没有梯度。
- active band logits 取平均，不直接求和。
- FedAvg 是完整 state 的样本加权平均；inactive 参数保留 global 值后仍参与平均，
  因而保留理论上的 update dilution。
- FedNyq 对 band 参数只在 Nyquist 覆盖 band 上界的客户端之间重新归一化。
- consistency 只比较高 rate 输入和降采样输入的共同 Nyquist support。
- 相同 seed 的 split、partition、rate assignment、初始化和 client schedule
  均固定，并把四个哈希写入配置和最终结果。
- 支持 CPU、CUDA、`--device auto/cuda/cpu`、`--amp`、DataLoader workers、
  pinned memory、异步传输和客户端峰值 CUDA 显存记录。

## 4. 合成数据

### Simple

三类信号：

- class 0：3 Hz；
- class 1：3 Hz + 18 Hz；
- class 2：6 Hz + 30 Hz。

生成：

```powershell
python make_synthetic.py --output data/synthetic.npz --seed 1234
```

### Complex

复杂档保留上述 Nyquist-dependent 判别结构，同时加入：三通道混合、频率抖动、
幅度调制、间歇性高频、弱谐波、彩色噪声、基线漂移、脉冲伪迹、传感器增益变化
和少量通道丢失。生成器会打印每类的 B1/B2/B3 相对能量。

```powershell
python make_synthetic.py `
  --profile complex `
  --num_samples 6000 `
  --channels 3 `
  --noise 0.65 `
  --output data/synthetic_complex.npz `
  --seed 1234
```

更难版本可使用 `--noise 0.9 --num_samples 10000`。

## 5. 命令注意事项

- FedAvg 必须使用 `--lambda_cons 0`；代码会拒绝非零 consistency。
- Full FedNyq 使用 `--method fednyq --lambda_cons 0.2`。
- PowerShell 多行续行符是反引号 `` ` ``，不是反斜杠 `\`。
- 参数名不得写成 `--experiment\_name`，必须是 `--experiment_name`。
- 当前重采样器只支持 base rate 到目标 rate 的整数倍率降采样。

FedAvg 示例：

```powershell
python train.py `
  --data data/synthetic_complex.npz `
  --method fedavg `
  --experiment_name complex_fedavg `
  --rates 25 50 100 `
  --rate_probs 0.4 0.4 0.2 `
  --lambda_cons 0 `
  --seed 2025 `
  --partition dirichlet `
  --dirichlet_alpha 0.2 `
  --device cuda `
  --amp
```

FedNyq-Agg 示例：

```powershell
python train.py `
  --data data/synthetic_complex.npz `
  --method fednyq `
  --experiment_name complex_fednyq_agg `
  --rates 25 50 100 `
  --rate_probs 0.4 0.4 0.2 `
  --lambda_cons 0 `
  --seed 2025 `
  --partition dirichlet `
  --dirichlet_alpha 0.2 `
  --device cuda `
  --amp
```

Full FedNyq 只需改成：

```text
--experiment_name complex_fednyq_full --lambda_cons 0.2
```

## 6. 已完成的工程验证

- 所有 Python 文件通过语法检查。
- simple/complex 数据均验证 shape、dtype、finite values 和 metadata。
- 25/50/100 Hz 模型路由分别验证为 1/2/3 个 active bands。
- 100→25 Hz 输出长度验证为 400→100。
- 30 Hz 正弦降到 25 Hz 后 RMS 从约 0.707 降至约 0.0093，确认抗混叠有效。
- FedAvg 和 FedNyq 均完成过缩小规模的真实端到端训练、CSV/JSON/PNG 输出。
- consistency 客户端训练分支已单独执行验证。
- 构造 delta 测试中，当 `rho_band3=0.25` 时，FedAvg band3 更新为 0.25，
  FedNyq 更新为 1.0，empirical ratio 为 0.25，与公式一致。
- 多 seed summary 和 rate-sweep summary 命令均完成过 smoke test。

## 7. 用户已报告的实验结果

以下结果来自用户粘贴的终端输出，当前工作区未找到相应 `outputs/` 文件，不能在本机
进一步检查逐轮 CSV。

### 实验 A

```text
FedAvg: accuracy 0.9983, macro-F1 0.9983
         25/50/100 Hz F1 = 0.5556 / 0.6301 / 0.9983
FedNyq: accuracy 0.9950, macro-F1 0.9950
         25/50/100 Hz F1 = 0.6565 / 0.9649 / 0.9950
```

初步解释：FedNyq 基本保持 100 Hz 性能，显著改善 50 Hz，并把 25 Hz 的类别
预测变得更均衡。25 Hz 下 class 0 和 class 1 在抗混叠后都主要只剩 3 Hz，理论
accuracy 上限接近 2/3；Macro-F1 比 accuracy 更能暴露某一类别是否完全塌缩。

### 实验 B（强 non-IID，疑似 `dirichlet_alpha=0.2`）

```text
FedAvg: accuracy 0.9917, macro-F1 0.9916
         25/50/100 Hz F1 = 0.5556 / 1.0000 / 0.9916
FedNyq: accuracy 0.7883, macro-F1 0.7646
         25/50/100 Hz F1 = 0.5556 / 0.9983 / 0.7646
```

这里的下降几乎完全位于 100 Hz 专属 band3。当前默认 20 个客户端、high-rate
比例 0.2 时只有 4 个 100 Hz 客户端；每轮选择 10 个时，约 4.3% 的轮次没有
high-rate client，约 24.8% 只有一个。再叠加 `alpha=0.2` 的强 label skew，
FedNyq 会把少数、高方差、可能类别偏置的 band3 更新从约 `rho=0.2` 放大到完整
尺度。FedAvg 的 dilution 在这种情形下反而充当了隐式正则化。

这说明 support normalization 修正幅值偏差，但不会自动修正 eligible client
稀少导致的方差或标签偏差。该解释必须用实际 CSV 和消融进一步验证，不能仅凭单 seed
定论。

## 8. 新对话建议优先执行的检查

1. 让用户提供或把对应 `outputs/<experiment>/<seed>/` 放入当前工作区。
2. 在 `client_metadata.csv` 中筛选 `sampling_rate=100`，检查仅有的 high-rate
   客户端是否整体或分别缺少某些标签。
3. 在 `round_metrics.csv` 检查：
   `rho_band3`、`eligible_clients_band3`、`mean_eligible_delta_norm_band3`、
   `aggregated_delta_norm_band3`、`empirical_update_ratio_band3` 和 100 Hz F1 曲线。
4. 先跑 `FedNyq-Agg --lambda_cons 0`，将 aggregation 与 consistency 分离。
5. 做单变量对照：
   - `dirichlet_alpha`: 0.2、0.5、IID；
   - high-rate fraction: 0.2、0.5、0.8；
   - `client_fraction`: 0.5、0.8；
   - `local_epochs=1` 或 `lr=5e-4` 检查更新强度。
6. 使用 2025/2026/2027 matched seeds；单 seed 不应声称统计显著。

## 9. 推荐下一步代码增强

在不改变论文核心默认方法的前提下，优先考虑：

- 增加 per-class precision/recall/F1 和每个 rate 的 confusion matrix，以确认
  25 Hz 下是否牺牲某个不可区分类别；
- 在逐轮 CSV 中记录 selected client IDs、eligible band client IDs 或对应哈希，
  便于追踪 band3 振荡；
- 研究并作为额外消融加入 server damping、minimum eligible mass 或 variance-aware
  stabilization；不要悄悄改变默认 FedNyq 公式；
- 对 complex 数据运行三方法 × 三 seeds × rate/non-IID sweep。

## 10. CUDA检查

峰值 GPU 显存为 `0.00` 表示实际使用 CPU。检查：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

运行时使用：

```text
--device cuda --amp --num_workers 4
```

实际解析设备会保存在每次运行的 `config.json` 中：`device_resolved`。

## 11. 输出与版本管理

每个实验输出到：

```text
outputs/<experiment_name>/<seed>/
```

核心文件包括 `config.json`、`client_metadata.csv`、`round_metrics.csv`、
`final_metrics.json` 和多张 PNG。通信模型对 FedAvg/FedNyq 相同，均假设选中客户端
上传和下载完整 state。

建议 Git 仓库忽略 `.venv/`、`__pycache__/`、`data/*.npz`、`outputs/`、`*.pt`、
`*.pth`、`*.ckpt`。另一台电脑首次使用 `git clone`，之后工作前
`git pull --rebase origin main`，工作后提交并 `git push origin main`。

## 12. 给新对话的直接提示词

```text
请先完整阅读项目根目录 HANDOFF.md 和 README.md，再检查现有代码。继续分析并完善
FedNyq 实验，重点调查在 high-rate proportion=0.2、Dirichlet alpha=0.2 时 band3
support-normalized aggregation 的高方差问题。不要在没有读取 round_metrics.csv 和
client_metadata.csv 的情况下下确定性结论；保持 FedAvg/FedNyq 公平性哈希和默认核心
公式不变，新增稳定化方法只能作为显式消融。
```
