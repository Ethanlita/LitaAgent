# LitaAgent-HRL 训练与架构详解（面向初学者的详细说明）

> 目标：利用 SCML 2025 Standard 赛道数据，训练分层强化学习代理 LitaAgent-HRL（模式 B：L1 微观安全基准，L2 生成日级目标，L3 残差微调，预留 L4 并发协调）。

## 目录
1. 架构概览与数学定义  
2. 动作空间与状态空间  
3. 数据流水线：从锦标赛到训练样本  
4. 训练流程：预训练与在线微调  
5. 如何运行数据采集 runner  
6. 如何训练模型并接入代理  
7. 常见问题与数据需求

---

## 1. 架构概览与数学定义
LitaAgent-HRL 由四层组成，时间尺度与职责不同：
- **L1 安全/基准层**：提炼自 Penguin 的微观谈判逻辑，输出安全掩码与基准报价，防止超量、超价、超资金。  
- **L2 战略管理层**：每日决策（Day 级），生成目标向量  
  \( g_t = [Q^{buy}_t, P^{buy}_t, Q^{sell}_t, P^{sell}_t] \)。  
- **L3 残差执行层**：轮级决策（Round 级），在 L1 基准上输出残差  
  \( \Delta a = [\Delta q, \Delta p] \)，合成最终报价  
  \( a_{final} = \text{clip}(a_{base} + \Delta a, \text{mask}) \)。  
- **L4 并发协调层（预留）**：注意力网络，在多线程并发谈判中给每个线程分配权重 \( \alpha_k \)，调节其激进程度。

时间顺序：  
1) 每日开始：L2 读取宏观状态，输出 \( g_t \)；L1 不改宏观，仅提供安全边界。  
2) 每场谈判：L3 在 L1 基准上微调；L4 可调节各线程的残差缩放。  
3) 所有动作最终经 L1 裁剪，保证安全。

## 2. 动作空间与状态空间
- **L2 动作（连续）**：\( g_t = [Q^{buy}, P^{buy}, Q^{sell}, P^{sell}] \)。  
- **L3 动作（连续 + 接受/拒绝）**：报价 \((q, t_{deliv}, p)\) 或 Accept/End；实现中以残差方式输出数量/价格，交付时间默认当前步+1。  
- **状态（宏观）**：库存 \(I\)、资金 \(B\)、市场均价 \(P_{mkt}\)、生产线 \(n_{lines}\)、已承诺合约。  
- **状态（微观序列）**：对手历史报价序列 \((q, p, round)\)、剩余轮次、L2 目标 \(g_t\)。

## 3. 数据流水线：从锦标赛到训练样本
### 3.1 采集数据
运行 `runners/hrl_data_runner.py`：  
- 参赛：全部 LitaAgent（tracked 版，除 HRL）、Penguin、2025 标准前 5、内置 Random/Sync/Decay。  
- 开启 `log_negotiations=True`、`log_ufuns=True`，输出至 `tournament_history/hrl_data_<timestamp>_std`。  
- 若安装 `scml_analyzer`，自动记录 Tracker。

### 3.2 解析日志
使用 `litaagent_std/hrl_x/data_pipeline.py`：  
```python
from litaagent_std.hrl_x.data_pipeline import load_negotiation_csv, build_macro_dataset, build_micro_dataset
df = load_negotiation_csv("./tournament_history/hrl_data_XXXX_std")
macro_ds = build_macro_dataset(df)   # L2 标签：按日聚合成交量/价格
micro_ds = build_micro_dataset(df)   # L3 序列：按谈判聚合轮次
```
- Macro 样本（L2）：`MacroSample(day, state_macro, goal)`，goal 是反推的日级目标。  
- Micro 样本（L3）：`MicroSample(negotiation_id, history, action, baseline)`，history 是轮级 DataFrame。

## 4. 训练流程：预训练与在线微调
### 4.1 预训练（离线）
- **L3 残差监督**：  
  - 标签：`residual = action_expert - baseline`（baseline 由 L1 生成）。  
  - 损失：MSE + 安全正则（越界惩罚），可加 ROL/ensemble 方差惩罚防 OOD。  
- **L2 目标生成 BC/CQL**：  
  - 标签：`g_t`（从成交反推）。  
  - 奖励塑形（在线用）：势能函数 \( \Phi = I \times P_{mkt} \)，风险惩罚（短缺预测）。

占位示例（`litaagent_std/hrl_x/training.py`）：  
```python
from litaagent_std.hrl_x.training import TrainConfig, train_l2_bc, train_l3_bc, save_model
cfg = TrainConfig(lr=1e-3, epochs=10, batch_size=32)
l2_model = train_l2_bc(macro_ds, cfg)
l3_model = train_l3_bc(micro_ds, cfg)
save_model(l2_model, "./out/l2_baseline.npz")
save_model(l3_model, "./out/l3_baseline.npz")
```
> “占位模型”意为简单可运行的线性回归示例；实际项目中应替换为 torch/tf 的 Transformer/PPO 等模型。

### 4.2 在线微调（分层/多智能体 PPO 示意）
目标：在真实环境（2025 规则：非易腐、期货、短缺惩罚，多线程并发）下，联合更新 L2/L3/L4，使收益更高、违约率更低、库存/资金曲线更稳。

**核心思路：CTDE（集中式训练、分布式执行）**
- L2（Day 粒度）单独 PPO；L3/L4（Round 粒度）用 MAPPO，critic 可观察全局。
- 动作安全由 L1 裁剪，始终开启。

**步骤流程**
1) 环境与初始化  
   - 载入离线预训练好的 L2/L3 权重，L4 初始均匀或简单启发式。  
   - 对手池：Penguin/AS0/随机/历史自我版本，提升鲁棒性。
2) 采样与拆分轨迹  
   - 一次迭代跑 3-5 场标准赛（n_steps≈90-100），收集轨迹。  
   - 按日拆分 macro 轨迹给 L2；按谈判轮拆分 micro 轨迹给 L3/L4。
3) 奖励设计  
   - L2：\( R_{L2} = \Delta cash + \gamma(\Phi_{t+1}-\Phi_t) - \text{risk\_penalty} + \epsilon \cdot \text{liquidity} \)，其中 \(\Phi = I \times P_{mkt}\)；risk_penalty 基于短缺预测；liquidity 为成交微奖。  
   - L3：目标对齐（罚 \(|q_{exec}-Q_{target}|\)、\(|p_{deal}-P_{target}|\)），成交优势（买入价低于市场/卖出价高于市场给正奖励），流动性微奖。  
   - L4：共享全局回报；注意力权重 \(\alpha\) 调节 L3 激进度，高 \(\alpha\) 成功成交则正反馈，盲目高 \(\alpha\) 导致亏损/违约则负反馈。
4) 更新规则  
   - L2：PPO clip 更新，输入日级状态 \(s_{macro}\)，动作 \(g_t\)。  
   - L3/L4：MAPPO 更新，输入轮级状态（历史序列、L2 目标、L4 权重），动作（残差/权重），critic 看到全局。  
   - 学习率：L3 小于 L2（保护已学微观技巧），L4 可略大以快速适应并发调度。
5) 断点续训  
   - 定期保存权重 + 优化器状态 + 种子；记录已完成场次/epoch。  
   - 恢复时加载最新快照继续采样-更新。  
   - 监控：平均利润、违约率、库存爆仓次数、对手池胜率；指标变差则回退。
6) 终止/早停  
   - 5-10 轮锦标赛后若收益趋稳且违约低，可停止；若过拟合则增加对手多样性或回退权重。

## 5. 如何运行数据采集 runner
```bash
python runners/hrl_data_runner.py
```
输出目录：`tournament_history/hrl_data_<timestamp>_std`，含 `logs/*/negotiations.csv`（训练所需）、tracker 日志（若可用）、scores/params 统计。

## 6. 如何训练模型并接入代理
1) 解析日志生成 `macro_ds`/`micro_ds`。  
2) 训练 L2/L3 模型并保存权重。  
3) 修改 `litaagent_std/hrl_x/agent.py`：  
   - `_heuristic_manager` -> 使用 L2 模型 `predict` 替换启发式。  
   - `propose/respond` -> 在 L1 基准报价上叠加 L3 残差，再调用 `clip_offer`。  
4) 在 runner 中注册 `LitaAgentHRL`/`Tracked` 进行仿真验证。

## 7. 常见问题与数据需求
- **需要多少数据？** 建议至少 30-50 场完整标准赛（每场 ~90-100 天），总谈判记录量达到数千到上万条。数据越多越有利于离线预训练；在线微调时可在 10-20 场后开始观察收敛。  
- **需要训练多少轮？**  
  - 离线预训练：L2/L3 各训练 10-50 epoch（视模型复杂度和数据量），以验证集指标稳定为准。  
  - 在线微调：跑 5-10 轮锦标赛检查收益/违约率，若仍在上升可继续，若过拟合则提前停止。  
- **如何中断续训？**  
  - 定期保存模型权重与优化器状态（torch/tf）；当前示例线性回归可用 `save_model()`。  
  - 记录训练 step/epoch 及随机种子；恢复时加载最新权重，从中断的 epoch/场次继续。  
- **为什么分层？** 解决长时间跨度的信用分配、并发耦合、安全探索问题。L1 保证安全，L2 管跨日规划，L3 做微调，L4 管并发资源分配。  
- **如果没有 negotiations.csv？** 必须重跑锦标赛并开启 `log_negotiations=True`，否则无法训练微观模型。
