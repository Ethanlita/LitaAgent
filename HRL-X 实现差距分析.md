# HRL-X 设计与实现差距分析报告

> 生成日期：2025年12月12日  
> 分析范围：5份设计文档 vs `litaagent_std/hrl_x/` 实现代码

---

## 📋 总体评估

当前实现处于**骨架阶段**，大量核心功能仅以占位符形式存在。总体完成度约 **15%**。

| 组件 | 设计完成度 | 状态 |
|------|------------|------|
| L1 安全护盾 | ~30% | 🟡 部分实现 |
| L2 战略管理 | ~10% | 🔴 仅启发式 |
| L3 残差执行 | ~5% | 🔴 残差=0 |
| L4 全局协调 | 0% | ⚫ 完全缺失 |
| 数据流水线 | ~25% | 🟡 骨架 |
| 训练流程 | ~5% | 🔴 仅占位符 |
| 主动协商 | 0% | ⚫ 完全缺失 |

---

## 🔴 严重缺失（核心架构组件）

### 1. L1 安全护盾层 - 重大功能缺失

**设计文件引用**：`HRL-X 架构实现与训练方案.md` 第4节、`L1-L4 层设计与离线强化学习.md` 第2节

**设计要求**：

```python
# 约束一：最大安全买入量
Q_max_buy = C_total - I_current - I_incoming + O_committed

# 约束二：最小必要买入量
Q_min_buy = max(0, O_committed - I_current - I_incoming)

# 约束三：破产保护价格
P_limit(q) = (B_t - Reserve) / q
```

- 生成**动作掩码张量（Action Mask Tensor）**，作用于 L3 的 Softmax 层
- 生成**基准动作（Baseline Action）**：`a_base = (Q_min_buy, Cost_production × (1 + Margin_min))`
- 实现为 TensorFlow `SafetyMaskingLayer` 自定义层

**当前实现**（`l1_safety.py`）：

```python
class PenguinMicroBaseline:
    def baseline_offer(self, target, delivery_time):
        qty = max(0, target.remaining)
        price = target.price_limit
        return qty, delivery_time, price

    def clip_offer(self, offer, wallet, target, is_buying, inventory_capacity):
        # 简单裁剪，无合约约束
```

**缺失清单**：

- [ ] `I_incoming`（在途原材料）追踪逻辑
- [ ] `O_committed`（已承诺订单）计算逻辑
- [ ] 基于已签合同队列的约束计算
- [ ] 动作掩码张量生成（`mask_tensor`）
- [ ] TensorFlow/PyTorch `SafetyMaskingLayer` 类
- [ ] 与 L3 输出层的 Logit Masking 集成
- [ ] 破产保护价格的动态计算

---

### 2. L2 战略管理层 - 几乎完全缺失

**设计文件引用**：`HRL-X 架构实现与训练方案.md` 第5节

**设计要求**：

```python
class ManagerPPOAgent(tf.keras.Model):
    def __init__(self):
        # 期货承诺向量卷积层
        self.future_conv = tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu')
        # Actor 网络（高斯分布）
        self.actor_out = tf.keras.layers.Dense(action_dim * 2)  # mean + log_std
        # Critic 网络
        self.critic_out = tf.keras.layers.Dense(1)
```

**状态空间 `S_high`**：
- 库存势能特征：`Φ(s) = I_total × P_avg`
- 资金健康度：`B_t / B_initial`
- 期货承诺向量：长度为 H（如10天）的向量，需 **Conv1D** 处理
- 过去10天的市场均价与成交量趋势

**动作空间**：
```python
g_t = [Q_target_buy, P_limit_buy, Q_target_sell, P_limit_sell]
```

**当前实现**（`agent.py`）：

```python
def _heuristic_manager(self, obs):
    capacity = max(1, int(obs["capacity"]))
    need_buy = max(0, capacity - int(obs["inventory_in"]))
    buy_price = obs["market_price_in"] * 1.05
    # ... 简单规则
```

**缺失清单**：

- [ ] `ManagerPPOAgent` 类（Actor-Critic 网络）
- [ ] Conv1D 层处理期货承诺向量
- [ ] 高斯分布采样机制（`tfp.distributions.Normal`）
- [ ] PPO Clip 损失函数（`compute_loss` 方法）
- [ ] 状态特征扩展：
  - [ ] 库存势能 `Φ(s)` 计算
  - [ ] 未来 H 天的订单承诺向量
  - [ ] 过去 10 天市场趋势
  - [ ] 资金健康度比例
- [ ] 价值函数 (Critic) 网络
- [ ] 熵正则化项

---

### 3. L3 残差执行层 - 核心未实现

**设计文件引用**：`HRL-X 架构实现与训练方案.md` 第6节、`HRL-X 研究：强化学习问题解决.md` 第3.3节

**设计要求**：

```python
class ResidualDecisionTransformer(tf.keras.Model):
    def __init__(self, d_model=128, n_heads=4, n_layers=2, max_len=20, action_dim=2):
        # 状态嵌入
        self.state_emb = tf.keras.layers.Dense(d_model)
        # 目标嵌入（L2 Goal 注入）
        self.goal_emb = tf.keras.layers.Dense(d_model)
        # 位置编码
        self.pos_emb = tf.keras.layers.Embedding(max_len, d_model)
        # Transformer Blocks
        self.blocks = [...]  # MultiHeadAttention, FFN, LayerNorm
        # 残差输出头
        self.action_head = tf.keras.layers.Dense(action_dim, activation='tanh')
        # 可学习缩放因子
        self.residual_scale = tf.Variable([5.0, 10.0], trainable=True)
```

**核心机制**：
```python
A_final = Clip(A_base + Δa, M_safe)
```

- 输入：谈判历史序列 `H_k = {o_{t-N}, ..., o_t}` + L2 目标向量 `g_t`
- 因果掩码（Causal Mask）防止信息泄露
- 自注意力机制捕捉对手行为模式

**当前实现**（`agent.py`）：

```python
def respond(self, negotiator_id, state):
    # ...
    baseline = self.l1.baseline_offer(target, ...)
    clipped = self.l1.clip_offer(baseline, ...)
    # 残差 = 0，直接返回基准
    return SAOResponse(ResponseType.REJECT_OFFER, clipped)
```

**缺失清单**：

- [ ] `ResidualDecisionTransformer` 类
- [ ] Transformer Blocks（MultiHeadAttention, FFN, LayerNorm）
- [ ] 状态嵌入层（`state_emb`）
- [ ] 目标条件注入机制（`goal_emb`）
- [ ] 位置编码（Positional Encoding）
- [ ] 因果掩码（Causal Mask）实现
- [ ] 可学习 `residual_scale` 参数
- [ ] 谈判历史序列的滑动窗口管理
- [ ] 隐状态 `h_k` 输出（供 L4 使用）

---

### 4. L4 全局协调层 - 完全缺失 ⚫

**设计文件引用**：`PenguinAgent 目标与谈判机制.md` 第4节、`HRL-X 架构实现与训练方案.md` 第7节

**设计要求**：

```python
class GlobalCoordinator(tf.keras.layers.Layer):
    def __init__(self, d_model=64, n_heads=4):
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model)
        self.dense_q = tf.keras.layers.Dense(d_model)  # Query from global state
        self.dense_k = tf.keras.layers.Dense(d_model)  # Key from thread states
```

**核心机制**：
```python
# 输入：所有活跃 L3 线程的隐状态
H_in = {h_1, h_2, ..., h_K}

# 注意力权重计算
α = Softmax(Q @ K^T / √d_k)

# 输出：线程重要性权重
# α_k 高 → L3 变得激进，确保成交
# α_k 低 → L3 变得保守，可放弃
```

**当前实现**：

**完全不存在任何 L4 相关代码**

**缺失清单**：

- [ ] `GlobalCoordinator` 类
- [ ] 多头自注意力计算
- [ ] 全局状态编码（Query 生成）
- [ ] 线程隐状态收集机制
- [ ] 注意力权重 `α_k` 分配
- [ ] 权重到 L3 策略调制的映射
- [ ] 端到端训练集成（与 L3 联合反向传播）
- [ ] 并发资源争夺解决机制

---

### 5. 主动协商发起机制 - 完全缺失 ⚫

**设计文件引用**：`PenguinAgent 目标与谈判机制.md` 第3节

**设计要求**：**"广播-过滤"（Broadcast-Filter）协议**

```python
def before_step(self):
    # 1. L2 设定目标
    goals = self.l2_manager.predict(macro_state)
    
    # 2. 获取所有潜在伙伴
    partners = self.awi.my_suppliers  # 或 my_consumers
    
    # 3. 向所有伙伴发起协商请求（饱和式请求）
    for partner in partners:
        self.awi.request_negotiation(
            partner=partner,
            product=self.awi.my_input_products[0],
            quantity=goals.Q_target_buy,
            unit_price=goals.P_limit_buy,
            time=self.awi.current_step + 1,
        )
```

**核心逻辑**：
- 不主观挑选对象，向**所有**潜在供应商/消费者广播
- 由 L4 在协商过程中动态决定哪些线程值得成交
- 总请求量 = N × Q_target（有过度承诺风险，需 L4 解决）

**当前实现**（`agent.py`）：

```python
def before_step(self):
    super().before_step()
    self._ensure_targets()
    state_h = self._macro_obs()
    goals = self._heuristic_manager(state_h)
    self.buy_target = DailyTarget(...)
    self.sell_target = DailyTarget(...)
    # ❌ 没有主动发起协商，完全被动等待
```

**缺失清单**：

- [ ] `awi.request_negotiation()` 调用
- [ ] 潜在伙伴列表获取（`awi.my_suppliers` / `awi.my_consumers`）
- [ ] 饱和式协商请求发起
- [ ] 与 L4 过滤机制的配合

---

## 🟠 训练流程缺失

### 6. 离线强化学习 - 仅有骨架

**设计文件引用**：`L1-L4 层设计与离线强化学习.md` 第4节、`HRL-X 研究：强化学习问题解决.md` 第5节

**设计要求**：**ROL (Reward-on-the-Line)** 算法

```python
# 集合一致性：训练 N=5 个 Q 网络
Q_ensemble = [Q_1, Q_2, ..., Q_5]

# 不确定性惩罚
Q_target(s, a) = min_i Q_i(s, a) - λ × Var(Q_i(s, a))

# 优势加权行为克隆
L_ROL = ||a_pred - a_expert||² + λ_var × Var(Q_ensemble(s, a))
```

**当前实现**（`training.py`）：

```python
class SimpleRegressor:
    def __init__(self, input_dim, output_dim):
        self.W = rng.standard_normal((input_dim, output_dim)) * 0.01
        self.b = np.zeros(output_dim)
    
    def fit(self, x, y, lr, epochs):
        for _ in range(epochs):
            pred = self.predict(x)
            grad = (pred - y) / len(x)
            self.W -= lr * x.T @ grad
```

**缺失清单**：

- [ ] ROL 算法核心实现
- [ ] Q 网络集合（Ensemble）
- [ ] 不确定性惩罚机制
- [ ] 优势加权行为克隆
- [ ] PyTorch/TensorFlow 深度学习模型替换
- [ ] CQL (Conservative Q-Learning) 备选实现
- [ ] 数据集优势过滤（只模仿 A(s,a) > 0 的样本）

---

### 7. 分层联合微调 - 未实现

**设计文件引用**：`HRL-X 架构实现与训练方案.md` 第8.3节

**设计要求**：**MAPPO (Multi-Agent PPO)** + 复合奖励函数

```python
R_t = R_profit + λ1 × R_liquidity - λ2 × R_risk + λ3 × R_intrinsic
```

**奖励分量**：

| 分量 | 公式 | 作用 |
|------|------|------|
| `R_profit` | `(B_{t+1} - B_t) + γΦ(s_{t+1}) - Φ(s_t)` | 势能函数解决短视问题 |
| `R_liquidity` | `ε if deal else 0` | 防止策略冻结 |
| `R_risk` | `-β × exp(max(0, -I_future))` | 前瞻性风险惩罚 |
| `R_intrinsic` | `-||q_executed - q_goal||²` | L3 与 L2 目标对齐 |

**缺失清单**：

- [ ] MAPPO 算法实现
- [ ] CTDE（集中式训练，去中心化执行）架构
- [ ] 势能函数 `Φ(s) = I × P_avg` 计算
- [ ] 复合奖励函数各分量实现
- [ ] L2/L3/L4 联合训练循环
- [ ] GAE (Generalized Advantage Estimation)

---

### 8. 自博弈训练 - 未实现

**设计文件引用**：`HRL-X 架构实现与训练方案.md` 第8.4节

**设计要求**：

```python
# 对手池
opponent_pool = [
    PenguinAgent,           # 静态基准
    AS0,                    # 静态基准
    LitaAgentHRL_v1,        # 历史版本
    LitaAgentHRL_v2,        # 历史版本
    ...
]

# 训练循环
for epoch in training:
    opponent = random.choice(opponent_pool)
    run_episode(current_agent, opponent)
    if epoch % save_interval == 0:
        opponent_pool.append(copy(current_agent))
```

**缺失清单**：

- [ ] 对手池管理机制
- [ ] 模型版本保存与加载
- [ ] 随机对手采样
- [ ] 纳什均衡逼近评估

---

## 🟡 数据流水线缺失

### 9. 宏观目标取证重构 - 不完整

**设计文件引用**：`PenguinAgent 目标与谈判机制.md` 第2节

**设计要求**：

```python
# 买入量目标重构
Q_target_buy = min(Q_max_safe, Q_needed)

where:
    Q_max_safe = C_total - I_current - I_incoming + O_committed
    Q_needed = max(0, Σ D_future - I_current - I_incoming)

# 买入限价重构
P_limit_buy = max(当天所有出价)  # 或 P_market_sell - C_process - Margin
```

**需要从日志提取**：
- 在途原材料 `I_incoming`
- 已承诺订单 `O_committed`
- 未来 H 天的销售合同需求量 `D_future`
- `world_stats.csv` 中的状态快照

**当前实现**（`data_pipeline.py`）：

```python
def build_macro_dataset(df):
    grouped = df[df["response"] == "accept"].groupby("time")
    for day, g in grouped:
        buy_qty = int(buy_deals["quantity"].sum())
        buy_price = float(buy_deals["price"].max())
        # ❌ 简单聚合，无合约逆向工程
```

**缺失清单**：

- [ ] `I_incoming` 追踪（已签未交付合同）
- [ ] `O_committed` 计算（已签销售合同）
- [ ] 未来 H 天订单需求量提取
- [ ] `world_stats.csv` / `contracts.csv` 解析
- [ ] 完整宏观状态特征提取：
  - [ ] 当日库存水平
  - [ ] 当日资金余额
  - [ ] 市场价格指数
  - [ ] 生产线闲置率

---

### 10. 微观序列数据集 - 过于简化

**设计文件引用**：`HRL-X 代理实现与数据收集.md` 第3.3节

**设计要求**：

| 字段 | 含义 | 用途 |
|------|------|------|
| `id` | 谈判唯一标识 | 关联不同轮次 |
| `round` | 谈判轮次 | 让步曲线分析 |
| `offer` | 提议 `(q, t, p)` | 模仿学习目标 |
| `response` | 回应类型 | 学习接受阈值 |
| `time` | 响应时间 | 对手急迫度推断 |

**需要提取**：
- 完整的出价序列（供 Transformer 输入）
- 每轮响应时间
- 让步曲线（Concession Curve）
- L1 基准的预计算

**当前实现**：

```python
def build_micro_dataset(df):
    for nid, g in df.groupby("id"):
        g_sorted = g.sort_values("round")
        last = g_sorted.iloc[-1]
        action = {"quantity": last.get("quantity"), ...}
        # ❌ 仅取最后一轮，丢失序列信息
```

**缺失清单**：

- [ ] 完整历史序列保留（不只是最后一轮）
- [ ] 响应时间特征提取
- [ ] 让步曲线计算（`price[t] - price[t-1]`）
- [ ] L1 基准预计算并存储在样本中
- [ ] 序列截断/填充到固定长度（供 Transformer 使用）

---

## 🔵 Agent 实现细节问题

### 11. 状态观测不完整

**设计要求 vs 当前实现对比**：

| 特征 | 设计要求 | 当前实现 | 状态 |
|------|----------|----------|------|
| `step_progress` | ✓ | ✓ | ✅ |
| `balance` | ✓ | ✓ | ✅ |
| `inventory_in` | ✓ | ✓ | ✅ |
| `inventory_out` | ✓ | ✓ | ✅ |
| `market_price_in` | ✓ | ✓ | ✅ |
| `market_price_out` | ✓ | ✓ | ✅ |
| `capacity` | ✓ | ✓ | ✅ |
| **期货承诺向量** | ✓ | ❌ | 🔴 缺失 |
| **市场历史趋势** | ✓ | ❌ | 🔴 缺失 |
| **已签合同队列** | ✓ | ❌ | 🔴 缺失 |
| **库存势能 Φ(s)** | ✓ | ❌ | 🔴 缺失 |
| **资金健康度比例** | ✓ | ❌ | 🔴 缺失 |

**缺失清单**：

- [ ] 未来 H 天期货承诺向量
- [ ] 过去 10 天市场趋势（均价、成交量）
- [ ] 已签合同信息（买入/卖出队列）
- [ ] 库存势能 `Φ(s) = I × P_avg`
- [ ] 资金健康度 `B_t / B_initial`

---

### 12. 低层状态观测缺失

**设计要求的 `S_low`**：

| 特征 | 当前状态 |
|------|----------|
| `subgoal_remaining` | 🟡 有 `target.remaining`，未归一化 |
| `negotiation_time` | 🔴 缺失 |
| `current_offer` | ✅ 已有 |
| `opponent_history` | 🔴 缺失 |

**缺失清单**：

- [ ] 谈判剩余时间/轮次特征
- [ ] 归一化的子目标剩余量 `remaining / target`
- [ ] 对手历史出价序列缓存

---

### 13. 乐观并发控制增强

**当前实现**：

```python
if price_ok and qty <= target.remaining:
    target.register_deal(qty)
    return SAOResponse(ResponseType.ACCEPT_OFFER, offer)
```

**存在问题**：
- 虽然有 `register_deal`，但没有与 L4 配合
- 无法在多线程间协调资源分配

**缺失清单**：

- [ ] 与 L4 协调器的集成
- [ ] 跨线程资源分配机制
- [ ] 过度承诺回滚逻辑（当合同失败时）

---

## 🎯 实施优先级建议

### P0 - 最高优先级（架构核心）

1. **实现 L4 全局协调层** - 解决并发资源耦合
2. **实现 L3 Decision Transformer** - 核心智能能力
3. **实现主动协商发起（广播-过滤协议）** - 打破被动模式

### P1 - 高优先级

4. **完善 L1 动作掩码和合约约束** - 安全基础
5. **实现 L2 PPO 战略管理器** - 跨期规划
6. **扩展状态观测空间** - 信息完备性

### P2 - 中优先级

7. **完善数据流水线** - 训练数据质量
8. **实现 ROL 离线 RL** - 冷启动解决方案
9. **实现复合奖励函数** - 训练信号

### P3 - 后续优化

10. **实现 MAPPO 在线训练**
11. **实现自博弈机制**
12. **性能调优与超参搜索**

---

## 📂 文件变更清单

需要新建/大改的文件：

| 文件 | 状态 | 说明 |
|------|------|------|
| `hrl_x/l4_coordinator.py` | 🆕 新建 | 全局协调层 |
| `hrl_x/l3_transformer.py` | 🆕 新建 | Decision Transformer |
| `hrl_x/l2_manager.py` | 🆕 新建 | PPO 战略管理器 |
| `hrl_x/l1_safety.py` | 🔄 大改 | 添加掩码、合约追踪 |
| `hrl_x/agent.py` | 🔄 大改 | 集成 L2/L3/L4，添加主动协商 |
| `hrl_x/data_pipeline.py` | 🔄 大改 | 完善特征提取 |
| `hrl_x/training.py` | 🔄 大改 | 实现 ROL/MAPPO |
| `hrl_x/rewards.py` | 🆕 新建 | 复合奖励函数 |
| `hrl_x/self_play.py` | 🆕 新建 | 自博弈训练 |

---

## 📝 附录：设计文档速查

| 设计点 | 主要参考文档 |
|--------|--------------|
| L1 安全护盾 | `HRL-X 架构实现与训练方案.md` §4, `L1-L4 层设计与离线强化学习.md` §2 |
| L2 战略管理 | `HRL-X 架构实现与训练方案.md` §5 |
| L3 残差执行 | `HRL-X 架构实现与训练方案.md` §6, `HRL-X 研究：强化学习问题解决.md` §3.3 |
| L4 全局协调 | `PenguinAgent 目标与谈判机制.md` §4, `HRL-X 架构实现与训练方案.md` §7 |
| 主动协商 | `PenguinAgent 目标与谈判机制.md` §3 |
| 离线 RL | `L1-L4 层设计与离线强化学习.md` §4, `HRL-X 研究：强化学习问题解决.md` §5 |
| 奖励函数 | `HRL-X 研究：强化学习问题解决.md` §6 |
| 数据流水线 | `HRL-X 代理实现与数据收集.md` §3.3, `PenguinAgent 目标与谈判机制.md` §2 |

---

*本文档应随实现进度更新，完成项请打勾 ✅*
