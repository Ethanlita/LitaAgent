最后更新：2026-01-10 12:00

已决事项：
1. 模型工件目录与命名统一为 `assets/models/accept/` 与 `assets/models/breach/`，文件名固定 `model.bin` + `model_meta.json`，可选 `calibration.json`；目录与模型类型分离以保证互换一致。
2. `predict_accept_strength` 默认强度固定为常数，默认 `s=8`，推荐范围 `[2, 20]`。
3. `buffer(t)` 固定为单调上升函数，采用 `buffer(t)=b_min+(b_max-b_min)*t^gamma`，`t=round_rel`；默认 `b_min=0.05`、`b_max=0.35`、`gamma=2`。
4. `m_i(o_i)` 采用 trading_price 近似（在线可得）并保留单调兜底：卖方 `m=(p-trading_price)/max(1e-6,trading_price)`，买方 `m=(trading_price-p)/max(1e-6,trading_price)`，再乘数量得到 `m_i=q*m`。
5. counter_all 的子集选择：若是"接受对方 offer"，视为 `P_sign=1`，因此 `\tilde q=q*LCB(P_fulfill)`；若是"我方发出 counter_offer"，仍用 `P_eff=LCB(P_sign)*LCB(P_fulfill)`。
6. terminal negative 只在"对手 END / timeout / failed 且无 agreement"时启用，并使用 `w_end` 作为软负权重；我方主动 END 不计入。

---

**7. price_concession_gamma = 0.5（而非 2.0）** (2026-01-10)

问题背景：
- 原设计假设谈判有多轮，gamma=2.0 使让步曲线平缓
- 实际分析 2024 OneShot 冠军日志发现：**99.4% 的谈判在 round 1 结束**
- 平均 `round_rel ≈ 0.03`（谈判进度仅 3%）

计算对比：
```
gamma=2.0: concession = 0.03^2.0 = 0.0009 (几乎无让步)
gamma=0.5: concession = 0.03^0.5 = 0.173  (有效让步)
```

结论：OneShot 谈判极短，需要激进让步策略，故改为 gamma=0.5。

---

**8. panic_round_rel_threshold = 0.1（而非 0.6）** (2026-01-10)

问题背景：
- 原设计在 `round_rel > 0.6` 时触发 panic 模式
- 但数据显示只有 **0.6% 的谈判**能达到 `round_rel > 0.1`
- 阈值 0.6 在 OneShot 中**永远不会触发**

设计含义：
- 新阈值 0.1 针对"极少数长谈判"（约 0.6% 的情况）
- 这些长谈判通常是双方僵持不下的情况
- 此时应采取更激进策略，避免 shortfall/disposal 惩罚
- 0.1 **并非"谈判刚开始就 panic"**，而是"比平均谈判长 3 倍以上才 panic"

---

**9. use_exo_price_for_utility = True（使用外生价格计算真实利润）** (2026-01-10)

问题背景：
- 原实现使用 `trading_price`（市场平均价）计算边际收益
- 导致决策错误，例如：
  - 卖家以 16 卖出，`trading_price=17` → 边际收益 = -1（负值）
  - 但实际采购成本 `exo_input=10` → 真实利润 = 16-10 = 6（正值）
  - Agent 错误拒绝了有利可图的交易

OneShot 外生价格结构：
- Level 0 (卖家): 外生**输入**合同 → `current_exogenous_input_price` = 采购成本
- Level 1 (买家): 外生**输出**合同 → `current_exogenous_output_price` = 销售收入

真实利润计算：
- 卖家: `profit = sell_price - exo_input_price`
- 买家: `profit = exo_output_price - buy_price`

---

**10. 外生价格是总价格，需转换为单价** (2026-01-10, bug fix)

问题发现：
- 比赛 `20260110_155408_oneshot` 中 LOS 得分垫底（0.89 vs 1.07+）
- 日志显示 `offers_accepted: 0` — LOS 从未接受任何 offer
- 调查发现：`current_exogenous_input_price = 110`，但这是 **10 单位的总价格**

错误逻辑：
```python
# 旧代码（错误）
exo_price = getattr(self.awi, "current_exogenous_input_price", None)  # = 110
profit = 16 - 110 = -94  # 巨额亏损 → 拒绝所有 offer
```

正确逻辑：
```python
# 新代码（正确）
total_price = getattr(self.awi, "current_exogenous_input_price", None)  # = 110
quantity = getattr(self.awi, "current_exogenous_input_quantity", None)  # = 10
unit_price = total_price / quantity  # = 11
profit = 16 - 11 = 5  # 正常利润 → 可能接受
```

SCML OneShot AWI 属性说明：
- `current_exogenous_input_price`: **总采购成本**（不是单价！）
- `current_exogenous_input_quantity`: 采购数量
- `current_exogenous_output_price`: **总销售收入**（不是单价！）
- `current_exogenous_output_quantity`: 销售数量

---

**11. 移除 `_select_subset` 中的多余惩罚项** (2026-01-10)

移除项：
1. **risk_penalty**: 原本 `risk_penalty = risk_lambda * (1 - p_eff) * q`
   - 冗余：`q_eff = q * p_eff` 已经体现了风险折扣
   - 再加 risk_penalty 是双重惩罚

2. **over_penalty 的 10% 乘数**: 原本 `over_penalty = 0.1 * disposal_cost * overfill`
   - 冗余：`disposal_cost` 本身就是超量惩罚单位
   - 乘 10% 导致对超量风险低估

修正后公式：
```python
score = utility - penalty_cost - over_penalty
# utility = profit * q (使用外生价格)
# penalty_cost = shortfall_unit * max(0, need - q_eff)
# over_penalty = disposal_unit * max(0, q_eff - need)
```

---

**12. buyer_overordering_ratio = 0.1（BUYER 超量采购，移除 buffer）** (2026-01-10)

灵感来源：RChan (SCML 2025 竞争对手)

罚款分析结论（货币量纲）：
```
SELLER: Disposal=0.83, Shortfall=8.69 (比例 10.4x)
BUYER:  Disposal=1.49, Shortfall=16.17 (比例 10.9x)
```

结论：Shortfall penalty 约为 Disposal cost 的 **10 倍**！

策略启示：
- **BUYER**: 应容忍超量采购 (overbuy)，宁可多买也不要 shortfall
- **SELLER**: 应保守，不超卖 (oversell 导致高额 shortfall 惩罚)

RChan 参数：
```python
overordering_max_selling = 0.0  # 卖家不超量
overordering_max_buying = 0.2   # 买家超量 20%
```

我们的实现：
```python
buyer_overordering_ratio = 0.1  # 先用 10% 做实验，后续可调整到 20%
```

Target 计算逻辑：
```python
if role == "BUYER":
    target = int(need_remaining * (1.0 + buyer_overordering_ratio))
else:  # SELLER
    target = int(need_remaining)  # 不超量
```

buffer 机制已移除（与 overordering 功能重复）。

---

**13. `_select_subset` Powerset 搜索已存在** (2026-01-10 确认)

当前实现已使用 Powerset 枚举 (`for mask in range(1 << n)`) 搜索最优接受子集，
组合数 = 2^portfolio_k = 2^6 = 64（可接受）。

惩罚逻辑已正确区分 BUYER 和 SELLER：
- **BUYER overfill** → `disposal_unit`（较低惩罚）
- **BUYER underfill** → `shortfall_unit`（较高惩罚）
- **SELLER overfill** → `shortfall_unit`（较高惩罚）
- **SELLER underfill** → `disposal_unit`（较低惩罚）

这意味着 Powerset 会自动倾向于：
- BUYER: 接受更多 offer（即使超过 need_remaining）
- SELLER: 保守接受（避免 oversell）

---

**14. Panic 模式直接使用对手最优价格** (2026-01-10)

问题背景：
- `_snap_price()` 使用 `round()` 进行银行家舍入（round half to even）
- Python: `round(16.5) = 16`（不是 17，因为 16 是偶数）
- 当 `price_range=1`（如 p_min=16, p_max=17）时：
  - p_mid = 16.5 被 round 回 16
  - **panic/让步完全失效**，对手看到的仍是 p_min

原方案（已废弃）：
```python
# Panic 时跳到中间价
price = (base_price + opp_price) / 2.0  # 可能产生 x.5 被压回
```

新方案：
```python
# Panic 时直接使用对手最优价格，完全绕过 rounding 问题
if role == "BUYER":
    return p_max  # 买家出最高价，对卖方最有利
else:  # SELLER
    return p_min  # 卖家出最低价，对买方最有利
```

这样 panic 模式保证能给出对手愿意接受的价格，避免"长期买不够/卖不出"。

未决事项：
- 暂无
