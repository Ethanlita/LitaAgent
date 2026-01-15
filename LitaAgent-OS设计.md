最后更新：2026-01-14 13:41
**D‑NB（Neural‑Bayesian Opponent Model + Heuristic Planner）具体实施方案**

* 设计总览（模块/数据流/职责）
* 每个策略的文字 + 数学形式
* 代码层面要做哪些文件/类/接口（不写代码，但写到“应该覆写哪个方法、记录什么字段”）
* 逐阶段实施路线（先 Logistic 验证，再 Transformer，上线 BOU，再启发式优化，再调参）
* **非常细的验收清单**（可逐条勾选）

> 重要提醒：你们的 `scml_analyzer` 里其实已经有一个 `log_parser.py`，它会去解析 NegMAS/SCML 自动生成的 `actions.csv / contracts.csv / breaches.csv / negotiations.csv / stats.csv` 或 `stats.csv.csv`（以实际输出为准）。这意味着：
> **我们已用最小 OneShot 运行验证：系统日志中可以拿到 OM 训练的最低字段**（offer/时间/角色/谈判序列/协议/合同/违约）。
> 因此这里改成**单通道数据源：系统日志为主**；Tracker 变为可选，仅用于记录线上决策时的内部变量（如 `p_accept_prior/LCB/decision_meta`）与调试。
> 若后续需要这些内部变量，再补 Tracker，不影响 OM 训练主线。

---

## 0. D‑NB 的目标与边界

### 0.1 目标

在 OneShot Track 中，你们最关键的痛点是：

1. **对手接受概率不确定** → 影响能否按时买够/卖出
2. **对手违约风险**（尤其弱对手）→ 会导致你买不到/卖不出并被罚
3. 多谈判并行 → 需要跨线程协调（portfolio/组合层面做决策）

D‑NB 的设计目标是：

* **学一个“对手响应模型”**：预测对手对某个 offer 在某上下文下的 `P(accept)`
* **学/取一个“对手可靠性”**：系统若给 breach_prob 则用系统；否则自建 `P(breach)` 或 `P(fulfill)`
* **在线贝叶斯更新（BOU）**：把模型输出当先验，用本局数据快速自适应，并拿到不确定性
* **启发式决策**：把概率/不确定性“吃进组合优化”，做稳健的下界决策（LCB），避免被差对手坑

---

## 1) 数据与 Tracker：你们距离 D‑NB 训练与上线差什么？

你提的第 1 点非常关键：**现有 Tracker 是否适用于 OneShot？是否记录 accept/breach？**

### 1.1 现状评估（基于你们 repo 里的实现）

你们目前 `scml_analyzer/auto_tracker.py` 的 `AgentLogger` **定义了**这些能力：

* `negotiation_accept/reject/offer_made/offer_received/...`
* `contract_signed/executed/breached`

但 `TrackedAgent` mixin 目前只自动覆盖并记录了：

* `before_step()`：库存（OneShot 可能没有，try/except 会跳过）
* `on_negotiation_success()`：记录 contract_signed
* `on_negotiation_failure()`：记录 failure

**它没有自动记录：**

* 你发出的 offer（first_proposals / counter_all）
* 你对对手 offer 的 accept/reject 响应（counter_all 内部的 response）
* 合同是否 executed / breached（缺少 on_contract_executed / on_contract_breached 的覆写）

所以你的担心是对的：**在 OneShot 下，现有 Tracker 很可能不足以直接产出可用训练数据。**

---

## 2) D‑NB 的数据源策略：系统日志为主，Tracker 可选

### 2.1 数据源 A：NegMAS/SCML 官方世界日志（强烈建议作为“标签来源”）

你们 `scml_analyzer/log_parser.py` 明确支持读取：

* `actions.csv`：每个 negotiation action（offer、accept、reject、agreement…）
  * agreement 表示达成协议，可视为 ACCEPT（若缺少 response 记录，用它回填）
* `contracts.csv`：合同签署与执行状态
* `breaches.csv`：违约记录
* `negotiations.csv`：谈判会话、历史
* `stats.csv` 或 `stats.csv.csv`：每步统计（列可能因 track 不同而不同，但 parser 有 fallback；以实际输出为准）

这意味着：
只要你们跑 tournament 时保存了 world 日志（一般 anac runner 会保存到 tournament 目录里），你就能稳定地得到：

* “某个 offer 是否被接受” 的标签（从 actions/negotiations 序列推断）
* “某个 contract 是否 breach” 的标签（contracts + breaches）

**优势：**

* 覆盖所有 agent 的交互（不只是你们的 agent）
* 标签是系统产出（不会因为你没打点而丢）

**你们需要做的“确认动作”**（后面验收清单会写得很具体）：

* 跑一个最小 OneShot tournament，确认 tournament/world 目录里确实生成了上述 csv
* 用你们已有 `LogParser.parse_directory()` 解析一个 world 目录，确认能得到 `actions_df/contracts_df/breaches_df`

> 若发现 OneShot world 没生成 breaches.csv：也不致命，因为 contracts.csv 里可能有 is_breached 字段；再不行还有 world_stats.csv 的 n_contracts_breached（但那是聚合级，不能做逐 contract 标签）。


### 2.1.1 系统日志字段覆盖结论（已验证）
最小 OneShot 运行已验证：world 日志会生成 `actions.csv` / `negotiations.csv` / `negotiations/<id>.csv` / `negs.csv` / `contracts.csv`（`breaches.csv` 可能为空）。
**这些字段足以构建 Accept OM 的最低训练集**，唯一需要注意的是 `actions.csv` 的 `state` 不是直接的 REJECT，需要用 `negotiations/<id>.csv` 的 offer 序列重建。

### 2.1.2 系统日志 -> OM 字段重建规则（Accept 最低集，offer-centric）

说明：Accept 数据集采用 **offer-centric** 口径：样本=我方发出的一个 offer（proposal_sent/counter_offer），标签=对手是否接受该 offer。

| 训练字段 | 来源文件 | 重建方法 |
| --- | --- | --- |
| `world_id` | `negs.csv` | 直接取 world 字段（备选：`events.json.sender` 或 log folder） |
| `negotiation_id` | `negotiations.csv` | `actions.csv.neg_id` -> `negs.csv.id` -> `negs.csv.name` -> `negotiations.csv.id` |
| `step`(谈判回合), `round_rel` | `negotiations/<id>.csv` | `step` 用 `negotiations/<id>.csv.step`；仿真步用 `negotiations.csv.sim_step`/`negs.csv.sim_step`；`round_rel` 取 `relative_time` |
| `proposer_id`, `partner_id` | `negotiations/<id>.csv` | 由 offer 序列 proposer/对手推导（proposer 为样本主体） |
| `role` | `negotiations.csv` | `proposer_id==buyer` -> BUYER，否则 SELLER（`is_buy` 仅 caller 角色） |
| `price_bounds`, `q_max` | `negotiations.csv` | 解析 issues 中 quantity/unit_price 区间 |
| `offer(q,p,time)` | `negotiations/<id>.csv` | 日志顺序 `(q,time,p)` -> `(q,p,time)`；t 从日志读取，OneShot 常为常数但不要硬编码（用于 agreement 匹配，建模可忽略） |
| `q_bucket`, `p_bin`, `round_bucket` | 规则 | 按离散化定义计算 |
| `trading_price` | `negs.csv` | `negs.csv.trading_price`；`stats.csv`（或 `stats.csv.csv`）时按 product 选 `trading_price_<product>` |
| `need_remaining` | `negs.csv` | proposer 对应 `needed_*`（`agent_time0/1` 名称匹配） |
| `history` | `negotiations/<id>.csv` | 依事件重建规则生成 token |
| `y_accept` | `negotiations.csv`/`negs.csv` | 见 offer-centric 标签规则 |

**事件重建规则（response_type ∈ {ACCEPT, REJECT, END}；history 的 action_type ∈ {OFFER, ACCEPT, END}）**
1. 从 `negotiations/<id>.csv` 读取 `new_offers` 生成 OFFER token：记录 proposer、offer(q,time,p)、round_rel 等；若 `new_offers` 缺失，用 `current_offer/current_proposer` 变化补齐；谈判首个报价标记 `is_first_proposal=True`。
2. 当 responder 返回 `REJECT` 时，记录 `response_sent`（response_type=REJECT，responded_offer=上一条对手 offer，counter_offer=当前 responder 的 offer）；确保 counter_offer 对应的 OFFER token 存在且标记 `is_counter=True`，若该 token 已由 new_offers 生成，则仅做标注不重复插入。
3. 若 negotiation 有 agreement，则生成 `ACCEPT`：responder=接受方，responded_offer=agreement（或最后一条对手 offer）。
4. negotiation failed/timedout/broken/ended 生成 `END` 事件；只入 history。
5. `event_index` 按事件序列顺序自增。

**offer-centric 标签规则（y_accept）**
- agreement 匹配我方 offer -> `y_accept=1`（如有重复，取最后一次 match；p 浮点容忍：round(p,k) 或 |p1-p2|<eps）。
- 我方 offer 后紧跟对手 counter_offer -> `y_accept=0`。
- END/timeout 默认不入样本；仅在对手 END/timeout/failed 且无 agreement 时，才可选 `add_terminal_negative=True` 给最后 outstanding offer 弱负样本（weight≈0.2）；我方主动 END 不计入。
- 落地规则（监督训练）：terminal negative 的 `sample_weight = w_end`（默认 0.2）。
- 落地规则（BOU）：terminal negative 计为软更新，$\beta \leftarrow \beta + w_{\text{end}}$（允许非整数 Beta 参数）。

### 2.1.3 系统日志字段重建清单（全字段）

**使用到的系统日志文件**
- `actions.csv`（动作状态：continuing/ended/agreement 等）
- `negotiations.csv`（谈判级汇总：partners/role/issue/agreements/flags）
- `negotiations/<id>.csv`（谈判回合明细：new_offers/current_offer/relative_time）
- `negs.csv`（谈判快照：agent0/1 数值、agent_time0/agent_time1 名称、needed_*、trading_price 等）
- `contracts.csv`（合同条款与执行状态）
- `breaches.csv`（可选：逐合同违约记录）
- `stats.csv` 或 `stats.csv.csv`（可选：每步统计，含 shortfall/disposal 等；以实际输出为准）
- `params.json` / `info.json` / `agents.json` / `agents.csv` / `events.json`（可选：全局配置/agent 名称映射）

**ID 对齐路径（关键）**
- `actions.csv.neg_id` (int) → `negs.csv.id` (int) → `negs.csv.name` (uuid) → `negotiations.csv.id` (uuid) → `negotiations/<id>.csv`

**字段重建规则（按功能分组）**

A) 训练样本主键与时间
- `world_id`: 取 `negs.csv.world`（备选：`events.json.sender` 或 log folder）。
- `negotiation_id`: 通过上面的 ID 对齐路径得到 UUID。
- `step`: 优先用 `negotiations/<id>.csv.step`（谈判回合步）；仿真步用 `negotiations.csv.sim_step` 或 `negs.csv.sim_step`。
- `round_rel`: 用 `negotiations/<id>.csv.relative_time`，若缺失用 `actions.csv.relative_time`；应在 [0,1] 内，超界需报警，构建时可 clip 到 [0,1] 保命。
- `round_bucket`: `round_bucket = min(T-1, floor(round_rel * T))`，默认 `T=5`。
- `event_index`: 按“事件重建规则”生成的事件序列顺序自增。

B) 参与方与角色
- `partner_id`, `proposer_id`: 从 `negotiations/<id>.csv` 的 offer 序列恢复：当前 offer 的 proposer 为 proposer_id，对手为 partner_id。
- `role`: 用 `negotiations.csv.buyer/seller` 判定；`proposer_id == buyer` → BUYER，否则 SELLER（`is_buy` 仅表示 caller 角色）。

C) 价格/数量边界与离散化
- `q_min/q_max`: 从 `negotiations.csv.issues` 解析 quantity 区间。
- `p_min/p_max`: 从 `negotiations.csv.issues` 解析 unit_price 区间。
- `q_bucket`: `q_bucket = clip(int(q), 0, q_max)`。
- `q_norm`: `q_norm = q / max(1, q_max)`。
- `p_bin`: `p_bin = 0` 表示 `p==p_min`，`p_bin = 1` 表示 `p==p_max`（OneShot 常见 `p_max=p_min+1`）。
- `p_norm`: `p_norm = (p - p_min) / max(1, p_max - p_min)`。
- `price_min/price_max`: 即 `p_min/p_max`。
- `q_max`: 即 `q_max`（也可作为 n_lines 近似）。

D) Offer 与响应字段（response_type: REJECT/ACCEPT/END）
- `offer(q,p,time)`: 来自 `negotiations/<id>.csv.new_offers` 或 `current_offer`；日志顺序为 `(q,time,p)`，读取后映射为 `(q,p,time)`（t 从日志读取，OneShot 常为常数但不要硬编码；用于 agreement 匹配，建模可忽略）。
- `offer_sent`: 由 `proposal_sent`/`counter_offer` 事件写入；Accept 样本来自 proposer_id 视角。
- `responded_offer`: 仅 `response_sent` 事件；REJECT 用上一条对手 offer，ACCEPT 用 agreement 或最后一条对手 offer。
- `counter_offer`: REJECT 时用当前 proposer 的 offer。
- `response_type`: 由事件重建规则得到 `REJECT/ACCEPT/END`。
- END/timeout 默认不入样本；仅在对手 END/timeout/failed 且无 agreement 时，才可选 `add_terminal_negative=True` 给最后 outstanding offer 弱负样本（weight≈0.2）；我方主动 END 不计入。

E) Context 特征（尽量从系统日志取，取不到可空）
- `trading_price`: 取 `negs.csv.trading_price`（已按产品）；若用 `stats.csv`（或 `stats.csv.csv`）需按 product 选 `trading_price_<product>`。
- `need_remaining`: 取 `negs.csv.needed_sales* / needed_supplies*`，按 `agent_time0/agent_time1`(agent 名称)对应 proposer，并根据 role 选择 sales(SELLER)/supplies(BUYER)。
- `pen_shortfall`, `cost_disposal`: **系数而非货币**。若从 `stats.csv`（或 `stats.csv.csv`）读取的是货币总惩罚（`shortfall_penalty_<agent>` / `disposal_cost_<agent>`），需还原系数：  
  - `pen_shortfall = shortfall_penalty_money / max(1e-6, shortfall_qty * penalty_multiplier_out)`  
  - `cost_disposal = disposal_cost_money / max(1e-6, inventory_penalized * penalty_multiplier_in)`  
  - `penalty_multiplier` 由 `penalties_scale` 决定：`trading`→`trading_price`（短缺用 `trading_price_{level+1}`，处置用 `trading_price_{level}`），`catalog`→`catalog_price`，`none`→1，`unit`→`unit_price`（缺失时可用当前报价近似）
- `pen_shortfall_norm`, `cost_disposal_norm`: 直接使用上述系数（已是相对价格尺度），不再除 `trading_price`。
- `system_breach_prob/level`: 系统日志通常不提供，置空；若运行时可从 bulletin-board 取到，记录到 Tracker。

F) History Tokens（NEGOTIATION scope）
- 以“事件重建规则”生成事件序列：每个事件包含 `action_type`（OFFER/ACCEPT/END）+ 对应 offer 的 `q_bucket/p_bin/round_bucket`，可附 `is_counter`/`is_first_proposal`。
- `speaker`: 以 proposer 为视角：事件 proposer == 当前样本 proposer -> ME，否则 OPP。
- `history_len`: 实际事件长度；`history_tokens` 长度不足 N 左侧 padding。
- `history_scope`: 固定为 `NEGOTIATION`。

G) 合同/违约（Breach 数据集）
- `contract_id/terms`: 来自 `contracts.csv`。
- `breach`/`breach_level`: 先看 `contracts.csv.breaches` 字段；不足时合并 `breaches.csv`。
- `executed`: 来自 `contracts.csv.executed_at` 或执行相关字段。

H) Partner 统计特征（PARTNER scope，派生）
- 由重建事件 + 合同执行统计得到：
  - `n_deals_with_pid`, `n_breaches_with_pid`
  - `accept_rate_first_proposal`：统计 `is_first_proposal` 的 ACCEPT 比率
  - `accept_rate_overall`：ACCEPT / (ACCEPT + REJECT)
  - `avg_rounds_to_agreement`：在 agreement 的谈判中平均 round 数
  - `recent_accept_rate_lastM` / `recent_breach_rate_lastM`：滑动窗口或 EMA
- 统计键建议用 `(partner_id, role)` 分开存。
- 统计必须按事件时间顺序做前缀累计：对每条样本写入时只用“截至该事件之前”的统计快照；写样本后再更新统计，避免泄漏。

I) 无法从系统日志重建的字段（需 Tracker 或置空）
- `model_prior_accept_mu`, `p_accept_prior/post/LCB`, `p_breach`, `p_eff`, `decision_meta`, `beta_state` 等内部变量。

J) Schema 元字段（系统日志可给或可常量填充）
- `schema_version`: 常量 `v0`（或当前版本号）。
- `event_type`: 事件类型来自重建 `proposal_sent`/`offer_received`/`response_sent`；Accept 样本取 `proposal_sent`/`counter_offer`。
- `agent_id`: proposal 事件用 `proposer_id`，response 事件用 `responder_id`；其他快照按 buyer/seller 字段填。
- `n_steps`: 来自 `params.json.n_steps`（仿真步）；谈判步上限来自 `negs.csv.n_steps`。
- `n_lines`: 可由 `info.json.active_lines` 取最大值/均值（若有）；否则用 `q_max` 近似。
- `mechanism_id`: 系统日志无该字段，使用 `negotiation_id` 或 `actions.csv.neg_id` 作为替代。

---

### 2.2 数据源 B（可选）：你们自己的 OneShot Tracker

官方日志的缺点是：它记录的是事件，但**不一定包含你们训练对手模型所需的完整上下文特征**，例如：

* 你方当时的 `remaining_need`
* 你方当时的 penalty 参数（shortfall_penalty/disposal_cost）
* 你方当时使用的策略阶段（early/mid/late），以及你方“计划的目标量”
* 你方当时对每个对手的 Beta(α,β) 状态、LCB 值、模型 prior 输出

这些属于线上决策内部变量，系统日志无法提供；只有在需要线上调试/复现时才需要 Tracker 记录。

因此：*Tracker 可选，仅在需要内部特征时再做 OneShotTrackerMixin（或扩展现有 mixin）。*

---

## 3) OneShot Tracker 设计（可选）：必须记录什么？怎么记录？
如果仅依赖系统日志训练 OM，这一节可以跳过；仅在需要记录线上决策内部变量/调试复现时启用。


### 3.1 Tracker 的目标

让任何人拿到日志都能回答：

* 我们在 step t 对 opponent i 提出了哪些 offer？（first/还价）
* opponent i 对这些 offer 的响应是什么？（accept/reject/end，timeout 视为 end）
* 如果 accept 了，是接受了哪一个 offer（精确到 q/p/time）
* 最终合同是否执行/违约？违约程度？
* 当时我们对它的 `p_accept_prior/posterior/LCB` 是多少？对 `p_breach` 的估计是多少？
* 当时的上下文：need、角色、价格边界、公告板 breach_prob（如果有）…

---

### 3.2 需要实现的 OneShotTrackerMixin（不写代码，但写“要覆写哪些方法”）

你们 OneShot agent 通常继承 `OneShotSyncAgent`，决策入口主要是：

* `first_proposals()`：你给所有对手的首轮 offer（批量）
* `counter_all(offers, states)`：收到对手 offers 后，你返回对每个对手的 response（accept/reject/end）

此外还有回调：

* `on_negotiation_success(contract, mechanism)`
* `on_negotiation_failure(partners, annotation, mechanism, state)`
* **可能存在**（不同版本/track）：
  `on_contract_executed(contract)`
  `on_contract_breached(contract, breaches, resolution)`
  `on_contracts_finalized(...)`（有些版本会在一天结束给总结）

**OneShotTrackerMixin 必须覆盖并记录：**

#### (A) init() / before_step()

记录：

* world_id（你们已有 world hash 逻辑很好）
* n_steps、n_lines、is_first_level（买/卖角色）
* 当步的公共参数快照：trading_price、min_price/max_price、penalties、exogenous quantities（如可得）

#### (B) first_proposals()

对每个 partner 产生一个事件：`proposal_sent`
字段建议包含：

* `step`, `partner_id`, `offer = (q, p, time)`
* `round_rel`, `round_bucket`（谈判进度；time 仍从日志读取但不用于建模）
* `is_first_proposal = True`
* `role`（buyer/seller）
* `need_remaining`（买方 needed_supplies / 卖方 needed_sales）
* `price_bounds`：min_price/max_price（来自 issues）
* `system_breach_prob`（如果能拿到）
* `model_prior_accept_mu`（Logistic/Transformer 输出）
* `prior_strength_s`（如果你用 evidential/固定强度）
* `beta_state_before = (α,β)`（对应 bucket）
* `accept_LCB_before`（用于 debug）
* `decision_meta`：你用了哪种启发式（portfolio/topK/探测单等）

#### (C) counter_all(offers, states)

在进入 counter_all 时，先记录每个 `offer_received`：

* `partner_id`, `offer = (q, p, time)`, `round_rel`, `round_bucket`（如可得）

然后对你返回的 response，记录 `response_sent`：

* `partner_id`
* `response_type` ∈ {ACCEPT, REJECT, END}
* `responded_offer`：被回应的对手报价
* 如果 REJECT：`counter_offer`
* `need_remaining`、penalty、price_bounds
* **此时的对手模型估计**：`p_accept_prior/post/LCB`、`p_breach`、`p_eff = p_accept* (1-p_breach)`
* `reason_code`：比如 “portfolio_optimal / risk_avoid / fill_shortfall / probe_explore …”

#### (D) on_negotiation_success()

记录 `negotiation_success` + `contract_signed`：

* contract_id, partner_id, agreement(q,p,time)
* 谁是最后出价者（如果机制状态提供）：用于判定“对手接受的是不是我们的 offer”
* 如果缺少 response_sent 的 ACCEPT，可用 agreement 回填（用于 offer-centric 标签对齐；agreement/最后一条 offer -> 对应我方 offer）。

#### (E) on_contract_executed / on_contract_breached（如果被调用）

这两条是你担心 “Tracker 可能没记录违约” 的关键补丁：

* `contract_executed`: contract_id, executed_quantity（如果有）
* `contract_breached`: contract_id, breach_level/breaches list

> 若实际运行发现 OneShot 不触发这两个回调：就要走下面的 **“日终从系统报告补标签”**。

#### (F) 日终（step 结束）从系统报告/公告板补齐 breach 标签

实现方式：在 `step()` 末尾或 `before_step()` 读取 “上一日” 的系统报告（如果接口存在），把：

* `breach_list`（谁违约、违约程度）
* `breach_prob`（累计违约比例）
  写入 tracker，并且把与你方已签约的 contract_id 对齐打标。

---

## 4) 系统 breach_prob 接口：先探测，再决定用系统还是用网络

你提的第 4 点我们要做成“工程可执行”的探测流程：**初始化时检测接口，运行时分支。**

### 4.1 你们必须实现一个 `CapabilityProbe`（能力探测）

在 agent `init()` 执行一次（并写入 tracker）：

探测清单（按优先级）：

1. `awi.reports_at_step(step)` 是否存在？

   * 若存在：返回 dict，value 是否有 `breach_prob` / `breach_level` 字段
   * （你们 scml-agents 旧代码里有人这么用过，这是最可能存在的接口）

2. `awi.reports` / `awi.reports_of_agent` / `awi.reports_for` 类似方法是否存在？

3. `awi.bulletin_board` / `awi.bb` / `awi.board` 是否存在？

   * 是否能读到 per-agent 的 breach_prob / breach_level / breach_list

4. 若以上都没有：记录 `system_breach_unavailable=True`

把探测结果写入日志：

* `cap.system_breach_source = "reports_at_step" | "bulletin_board" | "none"`
* `cap.fields = {...}`（实际可读字段名）

### 4.2 运行时的使用策略（分世界配置自适应）

实现一个 `BreachInfoProvider`，统一接口：

* `get_system_breach_prob(pid, step)` → `Optional[float]`
* `get_system_breach_level(pid, step)` → `Optional[float]`
* `get_system_breach_list(step)` → `Optional[list]`

策略：

* 如果系统能给：**优先用系统**（它全局视角更准）
* 如果系统不给：用你们的 `breach_model` + `online_beta_breach`（见后文）

---

## 5) 概率模型：Logistic 先行 + Transformer 同步开发

你提的第 3 点（并行 Logistic）非常正确：它能把“数据/标签/对齐/在线更新/决策接口”先跑通，避免 Transformer 训练周期拖累工程进度。

### 5.1 对手模型接口拆分（关键工程点）

接受概率与违约概率使用 **不同 history 口径**，因此拆为两个模型：

**AcceptModelInterface（必需）**

* `predict_accept_mu(context, offer, history_neg) -> μ ∈ (0,1)`
* `predict_accept_strength(context, offer, history_neg) -> s ≥ 2`（可选；先固定常数）

**BreachModelInterface（可选）**

* `predict_breach_mu(context, partner_features, history_partner) -> μ_breach`
* 若系统可提供 `breach_prob/level`，则此模型可用 placeholder 跳过训练。

启发式与 BOU 对 accept 模型只吃 (μ,s)；违约风险由 `BreachInfoProvider` 统一提供（系统优先，其次 breach 模型）。

### 5.1.0 模型工件目录与加载规范（固定）

* 统一模型工件目录：`assets/models/accept/` 与 `assets/models/breach/`。
* 目录下固定文件：`model.bin` + `model_meta.json`（必需），`calibration.json`（可选）。
* `model_meta.json` 必含字段：
  * `model_type`（logistic/transformer）
  * `schema_version`（如 `v0-A2`）
  * 离散化参数（如 `round_bucket_T`、`q_bucket_spec`、`p_bin_edges`）
  * `is_counter`/`is_first_proposal` 是否作为 token
  * 归一化口径（如 `q_norm = q / n_lines`）
* `assets/agent_config.json`：记录 `accept_model_dir`/`breach_model_dir` 与关键超参（`prior_strength_s`、`LCB_delta_accept`、`w_end`、`add_terminal_negative`、`buffer_*`、`overfill_penalty_ratio`、`q_candidate`、`price_concession_gamma` 等）。
* 路径解析必须相对模块文件所在目录，不依赖运行时 CWD；允许环境变量 `LITA_ACCEPT_MODEL_DIR`/`LITA_BREACH_MODEL_DIR` 覆盖。

---

### 5.1.1 时间与离散化规则（必须一致）

**时间（谈判进度）**

* `round_rel`（连续，0..1）：若 state 提供 `relative_time`，则 `round_rel = state.relative_time`；否则 `round_rel = state.step / max(1, state.n_steps - 1)`。
* `round_bucket`（离散桶）：`round_bucket = min(T-1, floor(round_rel * T))`，默认 `T=5`（早/中早/中/中晚/晚）。
* 用途：`round_rel` 用于 Logistic/Transformer；`round_bucket` 用于 BOU 分桶与回退。
* 在线决策：`first_proposals` 固定 `round_rel=0`；`counter_all/子集选择` 以 `states.relative_time` 为准，缺失时回退到 0。

**价格离散化**

* `p_bin`：在合法价格区间 `[p_min, p_max]` 内，p 更接近下界还是上界。
  * OneShot 常见 `p_max = p_min + 1` 时：`p_bin=0` 表示 `p==p_min`，`p_bin=1` 表示 `p==p_max`。
  * 一般情况：`p_bin = 0` 若 `p <= (p_min + p_max)/2`，否则 `p_bin = 1`。

**数量离散化**

* `q_max = nmi.issues[QUANTITY].max_value`（或用 `awi.n_lines`）。
* `q_clamped = clip(int(q), 0, q_max)`，`q_bucket = q_clamped`。
* 取值范围：`q_bucket ∈ {0,1,2,...,q_max}`。

---

### 5.1.2 history 定义（模型输入口径）

**核心定义**

* history 是按时间顺序排列的事件序列，用来预测对手响应或可靠性。
* 两类概率对应两种 history 口径：
  * `P_accept`: 以 **NEGOTIATION scope** 为主（当前谈判内的交互轨迹）。
  * `P_breach/Trust`: 以 **PARTNER scope** 为主（跨谈判/跨天的统计）。
* 因为 history 口径不同，accept 与 breach 模型必须拆分。

**NEGOTIATION scope（用于 accept）**

* 定义：对某个对手 `pid`，当前谈判 `negotiation_id` 内，从开始到当前的事件序列。
* 推荐存储：固定长度 ring buffer（例如 K=100），模型输入取最近 N 条（先用 N=16，后续可 32）。
* token 最小字段：
  * `speaker` ∈ {ME, OPP}
  * `action_type` ∈ {OFFER, ACCEPT, END}
  * `q_bucket`, `p_bin`, `round_bucket`
  * （可选）`is_counter`, `is_first_proposal`
* OFFER 口径：所有报价统一编码为 OFFER；counter_offer 标记 `is_counter=True`，首轮 offer 标记 `is_first_proposal=True`。
* REJECT 仅存在于 `response_type`（response_sent），不作为 token 的 `action_type`。
* 必须包含我方行为：对手反应依赖我方让步轨迹，单边序列会误判对手类型。

**PARTNER scope（用于 breach/可靠性）**

* 形式：统计量优先（不直接喂长序列），例如：
  * `n_deals_with_pid`, `n_breaches_with_pid`, `avg_breach_level`
  * `accept_rate_first_proposal`, `accept_rate_overall`
  * `avg_rounds_to_agreement`, `recent_accept_rate_lastM`, `recent_breach_rate_lastM`
* 对 Logistic/Transformer 都可作为 context/global 特征。

**ID 输入原则**

* 不输入原始 `partner_id` token（避免过拟合/泛化差）。
* `partner_id` 仅用于索引在线 Beta/统计状态（如 `H_neg[(pid, negotiation_id, role)]`, `H_partner[(pid, role)]`）。

---


### 5.1.3 特征可观测性矩阵与输入约束
Online 可得（推理必然可拿）：`round_rel/round_bucket`、`price_min/price_max`、`role`、history（双方交互）、我方 `need_remaining`/`pen_shortfall`/`cost_disposal`（系数口径）/`trading_price`、offer(`q`,`p`)、`partner_stats_*`
Online 可能可得（需 CapabilityProbe）：`system_breach_prob/level`（若系统提供，需 mask）
Online 不可得：对手 `need_remaining`、对手 penalties（shortfall/disposal）
硬规则：AcceptModel 只吃 Online 可得 + 可能可得（带 mask）特征；对手私有量不进模型输入（可用于分析/评估）

### 5.2 Logistic baseline（可快速验证的一版） Logistic baseline（可快速验证的一版）

**目标：** 先能学到 “对手在什么情况下更可能 accept”，并且能在线更新 + LCB 决策。

输入特征建议（从易到难）：

* 时间：`round_rel`（谈判进度，0..1；定义见 5.1.1）
* 角色：buyer/seller（one-hot）
* offer：

  * `q_norm = q / n_lines`
  * `p_bin`（见 5.1.1；OneShot 常见就两档）或 `p_norm`（若连续）
* 我方状态：

  * `need_norm = remaining_need / n_lines`
  * `pen_shortfall_norm, cost_disposal_norm`（系数口径，无需再除 trading_price）
* 对手公开风险：

  * `system_breach_prob`（如果可得；否则置空/0 并加 mask）
* 对手近期行为统计（来自 history 或 online stats）：

  * `opp_accept_rate_lastK`
  * `opp_counter_rate_lastK`（counter / (counter + accept)）
  * `opp_concession_speed`（对方出价从 max→min 的速度）
  * 冷启动默认 0.5（避免全 0 输入导致极端偏置）

输出：

* `μ_accept`

训练：

* 二分类交叉熵（label 是 “该 offer 是否被接受”）
* 需要做 class imbalance 处理（accept 通常偏少）：

  * weighted BCE 或者 focal loss（Logistic 也可用 sample weight）

验收标准（先设“工程可用线”，不是最强）：

* logloss 明显低于 “恒定预测”（比如永远 0.1）
* 校准曲线不要崩（后面会加温度缩放）

---

### 5.3 Transformer（对手类型识别 + 条件概率更强）

Transformer 的关键不是“比 Logistic 大”，而是它能吃 **序列**：

输入：

* `Context token`：`round_rel`、need、`role`、penalties、price bounds、system breach 等（仅在线可观测/可探测）。
* 最近 N 条事件 token（建议 N=16 或 32）：

  * speaker（me/opp）
  * action_type（offer/accept/end）
  * （可选）is_counter / is_first_proposal
  * q_bucket（0..q_max）
  * p_bin
  * round_bucket
输出：
* head1：`μ_accept`
* breach 模型独立实现（或由系统提供），不在该 Transformer 里做多头输出

训练同 Logistic（BCE/NLL）但要加：

* **校准**（温度缩放 / isotonic）
* **分层评估**（按对手 breach_prob、按 step 分层）

---

## 6) Bayesian Online Update（BOU）如何与模型协作？意义与理论是什么？

你提的第 5 点我这里给一个“工程上可落地、理论上站得住”的解释与用法。


### 6.0 BOU 分桶与更新方向（写死）
目标随机变量：
p_pid,b,sign = P(pid 接受我方 offer | b)

**BOU key**
key = (pid, role, round_bucket, p_bin, q_bucket_coarse)

**q_bucket_coarse（用于 BOU 粗桶）**
- q=1
- q=2
- q in [3,4]
- q in [5,7]
- q>=8
- q<=0 不更新

**更新触发方向**
- 仅当 pid 作为 responder 回应我方 offer 时更新；我方作为 responder 的事件不更新。
- END/timeout 默认不更新；仅对对手 END/timeout/failed 且无 agreement 的情形启用 `add_terminal_negative`；我方主动 END 不更新。

### 6.1 用模型输出构造 Beta 先验（核心做法） 用模型输出构造 Beta 先验（核心做法）

我们把 “对手接受某类 offer 的概率” 看成伯努利参数 (p)。

* 先验：(p \sim \mathrm{Beta}(\alpha_0, \beta_0))
* 观测：每次你对该对手发出某类 offer，会得到结果 (y \in {0,1})（是否 accept）

**神经模型输出：**

* ( \mu = \hat{p}_\text{model}(x) )
* ( s )（先验强度，类似“伪样本量”，可常数）

构造先验：
[
\alpha_0 = \mu \cdot s,\quad \beta_0 = (1-\mu)\cdot s
]

观测到 success (S)、failure (F) 后，后验：
[
p \mid D \sim \mathrm{Beta}(\alpha_0+S, \beta_0+F)
]

**意义：**

* 数据少时，后验会更接近模型先验（避免极端过拟合）
* 数据多时，(S,F) 主导，后验逐渐“听本局现实”（快速适应）

这就是 “神经网络提供归纳偏置 + 贝叶斯提供在线自适应与不确定性”。

### 6.2 为什么要用 LCB（下置信界）做决策？

你们的损失不是对称的：**缺量/卖不出会被罚**，而且罚可能很大。
所以用后验均值 (\mathbb{E}[p]) 可能过于激进（尤其在样本少/模型 OOD 时）。

因此我们用 **后验的下置信界**（Lower Credible Bound）：
[
\mathrm{LCB}*\delta(p) = \text{Quantile}*{\delta}\big(\mathrm{Beta}(\alpha,\beta)\big)
]
比如 (\delta=0.2) 表示：我们取 “有 80% 概率高于它” 的保守估计。

工程意义：

* 对“没交互过的陌生对手”，LCB 会很低 → 自动保守（避免大单）
* 对“已验证的稳定对手”，LCB 会快速上升 → 自动加大订单
* 这等价于一种 **风险厌恶 + 自动探索** 的机制（和 Bayesian UCB / Thompson 思路同源）

> 你们也可以同时实现两种模式：
>
> * **Thompson**：从 Beta 采样 (p) 来做探索（早期探测单）
> * **LCB**：用分位数做稳健下界（大单/关键决策时）

---

## 7) D‑NB 决策启发式：如何“吃到”模型 + BOU 输出？

你们明确要“决策用启发式”，所以这里必须写到能实现。

### 7.1 定义核心概率：签约概率 × 履约概率

对某个对手 (i)、某个候选 offer (o=(q,p))：

* (p^{\text{sign}}_i(o))：对手会接受并达成合同的概率
  → 用 accept 模型 + Beta 后验 LCB：(\mathrm{LCB}(p^{sign}))

* (p^{\text{fulfill}}_i)：对手会履约（不违约）的概率

  * 若系统给 breach_prob：(p^{fulfill}=1-\text{breach_prob})
  * 否则：用 breach 模型 / breach Beta 后验给 LCB

组合成“有效成功概率”：
[
p^{\text{eff}}_i(o) = \mathrm{LCB}(p^{\text{sign}}_i(o)) \cdot \mathrm{LCB}(p^{\text{fulfill}}_i)
]

有效期望交付量：
[
\tilde q_i(o)= q \cdot p^{\text{eff}}_i(o)
]

---

### 7.2 First proposals 的 portfolio 分配（替代硬 RiskGate）

你之前担心 “RiskGate 直接裁剪会导致买不够/卖不出”，所以我们不用硬裁剪，而用 **组合约束 + 风险权重**：

#### 目标（一个可实现版本）

在所有对手上选一组 offer (o_i)（很多对手可选 q=0 表示不出价），最大化：
[
\max \sum_i \Big( \tilde q_i(o_i)\cdot m_i(o_i) \Big) - \lambda \sum_i q_i \cdot (1-\mathrm{LCB}(p^{\text{fulfill}}_i))
]

其中：

* (m_i(o_i)) 用 trading_price 近似：卖方 `m=(p-trading_price)/max(1e-6,trading_price)`，买方 `m=(trading_price-p)/max(1e-6,trading_price)`，再乘数量；若无 trading_price 用单调近似。
* (\tilde q_i(o_i)= q_i \cdot \mathrm{LCB}(p^{\text{sign}}_i)\cdot \mathrm{LCB}(p^{\text{fulfill}}_i))，即同时考虑签约与履约（我方作为 proposer 时适用）。
* 第二项是“残余履约风险惩罚”（可选；若 \tilde q 已含 p^{eff}，默认可设 \lambda=0，避免双计）

#### 关键约束（保证不缺量）

你们最需要的是：**在高置信下满足 need**。用 LCB 的有效交付量做约束：

[
\sum_i \tilde q_i(o_i) \ge \text{need} \cdot (1 + \text{overordering\_ratio})
]

**2026-01-10 更新：移除 buffer 机制，改用 buyer_overordering_ratio**

罚款分析结论（RChan 启发）：
- Shortfall penalty 约为 Disposal cost 的 **10 倍**（货币量纲）
- 因此 BUYER 应容忍超量采购，SELLER 应保守

新策略：
- BUYER: `target = need_remaining * (1.0 + buyer_overordering_ratio)`，默认 10%
- SELLER: `target = need_remaining`（不超量）

原 buffer(t) 已移除（与 overordering 功能重复）。

#### 实现要点（与代码一致）

* 首轮报价 `round_rel` 固定 0（`round_bucket=0`），不要用仿真日进度代替谈判进度。
* 价格不再固定极值：按让步曲线 `round_rel^price_concession_gamma` 在 `[p_min,p_max]` 内线性移动（买方从低到高，卖方从高到低）。
* 先用候选数量 `q_candidate` 计算 `μ/LCB` 与排序，再用最终 `q` 重新估计 `p_eff` 做分配与更新。
* 报价的 `time` 只按 issues 的合法范围读取与校验（OneShot 常为常数，但不能硬编码为 `current_step`）。

超量控制：不做硬上限，在 counter_all 的子集评分里加入 overfill 惩罚（`overfill_penalty_ratio <= 0.1`，实际货币惩罚再乘 `penalty_multiplier`）。

#### 为什么这比裁剪更好？

* 裁剪是 “把某些对手变成 0”，但你在缺量时仍然需要“不得不交易的坏对手”
* portfolio 约束会在缺量时自动启用坏对手，但给它 **更小的单**，并通过 buffer 对冲

---

### 7.3 counter_all 的接受/拒绝：子集选择（组合层面最重要）

在 counter_all 里，你会同时收到多个对手的 offer（或你自己的上轮 offer）。你需要决定接受哪些。

可实现策略（工程上简单且强）：

1. 构造候选集合 (C)：所有当前 step 的对手 offer
2. 枚举子集 (S \subseteq C)（若对手数多，用 beam search/贪心近似）
3. 对每个子集算一个风险调整后的分数：

买方（需要买够）：
[
\text{Score}(S) = \text{Utility}(S) - \text{ShortfallPenalty}(\text{need}-\sum_{o\in S}\tilde q(o)) - \text{OverfillPenalty}(\sum_{o\in S}\tilde q(o)-\text{need}) - \text{RiskPenalty}(S)
]

卖方（需要卖出）：
[
\text{Score}(S) = \text{Utility}(S) - \text{DisposalCost}(\text{need}-\sum_{o\in S}\tilde q(o)) - \text{OverfillPenalty}(\sum_{o\in S}\tilde q(o)-\text{need}) - \text{RiskPenalty}(S)
]

其中：

* 接受对方 offer 时视为 `P_sign=1`，因此 \tilde q = q*LCB(P_fulfill)；若是我方发出 counter_offer，则 \tilde q = q*LCB(P_sign)*LCB(P_fulfill)。
* 超量惩罚：`over = max(0, sum(tilde q)-need)`；`OverfillPenalty = overfill_penalty_ratio * penalty_unit * over`，其中 `penalty_unit = pen_shortfall * penalty_multiplier_out`（买方）或 `cost_disposal * penalty_multiplier_in`（卖方），且 `overfill_penalty_ratio <= 0.1`。
* `RiskPenalty` 可按 breach_prob 做加权

4. 选分数最高的子集，接受其 offers，其余拒绝（带 counter_offer）

5. 接受子集后更新剩余需求（与实现对齐）：
   * `accepted_q_eff = Σ(q × fulfill)`（若禁用 breach 概率，则 `fulfill=1`）
   * `committed` 为当天已签约累计量（包含：counter_all 中我方接受对方报价；对方接受我方报价在 `on_negotiation_success` 中累计；已用标记避免双计）
   * `need_remaining_raw = max(0, need - accepted_q_eff)`
   * `need_live = max(0, need_remaining_raw - committed)`
   * `pending_expected/pending_worst` 来自未回复的已发报价
   * BUYER：`need_adj = max(0, need_live - pending_expected)`；SELLER：`need_adj = max(0, need_live - pending_worst)`
   * `need_remaining = ceil(need_adj - 1e-9)`
   若 `need_adj <= 0` 则对其余对手直接 END；否则用 `need_remaining` 重新生成 counter_offer，**必须返回 `REJECT + counter_offer`，禁止 `REJECT + None`**（否则会被视为 END）。
6. 本轮 `round_rel` 取 `states` 的 `relative_time`（缺失时取最小值或 0），不要用仿真日进度。

> 这一步就是把 “违约概率” 真正用在决策里：不是只过滤对手，而是决定**接受组合**。

---

### 7.4 探测单（Exploration）机制：解决“新对手类型识别”

OneShot 中你经常遇到你从未交互过的对手。你要避免一上来给大单。

可实现规则（非常工程化）：

* 当某对手对应 bucket 的 (\alpha+\beta < N_\text{min})（样本不足）：

  * 只给 `q=1` 或 `q=2` 的 probe offer
  * price 用对你最有利的档（买方给低价，卖方给高价）
* 每次 probe 的 accept/reject 更新 Beta
* 当 LCB 超过阈值（比如 0.4）且 breach 风险低，再逐步放大 q

这能让 Transformer 识别对手类型，同时 **不把探索成本变成罚分风险**。

---

## 8) 训练方案：这里不是 BC，而是“监督学习 + 调参优化”

你提的第 2 点：你觉得没有要 clone 的 agent。完全同意。这里我们做的是：

* **对手响应监督学习**（accept/breach）
* **启发式超参黑盒优化**（比 BC 更适合工程冲刺）

### 8.1 数据集构建（必须写到字段级）

你们需要两个主要数据集（Schema v0-A/B）：

#### Schema v0-A：Accept 数据集（offer-centric，逐 offer_sent）

样本单位：一次 `offer_sent`（`proposal_sent` / `counter_offer`），预测对手是否接受该 offer。
一致性要求：DatasetBuilder/Trainer/Schema 统一采用 offer-centric 口径，禁止混用 response_sent。
说明：`response_type`/`responder_id` 仅用于事件重建与标签对齐，不作为样本主体定义。

**主键（建议）**：`world_id`, `negotiation_id`, `step`, `proposer_id`, `partner_id`（可加 `event_index`）

**必需字段**：
- Context（在线可观测）：`round_rel/round_bucket`, `role`, 我方 `need_remaining`, `trading_price`, `price_min/price_max`
- Offer：`q`, `p`, `t` + `q_bucket/p_bin`（或 `q_norm/p_norm`）
- History tokens：`speaker`（以 proposer 为视角 ME/OPP）、`action_type`、`q_bucket/p_bin/round_bucket`
- Metadata：`history_len`, `history_scope`

**可选字段**：
- 我方 penalties（`pen_shortfall/cost_disposal`）
- `system_breach_prob/level`（若可得，需 mask）
- `partner_stats_*`

**标签**：
- `y_accept=1`：agreement 匹配我方 offer
- `y_accept=0`：对手 counter_offer
- END/timeout 默认不入样本；仅在对手 END/timeout/failed 且无 agreement 时，才可选 `add_terminal_negative=True` 给最后 outstanding offer 弱负样本（weight≈0.2）；我方主动 END 不计入。

#### Schema v0-B：Breach 数据集（违约/履约）

**两种口径：contract 级 / partner-day 级**

1) **按 contract**
* 主键：`world_id`, `partner_id`, `contract_id`
* 输入：context + PARTNER 统计 + 合同条款 `q,p,t`
* 标签：`y_breach` 或 `breach_level`

2) **按 partner-day**
* 主键：`world_id`, `partner_id`, `day_id`
* 输入：当日成交/执行/违约汇总（PARTNER scope 统计）
* 标签：`y_breach` 或 `breach_level`

---

### 8.2 调参（黑盒优化）必须写进方案 调参（黑盒优化）必须写进方案

你们最终强不强，很大程度取决于启发式超参。建议把以下参数暴露成 config：

* `LCB_delta_accept`（0.1~0.4）
* `LCB_delta_fulfill`
* `prior_strength_s_accept`（默认 8，推荐 2~20）
* `prior_strength_s_breach`
* `buffer_schedule`：固定 `b_min/b_max/gamma`，默认 0.05/0.35/2
* `probe_q`、`probe_steps`（前几步强制探测）
* `q_candidate`（候选数量，用于先验估计）
* `portfolio_K`（计算 budget：topK 进入枚举，其他用贪心）
* `risk_lambda`（违约风险惩罚系数）
* `overfill_penalty_ratio`（超量惩罚系数，≤短缺惩罚的 10%）
* `price_concession_gamma`（让步曲线幂指数）
* `price_policy`（min/max 切换时机阈值）

调参方法（工程上可快速落地）：

* 先 grid/随机搜索（几十组）
* 再用 CMA‑ES / Bayesian Optimization（如果你们已有框架）

目标函数建议不是只看均值，要看稳健性：
[
J = \mathbb{E}[\text{score}] - \eta \cdot \text{Std}[\text{score}] + \gamma \cdot \text{P10}[\text{score}]
]
（P10 是第 10 分位数，避免“偶尔爆炸”）

---

## 9) 代码实施路线：按这个顺序做，最稳

下面是“无需写代码也能照做”的实施路线。每一步都有明确产物与验收点。

### Phase 1：先确认日志与接口（不改模型）

1. 跑一个最小 OneShot tournament（n_configs=1, n_runs=1, n_steps=5）
2. 在输出的 negmas world 目录中确认是否存在：

   * actions.csv、contracts.csv、negotiations.csv、breaches.csv（若无 breach 看 contracts 字段）
3. 用 `scml_analyzer/log_parser.py` 解析该 world 目录，确认：

   * actions_df 非空
   * contracts_df 非空
   * breaches_df 若存在能读
4. 在你们 agent `init()` 打印/记录一次 `dir(self.awi)`（仅 debug），完成 breach 接口探测草表

**产物：** 一份 “OneShot 环境可用字段清单”（写到 README 或 DESIGN.md）

---

### Phase 2：系统日志数据管线（不依赖 Tracker）

1. 用 2.1.2 的重建规则从 world logs 生成事件序列与 offer-centric 标签（proposal_sent/counter_offer -> y_accept）
   * 输入：`actions.csv` / `negotiations.csv` / `negotiations/*.csv` / `negs.csv` / `contracts.csv`
   * 输出：`accept_dataset.parquet/jsonl`
2. 抽样对齐检查：随机挑几场 negotiation，把“某个 offer 是否被接受”的标签与 agreement/响应对齐

**产物：**
* accept_dataset_v0（system logs）
* 字段重建对齐检查记录

---

### Phase 2b（可选）：实现 OneShotTrackerMixin（仅在需要内部特征时）

实现一个新的 mixin（建议新文件，不要硬改旧的）：

* `scml_analyzer/oneshot_tracker_mixin.py`

  * 覆写：init, before_step, first_proposals, counter_all, on_negotiation_success, on_negotiation_failure
  * 可选覆写：on_contract_executed, on_contract_breached, on_contracts_finalized
  * before_step 还需要清空会话级缓存（等待响应/最后报价/已接受/已结束）与历史，避免跨天“幽灵拒绝”

并把它接入你们 OneShot agent（LitaAgent‑OS 或临时基于 Cautious/Rchan 的壳子）。

**产物：**

* （可选）每场比赛每个 agent 输出 tracker_logs/agent_*.json（或 jsonl）
* 事件里包含 proposal_sent/offer_received/response_sent 等关键字段

---

### Phase 3：做 Accept 数据集管线 + Logistic baseline（最关键的“先跑通”）

1. 写一个 dataset builder（建议独立脚本）：

   * 输入：tournament_history 下的 world logs（actions/negotiations/negs/contracts/negotiations/*.csv）
* 口径：offer-centric（proposal_sent/counter_offer）；可选 `add_terminal_negative`
   * 输出：`accept_dataset.parquet/jsonl`
2. 训练 Logistic accept 模型
3. 离线评估：logloss、Brier、ECE（至少能看 reliability curve）
4. 接入 agent：让 agent 在决策时调用 logistic → 得到 μ → 构造 Beta prior → 在线更新 → LCB 决策
5. 跑小规模 tournament，确认分数不崩，且日志中 LCB/αβ 有随交互变化

**产物：**

* accept_model_v0（logistic）
* online_beta_accept 生效的日志证据

---

### Phase 4：加入 breach 风险（先系统后自建）

1. 实现 `BreachInfoProvider`：

   * 若系统可给：直接用系统 breach_prob
   * 否则：用你们的 breach Beta（从观测 breach 事件更新）
2. 把 breach 融合进 (p^{eff}) 与 portfolio 约束
3. 跑 tournament，观察：

   * 是否减少“与高 breach 对手的大单”
   * 缺量罚是否降低（看 shortfall 相关统计）

**产物：**

* 可运行的 D‑NB v0（Logistic + BOU + breach provider + portfolio）

---

### Phase 5：Transformer 训练与替换（在不影响主线的情况下并行）

1. 用同一套 dataset builder 输出序列数据（最近 N 事件 token）
2. 训练 Transformer accept 模型（head1）
3. 校准（温度缩放）
4. 用统一接口替换 logistic（保持 BOU 与决策不变）
5. 做 ablation：

   * logistic vs transformer（同样的决策器）
   * transformer without BOU vs transformer+BOU

**产物：**

* accept_model_v1 (transformer)
* 清晰的 ablation 报告（哪部分带来收益）

---

### Phase 6：黑盒调参（把启发式“拧到最强”）

1. 把所有关键超参 config 化
2. 搭建批量评估脚本（多 seed、多 config）
3. 先随机搜索 50~200 组（很快能找到明显提升）
4. 再做 CMA‑ES/BO（如果时间允许）

**产物：**

* best_config.yaml（你们可直接用于提交比赛）

---

## 10) 非常具体的验收清单（逐条勾选）

### A. 环境与日志（必须全绿）

* [ ] OneShot tournament 的 world 目录中存在 actions.csv
* [ ] contracts.csv 可读，且能识别 executed/breached（字段或 breaches.csv）
* [ ] scml_analyzer 的 LogParser 能解析该目录，actions_df/contracts_df 非空
* [ ] 统计文件名已确认（stats.csv 或 stats.csv.csv），DatasetBuilder 可兼容读取
* [ ] `CapabilityProbe` 已实现并在 tracker 中记录系统 breach 数据源（reports/bb/none）
* [ ] 若系统 breach 可得：至少能拿到 breach_prob（0~1）

### B. OneShotTrackerMixin（可选）
仅在你们决定补齐内部决策变量时才需要完成此项。


* [ ] first_proposals 中每个对手都有 `proposal_sent` 事件，包含 q/p/time
* [ ] counter_all 中每个收到的 offer 都有 `offer_received`
* [ ] counter_all 中每个返回 response 都有 `response_sent`，包含 response_type
* [ ] on_negotiation_success 有 `contract_signed`，且能定位 partner_id
* [ ] negotiation_id / mechanism_id（至少一种）能把 proposal 与 success 对齐
* [ ] 若 on_contract_breached/executed 不触发：日终补标逻辑能从系统报告/统计里写入 breach 事件

### C. 数据集构建（必须全绿）

* [ ] accept_dataset 每条样本都有：context + offer + label y_accept
* [ ] context 仅使用在线可观测特征（对手私有量不得进入模型输入）
* [ ] agreement 能回溯到某条 offer_sent（价格匹配容忍），匹配率接近 100%，否则报错
* [ ] END/timeout 场景下不系统性高估 accept（按 outcome 分桶做校准检查）
* [ ] 样本能覆盖多种对手/多种时间段（不是集中在某一步）
* [ ] 序列特征不会泄漏未来（history 只取当前之前的事件）
* [ ] breach_dataset（若做）能对齐到 partner‑day 或 contract_id
* [ ] train/val/test 按 world_id 或 tournament_id 分割（避免同一世界泄漏）

### D. Logistic baseline（必须全绿）

* [ ] 训练可收敛（logloss 低于常数基线）
* [ ] 有校准评估（至少 ECE 或 reliability 分桶表）
* [ ] 能导出模型并在 agent 推理时加载
* [ ] 推理速度满足 step 时限（OneShot 很紧，logistic 必须是毫秒级）

### E. BOU（必须全绿）

* [ ] 对每个 opponent‑bucket 维护 (α,β)，并在每次观测后更新
* [ ] α+β 随交互次数增长（日志可见）
* [ ] LCB 在样本少时更保守、样本多时收敛（日志可见）
* [ ] prior_strength_s 可配置，且调大/调小能影响早期保守程度（可验证）
* [ ] 仅 responder != me 才允许更新 sign
* [ ] q<=0 不更新
* [ ] 仅在 add_terminal_negative 开启时允许对 outstanding offer 记弱负

### F. 决策器（必须全绿）

* [ ] portfolio 约束生效：在缺量风险高时，会自动增大 buffer 或扩大对手集合
* [ ] 不会因为风险过滤导致“永远买不够/卖不出”（要有 fallback：允许低可靠对手但小单 + buffer）
* [ ] probe 单机制生效：未知对手先小单，LCB 上升后再放大
* [ ] 与 breach 高的对手不会频繁下大单（日志统计可验证）

### G. 端到端性能（阶段性目标）

* [ ] D‑NB v0（logistic+BOU）分数 ≥ 你们当前启发式 baseline
* [ ] 加入 breach 风险后：shortfall/违约相关惩罚下降（从 stats 或日志对比）
* [ ] transformer 替换后：accept 预测指标提升，且端到端分数不回退
* [ ] 调参后：均值提升且 P10 提升（更稳健）

---

## 你提到的“参考 D 方案”在本方案中的落点

你之前列的 D 方案里强调的几点，我已经在这份实施方案里对应落地了：

* **换价时机**：在 decision_meta + price_policy 作为可调超参进入调参环节
* **q 尺寸策略（探测/小单/大单）**：probe 机制 + LCB 阈值放大
* **不确定性进组合优化（portfolio）**：用 LCB 的 (\tilde q) 做硬约束
* **探索‑利用（bandit）**：Thompson/LCB 两种模式可并存（探测用采样，大单用 LCB）
* **不是只过滤对手，而是改变交易结构**：portfolio 目标函数 + 风险惩罚项做到了


