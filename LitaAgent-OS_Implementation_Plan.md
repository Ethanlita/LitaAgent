最后更新：2026-01-09 00:30
# LitaAgent-OS 详细实施方案

本方案基于《LitaAgent-OS设计》文档，将 D-NB (Neural-Bayesian Opponent Model + Heuristic Planner) 的设计转化为可执行的工程实施步骤。

## 口径与前提（统一接口与标签定义）

* Accept 定义为动作层面的接受（无论最终是否执行），breach 作为独立标签处理。
* Response 口径：ACCEPT=接受；REJECT=拒绝并提出新报价（可带 counter_offer）；END=结束谈判。END 不计入 accept 监督训练与 BOU 更新。
* offer 不是动作，仅作为事件/字段出现；决策动作仍是 ACCEPT/REJECT/END。
* 历史序列 token 的 `action_type` 采用 {OFFER, ACCEPT, END}；REJECT 仅出现在 `response_type` 中，counter_offer 标记 `is_counter=True`（首轮用 `is_first_proposal`）。
* 系统日志可覆盖 OM 训练最低字段，Tracker 仅用于内部特征；因此 LitaAgent-OS 专有字段允许缺失，最低线是能支持 OM/Logistic 监督学习的通用字段。
* breach 数据集+模型为设计要求；若系统提供 breach_prob/level，则模型可用 placeholder（跳过训练，直接走系统）。
* 接受概率与违约概率使用两套模型：AcceptModelInterface（必需）+ BreachModelInterface（可选，系统可用时占位）。

## Transformer 字段分析与统一 Schema（Trainer/WorldLogs，Tracker 可选）
**目标**：对齐设计中的 Transformer 输入要求，明确 World Logs 重建字段与（可选）Tracker 字段，降低对齐成本。

**设计中的 Transformer 输入要素**（来自《LitaAgent-OS设计》）：
* Context token：时间、need、role、penalties、price bounds、system breach 等。
* 最近 N 条事件 token（建议 N=16 或 32）：speaker（me/opp）、action_type（offer/accept/end）、is_counter/is_first_proposal、q_bucket、p_bin、round_bucket。
* 输出：`mu_accept`（接受概率模型）；违约概率由独立 BreachModel 或系统提供。

**时间与离散化定义（与设计一致）**：
* `round_rel`（连续，0..1）：若 state 提供 `relative_time`，则 `round_rel = state.relative_time`；否则 `round_rel = state.step / max(1, state.n_steps - 1)`。
* `round_bucket`（离散桶）：`round_bucket = min(T-1, floor(round_rel * T))`，默认 `T=5`。
* `p_bin`：在 `[p_min, p_max]` 内更接近下界还是上界；OneShot 常见 `p_max = p_min + 1` 时，`p_bin=0` 表示 `p==p_min`，`p_bin=1` 表示 `p==p_max`。
* `q_max = nmi.issues[QUANTITY].max_value`（或 `awi.n_lines`），`q_bucket = clip(int(q), 0, q_max)`，取值范围 `0..q_max`。

**特征可观测性矩阵（AcceptModel 输入约束）**
* Online 可得：`round_rel/round_bucket`、`price_min/price_max`、`role`、history（双方交互）、我方 `need_remaining`/`pen_shortfall`/`cost_disposal`/`trading_price`、offer(`q`,`p`)、`partner_stats_*`
* Online 可能可得（需 CapabilityProbe）：`system_breach_prob/level`（若系统提供，需 mask）
* Online 不可得：对手 `need_remaining`、对手 penalties（shortfall/disposal）
* 规则：AcceptModel 只吃 Online 可得 + 可能可得（带 mask）的特征

### Schema v0-A：Accept 数据集（事件序列 + Trainer 样本）
**口径要求**：Accept 数据集采用 offer-centric（样本=offer_sent），DatasetBuilder/Trainer/Schema 必须保持一致并闭环。
#### v0-A1：事件序列（system logs 重建 / Tracker 可选）
**用途**：构建事件序列 + 训练样本对齐键。事件可由 system logs 重建；若启用 Tracker，以下为最低线字段（未列出者可缺省）。

**通用字段（所有事件）**：
* `schema_version` (str, e.g. "v0")
* `event_type` ∈ {`step_snapshot`, `proposal_sent`, `offer_received`, `response_sent`, `contract_signed`, `negotiation_failed`, `contract_executed`, `contract_breached`}
* `world_id` (str), `agent_id` (str), `partner_id` (str)
* `step` (int), `n_steps` (int), `n_lines` (int), `role` ∈ {BUYER, SELLER}
* `negotiation_id` (str) 或 `mechanism_id` (str) 至少一个

**上下文快照字段（建议随 `step_snapshot` 或每个事件写入）**：
* `round_rel` (float), `round_bucket` (int)
* `need_remaining` (int/float)
* `trading_price` (float)
* `pen_shortfall` (float), `cost_disposal` (float)
* `price_min` (float), `price_max` (float)
* `system_breach_prob` (float, optional), `system_breach_level` (float, optional)

**Offer 事件字段（proposal/offer/response）**：
* `offer` = `{q, p, t}`（用于 `proposal_sent`/`offer_received`；t 从日志读取，OneShot 常为常数但不要硬编码；建模可忽略/遮罩，用于 agreement 匹配）
* `offer_sent`（proposal_sent/counter_offer）：Accept 数据集样本来源
* `responded_offer` = `{q, p, t}`（仅 `response_sent`，用于标注对手是否接受我方 offer）
* `counter_offer`（仅 REJECT 时）
* `round_rel` (float), `round_bucket` (int)
* `response_type` ∈ {ACCEPT, REJECT, END}（仅 `response_sent`）
* `event_index` (int, optional；每个 negotiation 单调递增，可由日志顺序派生)

**系统日志重建注意事项（与设计 2.1.2/2.1.3 对齐）**
* `world_id` 优先 `negs.csv.world`，缺失时可用日志目录名或 `events.json.sender` 兜底。
* `negotiation_id` 需按 `actions.csv.neg_id -> negs.csv.id -> negs.csv.name -> negotiations.csv.id -> negotiations/<id>.csv` 对齐。
* `round_rel` 优先 `negotiations/<id>.csv.relative_time`，缺失时回退 `actions.csv.relative_time`。
* `price_min/price_max/q_max` 从 `negotiations.csv.issues` 解析（quantity/unit_price）；`n_lines` 若有 `info.json.active_lines` 优先使用，否则近似为 `q_max`。
* `actions.csv` 可能没有 agreement，ACCEPT 需以 `negotiations.csv.agreement` 或 `negs.csv.has_agreement` 为准。
* Accept 数据集样本口径为 `offer_sent`（offer-centric；proposal_sent/counter_offer），标签由 agreement/response 对齐得到。
        *   END/timeout 默认不更新；若启用 `add_terminal_negative`，可对最后 outstanding offer 做弱负更新。
* AcceptModel 输入仅使用在线可观测特征（对手私有量不得进入模型输入）。
* `negotiations/<id>.csv` 的 offer 顺序是 `(q, time, p)`，读取后需映射为 `(q, p, time)`。
* `new_offers` 为空时，用 `current_offer/current_proposer` 的变化补充 offer 序列。
* `proposer_id/partner_id` 由重建的 offer/proposer 序列推导。
* `responded_offer`：REJECT 用上一条对手 offer；ACCEPT 用 `negotiations.csv.agreement` 或最后一条 offer；END 不生成样本。
* `role` 用 `negotiations.csv.buyer/seller` 判定（`is_buy` 只是 caller 角色）。
* `need_remaining` 用 `negs.csv.agent_time0/agent_time1` 名称匹配 `needed_*`，并按 proposer 的 role 选择 sales/supplies。
* `trading_price` 优先 `negs.csv.trading_price`；若用 `stats.csv` 或 `stats.csv.csv`，需按 product 选 `trading_price_<product>`。
* `shortfall_penalty_<agent>` / `disposal_cost_<agent>` 的 `<agent>` 为 agent 名称（如 `00Gr@0`），按 `sim_step` 对齐。
* `step`（谈判回合）用 `negotiations/<id>.csv.step`，仿真步用 `negotiations.csv.sim_step` 或 `negs.csv.sim_step`。
* `n_steps` 用 `params.json.n_steps`（仿真步），谈判步上限用 `negs.csv.n_steps`。
* `END` 判定包含 `ended`（不仅是 `failed/timedout/broken`）。
**合同事件字段**：
* `contract_id` (str)
* `agreement` = `{q, p, t}`（`contract_signed`）
* `breach_level` / `breach_list`（`contract_breached` 可用时记录）

#### v0-A2：Trainer 样本（accept_dataset.parquet/jsonl）
**样本单位**：一次 offer_sent（proposal_sent / counter_offer），预测对手是否接受该 offer。

**键字段（对齐用，必需）**：
* `world_id`, `negotiation_id`（或 `mechanism_id`）, `step`, `proposer_id`, `partner_id`
* `event_index`（可选；若无，可由日志顺序派生）

**标签**：
* `y_accept`=1：对手接受该 offer（agreement 匹配；p 需浮点容忍：round(p,k) 或 |p1-p2|<eps）
* `y_accept`=0：对手显式拒绝并给出 counter
        *   END/timeout 默认不更新；若启用 `add_terminal_negative`，可对最后 outstanding offer 做弱负更新。

**Context（在线可观测，Logistic 直接使用）**：
* `round_rel`, `round_bucket`, `role`
* `need_remaining`, `need_norm`（我方）
* `trading_price`, `pen_shortfall_norm`, `cost_disposal_norm`（我方）
* `price_min`, `price_max`
* `system_breach_prob` (optional), `system_breach_level` (optional, maskable)
* `partner_stats_*` (optional; PARTNER scope 统计)

**Offer（原始数值）**：
* 来自 `offer_sent` 的 `q`, `p`, `q_norm`, `q_bucket`, `p_bin`（或 `p_norm`）

**History Tokens（Transformer 直接使用）**：
* `history_tokens`: 长度 N 的 list（左侧 padding），每个 token：
  * `speaker` ∈ {ME, OPP}（以 proposer_id 视角重标）
  * `action_type` ∈ {OFFER, ACCEPT, END}
  * `is_counter` (bool, optional), `is_first_proposal` (bool, optional)
  * `q_bucket`, `p_bin`, `round_bucket`
* `history_len` (int), `history_scope` 固定为 NEGOTIATION
* OFFER 口径：所有报价统一编码为 OFFER；counter_offer 记 `is_counter=True`，首轮 offer 记 `is_first_proposal=True`。
* REJECT 仅存在于 `response_type`（response_sent），不作为 history 的 `action_type`。
* `partner_id` 不作为模型输入 token，仅用于索引 BOU/统计状态
### Schema v0-B：Breach 数据集（可选）
**仅在系统不提供 breach_prob/level 时启用**：
* 口径：`contract` 或 `partner-day`（二选一）
* 关键字段：`world_id`, `partner_id`, `contract_id`/`day_id`, `context`, `y_breach` 或 `breach_level`

## 阶段一：环境验证与日志接口确认
**目标**：确保 NegMAS/SCML OneShot 环境产生的日志包含所有必要标签（Accept/Breach），并验证现有解析工具的兼容性。

1.  **运行最小化 OneShot锦标赛**
    *   **操作**：编写脚本运行一个极小规模的 OneShot Tournament（例如：1次运行，5步，只有标准 Agent）。
    *   **验证点**：检查生成的 World 目录（通常在 `tournament_results/world_logs` 下）。
    *   **检查清单**：
        *   `actions.csv` 是否存在？能否与 `negotiations.csv`/`negotiations/*.csv` 对齐以重建 ACCEPT/REJECT/END？
        *   备注：在建模/日志对齐时，报价事件统一映射为 OFFER；`agreement` 视为 ACCEPT。REJECT 仅在 `response_type` 中出现。
        *   `contracts.csv` 是否存在？是否有 `executed`, `breached` 状态标记？
        *   `breaches.csv` 是否存在？（如果不存在，确认 `contracts.csv` 中是否包含违约信息）。
        *   `negotiations.csv` 是否存在？
        *   统计文件名是否为 `stats.csv` 或 `stats.csv.csv`？记录实际文件名，DatasetBuilder 需兼容两者。

2.  **验证 `scml_analyzer` 解析能力**
    *   **操作**：使用现有的 `scml_analyzer.log_parser.LogParser` 解析上述 World 目录。
    *   **验证点**：
        *   `actions_df`, `contracts_df` 是否非空？
        *   确保能从 DataFrame 中提取出“谁在第几步对谁发了什么 Offer”以及“该 Offer 是否被对方接受（accept 动作）”。

3.  **系统 Breach 信息探测**
    *   **操作**：创建一个简单的探测 Agent，在 `init()` 中检查 `self.awi` 对象。
    *   **验证点**：
        *   `awi.reports_at_step(step)` 是否可用？
        *   `awi.reports` / `awi.reports_of_agent` / `awi.reports_for` 是否可用？
        *   `awi.bulletin_board` / `awi.board` / `awi.bb` 是否可用？
        *   系统是否提供 `breach_prob` / `breach_level` / `breach_list`？

## 阶段二：Agent 框架与（可选）OneShot Tracker 实现
**目标**：搭建 Agent 骨架，并实现专门的 Tracker 以记录训练模型所需的“上下文特征”和“决策时状态”。

1.  **建立代码结构**
    *   创建 `litaagent_os` 包。
    *   创建 `LitaAgentOS` 类，继承自 `OneShotSyncAgent`。

2.  **实现 `CapabilityProbe` 类**
    *   **功能**：在 Agent 初始化时运行，探测系统提供的 breach 信息接口。
    *   **逻辑**：按优先级检查 `reports_at_step` → `reports/reports_of_agent/reports_for` → `bulletin_board/board/bb`，记录可用字段（breach_prob/breach_level/breach_list）供后续 `BreachInfoProvider` 使用。

3.  **实现 `OneShotTrackerMixin`（可选）**
    *   TrackerMixin 为可选：用于记录官方日志没有的内部决策变量（LCB/先验/策略分支等），不是训练数据的必要依赖。
    *   **文件**：`scml_analyzer/oneshot_tracker_mixin.py`
    *   **最低线（可用于 OM/Logistic 监督训练）**：
        *   `before_step()`: 记录 `world_id`, `step`, `round_rel`, `round_bucket`, `role`, `n_lines`, `trading_price`, `penalties`。
        *   `first_proposals()`: 对每个对手记录 `proposal_sent`。
            *   字段：`negotiation_id`（或 `mechanism_id`）、`partner_id`, `offer(q,p,time)`, `round_rel`, `round_bucket`, `is_first_proposal`, `role`, `need_remaining`, `price_bounds`。
        *   `counter_all(offers, states)`:
            *   对每个收到的 offer 记录 `offer_received`（包含 `negotiation_id`, `round_rel`, `round_bucket`）。
            *   对每个返回的 response 记录 `response_sent`，`response_type` ∈ {ACCEPT, REJECT, END}。
            *   若为 REJECT：写入 `counter_offer`（即“拒绝并提出新报价”的新 offer）。
        *   `on_negotiation_success()`: 记录 `contract_signed`，包含 `contract_id`, `partner_id`, `agreement(q,p,time)`；若缺少 `response_sent` 的 ACCEPT，可用该事件回填 ACCEPT（`responded_offer` 取 agreement 或最后一条对手报价）。
        *   `on_negotiation_failure()`: 记录 `negotiation_failed`（END/超时），不作为 reject 训练样本。
        *   `on_contract_executed` / `on_contract_breached`（若可用）: 记录执行/违约事件。
    *   **可选扩展（LitaAgent-OS 专有，可为空）**：
        *   `model_prior_accept_mu`, `p_accept_prior/post/LCB`, `p_breach`, `p_eff`, `beta_state`, `decision_meta` 等。
    *   **数据补全逻辑**：
        *   若 `on_contract_breached` 未触发，需要在每日结束时从系统报告中读取 breach 信息并写入 Tracker 日志。
        *   Tracker 必须允许上述可选字段缺失，DatasetBuilder 要能容忍空值。

4.  **集成测试**
    *   将 `OneShotTrackerMixin` 混入 `LitaAgentOS`。
    *   运行一场比赛，检查输出的 `tracker.json` / `logs` 是否包含最低线字段，可选字段允许为空。

## 阶段三：Logistic 模型基线与数据管线
**目标**：跑通“数据收集 -> 模型训练 -> 在线推理”的完整闭环。

1.  **构建 Accept 数据集管线**
    *   **工具**：编写 `DatasetBuilder` 脚本。
        *   END/timeout 默认不更新；若启用 `add_terminal_negative`，可对最后 outstanding offer 做弱负更新。
        *   落地规则：监督训练中 terminal negative 的 `sample_weight = w_end`（默认 0.2）。
    *   **输入**：优先使用 World Logs（Tracker Logs 仅在需要内部特征时使用）。
    *   **对齐逻辑**：以 `offer_sent` 为样本主表，通过 `world_id`, `negotiation_id`（或 `mechanism_id`）, `step`, `proposer_id`, `partner_id` 对齐上下文与 history；必要时用 `event_index`（日志顺序派生）；label 由 `response_sent`/`contract_signed` 对齐。
    *   **口径约束**：`response_sent`/`contract_signed` 只用于打标签（y_accept），不作为样本单位。
    *   **对齐校验**：agreement 能回溯到某条 offer_sent（考虑浮点容忍），匹配率应接近 100%，否则直接报错。
    *   **输出**：`accept_dataset.csv` / `.parquet`。
    *   **特征工程**：
        *   `q_norm`, `p_bin` (Unit Price vs Limits)
        *   `round_rel`（谈判进度 ∈ [0,1]；来自 `relative_time`，缺失时用 `step/(n_steps-1)` 近似）
        *   `role` (Buyer/Seller)
        *   `need_urgency` (Need / TimeLeft; TimeLeft 由 round_rel 推导)
        *   `opp_behavior_stats` (如果有)
        *   `history_tokens`（NEGOTIATION scope 最近 N 条；token 含 speaker/action_type/q_bucket/p_bin/round_bucket；action_type ∈ {OFFER, ACCEPT, END}）
        *   `partner_stats_*`（PARTNER scope 统计，供 accept/可靠性特征使用；必须按事件顺序生成 prefix stats，写样本后再更新统计，避免泄漏）
    *   **数据切分**：按 `world_id`/`tournament_id` 切分 train/val/test，避免同一世界泄漏。

2.  **构建 Breach 数据集（条件）**
    *   **系统可用**：若系统提供 `breach_prob/level`，构建 placeholder，不训练 breach 模型。
    *   **系统不可用**：从 `contracts.csv`/`breaches.csv` 构建 breach 数据集（partner-day 或 contract 口径均可），输出 `breach_dataset.parquet`。

3.  **训练 Logistic Regression 模型**
    *   **目标**：预测 $P(\text{accept} | \text{context}, \text{offer})$。
    *   **指标**：LogLoss (BCE), Calibration Curve。
    *   **处理**：必须处理类别不平衡 (Class Imbalance)。
    *   **Breach 模型（可选）**：若系统不可用，可训练简化 breach 模型或使用 Beta 先验作为 `predict_breach_mu` 的来源。

4.  **实现两套模型接口**
    *   **AcceptModelInterface**（必需，Logistic/Transformer 互换）：
        *   `predict_accept_mu(context, offer, history_neg) -> float (0~1)`
        *   `predict_accept_strength(context, offer, history_neg) -> float (s >= 2)`
    *   **BreachModelInterface**（可选，系统可用时占位）：
        *   `predict_breach_mu(context, partner_features, history_partner) -> Optional[float]`

5.  **Agent 接入推理**
    *   在 Agent 中加载 accept 模型；breach 由系统提供或加载 breach 模型（可选）。
    *   在 `first_proposals` 和 `counter_all` 中调用 accept 模型；breach 通过 `BreachInfoProvider` 统一获取。
    *   Tracker 仅在可用时记录预测值（其他 Agent 可为空）。

## 阶段四：贝叶斯在线更新 (BOU) 模块
**目标**：利用在线交互数据修正模型偏差，计算 LCB (下置信界)。

1.  **实现 `BOUEstimator` 类**
    *   **状态维护**：为每个 `key = (pid, role, round_bucket, p_bin, q_bucket_coarse)` 维护 Beta 分布参数 $\alpha, \beta$。
    *   **分桶口径**：包含 `round_bucket`, `p_bin`, `q_bucket_coarse`（用于稳定接受行为的统计）。
    *   **q_bucket_coarse**：`q=1`、`q=2`、`q in [3,4]`、`q in [5,7]`、`q>=8`；`q<=0` 不更新。
    *   **初始化逻辑**：
        *   **数值安全**：$\mu_{\text{model}} \leftarrow \text{clip}(\mu_{\text{model}}, \epsilon, 1-\epsilon)$（建议 $\epsilon=1e^{-4}$ 或 $1e^{-3}$）
        *   $\alpha_0 = \mu_{\text{model}} \cdot s$
        *   $\beta_0 = (1 - \mu_{\text{model}}) \cdot s$
    *   **更新逻辑**：
        *   **更新触发方向**：仅在 pid 作为 responder 回应我方 offer 时更新；我方作为 responder 的事件不更新。
        *   观测到 Accept: $\alpha \leftarrow \alpha + 1$
        *   观测到 Reject: $\beta \leftarrow \beta + 1$
        *   END/timeout 默认不更新；若启用 `add_terminal_negative`，可对最后 outstanding offer 做弱负更新。
        *   弱负更新：$\beta \leftarrow \beta + w_{\text{end}}$（允许非整数 Beta 参数，默认 $w_{\text{end}}=0.2$）。
        *   **在线校验（断言/日志）**：responder != me 才允许更新；q<=0 不更新；仅在 `add_terminal_negative` 开启时允许对 outstanding offer 记弱负。
    *   **推理逻辑 (LCB)**：
        *   计算 $P_{\text{sign}} = \text{BetaQuantile}(\alpha, \beta, \delta=0.2)$

2.  **集成入 Agent**
    *   Agent 决策不再直接使用模型输出，而是使用 `BOUEstimator.get_lcb()`。
    *   确保 Tracker 记录此时的 LCB 值。

## 阶段五：启发式决策与组合优化
**目标**：基于 LCB 概率进行风险规避的决策。

1.  **实现 `BreachInfoProvider`**
    *   逻辑：若系统提供 `breach_prob/level/list` 则直接使用；否则使用本地 breach 模型或在线 Beta 维护 breach 记录。
    *   计算有效概率：$P^{\text{eff}}(o) = \text{LCB}(P^{\text{sign}}) \times \text{LCB}(P^{\text{fulfill}})$。
    *   $P^{\text{fulfill}}$ 定义（写死口径，避免分叉）：
        *   系统给 breach_prob：$P^{\text{fulfill}} = 1 - \text{breach_prob}$。
        *   系统给 breach_level（0~1）：$P^{\text{fulfill}} = 1 - \text{breach_level}$。
        *   本地 breach 模型：接口统一输出 $P^{\text{fulfill}}$。

2.  **实现 `first_proposals` 策略 (Portfolio Optimization)**
    *   **数学模型**：
        $$ \max \sum_i (\tilde{q}_i(o_i) \cdot m_i(o_i)) - \lambda \sum_i q_i \cdot (1 - \mathrm{LCB}(P^{\text{fulfill}}_i)) $$
        $$ \text{s.t.} \sum_i \tilde{q}_i(o_i) \ge \text{need} \cdot (1 + \text{buffer}) $$
        *   其中 $m_i(o_i)$ 是边际收益（买方：越便宜越好；卖方：越贵越好），$\tilde{q}_i(o_i) = q_i \cdot \text{LCB}(P^{\text{sign}}_i) \cdot \text{LCB}(P^{\text{fulfill}}_i)$。
        *   若 $\tilde{q}$ 已包含 $P^{\text{eff}}$，第二项视为“残余履约风险惩罚”（不包含签约风险）；默认可设 $\lambda=0$，避免双计。
    *   **实现**：由于是 OneShot，对手数量有限，可用简化的贪心或 Top-K 枚举求解。
    *   **Fallback**：如果无法满足约束（太保守），则降低 Buffer 或 LCB 阈值。

3.  **实现 `counter_all` 策略 (Subset Selection)**
    *   **逻辑**：
        1.  收集所有收到的 Offer。
        2.  生成可能的接受子集 (Subset)。
        3.  对每个子集评分：$\text{Score}(S) = \text{Utility} - \text{ShortfallPenalty} - \text{RiskPenalty}$。
        4.  选择分数最高的子集 Accept，其余 Reject（带 counter_offer）。

4.  **实现 Probe (探测) 机制**
    *   对于 $\alpha + \beta$ 小于阈值 (新对手) 的情况，强制限制 $q$ 为极小值 (1或2)，进行低风险探测。

## 阶段六：Transformer 模型升级 (并行或延后)
**目标**：利用序列历史信息提升预测准确率。

1.  **更新数据集管线**：增加 `sequence_history` 特征（NEGOTIATION scope 过去 N 个事件 token）。
2.  **训练 Transformer**：用于 AcceptModel（替换 Logistic）；BreachModel 仍独立或由系统提供。
3.  **接口兼容**：`AcceptModelInterface` 内部切换实现，Agent 逻辑代码无需改动。

## 阶段七：超参调优
**目标**：确定最佳配置。

1.  **配置化**：将 `LCB_delta_accept`, `LCB_delta_fulfill`, `prior_strength_accept`, `prior_strength_breach`, `buffer_factor`, `risk_lambda`, `probe_q`, `portfolio_K` 等提取到 `agent_config.json`。
2.  **搜索**：运行大规模 Tournament，使用 Grid Search 或 CMA-ES 寻找最优参数组。

---
**后续工作流**：
请确认上述方案是否清晰。如果无异议，我们将从【阶段一：环境验证与日志接口确认】开始执行。
