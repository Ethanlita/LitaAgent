from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class LitaOSConfig:
    """LitaAgent-OS 超参数与阈值配置"""

    # 模型工件与配置
    model_root_dir: str = "assets/models"
    accept_model_dir: str = "assets/models/accept"
    breach_model_dir: str = "assets/models/breach"
    agent_config_path: str = "assets/agent_config.json"

    # 时间与分桶
    round_bucket_T: int = 5

    # BOU 相关
    prior_strength_s: float = 8.0
    lcb_delta_accept: float = 0.2
    lcb_delta_fulfill: float = 0.2
    w_end: float = 0.2
    add_terminal_negative: bool = False

    # 概率数值安全
    mu_eps: float = 1e-4

    # probe 机制
    # =========================================================================
    # probe 阶段说明 (2026-01-10 更新):
    # 
    # 目的：在比赛早期收集数据来改进 BOU 估计
    # 
    # 问题：早期模型（Logistic + BOU LCB）无法准确预测接受概率，导致：
    # - 使用 q = remaining / p_eff 会严重超量报价
    # - 例如：target=8, p_eff=0.3 → 发出 total_q=30+
    # 
    # 解决方案：
    # 1. probe 阶段使用"名义剩余"分配，保证 total_q <= target
    # 2. probe 阶段将数量分散给多个 partner，收集更多数据
    # 3. post-probe 阶段回到 q_eff 逻辑（此时 BOU 已有足够数据）
    # =========================================================================
    probe_q: int = 1
    probe_q_min: int = 1  # probe 时每个对手的最小 q
    
    # probe_steps: 静态配置的 probe 天数（已废弃，使用 probe_steps_ratio）
    probe_steps: int = 2
    
    # =========================================================================
    # Probe 阶段配置 (按角色分离)
    # =========================================================================
    # 实际 probe 天数 = max(probe_steps_min, int(n_steps * probe_steps_ratio))
    # 
    # 分离原因 (2026-01-13):
    #   - BUYER 和 SELLER 面临的市场环境不同
    #   - BUYER 可能需要更长的 probe 来探索供应商
    #   - SELLER 可能需要更短的 probe 以快速响应需求
    #   - 允许分别调优两个角色的探索策略
    # =========================================================================
    
    # BUYER probe 配置
    buyer_probe_steps_ratio: float = 0.1  # BUYER probe 阶段占总步数的比例 (10%)
    buyer_probe_steps_min: int = 10  # BUYER 最少 probe 天数
    
    # SELLER probe 配置
    seller_probe_steps_ratio: float = 0.1  # SELLER probe 阶段占总步数的比例 (10%)
    seller_probe_steps_min: int = 10  # SELLER 最少 probe 天数
    
    # 向后兼容的统一配置（如果角色配置未设置，使用这个）
    probe_steps_ratio: float = 0.1  # 统一 probe 比例（向后兼容）
    probe_steps_min: int = 10  # 统一最少天数（向后兼容）
    
    probe_lcb_threshold: float = 0.30  # 降低以避免先验触发 probe
    q_candidate: int = 2

    # 组合优化
    buffer_min: float = 0.05
    buffer_max: float = 0.35
    buffer_exp: float = 2.0
    portfolio_k: int = 6
    risk_lambda: float = 0.0
    overfill_penalty_ratio: float = 0.1

    # =========================================================================
    # 超量采购策略 (Overordering)
    # =========================================================================
    # 灵感来源：RChan (SCML 2025 竞争对手)
    # 
    # 分析结论 (2026-01-10):
    #   - Shortfall penalty 约为 Disposal cost 的 10 倍（货币量纲）
    #   - 因此 BUYER 应容忍超量采购（overbuy），宁可多买也不要 shortfall
    #   - SELLER 应保守，不超卖（oversell 会导致高额 shortfall 惩罚）
    # 
    # RChan 参数:
    #   - overordering_max_selling = 0.0 (卖家不超量)
    #   - overordering_max_buying = 0.2 (买家超量 20%)
    # 
    # 我们先用 10% 做实验，后续可调整
    # =========================================================================
    buyer_overordering_ratio: float = 0.1  # BUYER 超量采购比例 (10%，RChan 用 20%)
    
    # overordering_ensure_plus_one: 确保 overordering 至少 +1
    # 
    # 问题背景 (2026-01-10 Reviewer 反馈):
    #   原实现 target = int(need * (1 + ratio)) 使用 floor 截断
    #   导致小需求时 overordering 无效:
    #     - need=6, ratio=0.1 → 6*1.1=6.6 → int=6 (没有 overorder)
    #     - need=8 → 8.8 → int=8 (还是没有 overorder)
    #     - 只有 need≥10 才有效
    # 
    # 修复: target = need + max(1, ceil(need * ratio))
    # 确保即使小需求也至少 +1
    overordering_ensure_plus_one: bool = True
    
    # =========================================================================
    # Conditional Overordering (2026-01-13 Reviewer P1-1 建议)
    # =========================================================================
    # 问题背景:
    #   无条件的 +1 overordering 在某些市场条件下可能亩损
    #   例如: shortfall_unit < buy_price + disposal_unit 时
    #          超买的成本超过了短缺的惩罚
    # 
    # 解决方案:
    #   只有当 shortfall_unit > buy_price + disposal_unit + margin 时才 +1
    #   否则 target = need (不超买)
    # 
    # 公式: 
    #   if shortfall_unit > buy_price + disposal_unit + margin:
    #       target = need + 1
    #   else:
    #       target = need
    conditional_overordering_enabled: bool = True  # P1-1: 启用条件性 overordering
    conditional_overordering_margin: float = 0.1  # 安全边际 (比例)
    
    # =========================================================================
    # BUYER First Proposal Penalty-Aware 策略
    # =========================================================================
    # 问题背景 (2026-01-10 分析):
    #   - BUYER first_proposal 默认出 p_min (最低价)
    #   - 但 SELLER 同时收到多个 BUYER 的报价
    #   - SELLER 会优先接受高价 BUYER 的报价
    #   - 结果: LOS 作为 BUYER 只有 46% 的时间能收到 SELLER 回复
    # 
    # 解决方案: 如果 shortfall_penalty 明显大于 disposal_cost:
    #   - BUYER 在 first_proposal 时直接报 p_max 或更高价格
    #   - 这样更有竞争力，更可能获得 SELLER 回复
    # 
    # 触发条件: shortfall_unit / disposal_unit > buyer_fp_penalty_aware_threshold
    buyer_fp_penalty_aware_enabled: bool = True
    buyer_fp_penalty_aware_threshold: float = 2.0  # shortfall/disposal > 2 时触发

    # =========================================================================
    # Post-Probe Q 限制 (防止小单画像外推大单)
    # =========================================================================
    # 问题背景 (2026-01-10 Reviewer 反馈):
    #   - probe 阶段主要打 q=1~2 的小单
    #   - post-probe 用小单的 p_eff 外推大单 (如 q=10)
    #   - 但很多对手"小单愿接，大单完全不接"
    #   - 结果: 给对手报了 q=10 但对方完全不接
    # 
    # 解决方案 (2026-01-13 Reviewer P0-1 更新):
    #   1. post-probe 从 "1/p放量" 改为 "名义量控制+小颗粒分配"
    #   2. 使用 remaining_nominal -= q 而非 remaining_eff -= q * p_eff
    #   3. 限制 q 到 {1, 2, 3} (post_probe_max_q)
    #   4. p_eff 只用于 partner 排序，不用于数量放大
    post_probe_max_q_delta: int = 4  # 已废弃，保留为向后兼容
    post_probe_min_partners: int = 3  # 至少给前 M 个 partner 发单 (多样性约束)
    
    # post_probe_max_q: post-probe 每个 partner 最大发送 q
    # 
    # 设计决策 (2026-01-13 Reviewer P0-1):
    #   - probe 阶段主要打 q=1~2 的小单
    #   - post-probe 不应该外推大单 (对方可能小单愿接、大单不接)
    #   - 限制 q ∈ {1, 2, 3}，避免用小单画像外推大单
    #   - 配合名义量控制，确保 sum(q) ≈ target
    post_probe_max_q: int = 3  # post-probe 每个 partner 最大 q
    
    # post_probe_use_nominal_allocation: 是否使用名义量分配 (新逻辑)
    # True: remaining_nominal -= q (每次扣减名义 q)
    # False: remaining_eff -= q * p_eff (老逻辑，用期望值放量)
    post_probe_use_nominal_allocation: bool = True  # P0-1: 使用名义量控制
    
    # post_probe_dynamic_min_partners: 是否动态计算 min_partners
    # True: min_partners = min(n_partners, max(4, ceil(target/2)))
    # False: 使用静态的 post_probe_min_partners
    post_probe_dynamic_min_partners: bool = True  # P1-2: 动态计算 min_partners

    # =========================================================================
    # BUYER 接单硬上限 (2026-01-11 Reviewer 建议)
    # =========================================================================
    # 问题背景:
    #   _select_subset 的效用函数对超过 need 的部分仍计入正收益
    #   导致 BUYER 会接受对手给的大单，最终过量签约（如 need=8 却签 42）
    # 
    # 解决方案:
    #   1. 硬上限约束: 任何候选子集的 total_q > cap 时直接跳过
    #   2. cap = ceil(need * cap_mult) + cap_abs
    # 
    # 示例 (cap_mult=1.3, cap_abs=1):
    #   need=8 → cap = ceil(8*1.3) + 1 = 11
    #   need=6 → cap = ceil(6*1.3) + 1 = 9
    buyer_accept_cap_mult: float = 1.3  # BUYER 接单上限乘数
    buyer_accept_cap_abs: int = 1  # BUYER 接单上限加数
    
    # =========================================================================
    # BUYER Score 计算方式 (2026-01-11 Reviewer P0 建议)
    # =========================================================================
    # 问题背景:
    #   原 buyer_marginal_utility_fix 只对超买部分的收入抹零，但成本也被抹掉了
    #   导致"超买像免费保险"，BUYER 倾向于 overfill
    #
    # 解决方案: 超量成本计入
    #   - 有用部分：利润 = (exo_out_price - buy_price) × q_useful
    #   - 超量部分：只有成本 = -buy_price × q_excess
    #   - 总 utility = Σ[profit × q_useful - buy_price × q_excess]
    #
    # 这样超买的成本会被正确计入，而非被忽略
    buyer_score_rcp: bool = True  # 使用修正后的 Score 公式（计入超量成本）
    buyer_marginal_utility_fix: bool = True  # 已废弃，保留为向后兼容

    # =========================================================================
    # SELLER Score 计算方式 (2026-01-12 Reviewer 建议)
    # =========================================================================
    # 问题背景:
    #   SELLER 有固定的外生采购成本 (exo_input_price × exo_input_qty)
    #   如果 SELLER 卖不够 (underfill)，没卖出去的部分不仅要支付 disposal cost
    #   还要承担已经支付的采购成本 (这部分原先没有计入)
    #
    # 解决方案: 对 SELLER 的 underfill 部分扣除采购成本
    #   - 有用部分: 利润 = (sell_price - exo_input_price) × q_useful
    #   - 卖不够部分: 损失 = -exo_input_price × q_underfill (已采购但卖不掉)
    seller_score_rcp: bool = True  # SELLER 计入 underfill 采购成本

    # =========================================================================
    # BUYER 发出 offer 总量预算 (2026-01-11 Reviewer P1 建议)
    # =========================================================================
    # 问题背景:
    #   post-probe 阶段按 q_eff (期望) 去凑 target
    #   如果概率低估，会给很多人都发到 q_max_allowed
    #   比如需求 8，却给 10 个 partner 都发 q=6，对方如果都接受就签 60
    # 
    # 解决方案:
    #   对 BUYER 发出的 offer 总量加硬约束
    #   offer_budget = ceil(need * offer_budget_mult) + offer_budget_abs
    #   发出的 offer 总 q 超过 budget 后，后面的 partner 直接 None
    # 
    # 2026-01-13 Reviewer P0-2: 进一步收紧
    #   旧公式: budget = ceil(need*1.2) + 2 → 允许 overfill 最多 +4
    #   新公式: budget = target_nominal + 1 → 允许 overfill 最多 +1
    #   使用新配置项 buyer_offer_budget_use_target_plus_one
    buyer_offer_budget_enabled: bool = True
    buyer_offer_budget_mult: float = 1.2  # 已废弃 (如果 use_target_plus_one=True)
    buyer_offer_budget_abs: int = 2  # 已废弃 (如果 use_target_plus_one=True)
    
    # buyer_offer_budget_use_target_plus_one: 使用更严格的 budget 公式
    # True: budget = target_nominal + 1 (最多超 1 单)
    # False: budget = ceil(need * mult) + abs (老公式)
    buyer_offer_budget_use_target_plus_one: bool = True  # P0-2: 收紧 budget

    # =========================================================================
    # Breach 概率禁用 (2026-01-12 Reviewer P0 建议)
    # =========================================================================
    # 问题背景:
    #   OneShot 中 breach 概率来自 FinancialReport，但实际很少有有用信息
    #   使用 p_eff = lcb_sign × fulfill 会让 fulfill < 1 时放大 offer 量
    #   但 OneShot 的 breach 机制与 Standard 不同，这个折扣可能没必要
    #
    # 解决方案:
    #   禁用 breach 概率折扣，fulfill = 1 always
    #   这样 p_eff = lcb_sign，只反映对方接受概率
    disable_breach_probability: bool = True  # 禁用 breach 概率，fulfill = 1

    # =========================================================================
    # 价格策略
    # =========================================================================
    use_trading_price: bool = True
    
    # price_concession_gamma: 价格让步速度参数
    # 让步公式: concession = round_rel^gamma
    # 
    # 设计决策 (2026-01-10):
    # 原值 gamma=2.0 在实际比赛中几乎无效，因为：
    # - 分析 2024 OneShot 冠军日志发现 99.4% 的谈判在 round 1 结束
    # - 平均 round_rel ≈ 0.03（即谈判进度仅 3%）
    # - gamma=2.0 时: concession = 0.03^2 = 0.0009（几乎为零）
    # - gamma=0.5 时: concession = 0.03^0.5 ≈ 0.17（有意义的让步）
    # 
    # OneShot 谈判极短（通常 1-2 轮），需要激进让步策略
    price_concession_gamma: float = 0.5
    
    counter_anchor_eta_exp: float = 1.0  # eta = round_rel^k，用于 counter 价格锚定
    counter_monotonic: bool = True  # 是否强制单调让步

    # =========================================================================
    # Panic 模式
    # =========================================================================
    panic_enabled: bool = True
    panic_penalty_ratio_threshold: float = 3.0  # R > 此值触发 panic
    
    # panic_round_rel_threshold: 进入 panic 模式的谈判进度阈值
    # 
    # 设计决策 (2026-01-10):
    # 原值 0.6 在 OneShot 中永远不会触发，因为：
    # - 99.4% 的谈判在 round_rel ≈ 0.03 时就已结束
    # - 只有 0.6% 的谈判能到达 round_rel > 0.1
    # - 阈值 0.6 意味着 panic 模式形同虚设
    # 
    # 新值 0.1 的含义：
    # - 在极少数"长谈判"（约 0.6% 的情况）中启用 panic
    # - 这些长谈判往往是因为双方僵持不下
    # - 此时应采取更激进策略避免 shortfall/disposal 惩罚
    panic_round_rel_threshold: float = 0.1

    # =========================================================================
    # 组合优化 (Utility 计算)
    # =========================================================================
    # use_exo_price_for_utility: 是否使用外生价格计算真实利润
    # 
    # 设计决策 (2026-01-10):
    # 原实现使用 trading_price（市场平均价）计算边际收益，导致：
    # - 卖家以 16 卖出，trading_price=17 → 边际收益 = -1（负值）
    # - 但实际采购成本 exo_input=10 → 真实利润 = 16-10 = 6（正值）
    # - 导致 Agent 错误拒绝了有利可图的交易
    # 
    # OneShot 外生价格结构：
    # - Level 0 (卖家): 有外生输入合同，价格 = current_exogenous_input_price
    # - Level 1 (买家): 有外生输出合同，价格 = current_exogenous_output_price
    # 
    # 真实利润计算：
    # - 卖家: profit = sell_price - exo_input_price (卖价 - 采购成本)
    # - 买家: profit = exo_output_price - buy_price (销售收入 - 进货价)
    use_exo_price_for_utility: bool = True

    # 训练模型默认强度
    default_accept_strength: float = 8.0

    def load_overrides(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return
        self.apply_overrides(data)

    def apply_overrides(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if not hasattr(self, key):
                continue
            current = getattr(self, key)
            try:
                if isinstance(current, bool):
                    setattr(self, key, bool(value))
                elif isinstance(current, int):
                    setattr(self, key, int(value))
                elif isinstance(current, float):
                    setattr(self, key, float(value))
                elif isinstance(current, str):
                    setattr(self, key, str(value))
                else:
                    setattr(self, key, value)
            except Exception:
                continue
