"""模拟 LOS 的 p_eff 计算"""
from litaagent_os.config import LitaOSConfig
from litaagent_os.models import LinearLogisticAcceptModel
from litaagent_os.features import build_context_features, build_offer_features
from litaagent_os.bou import BOUEstimator
from litaagent_os.partner_stats import PartnerStatsStore

cfg = LitaOSConfig()
accept_model = LinearLogisticAcceptModel(cfg)
bou = BOUEstimator(cfg)
partner_stats = PartnerStatsStore(k=32)

# 模拟 first_proposal 场景
# Level 0 卖家，round_rel = 0.0（first proposal）
role = "SELLER"
round_rel = 0.0
p_min, p_max = 16.0, 17.0
q_max = 10
need_remaining = 8  # 假设外生输入是 8
trading_price = 16.5
shortfall_penalty = 0.2
disposal_cost = 0.1

# 使用默认权重（没有加载模型）
print("=== Accept Model 预测 ===")

# 测试不同的 offer
for q in [1, 5, 8, 10]:
    for p in [16.0, 16.5, 17.0]:
        stats = partner_stats.get("test_partner", role).snapshot()
        context = build_context_features(
            cfg,
            role,
            round_rel,
            p_min,
            p_max,
            need_remaining,
            q_max,
            trading_price,
            shortfall_penalty,
            disposal_cost,
            None,  # system_breach_prob
            None,  # system_breach_level
            stats,
        )
        offer = build_offer_features((q, 0, p), q_max, p_min, p_max)
        
        mu = accept_model.predict_accept_mu(context, offer, [])
        strength = accept_model.predict_accept_strength(context, offer, [])
        
        # BOU 的 LCB（没有数据时返回 None）
        lcb = bou.lcb(
            pid="test_partner",
            role=role,
            round_bucket=context.round_bucket,
            p_bin=offer.p_bin,
            q=offer.q,
            mu=mu,
            strength=strength,
            delta=cfg.lcb_delta_accept,
        )
        
        p_eff = (lcb if lcb is not None else mu) * 1.0  # fulfill = 1.0
        
        print(f"  Q={q}, P={p}: mu={mu:.3f}, strength={strength:.3f}, lcb={lcb}, p_eff={p_eff:.3f}")

print("\n=== 模拟 first_proposal 分配 ===")
# 模拟原始代码的分配逻辑
target = int(need_remaining * (1.0 + 0.05))  # buffer = 0.05 at round_rel=0
print(f"target = {target} (need_remaining={need_remaining} * 1.05)")

# 假设有 5 个 partner，每个的 p_eff 都是 ~0.3
n_partners = 5
p_eff_cand = 0.3
remaining_eff = float(target)

print(f"\n原始逻辑 (q = remaining_eff / p_eff):")
total_q = 0
for i in range(n_partners):
    if remaining_eff <= 0:
        q = 0
    else:
        q = min(q_max, max(1, int(round(remaining_eff / max(p_eff_cand, 1e-6)))))
    print(f"  Partner {i}: remaining_eff={remaining_eff:.1f}, q={q}")
    remaining_eff -= q * p_eff_cand
    total_q += q

print(f"Total Q = {total_q}")

print(f"\n修复后逻辑 v2 (按 partner 分配):")
import math
remaining_nominal = float(target)
total_q = 0
# 按 partner 均分，但根据 p_eff 排序后优先分配
q_per_partner = max(1, math.ceil(target / n_partners))
for i in range(n_partners):
    if remaining_nominal <= 0:
        q = 0
    else:
        q = min(q_max, max(1, min(q_per_partner, int(round(remaining_nominal)))))
    print(f"  Partner {i}: remaining_nominal={remaining_nominal:.1f}, q={q}")
    remaining_nominal -= q
    total_q += q

print(f"Total Q = {total_q}")
