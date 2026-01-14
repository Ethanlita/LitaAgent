"""分析 LOS 在每个 step 的合同情况，检查超卖/超买和 probe 阶段效果"""
import pandas as pd
from pathlib import Path
import sys

# 找到指定或最新的比赛
if len(sys.argv) > 1:
    tournament_name = sys.argv[1]
    stage_dir = Path(rf"D:\SCML_initial\tournament_history\{tournament_name}\{tournament_name}-stage-0001")
else:
    stage_dir = Path(r"D:\SCML_initial\tournament_history\20260110_171712_oneshot\20260110_171712_oneshot-stage-0001")

world_dirs = [d for d in stage_dir.iterdir() if d.is_dir()]

print(f"Found {len(world_dirs)} worlds")
print(f"Tournament: {stage_dir.parent.name}")

# 统计
total_oversell = 0
total_shortfall = 0
probe_oversell = 0
probe_shortfall = 0
postprobe_oversell = 0
postprobe_shortfall = 0
n_steps = 50
probe_days = max(10, int(n_steps * 0.1))  # 10% 或至少 10 天

for world_wrapper in world_dirs[:10]:
    # 找到内层的 world 目录
    inner_dirs = [d for d in world_wrapper.iterdir() if d.is_dir() and not d.name.startswith("_")]
    if not inner_dirs:
        continue
    world_dir = inner_dirs[0]
    
    # 读取合同
    contracts_file = world_dir / "contracts.csv"
    if not contracts_file.exists():
        continue
    
    df = pd.read_csv(contracts_file)
    
    # 找出 LOS 相关的合同
    los_as_seller = df[df["seller_name"].str.contains("LOS", na=False)]
    los_as_buyer = df[df["buyer_name"].str.contains("LOS", na=False)]
    
    # 外生合同
    los_exo_input = df[(df["seller_name"] == "SELLER") & (df["buyer_name"].str.contains("LOS", na=False))]
    los_exo_output = df[(df["buyer_name"] == "BUYER") & (df["seller_name"].str.contains("LOS", na=False))]
    
    # 判断 LOS 的角色
    is_seller = len(los_exo_input) > 0  # 有外生输入 → Level 0 卖家
    
    print(f"\n{'='*60}")
    print(f"World: {world_wrapper.name[:40]}...")
    print(f"LOS role: {'SELLER (Level 0)' if is_seller else 'BUYER (Level 1)'}")
    print(f"Probe phase: step 0-{probe_days-1}")
    
    # 按 step 统计
    for step in sorted(df["delivery_time"].unique())[:15]:
        in_probe = step < probe_days
        phase = "PROBE" if in_probe else "POST"
        
        if is_seller:
            # LOS 是卖家
            exo_in = los_exo_input[los_exo_input["delivery_time"] == step]
            exo_in_q = exo_in["quantity"].sum() if len(exo_in) > 0 else 0
            
            sales = los_as_seller[(los_as_seller["delivery_time"] == step) & (los_as_seller["buyer_name"] != "BUYER")]
            sales_q = sales["quantity"].sum() if len(sales) > 0 else 0
            
            diff = sales_q - exo_in_q
            if diff > 0:
                status = "⚠️ OVER"
                total_oversell += diff
                if in_probe:
                    probe_oversell += diff
                else:
                    postprobe_oversell += diff
            else:
                status = "✓"
            print(f"  [{phase}] Step {step:2d}: exo_in={exo_in_q:2d}, sold={sales_q:2d}, diff={diff:+3d} {status}")
        else:
            # LOS 是买家
            exo_out = los_exo_output[los_exo_output["delivery_time"] == step]
            exo_out_q = exo_out["quantity"].sum() if len(exo_out) > 0 else 0
            
            buys = los_as_buyer[(los_as_buyer["delivery_time"] == step) & (los_as_buyer["seller_name"] != "SELLER")]
            buys_q = buys["quantity"].sum() if len(buys) > 0 else 0
            
            diff = buys_q - exo_out_q
            if diff > 0:
                status = "⚠️ OVER"  # 买多了
                total_oversell += diff
                if in_probe:
                    probe_oversell += diff
                else:
                    postprobe_oversell += diff
            elif diff < 0:
                status = "⚠️ SHORT"  # 买少了
                total_shortfall += abs(diff)
                if in_probe:
                    probe_shortfall += abs(diff)
                else:
                    postprobe_shortfall += abs(diff)
            else:
                status = "✓"
            print(f"  [{phase}] Step {step:2d}: need={exo_out_q:2d}, bought={buys_q:2d}, diff={diff:+3d} {status}")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Probe phase (step 0-{probe_days-1}):")
print(f"  Oversell/Overbuy: {probe_oversell}")
print(f"  Shortfall: {probe_shortfall}")
print(f"Post-probe phase (step {probe_days}+):")
print(f"  Oversell/Overbuy: {postprobe_oversell}")
print(f"  Shortfall: {postprobe_shortfall}")
print(f"Total:")
print(f"  Oversell/Overbuy: {total_oversell}")
print(f"  Shortfall: {total_shortfall}")
