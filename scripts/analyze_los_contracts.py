"""分析 LOS 在每个 step 的合同情况，检查是否超卖"""
import pandas as pd
from pathlib import Path
import sys

# 允许指定比赛目录
if len(sys.argv) > 1:
    stage_dir = Path(sys.argv[1])
else:
    stage_dir = Path(r"D:\SCML_initial\tournament_history\20260110_163218_oneshot\20260110_163218_oneshot-stage-0001")
world_dirs = [d for d in stage_dir.iterdir() if d.is_dir()]

print(f"Found {len(world_dirs)} worlds")

for world_wrapper in world_dirs[:3]:
    # 找到内层的 world 目录
    inner_dirs = [d for d in world_wrapper.iterdir() if d.is_dir() and not d.name.startswith("_")]
    if not inner_dirs:
        continue
    world_dir = inner_dirs[0]
    
    print(f"\n=== {world_wrapper.name} ===")
    
    # 读取合同
    contracts_file = world_dir / "contracts.csv"
    if not contracts_file.exists():
        print("  No contracts.csv")
        continue
    
    df = pd.read_csv(contracts_file)
    
    # 找出 LOS 相关的合同
    los_as_seller = df[df["seller_name"].str.contains("LOS", na=False)]
    los_as_buyer = df[df["buyer_name"].str.contains("LOS", na=False)]
    
    print(f"  LOS as seller: {len(los_as_seller)} contracts")
    print(f"  LOS as buyer: {len(los_as_buyer)} contracts")
    
    # 外生合同 (seller=SELLER 或 buyer=BUYER)
    los_exo_input = df[(df["seller_name"] == "SELLER") & (df["buyer_name"].str.contains("LOS", na=False))]
    los_exo_output = df[(df["buyer_name"] == "BUYER") & (df["seller_name"].str.contains("LOS", na=False))]
    
    # 按 step 统计
    print("\n  Per-step analysis:")
    for step in sorted(df["delivery_time"].unique()):
        step_df = df[df["delivery_time"] == step]
        
        # LOS 的外生输入（采购）
        exo_in = los_exo_input[los_exo_input["delivery_time"] == step]
        exo_in_q = exo_in["quantity"].sum() if len(exo_in) > 0 else 0
        exo_in_p = exo_in["unit_price"].mean() if len(exo_in) > 0 else 0
        
        # LOS 的外生输出（如果是 level 1）
        exo_out = los_exo_output[los_exo_output["delivery_time"] == step]
        exo_out_q = exo_out["quantity"].sum() if len(exo_out) > 0 else 0
        
        # LOS 作为卖家的销售合同（非外生）
        sales = los_as_seller[(los_as_seller["delivery_time"] == step) & (los_as_seller["buyer_name"] != "BUYER")]
        sales_q = sales["quantity"].sum() if len(sales) > 0 else 0
        sales_p = sales["unit_price"].mean() if len(sales) > 0 else 0
        
        # LOS 作为买家的采购合同（非外生）
        buys = los_as_buyer[(los_as_buyer["delivery_time"] == step) & (los_as_buyer["seller_name"] != "SELLER")]
        buys_q = buys["quantity"].sum() if len(buys) > 0 else 0
        
        # 检查是否超卖
        if exo_in_q > 0:  # LOS 是 level 0 (seller)
            oversell = sales_q - exo_in_q
            status = "⚠️ OVERSELL" if oversell > 0 else "✓"
            print(f"    Step {step}: exo_in={exo_in_q}@{exo_in_p:.1f}, sold={sales_q}@{sales_p:.1f}, diff={oversell} {status}")
        elif exo_out_q > 0:  # LOS 是 level 1 (buyer)
            shortfall = exo_out_q - buys_q
            status = "⚠️ SHORTFALL" if shortfall > 0 else "✓"
            print(f"    Step {step}: need_buy={exo_out_q}, bought={buys_q}, shortfall={shortfall} {status}")
