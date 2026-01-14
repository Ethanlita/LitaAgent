"""检查外生价格属性"""
from scml.oneshot import SCML2024OneShotWorld

world = SCML2024OneShotWorld(**SCML2024OneShotWorld.generate(
    agent_types=['scml.oneshot.agents.rand.SyncRandomOneShotAgent'] * 4,
    n_steps=5,
    n_processes=2,
))
world.step()

print("=== Exogenous Contract Info ===")
for agent in world.agents.values():
    awi = agent.awi
    print(f"\nAgent: {agent.name}, Level: {awi.level}")
    print(f"  exo_input_price:    {getattr(awi, 'current_exogenous_input_price', 'N/A')}")
    print(f"  exo_input_quantity: {getattr(awi, 'current_exogenous_input_quantity', 'N/A')}")
    print(f"  exo_output_price:   {getattr(awi, 'current_exogenous_output_price', 'N/A')}")
    print(f"  exo_output_quantity:{getattr(awi, 'current_exogenous_output_quantity', 'N/A')}")
    print(f"  trading_prices:     {awi.trading_prices}")
    
    # 检查外生合同
    exo_in = getattr(awi, 'current_exogenous_input_quantity', 0) or 0
    exo_out = getattr(awi, 'current_exogenous_output_quantity', 0) or 0
    
    if awi.level == 0:
        print(f"  -> Seller: has exo INPUT (procurement), needs to SELL output")
        print(f"     Cost basis should be exo_input_price / exo_input_quantity")
        if exo_in > 0:
            avg_cost = getattr(awi, 'current_exogenous_input_price', 0) / exo_in
            print(f"     Average cost per unit: {avg_cost:.2f}")
    else:
        print(f"  -> Buyer: has exo OUTPUT (sales), needs to BUY input")
        print(f"     Revenue basis should be exo_output_price / exo_output_quantity")
        if exo_out > 0:
            avg_rev = getattr(awi, 'current_exogenous_output_price', 0) / exo_out
            print(f"     Average revenue per unit: {avg_rev:.2f}")
