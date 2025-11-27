import unittest
from unittest.mock import Mock, patch, MagicMock  # Added MagicMock
from collections import defaultdict

from scml.std import StdAWI

from litaagent_std.inventory_manager_n import InventoryManager
# Assuming litaagent_yr.py is in the same directory or accessible in PYTHONPATH
# 假设 litaagent_yr.py 在同一目录或 PYTHONPATH 中可访问
from litaagent_std.litaagent_yr import LitaAgentYR, HeuristicSettings, MaterialType, IMContract, IMContractType


class TestDynamicProfitMargin(unittest.TestCase):

    def setUp(self):
        """Set up for each test."""
        """为每个测试进行设置。"""
        self.agent = LitaAgentYR(name="test_agent")
        self.agent.h = HeuristicSettings()  # Use default heuristics / 使用默认启发式参数
        self.agent.initial_min_profit_ratio = self.agent.h.min_profit_ratio  # Should be 0.10 by default / 默认为0.10

        # Mock AWI (Agent World Interface)
        # 模拟 AWI (代理世界接口)
        self.agent.awi = MagicMock(spec=StdAWI)  # Use MagicMock for attribute access / 使用 MagicMock 进行属性访问
        self.agent.awi.n_steps = 20  # Default total days / 默认总天数
        self.agent.awi.current_step = 0  # Default current day / 默认当前日期

        # Mock InventoryManager
        # 模拟 InventoryManager
        self.agent.im = MagicMock(spec=InventoryManager)  # Use MagicMock / 使用 MagicMock
        self.agent.im.daily_production_capacity = 10  # Default capacity / 默认产能

        # Default mock returns for IM methods
        # IM 方法的默认模拟返回
        self.agent.im.get_inventory_summary.return_value = {'current_stock': 0, 'estimated_available': 0}
        self.agent.im.get_pending_contracts.return_value = []

        # Reset sales counters for each test
        # 为每个测试重置销售计数器
        self.agent._sales_successes_since_margin_update = 0
        self.agent._sales_failures_since_margin_update = 0

        # Ensure min_profit_ratio is reset to initial for each test run of the method
        # 确保在每次测试运行该方法时，min_profit_ratio 被重置为初始值
        # The method itself reads self.initial_min_profit_ratio as its starting point.
        # 该方法本身读取 self.initial_min_profit_ratio 作为其起点。
        self.agent.min_profit_ratio = self.agent.initial_min_profit_ratio

    def _configure_im_product_stock(self, stock_level):
        self.agent.im.get_inventory_summary.side_effect = lambda day, mat_type: \
            {'current_stock': stock_level, 'estimated_available': stock_level} if mat_type == MaterialType.PRODUCT else \
                {'current_stock': 0, 'estimated_available': 0}

    def _configure_im_future_demand(self, demand_per_day_list):
        """
        Configures mock for get_pending_contracts.
        demand_per_day_list: A list of (day_offset, quantity) tuples.
        配置 get_pending_contracts 的模拟。
        demand_per_day_list: 一个 (日期偏移, 数量) 元组的列表。
        """

        def side_effect_func(is_supply, day):
            if is_supply:
                return []

            contracts = []
            current_sim_day = self.agent.awi.current_step
            for d_offset, qty in demand_per_day_list:
                contract_day = current_sim_day + d_offset
                if contract_day == day:
                    mock_contract = Mock(spec=IMContract)
                    mock_contract.quantity = qty
                    mock_contract.material_type = MaterialType.PRODUCT
                    contracts.append(mock_contract)
            return contracts

        self.agent.im.get_pending_contracts.side_effect = side_effect_func

    def test_rule_A_no_stock_no_demand(self):
        """Rule A: No stock, no immediate demand. Should use base (initial_min_profit_ratio)."""
        """规则 A：没有库存，没有即时需求。应使用基础值 (initial_min_profit_ratio)。"""
        self.agent.awi.current_step = 5
        self._configure_im_product_stock(0)
        self._configure_im_future_demand([])  # No future demand / 没有未来需求

        # Rule B might increase it if capacity is low, let's assume capacity is not an issue here for Rule A focus
        # 如果产能低，规则B可能会增加它，这里为了关注规则A，假设产能不是问题
        # Default initial_min_profit_ratio is 0.10. Rule B (low inv vs cap) might make it 0.12.
        # 默认 initial_min_profit_ratio 是 0.10。规则B（低库存与产能）可能使其变为 0.12。
        self.agent._update_dynamic_profit_margin_parameters()
        self.assertAlmostEqual(self.agent.min_profit_ratio, 0.12)  # initial 0.10, Rule B makes it 0.12

    def test_rule_A_very_high_inventory(self):
        """Rule A: Very high inventory (>2x demand)."""
        """规则 A：库存非常高 (>2倍需求)。"""
        self.agent.awi.current_step = 5
        self._configure_im_product_stock(100)
        self._configure_im_future_demand([(1, 10), (2, 10), (3, 10), (4, 5), (5, 5)])  # Total 40

        self.agent._update_dynamic_profit_margin_parameters()
        # Rule A sets to 0.05. Rule B might increase it.
        # If capacity is 10, 100 stock is > 0.5 * capacity, so Rule B (low inv vs cap) won't trigger.
        # If future demand is 40, 100 stock is > 0.5 * 40, so Rule B (low inv vs demand) won't trigger.
        # Rule C (not late game) no effect. Rule D no effect.
        self.assertAlmostEqual(self.agent.min_profit_ratio, 0.05)

    def test_rule_A_moderately_high_inventory(self):
        """Rule A: Moderately high inventory (>1x demand)."""
        """规则 A：库存中等偏高 (>1倍需求)。"""
        self.agent.awi.current_step = 5
        self._configure_im_product_stock(50)
        self._configure_im_future_demand([(1, 10), (2, 10), (3, 10), (4, 5), (5, 5)])  # Total 40

        self.agent._update_dynamic_profit_margin_parameters()
        # Rule A sets to 0.07. Other rules assumed not to override significantly here.
        self.assertAlmostEqual(self.agent.min_profit_ratio, 0.07)

    def test_rule_A_no_demand_inv_gt_1_day_prod(self):
        """Rule A: No demand, but inventory > 1 day's production."""
        """规则 A：没有需求，但库存 > 1天的产量。"""
        self.agent.awi.current_step = 5
        self.agent.im.daily_production_capacity = 10
        self._configure_im_product_stock(15)  # More than 1 day's capacity / 超过1天的产能
        self._configure_im_future_demand([])

        self.agent._update_dynamic_profit_margin_parameters()
        # Rule A sets to 0.06.
        self.assertAlmostEqual(self.agent.min_profit_ratio, 0.06)

    def test_rule_B_low_inv_vs_demand(self):
        """Rule B: Low inventory vs demand (<0.5x), should be max(RuleA_val, 0.15)."""
        """规则 B：库存相对于需求较低 (<0.5倍)，应为 max(RuleA_val, 0.15)。"""
        self.agent.awi.current_step = 5
        self.agent.initial_min_profit_ratio = 0.10  # Rule A default
        self.agent.min_profit_ratio = 0.10
        self._configure_im_product_stock(10)
        self._configure_im_future_demand([(1, 10), (2, 10), (3, 10)])  # Demand = 30. Stock 10 < 0.5 * 30

        self.agent._update_dynamic_profit_margin_parameters()
        self.assertAlmostEqual(self.agent.min_profit_ratio, 0.15)

    def test_rule_B_low_inv_vs_capacity(self):
        """Rule B: Low inventory vs capacity (<0.5 day), should be max(RuleA_val, 0.12)."""
        """规则 B：库存相对于产能较低 (<0.5天)，应为 max(RuleA_val, 0.12)。"""
        self.agent.awi.current_step = 5
        self.agent.initial_min_profit_ratio = 0.06  # Assume Rule A set it low
        self.agent.min_profit_ratio = 0.06
        self.agent.im.daily_production_capacity = 10
        self._configure_im_product_stock(4)  # Stock 4 < 0.5 * 10
        self._configure_im_future_demand([])  # No demand to isolate capacity effect / 没有需求以隔离产能效应

        self.agent._update_dynamic_profit_margin_parameters()
        self.assertAlmostEqual(self.agent.min_profit_ratio, 0.12)

    def test_rule_C_end_game(self):
        """Rule C: End game (last 15%), should be min(RuleAB_val, 0.03)."""
        """规则 C：游戏结束阶段 (最后15%)，应为 min(RuleAB_val, 0.03)。"""
        self.agent.awi.current_step = 18  # 18/20 = 90% > 85%
        self.agent.initial_min_profit_ratio = 0.12  # Assume Rule A/B set it to this
        self.agent.min_profit_ratio = 0.12
        self._configure_im_product_stock(20)  # Non-triggering stock/demand for A/B
        self._configure_im_future_demand([(1, 5)])

        self.agent._update_dynamic_profit_margin_parameters()
        self.assertAlmostEqual(self.agent.min_profit_ratio, 0.03)

    def test_rule_C_late_mid_game(self):
        """Rule C: Late mid-game (>60%), should be min(RuleAB_val, 0.08)."""
        """规则 C：游戏中后期 (>60%)，应为 min(RuleAB_val, 0.08)。"""
        self.agent.awi.current_step = 13  # 13/20 = 65% > 60%
        self.agent.initial_min_profit_ratio = 0.12  # Assume Rule A/B set it to this
        self.agent.min_profit_ratio = 0.12
        self._configure_im_product_stock(20)
        self._configure_im_future_demand([(1, 5)])

        self.agent._update_dynamic_profit_margin_parameters()
        self.assertAlmostEqual(self.agent.min_profit_ratio, 0.08)

    def test_rule_D_successes_increase_margin(self):
        """Rule D: Sales successes increase margin."""
        """规则 D：销售成功增加利润率。"""
        self.agent.min_profit_ratio = 0.10  # Set by A/B/C
        self.agent._sales_successes_since_margin_update = 10
        self.agent._sales_failures_since_margin_update = 0

        self.agent._update_dynamic_profit_margin_parameters()
        # 0.10 (base after A/B/C) + (10//5)*0.005 = 0.10 + 0.01 = 0.11
        self.assertAlmostEqual(self.agent.min_profit_ratio, 0.11)
        self.assertEqual(self.agent._sales_successes_since_margin_update, 0)
        self.assertEqual(self.agent._sales_failures_since_margin_update, 0)

    def test_rule_D_failures_decrease_margin(self):
        """Rule D: Sales failures decrease margin."""
        """规则 D：销售失败降低利润率。"""
        self.agent.min_profit_ratio = 0.10  # Set by A/B/C
        self.agent._sales_successes_since_margin_update = 0
        self.agent._sales_failures_since_margin_update = 2

        self.agent._update_dynamic_profit_margin_parameters()
        # 0.10 (base after A/B/C) - 2*0.005 = 0.10 - 0.01 = 0.09
        self.assertAlmostEqual(self.agent.min_profit_ratio, 0.09)
        self.assertEqual(self.agent._sales_successes_since_margin_update, 0)
        self.assertEqual(self.agent._sales_failures_since_margin_update, 0)

    def test_final_clamping_too_low(self):
        """Final clamping: value too low."""
        """最终限制：值过低。"""
        # Setup conditions to make rules A, B, C, D result in < 0.02
        self.agent.awi.current_step = 19  # End game
        self.agent.initial_min_profit_ratio = 0.05  # Start low
        self.agent.min_profit_ratio = 0.05
        self._configure_im_product_stock(100)  # High stock
        self._configure_im_future_demand([(1, 10)])
        self.agent._sales_failures_since_margin_update = 2  # Further decrease

        self.agent._update_dynamic_profit_margin_parameters()
        # Rule A -> 0.05
        # Rule C -> min(0.05, 0.03) = 0.03
        # Rule D -> 0.03 - 2*0.005 = 0.02
        # Clamped to 0.02
        self.assertAlmostEqual(self.agent.min_profit_ratio, 0.02)

    def test_final_clamping_too_high(self):
        """Final clamping: value too high."""
        """最终限制：值过高。"""
        # Setup conditions to make rules A, B, C, D result in > 0.25
        self.agent.awi.current_step = 1
        self.agent.initial_min_profit_ratio = 0.24  # Start high
        self.agent.min_profit_ratio = 0.24
        self._configure_im_product_stock(1)  # Low stock vs capacity
        self._configure_im_future_demand([(1, 1)])  # Low stock vs demand
        self.agent._sales_successes_since_margin_update = 10  # Further increase

        self.agent._update_dynamic_profit_margin_parameters()
        # Rule A -> 0.24 (initial)
        # Rule B -> max(0.24, 0.15) = 0.24 (low inv vs demand)
        # Rule B -> max(0.24, 0.12) = 0.24 (low inv vs cap)
        # Rule C -> no effect
        # Rule D -> 0.24 + (10//5)*0.005 = 0.24 + 0.01 = 0.25
        # Clamped to 0.25
        self.assertAlmostEqual(self.agent.min_profit_ratio, 0.25)

    def test_combined_scenario_1(self):
        """Combined: Early game, low stock vs demand, 1 sales failure."""
        """组合场景1：游戏早期，库存相对于需求较低，1次销售失败。"""
        self.agent.awi.current_step = 5
        self.agent.awi.n_steps = 20
        self.agent.initial_min_profit_ratio = 0.10
        self.agent.min_profit_ratio = 0.10
        self.agent.im.daily_production_capacity = 10

        self._configure_im_product_stock(5)
        self._configure_im_future_demand([(1, 10), (2, 10)])  # Demand = 20. Stock 5 < 0.5 * 20

        self.agent._sales_successes_since_margin_update = 0
        self.agent._sales_failures_since_margin_update = 1

        self.agent._update_dynamic_profit_margin_parameters()
        # Rule A: Default (0.10)
        # Rule B: Low inv vs demand (5 < 0.5*20=10) -> max(0.10, 0.15) = 0.15
        # Rule B: Low inv vs cap (5 == 0.5*10) -> no change from 0.15
        # Rule C: No effect (early game)
        # Rule D: 0.15 - 1*0.005 = 0.145
        # Clamping: No effect
        self.assertAlmostEqual(self.agent.min_profit_ratio, 0.145)

    def test_combined_scenario_2(self):
        """Combined: Late game, high stock, some sales successes."""
        """组合场景2：游戏后期，高库存，一些销售成功。"""
        self.agent.awi.current_step = 19
        self.agent.awi.n_steps = 20
        self.agent.initial_min_profit_ratio = 0.10
        self.agent.min_profit_ratio = 0.10
        self.agent.im.daily_production_capacity = 10

        self._configure_im_product_stock(100)
        self._configure_im_future_demand([(1, 10)])  # Demand = 10. Stock 100 > 2 * 10

        self.agent._sales_successes_since_margin_update = 6
        self.agent._sales_failures_since_margin_update = 0

        self.agent._update_dynamic_profit_margin_parameters()
        # Rule A: Very high inv (>2x demand) -> 0.05
        # Rule B: No effect
        # Rule C: End game (19/20=95% > 85%) -> min(0.05, 0.03) = 0.03
        # Rule D: 0.03 + (6//5)*0.005 = 0.03 + 0.005 = 0.035
        # Clamping: No effect
        self.assertAlmostEqual(self.agent.min_profit_ratio, 0.035)


if __name__ == '__main__':
    unittest.main()