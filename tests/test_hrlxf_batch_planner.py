import unittest

import numpy as np

from litaagent_std.hrl_xf.batch_planner import plan_buy_offers_by_alpha
from litaagent_std.hrl_xf.l1_safety import L1SafetyLayer


class TestHRLXFBatchPlanner(unittest.TestCase):
    def test_plan_buy_offers_respects_q_safe_and_budget(self):
        l1 = L1SafetyLayer(horizon=5)
        Q_safe = np.array([0, 10, 10, 10, 10, 10], dtype=np.float32)
        B_free = 100.0

        buy_ids = ["a", "b", "c"]
        actions = {
            "a": (10.0, 5.0, 1),
            "b": (20.0, 4.0, 1),
            "c": (10.0, 10.0, 1),
        }
        alphas = {"a": 0.1, "b": 0.6, "c": 0.3}

        offers = plan_buy_offers_by_alpha(
            l1=l1,
            buy_ids=buy_ids,
            actions=actions,
            alphas=alphas,
            Q_safe=Q_safe,
            B_free=B_free,
            current_step=0,
        )

        self.assertIsNotNone(offers["b"])
        self.assertEqual(offers["b"][0], 10)  # Q_safe[1] 只有 10
        self.assertEqual(offers["b"][1], 1)
        self.assertAlmostEqual(offers["b"][2], 4.0, places=6)
        self.assertIsNone(offers["a"])
        self.assertIsNone(offers["c"])

        total_spend = sum(o[0] * o[2] for o in offers.values() if o is not None)
        total_qty_t1 = sum(o[0] for o in offers.values() if o is not None and o[1] == 1)
        self.assertLessEqual(total_spend, B_free + 1e-6)
        self.assertLessEqual(total_qty_t1, Q_safe[1] + 1e-6)

    def test_plan_buy_offers_leftover_flows(self):
        l1 = L1SafetyLayer(horizon=5)
        Q_safe = np.array([0, 100, 100, 100, 100, 100], dtype=np.float32)
        B_free = 100.0

        buy_ids = ["a", "b"]
        actions = {
            "a": (1.0, 10.0, 1),   # 花费 10
            "b": (100.0, 1.0, 2),  # 尽量吃满剩余预算
        }
        alphas = {"a": 0.9, "b": 0.1}

        offers = plan_buy_offers_by_alpha(
            l1=l1,
            buy_ids=buy_ids,
            actions=actions,
            alphas=alphas,
            Q_safe=Q_safe,
            B_free=B_free,
            current_step=0,
        )

        self.assertEqual(offers["a"][0], 1)
        self.assertEqual(offers["a"][1], 1)
        self.assertAlmostEqual(offers["a"][2], 10.0, places=6)

        self.assertEqual(offers["b"][0], 90)  # 剩余预算 90
        self.assertEqual(offers["b"][1], 2)
        self.assertAlmostEqual(offers["b"][2], 1.0, places=6)

        total_spend = sum(o[0] * o[2] for o in offers.values() if o is not None)
        self.assertAlmostEqual(total_spend, B_free, places=6)

    def test_plan_buy_offers_tie_breaker_is_deterministic(self):
        l1 = L1SafetyLayer(horizon=5)
        Q_safe = np.array([0, 100, 100, 100, 100, 100], dtype=np.float32)
        B_free = 50.0

        buy_ids = ["b", "a"]  # 故意打乱输入顺序
        actions = {
            "a": (10.0, 10.0, 1),
            "b": (10.0, 10.0, 1),
        }
        alphas = {"a": 0.5, "b": 0.5}

        offers = plan_buy_offers_by_alpha(
            l1=l1,
            buy_ids=buy_ids,
            actions=actions,
            alphas=alphas,
            Q_safe=Q_safe,
            B_free=B_free,
            current_step=0,
        )

        self.assertIsNotNone(offers["a"])
        self.assertEqual(offers["a"][0], 5)  # 预算只够 5
        self.assertIsNone(offers["b"])


if __name__ == "__main__":
    unittest.main()

