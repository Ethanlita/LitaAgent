import unittest

import numpy as np

from litaagent_std.hrl_xf.l3_executor import HeuristicL3Executor, TemporalDecisionTransformer
from litaagent_std.hrl_xf.l4_coordinator import HeuristicL4Coordinator, ThreadState


class TestHRLXFInterfaces(unittest.TestCase):
    def test_l3_heuristic_output_has_hidden_state(self):
        l3 = HeuristicL3Executor(horizon=10)
        out = l3.compute(
            history=[],
            goal=np.zeros(16, dtype=np.float32),
            is_buying=True,
            time_mask=np.zeros(11, dtype=np.float32),
            baseline=(10.0, 100.0, 1),
        )
        self.assertIsNotNone(out.hidden_state)
        self.assertEqual(out.hidden_state.shape, (128,))

    def test_l3_transformer_return_latent_flag(self):
        try:
            import torch
        except Exception as e:
            self.skipTest(f"torch not available: {e}")
            return

        model = TemporalDecisionTransformer(
            horizon=10,
            d_model=32,
            n_heads=4,
            n_layers=1,
            max_seq_len=4,
        )
        history = torch.zeros((2, 1, 3), dtype=torch.float32)
        goal = torch.zeros((2, 16), dtype=torch.float32)
        role = torch.zeros((2,), dtype=torch.long)
        time_mask = torch.zeros((2, 11), dtype=torch.float32)
        baseline = torch.zeros((2, 3), dtype=torch.float32)

        delta_q, delta_p, time_logits = model(history, goal, role, time_mask, baseline)
        self.assertEqual(delta_q.shape, (2, 1))
        self.assertEqual(delta_p.shape, (2, 1))
        self.assertEqual(time_logits.shape, (2, 11))

        delta_q, delta_p, time_logits, latent = model(
            history,
            goal,
            role,
            time_mask,
            baseline,
            return_latent=True,
        )
        self.assertEqual(latent.shape, (2, 32))

    def test_l4_output_thread_ids_aligned(self):
        l4 = HeuristicL4Coordinator(horizon=10)
        threads = [
            ThreadState(
                thread_id="a",
                thread_feat=np.zeros(24, dtype=np.float32),
                target_time=1,
                role=0,
            ),
            ThreadState(
                thread_id="b",
                thread_feat=np.zeros(24, dtype=np.float32),
                target_time=2,
                role=1,
            ),
        ]
        out = l4.compute(threads, np.zeros(30, dtype=np.float32))
        self.assertEqual(out.thread_ids, ["a", "b"])
        self.assertEqual(out.weights.shape, (2,))
        self.assertEqual(out.modulation_factors.shape, (2,))

    def test_global_coordinator_padding_mask(self):
        try:
            import torch

            from litaagent_std.hrl_xf.l4_coordinator import GlobalCoordinator
        except Exception as e:
            self.skipTest(f"torch not available: {e}")
            return

        model = GlobalCoordinator(
            d_model=16,
            n_heads=2,
            horizon=10,
            thread_feat_dim=8,
            global_feat_dim=12,
        )

        thread_feats = torch.randn((1, 4, 8), dtype=torch.float32)
        thread_times = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        thread_roles = torch.tensor([[0, 1, 0, 1]], dtype=torch.long)
        global_state = torch.zeros((1, 12), dtype=torch.float32)
        mask = torch.tensor([[True, True, False, False]])

        weights, _ = model(thread_feats, thread_times, thread_roles, global_state, thread_mask=mask)

        self.assertTrue(torch.allclose(weights[0, 2:], torch.zeros(2), atol=1e-6))
        self.assertAlmostEqual(weights[0, :2].sum().item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
