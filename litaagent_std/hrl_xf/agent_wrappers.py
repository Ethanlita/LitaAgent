from __future__ import annotations

from .agent import LitaAgentHRL


class LitaAgentHRLHeuristicL4(LitaAgentHRL):
    """HRL agent that forces heuristic L4 even in neural mode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self, "l4", None) is not None:
            self.l4.use_neural_alpha = False
            self.l4.coordinator = None


class LitaAgentHRLAlphaZero(LitaAgentHRL):
    """HRL agent that forces L4 alpha to zero for all threads."""

    def _compute_global_control(self):
        broadcast, alpha_map = super()._compute_global_control()
        if not alpha_map:
            return broadcast, alpha_map
        return broadcast, {k: 0.0 for k in alpha_map}
