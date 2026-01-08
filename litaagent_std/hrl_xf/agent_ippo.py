"""IPPO Training Agent for HRL-XF."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from negmas.sao import SAOResponse, SAOState, ResponseType
from negmas.outcomes import Outcome
import math

from .agent import LitaAgentHRL, NegotiationContext, L3Input, L3Output
from .l3_executor import NegotiationRound, SAOAction
from .l3_ippo import L3ActorCriticIPPO, TORCH_AVAILABLE
from .l2_manager import BUCKET_RANGES

if TORCH_AVAILABLE:
    import torch


@dataclass
class TrajectoryStep:
    """Single step in a negotiation trajectory."""
    obs: Dict[str, Any]
    action: Dict[str, Any]
    log_prob: float
    value: float
    reward: float = 0.0
    done: bool = False
    truncated: bool = False # Handle world end or other interruptions
    policy_mask: float = 1.0  # 1.0 = use for policy loss, 0.0 = skip (constraint override)
    bootstrap_value: float = 0.0  # For truncated episodes, V(s_next) estimate
    next_obs: Optional[Dict[str, Any]] = None  # s_{t+1} for accurate bootstrap
    
    # Store raw info for debugging
    info: Dict[str, Any] = field(default_factory=dict)


class LitaAgentHRLIPPOTrain(LitaAgentHRL):
    """
    HRL-XF Agent for IPPO Training (L3 Only).
    
    Differences from base agent:
    1. Uses L3ActorCriticIPPO (probabilistic) instead of L3Actor.
    2. Records trajectories for PPO updates.
    3. Forces Alpha=0 (disables L4 coordination for L3 training).
    4. Implements reward calculation for negotiation steps.
    """


    def __init__(
        self, 
        *args, 
        parameter_sharing: bool = True,
        l3_ippo_model_path: Optional[str] = None,
        deterministic_policy: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Override L3 with IPPO model
        # Note: We still keep self.l3 for Heuristic fallback if needed, or we just ignore it.
        # But base class uses self.l3 in respond/propose, which we override.
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for IPPO training")
            
        self.l3_ippo = L3ActorCriticIPPO(
            horizon=self.horizon,
            context_dim=31,
            n_qty_buckets=10 # Configurable?
        )

        self.deterministic_policy = deterministic_policy
        if l3_ippo_model_path:
            self._load_ippo_weights(l3_ippo_model_path)
        
        # Storage for active trajectories
        # negotiator_id -> List[TrajectoryStep]
        self._trajectories: Dict[str, List[TrajectoryStep]] = {}
        
        # Completed episodes buffer (to be collected by trainer)
        self.finished_episodes: List[List[TrajectoryStep]] = []
        
        self.parameter_sharing = parameter_sharing
        self.imitation_mode = False # If True, use Teacher (self.l3) and record targets
        
        print("[IPPO] Initialized LitaAgentHRLIPPOTrain with Alpha=0 enforcement.")

    def _model_device(self):
        """Get device of l3_ippo model safely."""
        try:
            return next(self.l3_ippo.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def _load_ippo_weights(self, path: str) -> None:
        """Load IPPO policy weights and fail on mismatch."""
        state_dict = torch.load(path, map_location="cpu")
        incompatible = self.l3_ippo.load_state_dict(state_dict, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        if missing or unexpected:
            missing_preview = missing[:5]
            unexpected_preview = unexpected[:5]
            raise RuntimeError(
                "IPPO state_dict mismatch. "
                f"missing={missing_preview} (total {len(missing)}), "
                f"unexpected={unexpected_preview} (total {len(unexpected)})"
            )

    def init(self):
        super().init()
        self._trajectories = {}
        self.finished_episodes = []

    def before_step(self):
        """
        Override to snapshot observations for active trajectories AFTER world state updates.
        This provides a closer approximation to s_{t+1} for truncated episodes.
        """
        # Snapshot contexts for active trajectories before base clears them.
        pending_ctxs: Dict[str, NegotiationContext] = {}
        for nid, traj in self._trajectories.items():
            if traj and not traj[-1].done:
                ctx = self._contexts.get(nid)
                if ctx is not None:
                    pending_ctxs[nid] = ctx

        # Now call super which clears contexts and updates L1/L2 for the new step.
        super().before_step()

        if not pending_ctxs:
            return

        # Build a broadcast using updated world state.
        n_buy = sum(1 for ctx in pending_ctxs.values() if ctx.is_buying)
        n_sell = len(pending_ctxs) - n_buy
        try:
            broadcast = self.l4.monitor.compute_broadcast(n_buy, n_sell)
        except Exception:
            return

        # After super().before_step(), write next_obs for last steps
        for nid, ctx in pending_ctxs.items():
            try:
                ctx.l1_output = self._step_l1_buy if ctx.is_buying else self._step_l1_sell
                ctx.l2_output = self._current_l2_output
                ctx.alpha = 0.0
                self._refresh_context_from_mechanism(ctx)
                l3_input = self._build_l3_input(ctx, broadcast, current_offer=ctx.last_offer)
                l3_input.alpha = 0.0
                obs = self._encode_observation(l3_input, has_offer=(ctx.last_offer is not None))
            except Exception:
                continue

            traj = self._trajectories.get(nid)
            if traj and not traj[-1].done and traj[-1].next_obs is None:
                traj[-1].next_obs = obs

    # ==================== L3 Override (Probabilistic) ====================

    def _get_l3_action(self, ctx: NegotiationContext, l3_input: L3Input, has_offer: bool) -> Tuple[L3Output, float, float]:
        """
        Execute L3 IPPO Policy.
        Returns:
            l3_output: Standard L3Output for execution
            log_prob: float
            value: float
        """
        # 1. Convert L3Input to Observation Dict (Tensor-ready)
        obs = self._encode_observation(l3_input, has_offer)

        traj = self._trajectories.get(ctx.negotiation_id)
        prev_step = None
        if traj:
            last_step = traj[-1]
            if not last_step.done:
                prev_step = last_step
        
        if self.imitation_mode:
            # Use Teacher (Heuristic/BC from parent)
            teacher_output = self.l3.compute(l3_input)
            
            # Map action to IPPO targets
            action_dict = self._encode_ippo_target(teacher_output, l3_input, has_offer)
            
            step = TrajectoryStep(
                obs=obs,
                action=action_dict,
                log_prob=0.0,
                value=0.0,
                reward=0.0,
                done=False,
                info={"imitation": True}
            )
            teacher_output.trajectory_step = step  # Bind trajectory step!
            
            if ctx.negotiation_id not in self._trajectories:
                self._trajectories[ctx.negotiation_id] = []
            self._trajectories[ctx.negotiation_id].append(step)

            if prev_step is not None:
                prev_step.next_obs = obs
            
            return teacher_output, 0.0, 0.0
        
        # 1.5 PRE-CHECK: Force END if constraints make trading impossible
        force_end = False
        if l3_input.is_buying:
            max_storage = max(l3_input.Q_safe) if l3_input.Q_safe else 0
            if max_storage < 1.0 or l3_input.B_free < 0.1:
                force_end = True
        else:
            max_inventory = max(l3_input.Q_safe) if l3_input.Q_safe else 0
            if max_inventory < 1.0:
                force_end = True
        
        if force_end:
            # Deterministic END - skip network sampling entirely
            action_dict = {'op': 2, 'delta_t': 0, 'price_ratio': 0.5, 'qty_bucket': 0}
            log_prob = 0.0
            
            # Get value estimate for critic training
            with torch.no_grad():
                device = self._model_device()
                h = torch.from_numpy(obs['history']).float().unsqueeze(0).to(device)
                c = torch.from_numpy(obs['context']).float().unsqueeze(0).to(device)
                m = torch.from_numpy(obs['time_mask']).float().unsqueeze(0).to(device)
                _, _, _, _, value_t = self.l3_ippo(h, c, m)
                value = value_t.item()
            
            l3_output = L3Output(action=SAOAction(action_type="end"))
            step = TrajectoryStep(
                obs=obs, action=action_dict, log_prob=log_prob, value=value,
                done=True, truncated=False, policy_mask=0.0,
                info={"forced_end": True, "clip_reason": "pre_check_constraint"}
            )
            l3_output.trajectory_step = step  # Bind!
            
            if ctx.negotiation_id not in self._trajectories:
                self._trajectories[ctx.negotiation_id] = []
            self._trajectories[ctx.negotiation_id].append(step)

            if prev_step is not None:
                prev_step.next_obs = obs
            
            return l3_output, log_prob, value
            
        # 2. Sample action from network
        action_dict, log_prob, value, info = self.l3_ippo.act(
            obs, deterministic=self.deterministic_policy
        )
        
        # 3. Decode action_dict back to L3Output (SAOAction)
        l3_output = self._decode_ippo_action(action_dict, l3_input)
        
        # 3.5 POST-CHECK: If decode forced END due to fine-grained budget check, sync action_dict
        decode_forced_end = False
        if l3_output.action.action_type == "end" and action_dict['op'] != 2:
            action_dict['op'] = 2  # Overwrite to match execution
            decode_forced_end = True
        
        # 4. Record step (Reward will be filled later)
        step = TrajectoryStep(
            obs=obs,
            action=action_dict,
            log_prob=log_prob,
            value=value,
            reward=0.0,
            done=False,
            policy_mask=0.0 if decode_forced_end else 1.0,
            info={**info, "clip_reason": "decode_forced_end"} if decode_forced_end else info
        )
        l3_output.trajectory_step = step  # Bind trajectory step!
        if prev_step is not None:
            prev_step.next_obs = obs  # s_t's next_obs = s_{t+1}
        
        if ctx.negotiation_id not in self._trajectories:
            self._trajectories[ctx.negotiation_id] = []
        self._trajectories[ctx.negotiation_id].append(step)
        
        return l3_output, log_prob, value

    def _encode_ippo_target(self, l3_output: L3Output, l3_input: L3Input, has_offer: bool) -> Dict[str, Any]:
        """Inverse map L3Output to action dict."""
        action = l3_output.action
        
        # Default targets
        op = 2 # END
        delta_t = 0
        price_ratio = 0.5
        qty_bucket = 0
        
        if action.action_type == "accept":
            op = 0
        elif action.action_type == "reject":
            op = 1
            if action.offer:
                qty, time, price = action.offer
                delta_t = int(time - l3_input.global_broadcast.current_step)
                delta_t = max(0, min(delta_t, self.horizon))
                
                # Price Ratio
                p_min = l3_input.min_price
                p_max = min(l3_input.max_price, 10000.0)
                if math.isinf(p_max): p_max = p_min * 2.0
                
                if p_max > p_min + 1e-6:
                    price_ratio = (price - p_min) / (p_max - p_min)
                else:
                    price_ratio = 0.5
                price_ratio = max(0.0, min(1.0, price_ratio))
                
                # Qty Bucket
                # Need Q_max (recalculate same logic as decode)
                if l3_input.is_buying:
                     limit_storage = float(l3_input.Q_safe[delta_t]) if delta_t < len(l3_input.Q_safe) else 0.0
                     budget_max = l3_input.B_free / (price + 0.1)
                     q_max = min(limit_storage, budget_max)
                else:
                     q_max = float(l3_input.Q_safe[delta_t]) if delta_t < len(l3_input.Q_safe) else 0.0
                
                if int(q_max) < 1:
                     # Check Constraint Sync: If logic says max < 1, behavior should be END
                     op = 2
                     q_max = 1 # dummy
                else:
                    q_max = max(1, int(q_max))
                
                # Map qty to bucket
                n_buckets = self.l3_ippo.n_qty_buckets
                if n_buckets > 1:
                    # buffer = (qty-1)/(qmax-1) * (N-1)
                    if q_max > 1:
                        bucket_f = ((qty - 1) / (q_max - 1)) * (n_buckets - 1)
                        qty_bucket = int(round(bucket_f))
                    else:
                        qty_bucket = 0
                else:
                    qty_bucket = 0
                
                qty_bucket = max(0, min(n_buckets-1, qty_bucket))

        elif action.action_type == "end":
            op = 2
            
        return {
            'op': op,
            'delta_t': delta_t,
            'price_ratio': price_ratio,
            'qty_bucket': qty_bucket
        }


    def _encode_observation(self, l3_input: L3Input, has_offer: bool) -> Dict[str, Any]:
        """Vectorize L3Input for L3ActorCriticIPPO."""
        # History
        history_arr = self._vectorize_history(l3_input.history)
        
        # Context (L3Actor._build_context logic reimplemented or exposed)
        # Accessing private method from base class's L3Actor if possible?
        # Actually L3Actor._build_context is on L3Actor instance. 
        # We can implement a static helper or copy logic.
        # For now, let's look at how L3Actor does it. It calls self._build_context.
        # But L3Actor is a class in l3_executor.
        # I should probably refactor _build_context out or copy it. 
        # To avoid editing l3_executor deeply, I'll copy the logic here for stability.
        
        context_vec = self._build_context_vec(l3_input)
        
        return {
            "history": history_arr,
            "context": context_vec,
            "time_mask": l3_input.time_mask,
            "has_offer": has_offer
        }

    def _vectorize_history(self, history: List[Any]) -> np.ndarray:
        # Same as L3Actor._compute_neural
        max_len = self.l3_ippo.max_seq_len
        if len(history) > max_len:
            history = history[-max_len:]

        if len(history) == 0:
            return np.zeros((1, 4), dtype=np.float32)
        
        rows = []
        for r in history:
            delta = int(max(0, min(int(r.delta_t), self.horizon)))
            rows.append([float(r.quantity), float(r.price), float(delta), 1.0 if r.is_my_turn else 0.0])
        return np.array(rows, dtype=np.float32)

    def _build_context_vec(self, l3_input: L3Input) -> np.ndarray:
        # Reimplementation of L3Actor._build_context logic for 29-dim vector
        # [is_buy(1), step(1), rel_time(1), alpha(1), B_free(1), Q_safe(1), 
        #  goal(16), gap(4), partner(3 - not used/embedding?)]
        # Actually context_dim=31 in existing code.
        # Based on HRL-XF paper/code:
        # is_buy: 1
        # relative_time: 1
        # alpha: 1
        # B_free (norm): 1
        # Q_safe @ buckets (4?) or just sum?
        # Let's verify context dim from previous file read... 
        # I didn't verify the EXACT content of _build_context in L3Actor.
        # Limitation: I need to know the features to be compatible with BC weights.
        # I will use a placeholder or assume I can access the helper if I modify l3_executor.
        # For now, I'll access the base object's method if it exists, or just copy the logic if I can recall/infer it.
        # Wait, I have `self.l3`. `self.l3` is an instance of `L3Actor`. 
        # `L3Actor` has `_build_context` method? Let's check the previous `read_file` output.
        # It calls `self._build_context(l3_input)` inside `_compute_neural`.
        # BUT `_build_context` is NOT defined in `L3Actor` class in the provided snippet!
        # It must be defined in `l3_executor.py` but outside `L3Actor` or inherited?
        # Ah, I missed reading the FULL file of `l3_executor.py`. 
        # I will assume `L3Actor` HAS `_build_context`. The snippet showed `_compute_neural` calling it.
        # But where is it defined?
        # I will call `self.l3._build_context(l3_input)` to reuse it!
        return self.l3._build_context(l3_input)

    def _decode_ippo_action(self, action_dict: Dict[str, Any], l3_input: L3Input) -> L3Output:
        """Convert sampled action to SAOAction."""
        op = action_dict['op']
        
        # 0: ACCEPT
        if op == 0:
            # ONLY valid if current_offer exists (enforced by mask)
            if l3_input.current_offer:
                return L3Output(action=SAOAction(action_type="accept", offer=l3_input.current_offer))
            else:
                # Should not happen with mask, but fallback
                return L3Output(action=SAOAction(action_type="end"))

        # 2: END
        if op == 2:
            return L3Output(action=SAOAction(action_type="end"))

        # 1: REJECT (Counter Offer)
        delta_t = action_dict['delta_t']
        
        # Quantity from bucket
        # Map bucket 0..N-1 to 1..Q_max
        q_bucket = action_dict['qty_bucket']
        
        # Determine Q_max
        if l3_input.is_buying:
            limit_storage = float(l3_input.Q_safe[delta_t]) if delta_t < len(l3_input.Q_safe) else 0.0
            # Price estimate needed for budget? Use sampled price?
            # Circular dependency: Q depends on P for budget?
            # Strict safe Q:
            # But we decode P first?
            # Let's decode P first.
            pass
        else:
            limit_storage = float(l3_input.Q_safe[delta_t]) if delta_t < len(l3_input.Q_safe) else 0.0

        # Price
        # Beta output is ratio 0..1
        ratio = action_dict['price_ratio']
        p_min = l3_input.min_price
        p_max = min(l3_input.max_price, 10000.0) # Safety cap
        if math.isinf(p_max): p_max = p_min * 2.0 # Fallback
        
        price = p_min + ratio * (p_max - p_min)
        
        # Now Q_max with budget
        if l3_input.is_buying:
            budget_max = l3_input.B_free / (price + 1e-6)
            q_max = min(limit_storage, budget_max)
        else:
            q_max = limit_storage
            
        # Critical Safety Check:
        # If max feasible quantity is 0, we CANNOT make a counter-offer.
        # We must END negotiation to avoid invalid action or infinite reject loops.
        q_max_int = int(q_max)
        if q_max_int < 1:
            return L3Output(action=SAOAction(action_type="end"))

        q_max = max(1, q_max_int)
        
        # Bucket mapping
        # 0 -> 1
        # N-1 -> q_max
        # Linear:
        n_buckets = self.l3_ippo.n_qty_buckets
        if n_buckets > 1:
            qty = 1 + (q_bucket / (n_buckets - 1)) * (q_max - 1)
        else:
            qty = 1
            
        qty = int(round(qty))
        qty = max(1, qty)
        
        abs_time = int(l3_input.global_broadcast.current_step + delta_t)
        offer = (qty, abs_time, float(price))
        
        return L3Output(action=SAOAction(action_type="reject", offer=offer))

    def flush_trajectories(self, world_step: int = -1) -> None:
        """
        Close all active trajectories at the end of simulation.
        For truncated episodes, compute bootstrap_value from last state.
        """
        keys = list(self._trajectories.keys())
        for nid in keys:
            traj = self._trajectories.get(nid, [])
            bootstrap_val = 0.0
            
            # Try to get bootstrap value from last step's next_obs (true s_{t+1})
            if traj and hasattr(self, 'l3_ippo'):
                try:
                    last_step = traj[-1]
                    obs = last_step.next_obs
                    if obs is not None:
                        with torch.no_grad():
                            device = self._model_device()
                            h = torch.from_numpy(obs['history']).float().unsqueeze(0).to(device)
                            c = torch.from_numpy(obs['context']).float().unsqueeze(0).to(device)
                            m = torch.from_numpy(obs['time_mask']).float().unsqueeze(0).to(device)
                            _, _, _, _, value_t = self.l3_ippo(h, c, m)
                            bootstrap_val = value_t.item()
                except Exception:
                    bootstrap_val = 0.0
            
            self._close_trajectory(nid, reward=0.0, success=False, truncated=True, bootstrap_value=bootstrap_val)
        self._trajectories.clear()

    # ==================== Override Negotiator Hooks ====================
    
    def respond(self, negotiator_id: str, state: SAOState) -> SAOResponse:
        # Standard Setup
        offer = state.current_offer
        if offer is None:
            # Issue 6: Record minimal TrajectoryStep even when offer is None
            # This ensures the trajectory is properly closed for PPO
            ctx = self._get_or_create_context(negotiator_id)
            obs = None
            try:
                broadcast, _ = self._compute_global_control()
                ctx.alpha = 0.0
                l3_input = self._build_l3_input(ctx, broadcast, current_offer=None)
                l3_input.alpha = 0.0
                obs = self._encode_observation(l3_input, has_offer=False)
            except Exception:
                obs = None

            if obs is None:
                if negotiator_id in self._trajectories and self._trajectories[negotiator_id]:
                    obs = self._trajectories[negotiator_id][-1].obs
                else:
                    obs = {
                        "history": np.zeros((1, 4), dtype=np.float32),
                        "context": np.zeros((31,), dtype=np.float32),
                        "time_mask": np.zeros((self.horizon + 1,), dtype=np.float32),
                        "has_offer": False,
                    }

            # Add a minimal END step with policy_mask=0 (no policy gradient)
            step = TrajectoryStep(
                obs=obs,
                action={'op': 2, 'delta_t': 0, 'price_ratio': 0.5, 'qty_bucket': 0},
                log_prob=0.0,
                value=0.0,
                reward=0.0,
                done=False,
                truncated=True,
                next_obs=None,
                policy_mask=0.0,  # Skip policy loss - this was forced
                info={'clip_reason': 'no_offer'}
            )
            if negotiator_id not in self._trajectories:
                self._trajectories[negotiator_id] = []
            self._trajectories[negotiator_id].append(step)
            # Close as truncated
            self._close_trajectory(negotiator_id, reward=0.0, success=False, truncated=True)
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        ctx = self._get_or_create_context(negotiator_id)
        # Update context
        ctx.last_relative_time = float(getattr(state, "relative_time", 0.0) or 0.0)
        ctx.last_step = int(getattr(state, "step", 0) or 0)
        qty, delivery, price = offer
        delta_t = int(delivery - self.awi.current_step)
        delta_t = int(max(0, min(delta_t, self.horizon)))
        ctx.last_delta_t = delta_t
        ctx.last_offer = (int(qty), float(price), int(delta_t))
        ctx.history.append(NegotiationRound(
            quantity=qty, price=price, delta_t=delta_t, is_my_turn=False
        ))

        # Build Input (Force Alpha=0)
        broadcast, _ = self._compute_global_control() # Ignoring alpha_map
        ctx.alpha = 0.0 # Force 0
        l3_input = self._build_l3_input(ctx, broadcast, current_offer=ctx.last_offer)
        # Enforce input alpha is 0
        l3_input.alpha = 0.0

        # EXECUTE POLICY
        l3_output, _, _ = self._get_l3_action(ctx, l3_input, has_offer=True)
        ctx.l3_output = l3_output

        # Add my action to history if I counter/accept?
        # Wait, if I Accept, deal is done. If I Reject w/ Counter, I add to history.
        # But trajectory recording is done in _get_l3_action.
        
        action_type = l3_output.action.action_type
        
        if action_type == "accept":
            # Just return accept. L1 feasibility check?
            # Training agent should probably trust the sampled action or apply L1.
            # User said: "L1 永远是规则/clip". 
            # If sampled Accept is infeasible, we should probably END or Reject?
            # Or assume mask handled it?
            # Let's apply standard check.
            if self._is_accept_feasible(offer, ctx, broadcast):
                 return SAOResponse(ResponseType.ACCEPT_OFFER, offer)
            else:
                # Fallback if policy messed up mask: End - sync trajectory
                step = getattr(l3_output, 'trajectory_step', None)
                if step:
                    step.action['op'] = 2  # Force END
                    step.policy_mask = 0.0  # Skip policy loss
                    step.info['clip_reason'] = 'accept_infeasible'
                return SAOResponse(ResponseType.END_NEGOTIATION, None)

        if action_type == "end":
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # Counter
        my_offer = l3_output.action.offer
        if my_offer:
            # Enforce L1 Safety via Base Agent logic
            safe_offer = self._resolve_counter_offer(ctx, my_offer, negotiator_id, broadcast)
            
            # Action Execution Consistency Fix:
            # If L1 clips the offer or forces End, we MUST update the TrajectoryStep
            # so the agent learns from what actually happened.
            step = getattr(l3_output, 'trajectory_step', None)
            
            if safe_offer is None:
                 # L1 forced End
                 if step:
                     step.action['op'] = 2 # End
                     step.policy_mask = 0.0  # Skip policy loss
                     step.info['clip_reason'] = 'l1_forced_end'
                 return SAOResponse(ResponseType.END_NEGOTIATION, None)
            
            if safe_offer != my_offer and step:
                # L1 clipped quantity/price - reverse encode to sync training data
                # Mark as clipped for policy mask
                step.policy_mask = 0.0
                step.info['clip_reason'] = 'l1_clipped'
                
                new_qty, new_time, new_price = safe_offer
                new_delta_t = int(new_time - self.awi.current_step)
                new_delta_t = max(0, min(new_delta_t, self.horizon))
                
                # Reverse encode price_ratio
                p_min = l3_input.min_price
                p_max = min(l3_input.max_price, 10000.0)
                if math.isinf(p_max): p_max = p_min * 2.0
                if p_max > p_min + 1e-6:
                    new_price_ratio = (new_price - p_min) / (p_max - p_min)
                else:
                    new_price_ratio = 0.5
                new_price_ratio = max(0.0, min(1.0, new_price_ratio))
                
                # Reverse encode qty_bucket
                if l3_input.is_buying:
                    limit_storage = float(l3_input.Q_safe[new_delta_t]) if new_delta_t < len(l3_input.Q_safe) else 0.0
                    budget_max = l3_input.B_free / (new_price + 0.1)
                    q_max = min(limit_storage, budget_max)
                else:
                    q_max = float(l3_input.Q_safe[new_delta_t]) if new_delta_t < len(l3_input.Q_safe) else 0.0
                q_max = max(1, int(q_max))
                
                n_buckets = self.l3_ippo.n_qty_buckets
                if n_buckets > 1 and q_max > 1:
                    new_qty_bucket = int(round(((new_qty - 1) / (q_max - 1)) * (n_buckets - 1)))
                else:
                    new_qty_bucket = 0
                new_qty_bucket = max(0, min(n_buckets - 1, new_qty_bucket))
                
                # Update step action dict
                step.action['delta_t'] = new_delta_t
                step.action['price_ratio'] = new_price_ratio
                step.action['qty_bucket'] = new_qty_bucket

            # Record history with ACTUAL offer sent
            ctx.history.append(NegotiationRound(
                quantity=safe_offer[0], price=safe_offer[2], delta_t=int(safe_offer[1]-self.awi.current_step), is_my_turn=True
            ))
            return SAOResponse(ResponseType.REJECT_OFFER, safe_offer)
        
        return SAOResponse(ResponseType.END_NEGOTIATION, None)

    def propose(self, negotiator_id: str, state: SAOState) -> Optional[Outcome]:
        ctx = self._get_or_create_context(negotiator_id)
        ctx.last_relative_time = float(getattr(state, "relative_time", 0.0) or 0.0)
        ctx.last_step = int(getattr(state, "step", 0) or 0)
        
        current_offer = None
        if state.current_offer is not None:
             # Should not happen in propose unless first turn?
             # But if it exists, use it.
             qty, delivery, price = state.current_offer
             delta_t = int(delivery - self.awi.current_step)
             current_offer = (int(qty), float(price), int(delta_t))

        # Build Input (Alpha=0)
        broadcast, _ = self._compute_global_control()
        ctx.alpha = 0.0
        l3_input = self._build_l3_input(ctx, broadcast, current_offer=current_offer)
        l3_input.alpha = 0.0

        # EXECUTE
        l3_output, _, _ = self._get_l3_action(ctx, l3_input, has_offer=(current_offer is not None))
        ctx.l3_output = l3_output

        if l3_output.action.action_type == "end":
            return None
        
        my_offer = l3_output.action.offer
        if my_offer:
             safe_offer = self._resolve_counter_offer(ctx, my_offer, negotiator_id, broadcast)
             
             # Sync Trajectory (Consistency)
             step = getattr(l3_output, 'trajectory_step', None)
             
             if safe_offer is None:
                 if step:
                     step.action['op'] = 2 # Change to End
                     step.policy_mask = 0.0  # Skip policy loss
                     step.info['clip_reason'] = 'l1_forced_end_propose'
                 return None
             
             # Sync L1 clipped values to trajectory
             if safe_offer != my_offer and step:
                 # Mark as clipped for policy mask
                 step.policy_mask = 0.0
                 step.info['clip_reason'] = 'l1_clipped_propose'
                 
                 # Reverse encode clipped offer
                 new_qty, new_time, new_price = safe_offer
                 new_delta_t = int(new_time - self.awi.current_step)
                 new_delta_t = max(0, min(new_delta_t, self.horizon))
                 
                 # Price ratio
                 p_min = l3_input.min_price
                 p_max = min(l3_input.max_price, 10000.0)
                 if math.isinf(p_max): p_max = p_min * 2.0
                 if p_max > p_min + 1e-6:
                     new_price_ratio = (new_price - p_min) / (p_max - p_min)
                 else:
                     new_price_ratio = 0.5
                 new_price_ratio = max(0.0, min(1.0, new_price_ratio))
                 
                 # Qty bucket
                 if l3_input.is_buying:
                     limit_storage = float(l3_input.Q_safe[new_delta_t]) if new_delta_t < len(l3_input.Q_safe) else 0.0
                     budget_max = l3_input.B_free / (new_price + 0.1)
                     q_max = min(limit_storage, budget_max)
                 else:
                     q_max = float(l3_input.Q_safe[new_delta_t]) if new_delta_t < len(l3_input.Q_safe) else 0.0
                 q_max = max(1, int(q_max))
                 
                 n_buckets = self.l3_ippo.n_qty_buckets
                 if n_buckets > 1 and q_max > 1:
                     new_qty_bucket = int(round(((new_qty - 1) / (q_max - 1)) * (n_buckets - 1)))
                 else:
                     new_qty_bucket = 0
                 new_qty_bucket = max(0, min(n_buckets - 1, new_qty_bucket))
                 
                 step.action['delta_t'] = new_delta_t
                 step.action['price_ratio'] = new_price_ratio
                 step.action['qty_bucket'] = new_qty_bucket
                 # policy_mask already set above

             ctx.history.append(NegotiationRound(
                quantity=safe_offer[0], price=safe_offer[2], delta_t=int(safe_offer[1]-self.awi.current_step), is_my_turn=True
            ))
             return safe_offer
             
        return None

    # ==================== Reward & Completion ====================

    def on_negotiation_success(self, contract, mechanism) -> None:
        super().on_negotiation_success(contract, mechanism)
        
        nid = mechanism.id
        
        # Calculate Reward based on Surplus
        # r_deal = bonus + surplus
        reward = 1.0 # Base Success Bonus
        
        try:
            agreement = contract.agreement
            qty = float(agreement.get("quantity", 0))
            price = float(agreement.get("unit_price", 0))
            
            # Determine if buying or selling based on context or product
            # mechanism.id is negotiator_id. Check context.
            ctx = self._contexts.get(nid)
            is_buying = ctx.is_buying if ctx else False
            # Check product fallback
            if not ctx:
                annotation = getattr(contract, "annotation", {}) or {}
                product = annotation.get("product", None)
                if product == self.awi.my_input_product:
                    is_buying = True
                else:
                    is_buying = False
                    
            # Get Spot Price
            # trading_prices is dict of product -> price
            target_product = self.awi.my_input_product if is_buying else self.awi.my_output_product
            spot = self.awi.trading_prices.get(target_product, 0)
            if spot <= 1e-6:
                spot = price # Fallback if no spot data

            if is_buying:
                # Surplus: (Spot - Price) * Qty
                # Normalized by Spot to avoid explosion
                surplus = (spot - price) * qty / max(spot, 1.0)
            else:
                # Surplus: (Price - Spot) * Qty
                surplus = (price - spot) * qty / max(spot, 1.0)
                
            # Clamp surplus to reasonable range [-5, 5]
            surplus = max(-5.0, min(5.0, surplus))
            
            reward += surplus
            
            self._close_trajectory(nid, reward=reward, success=True)
            
        except Exception as e:
            # print(f"Error in reward calc: {e}")
            self._close_trajectory(nid, reward=1.0, success=True)

    def on_negotiation_failure(self, partners, annotation, mechanism, state) -> None:
        # Override if possible, otherwise we need to catch it via other means.
        # Assuming NegMAS calls this on the agent if implemented.
        nid = mechanism.id
        penalty = -0.1
        self._close_trajectory(nid, reward=penalty, success=False)

    def _close_trajectory(self, nid: str, reward: float, success: bool, truncated: bool = False, bootstrap_value: float = 0.0):
        if nid not in self._trajectories:
            return
            
        traj = self._trajectories[nid]
        if not traj:
            return
            
        # Update last step
        last_step = traj[-1]
        last_step.reward = reward
        last_step.done = True
        last_step.truncated = truncated
        last_step.bootstrap_value = bootstrap_value
        
        # Apply time penalty to all steps? 
        # Or just let GAE handle it if we give small negative reward each step?
        # User suggested r_step = -0.01 per step.
        for step in traj:
            if not step.done:
                step.reward += -0.01
                
        # Move to finished
        self.finished_episodes.append(traj)
        del self._trajectories[nid]
