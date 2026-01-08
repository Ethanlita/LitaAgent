"""L3 IPPO Actor-Critic Implementation."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.distributions as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    dist = None

if TORCH_AVAILABLE:
    class L3ActorCriticIPPO(nn.Module):
        """L3 Actor-Critic for IPPO training.
        
        Outputs:
            - op: Categorical(3) [ACCEPT, REJECT, END]
            - time: Categorical(H+1)
            - price: Beta(alpha, beta) -> mapped to [min_price, max_price]
            - quantity: Categorical(N_qty_buckets) -> mapped to [1, Q_max]
            - value: Scalar
        """

        def __init__(
            self,
            horizon: int = 40,
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 4,
            max_seq_len: int = 20,
            context_dim: int = 31,
            n_qty_buckets: int = 10,
        ):
            super().__init__()
            self.horizon = horizon
            self.max_seq_len = max_seq_len
            self.n_qty_buckets = n_qty_buckets

            # Backbone (Reused from L3DecisionTransformer)
            self.history_embed = nn.Linear(4, d_model)
            self.context_embed = nn.Linear(context_dim, d_model)
            self.pos_embed = nn.Embedding(max_seq_len, d_model)
            
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
            )
            self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

            # Actor Heads
            self.op_head = nn.Linear(d_model, 3)     # [ACCEPT, REJECT, END]
            self.time_head = nn.Linear(d_model, horizon + 1)
            
            # Price: Output concentration parameters (alpha, beta) for Beta distribution
            # Softplus to ensure positivity + epsilon for stability
            self.price_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 2),
                nn.Softplus(), 
            )

            # Quantity: Discrete logits for buckets
            self.qty_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, n_qty_buckets),
            )

            # Critic Head (Central Value)
            # Uses the same backbone features (Shared parameter architecture)
            self.value_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1)
            )

        def forward(
            self,
            history: torch.Tensor,
            context: torch.Tensor,
            time_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Args:
                history: (B, T, 4)
                context: (B, context_dim)
                time_mask: (B, H+1) - optional mask for time logits

            Returns:
                op_logits: (B, 3)
                time_logits: (B, H+1)
                price_params: (B, 2)
                qty_logits: (B, N_buckets)
                value: (B, 1)
            """
            B, T, _ = history.shape
            if T > self.max_seq_len:
                history = history[:, -self.max_seq_len :, :]
                T = history.shape[1]

            h_tokens = self.history_embed(history)
            positions = torch.arange(T, device=h_tokens.device).unsqueeze(0).expand(B, T)
            h_tokens = h_tokens + self.pos_embed(positions)

            memory = self.context_embed(context).unsqueeze(1) # (B, 1, d_model)
            
            # Causal mask for history
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(h_tokens.device)
            
            # (B, T, d_model)
            feat = self.transformer(h_tokens, memory, tgt_mask=causal_mask)
            
            # Use last token embedding for prediction
            last_feat = feat[:, -1, :]

            # Heads
            op_logits = self.op_head(last_feat)
            
            time_logits = self.time_head(last_feat)
            if time_mask is not None:
                # time_mask should be logits mask (0 or -inf)
                time_logits = time_logits + time_mask

            price_params = self.price_head(last_feat) + 1.001  # Ensure alpha, beta > 1 for bell-shapes/stability
            qty_logits = self.qty_head(last_feat)
            value = self.value_head(last_feat)

            return op_logits, time_logits, price_params, qty_logits, value

        def act(
            self, 
            obs: Dict[str, Any], 
            deterministic: bool = False
        ) -> Tuple[Dict[str, Any], float, float, Dict[str, Any]]:
            """
            Execute policy to get action, log_prob, and value.
            
            Args:
                obs: Dict containing 'history', 'context', 'time_mask', 'has_offer'
                    'history': np.array or torch.Tensor
                    'context': np.array or torch.Tensor
                    'time_mask': np.array or torch.Tensor
                    'has_offer': bool (python scalar) or Tensor
                deterministic: If True, take argmax/mean.

            Returns:
                action_dict: {
                    'op': int (0=ACCEPT, 1=REJECT, 2=END),
                    'delta_t': int,
                    'price_ratio': float (0~1),
                    'qty_bucket': int (0~N-1)
                }
                log_prob: float (sum of log_probs of chosen branches)
                value: float
                aux_info: Dict with logits/dists for logging
            """
            self.eval() # Inference mode usually
            
            # 1. Prepare Inputs
            device = next(self.parameters()).device
            
            history = torch.as_tensor(obs['history'], dtype=torch.float32, device=device)
            if history.ndim == 2: history = history.unsqueeze(0) # (1, T, 4)
            
            context = torch.as_tensor(obs['context'], dtype=torch.float32, device=device)
            if context.ndim == 1: context = context.unsqueeze(0) # (1, D)

            time_mask = torch.as_tensor(obs['time_mask'], dtype=torch.float32, device=device)
            if time_mask.ndim == 1: time_mask = time_mask.unsqueeze(0) # (1, H+1)
            
            has_offer = obs.get('has_offer', True) # Default to True if not provided?
            # If has_offer is batch, handle it. Assuming batch=1 for act().

            # 2. Forward
            with torch.no_grad():
                op_logits, time_logits, price_params, qty_logits, value = self.forward(history, context, time_mask)

            # 3. Masking OP (If no offer, cannot ACCEPT)
            # ACCEPT is index 0. If has_offer is False, mask index 0.
            if not has_offer:
                op_logits[0, 0] = -1e9

            # 4. Sampling
            
            # --- OP ---
            op_dist = dist.Categorical(logits=op_logits)
            if deterministic:
                op = torch.argmax(op_logits, dim=-1)
            else:
                op = op_dist.sample()
            op_logp = op_dist.log_prob(op)

            # --- Time ---
            # time_logits is already masked in forward
            time_dist = dist.Categorical(logits=time_logits)
            if deterministic:
                delta_t = torch.argmax(time_logits, dim=-1)
            else:
                delta_t = time_dist.sample()
            time_logp = time_dist.log_prob(delta_t)

            # --- Price (Beta) ---
            # params shape (B, 2) -> alpha, beta
            alpha, beta = price_params[:, 0], price_params[:, 1]
            price_dist = dist.Beta(alpha, beta)
            if deterministic:
                price_ratio = (alpha - 1) / (alpha + beta - 2 + 1e-6) # Mode of Beta(a>1, b>1)
                price_ratio = torch.clamp(price_ratio, 0.0, 1.0)
                price_ratio = torch.where((alpha<=1)|(beta<=1), alpha/(alpha+beta), price_ratio) # Fallback to mean
            else:
                price_ratio = price_dist.sample()
            price_logp = price_dist.log_prob(price_ratio)

            # --- Quantity ---
            qty_dist = dist.Categorical(logits=qty_logits)
            if deterministic:
                qty_bucket = torch.argmax(qty_logits, dim=-1)
            else:
                qty_bucket = qty_dist.sample()
            qty_logp = qty_dist.log_prob(qty_bucket)

            # 5. Composite Logic
            # The user suggested: op -> if reject -> delta_t -> price -> ...
            # But standard PPO assumes independence or simple conditional.
            # If we mask "accept", we effectively only choose from reject/end.
            # If op == ACCEPT (0), then time/price/qty are irrelevant for the LOG PROB?
            # Actually, if op == ACCEPT, the action is just "ACCEPT". The offer params are fixed from input.
            # If op == END (2), action is END. 
            # Only if op == REJECT (1), we use the other branches.
            # But to keep tensor shape consistent for batch training, we usually compute log_prob for all heads
            # and then mask the loss. 
            # Or, we define log_prob = logp(op) + (logp(time)+logp(price)+logp(qty) if op==REJECT else 0)
            
            is_reject = (op == 1)
            total_logp = op_logp
            # Only add counter-offer logprobs if we rejected (and thus made a counter-offer)
            # This makes the probability valid for the composite action space.
            if is_reject.item():
                total_logp = total_logp + time_logp + price_logp + qty_logp

            action_dict = {
                'op': op.item(),
                'delta_t': delta_t.item(),
                'price_ratio': price_ratio.item(),
                'qty_bucket': qty_bucket.item()
            }
            
            return action_dict, total_logp.item(), value.item(), {}

        def evaluate_actions(
            self,
            obs: Dict[str, torch.Tensor],
            actions: Dict[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Evaluate a batch of actions for PPO update.
            
            Args:
                obs: Batch dict of tensors
                actions: Batch dict of tensors {op, delta_t, price_ratio, qty_bucket}
                
            Returns:
                log_prob: (B,)
                entropy: (B,)
                value: (B, 1)
            """
            history = obs['history']
            context = obs['context']
            time_mask = obs['time_mask']
            # has_offer mask needs to be applied to op_logits. 
            # Assume 'has_offer' is a boolean tensor in obs or we handle it via data?
            # For simplicity, if we recorded actions, they are valid relative to has_offer.
            # But to reproduce log_probs, we must mask op_logits same as in act().
            has_offer = obs.get('has_offer', None)

            op_logits, time_logits, price_params, qty_logits, value = self.forward(history, context, time_mask)

            # Mask OP
            if has_offer is not None:
                # has_offer shape (B,) or (B,1)
                mask_val = -1e9
                # If has_offer is False (0), mask index 0 (ACCEPT)
                # op_logits[:, 0] = op_logits[:, 0].masked_fill(~has_offer.bool().squeeze(), mask_val)
                # Ensure shapes match
                if has_offer.ndim == 1:
                    no_offer = ~has_offer.bool()
                    op_logits[no_offer, 0] = mask_val
            
            # Distributions
            op_dist = dist.Categorical(logits=op_logits)
            time_dist = dist.Categorical(logits=time_logits)
            alpha, beta = price_params[:, 0], price_params[:, 1]
            price_dist = dist.Beta(alpha, beta)
            qty_dist = dist.Categorical(logits=qty_logits)

            # Log Probs
            op = actions['op'] # (B,)
            op_logp = op_dist.log_prob(op)

            # For conditional branches, we only care about their logp if op == REJECT (1)
            # But technically PPO usually trains all heads if there are gradients flow? 
            # Or we mask the loss. 
            # Here let's compute the joint log_prob of the *executed* action.
            
            is_reject = (op == 1)
            
            # Time/Price/Qty are only relevant for counter-offers
            time_logp = time_dist.log_prob(actions['delta_t'])
            
            # CRITICAL: Clamp price_ratio to avoid NaN from Beta.log_prob at boundaries
            # Beta distribution is undefined at 0 and 1 for certain alpha/beta values
            price_ratio_clamped = torch.clamp(actions['price_ratio'], 1e-4, 1.0 - 1e-4)
            price_logp = price_dist.log_prob(price_ratio_clamped)
            
            qty_logp = qty_dist.log_prob(actions['qty_bucket'])
            
            # Total Log Prob
            # If accept/end, logp = op_logp
            # If reject, logp = op_logp + time + price + qty
            # Use torch.where instead of multiplication to avoid 0 * -inf = NaN
            branch_logp = time_logp + price_logp + qty_logp
            total_logp = op_logp + torch.where(
                is_reject, 
                branch_logp, 
                torch.zeros_like(branch_logp)
            )

            # Entropy
            # Composite entropy is tricky. simplest is sum of all entropies, 
            # but arguably we should only count entropy of active branches.
            # H(op) + P(reject) * (H(time) + H(price) + H(qty))
            op_probs = op_dist.probs # (B, 3)
            p_reject = op_probs[:, 1]
            
            branch_entropy = time_dist.entropy() + price_dist.entropy() + qty_dist.entropy()
            total_entropy = op_dist.entropy() + (p_reject * branch_entropy)

            return total_logp, total_entropy, value

else:
    class L3ActorCriticIPPO:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for L3ActorCriticIPPO")
