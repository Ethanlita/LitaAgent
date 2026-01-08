"""
IPPO Training Runner for HRL-XF (Standard Track).

Features:
- Warmup Imitation (Supervised Learning from Heuristic/BC)
- IPPO Training (Clipping, GAE, Entropy)
- done-aware GAE
- Serial Rollout (Stable)
"""

import os
import sys
import argparse
import random
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Add Root to Path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scml.std.agents import GreedyStdAgent, RandomStdAgent, SyncRandomStdAgent
from scml.utils import (
    DefaultAgentsStd,
    anac2024_std_world_generator,
    anac2024_config_generator_std,
    anac_assigner_std,
)

from litaagent_std.hrl_xf.agent_ippo import LitaAgentHRLIPPOTrain, TrajectoryStep
from litaagent_std.hrl_xf.l3_ippo import L3ActorCriticIPPO

# Config
class TrainConfig:
    # Env
    n_steps = (50, 200) # Official range by default
    n_worlds_per_update = 8 # Rollout size
    n_competitors_per_world = 4
    max_worlds_per_config = 1
    fair_assign = False
    
    # Warmup
    warmup_epochs = 3
    warmup_worlds = 4
    
    # PPO
    lr = 3e-4
    gamma = 0.98
    gae_lambda = 0.95
    clip_eps = 0.2
    entropy_coef = 0.01
    value_coef = 0.5
    max_grad_norm = 0.5
    ppo_epochs = 4
    batch_size = 256
    total_updates = 100
    
    # Checkpoint
    save_dir = "training_runs/ippo_v1"
    log_interval = 1
    
    # Pretrained BC Weights
    bc_model_path = None # Set to path of pretrained BC model if available (e.g. "results/bc_model.pt")
    l2_model_path = None
    l3_bc_model_path = None

def compute_gae(trajectories, gamma, lam):
    """
    Compute Generalized Advantage Estimation (GAE).
    Input: List of Trajectories (List[TrajectoryStep])
    Returns: List of processed samples with 'adv', 'ret', and 'policy_mask'
    """
    samples = []
    
    for traj in trajectories:
        if not traj:
            continue
            
        # Extract arrays
        rewards = [step.reward for step in traj]
        values = [step.value for step in traj]
        dones = [step.done for step in traj]
        truncated = [getattr(step, 'truncated', False) for step in traj]
        policy_masks = [getattr(step, 'policy_mask', 1.0) for step in traj]
        
        advantages = []
        last_gae_lam = 0
        
        steps_len = len(traj)
        for t in reversed(range(steps_len)):
            # GAE Logic with proper truncation handling
            non_terminal = 1.0
            if dones[t]:
                if truncated[t]:
                    non_terminal = 1.0 # Treat as running for bootstrapping
                else:
                    non_terminal = 0.0 # Truly terminal
            
            if t == steps_len - 1:
                # Last step: use bootstrap_value if truncated, else 0
                if truncated[t]:
                    next_value = getattr(traj[t], 'bootstrap_value', 0.0)
                else:
                    next_value = 0.0
            else:
                next_value = values[t+1]
            
            # Delta computation
            if dones[t] and not truncated[t]:
                # Natural termination: V_next contribution is 0
                delta = rewards[t] - values[t]
            else:
                # Running or truncated: bootstrap from next value
                delta = rewards[t] + gamma * next_value - values[t]
            
            # GAE chain propagation
            gae = delta + gamma * lam * non_terminal * last_gae_lam
            advantages.insert(0, gae)
            last_gae_lam = gae
            
        # Compute Returns
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        # Store processed info back to list
        for i, step in enumerate(traj):
            sample = {
                'obs': step.obs,
                'action': step.action,
                'log_prob': step.log_prob,
                'value': step.value,
                'return': returns[i],
                'advantage': advantages[i],
                'policy_mask': policy_masks[i]
            }
            samples.append(sample)
            
    return samples

class IPPODataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    # Batch is list of dicts. We need to stack tensors.
    # Obs is dict of arrays.
    
    obs_list = [b['obs'] for b in batch]
    # Stack history: (B, T, 4) -> Pad sequence?
    # History is array.
    histories = [torch.from_numpy(o['history']).float() for o in obs_list]
    # Pad history
    histories_padded = torch.nn.utils.rnn.pad_sequence(histories, batch_first=True)
    
    contexts = torch.stack([torch.from_numpy(o['context']).float().squeeze() for o in obs_list])
    time_masks = torch.stack([torch.from_numpy(o['time_mask']).float().squeeze() for o in obs_list])
    has_offers = torch.tensor([o['has_offer'] for o in obs_list])
    
    # Actions
    actions = [b['action'] for b in batch]
    op = torch.tensor([a['op'] for a in actions])
    delta_t = torch.tensor([a['delta_t'] for a in actions])
    price_ratio = torch.tensor([a['price_ratio'] for a in actions])
    qty_bucket = torch.tensor([a['qty_bucket'] for a in actions])
    
    old_log_probs = torch.tensor([b['log_prob'] for b in batch]).float()
    returns = torch.tensor([b['return'] for b in batch]).float()
    advantages = torch.tensor([b['advantage'] for b in batch]).float()
    policy_masks = torch.tensor([b.get('policy_mask', 1.0) for b in batch]).float()
    
    return {
        'obs': {
            'history': histories_padded,
            'context': contexts,
            'time_mask': time_masks,
            'has_offer': has_offers
        },
        'actions': {
            'op': op,
            'delta_t': delta_t,
            'price_ratio': price_ratio,
            'qty_bucket': qty_bucket
        },
        'old_log_probs': old_log_probs,
        'returns': returns,
        'advantages': advantages,
        'policy_masks': policy_masks
    }


def _parse_n_steps(values):
    if values is None:
        return None
    if isinstance(values, (tuple, list)):
        if len(values) == 1:
            return int(values[0])
        if len(values) >= 2:
            return (int(values[0]), int(values[1]))
    return int(values)


def _unwrap_agent(agent):
    obj = getattr(agent, "adapted_object", None)
    if obj is None:
        obj = getattr(agent, "_obj", None)
    return obj if obj is not None else agent


def _collect_training_agents(world, training_agent_type):
    agents = []
    for agent in world.agents.values():
        obj = _unwrap_agent(agent)
        if isinstance(obj, training_agent_type):
            agents.append(obj)
    return agents


def _require_file(path, label):
    if not path:
        raise ValueError(f"{label} path is required but not set")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} path not found: {path}")
    return path


def _find_latest_model(root, pattern):
    root_path = Path(root)
    if not root_path.exists():
        return None
    candidates = list(root_path.rglob(pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def _auto_resolve_bc_paths(cfg: TrainConfig) -> None:
    if cfg.l2_model_path is None:
        cfg.l2_model_path = _find_latest_model("training_runs", "l2_bc.pt")
        if cfg.l2_model_path:
            print(f"Auto-selected L2 BC: {cfg.l2_model_path}")
    if cfg.l3_bc_model_path is None:
        cfg.l3_bc_model_path = _find_latest_model("training_runs", "l3_bc.pt")
        if cfg.l3_bc_model_path:
            print(f"Auto-selected L3 BC: {cfg.l3_bc_model_path}")


def _load_state_dict_strict(model, state_dict, label: str) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        missing_preview = missing[:5]
        unexpected_preview = unexpected[:5]
        raise RuntimeError(
            f"{label} state_dict mismatch. "
            f"missing={missing_preview} (total {len(missing)}), "
            f"unexpected={unexpected_preview} (total {len(unexpected)})"
        )


def _build_training_agent_params(cfg: TrainConfig, horizon: int) -> dict:
    return {
        "mode": "neural",
        "horizon": horizon,
        "l2_model_path": cfg.l2_model_path,
        "l3_model_path": cfg.l3_bc_model_path,
    }


def _validate_training_agents(train_agents, require_neural: bool = True) -> None:
    if not train_agents:
        raise RuntimeError("No training agents found in world (weights not in loop)")
    if not require_neural:
        return
    for agent in train_agents:
        if getattr(agent, "mode", None) != "neural":
            raise RuntimeError("Training agent mode is not neural")
        l2 = getattr(agent, "l2", None)
        l3 = getattr(agent, "l3", None)
        if l2 is None or getattr(l2, "mode", None) != "neural":
            raise RuntimeError("Training agent L2 is not neural")
        if l3 is None or getattr(l3, "mode", None) != "neural":
            raise RuntimeError("Training agent L3 is not neural")


def create_world_anac_style(
    training_agent_type: type,
    n_steps: int = 50,
    n_competitors_per_world: int = 4,
    max_worlds_per_config: int = 1,
    fair_assign: bool = False,
    training_agent_params=None,
) -> tuple:
    """
    Create a World using ANAC 2024 Standard Track official configuration.
    
    This mimics how the actual tournament creates worlds:
    - Uses anac2024_config_generator_std for config generation
    - Uses anac2024_std_world_generator for world creation
    - Our training agent is injected as one of the competitors
    
    Args:
        training_agent_type: The agent class for training (LitaAgentHRLIPPOTrain)
        n_steps: Number of simulation steps (50-200 in official tournament)
        n_competitors_per_world: Number of competitors (default 4)
        training_agent_params: Keyword args injected into the training agent constructor
    
    Returns:
        (world, training_agents): The created world and list of our training agent instances
    """
    # Competitors: our agent + some from default pool
    # In official tournament, DefaultAgentsStd = (GreedyStdAgent, RandomStdAgent, SyncRandomStdAgent)
    default_agents = list(DefaultAgentsStd) if DefaultAgentsStd else [GreedyStdAgent, RandomStdAgent, SyncRandomStdAgent]
    
    # Build competitor list: 1 training agent + (n_competitors-1) defaults
    competitors = [training_agent_type]
    n_others = n_competitors_per_world - 1
    for _ in range(n_others):
        competitors.append(random.choice(default_agents))
    
    # Use official config generator
    # This generates proper multi-level factory layout
    configs = anac2024_config_generator_std(
        n_competitors=len(competitors),
        n_agents_per_competitor=1,  # 1 instance per competitor type
        n_steps=n_steps,
        compact=True,
    )
    
    if not configs:
        raise RuntimeError("anac2024_config_generator_std returned empty config")
    
    # Use official assigner to assign competitors to slots
    # This replaces placeholder positions with actual competitor types
    if fair_assign and max_worlds_per_config < len(competitors):
        max_worlds_per_config = len(competitors)

    params = []
    for i in range(len(competitors)):
        if i == 0:
            params.append(training_agent_params or {})
        else:
            params.append({})

    assigned = anac_assigner_std(
        config=configs,
        max_n_worlds=max_worlds_per_config,
        n_agents_per_competitor=1,
        competitors=competitors,
        params=params,
        fair=fair_assign,
    )
    
    if not assigned:
        raise RuntimeError("anac_assigner_std returned empty assignment")
    
    assignment_set = random.choice(assigned)
    if not assignment_set:
        raise RuntimeError("anac_assigner_std returned empty assignment set")
    assignment = random.choice(assignment_set)
    
    # Create world using official generator
    world = anac2024_std_world_generator(**assignment)
    
    # Find our training agent instances
    training_agents = _collect_training_agents(world, training_agent_type)
    
    return world, training_agents


def _parse_args():
    parser = argparse.ArgumentParser(description="HRL-XF IPPO 训练入口")
    parser.add_argument("--n-steps", nargs="+", type=int, default=None)
    parser.add_argument("--n-worlds-per-update", type=int, default=None)
    parser.add_argument("--n-competitors-per-world", type=int, default=None)
    parser.add_argument("--max-worlds-per-config", type=int, default=None)
    parser.add_argument("--fair-assign", action="store_true")
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--warmup-worlds", type=int, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--bc-model-path", type=str, default=None)
    parser.add_argument("--l2-model-path", type=str, default=None)
    parser.add_argument("--l3-bc-model-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _apply_args(cfg: TrainConfig, args):
    if args.n_steps is not None:
        cfg.n_steps = _parse_n_steps(args.n_steps)
    if args.n_worlds_per_update is not None:
        cfg.n_worlds_per_update = int(args.n_worlds_per_update)
    if args.n_competitors_per_world is not None:
        cfg.n_competitors_per_world = int(args.n_competitors_per_world)
    if args.max_worlds_per_config is not None:
        cfg.max_worlds_per_config = int(args.max_worlds_per_config)
    if args.fair_assign:
        cfg.fair_assign = True
    if args.warmup_epochs is not None:
        cfg.warmup_epochs = int(args.warmup_epochs)
    if args.warmup_worlds is not None:
        cfg.warmup_worlds = int(args.warmup_worlds)
    if args.updates is not None:
        cfg.total_updates = int(args.updates)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.lr is not None:
        cfg.lr = float(args.lr)
    if args.save_dir is not None:
        cfg.save_dir = str(args.save_dir)
    if args.bc_model_path is not None:
        cfg.bc_model_path = str(args.bc_model_path)
    if args.l2_model_path is not None:
        cfg.l2_model_path = str(args.l2_model_path)
    if args.l3_bc_model_path is not None:
        cfg.l3_bc_model_path = str(args.l3_bc_model_path)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)


def train_ippo():
    cfg = TrainConfig()
    args = _parse_args()
    _apply_args(cfg, args)
    _auto_resolve_bc_paths(cfg)
    warmup_enabled = cfg.warmup_worlds > 0 and cfg.warmup_epochs > 0
    
    _require_file(cfg.l2_model_path, "L2 BC")
    if warmup_enabled:
        _require_file(cfg.l3_bc_model_path, "L3 BC")
    elif cfg.l3_bc_model_path:
        _require_file(cfg.l3_bc_model_path, "L3 BC")
    if cfg.bc_model_path:
        _require_file(cfg.bc_model_path, "IPPO init")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    print(f"Starts Training on {device}")
    
    # 1. Initialize Agent & Model
    agent_name = "HRL_IPPO"
    agent = LitaAgentHRLIPPOTrain(
        name=agent_name,
        horizon=40,
        mode="neural",
        l2_model_path=cfg.l2_model_path,
        l3_model_path=cfg.l3_bc_model_path,
    )
    model = agent.l3_ippo.to(device)
    training_agent_params = _build_training_agent_params(cfg, agent.horizon)
    
    # Load BC weights if provided
    if cfg.bc_model_path:
        print(f"Loading BC weights from {cfg.bc_model_path}")
        try:
            state_dict = torch.load(cfg.bc_model_path, map_location=device)
            _load_state_dict_strict(model, state_dict, "IPPO init")
            print("BC weights loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load BC weights: {e}") from e
            
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    # 2. Opponent pool is now handled by create_world_anac_style using DefaultAgentsStd
    
    # ==================== WARMUP PHASE ====================
    print("=== Starting Warmup Imitation Phase (ANAC-style worlds) ===")
    agent.imitation_mode = True
    
    warmup_samples = []
    
    for i in range(cfg.warmup_worlds):
        # Create world using official ANAC 2024 Standard Track configuration
        world, train_agents = create_world_anac_style(
            training_agent_type=LitaAgentHRLIPPOTrain,
            n_steps=cfg.n_steps,
            n_competitors_per_world=cfg.n_competitors_per_world,
            max_worlds_per_config=cfg.max_worlds_per_config,
            fair_assign=cfg.fair_assign,
            training_agent_params=training_agent_params,
        )
        _validate_training_agents(train_agents, require_neural=True)
        
        # Share the model weights with all training agents
        for agent in train_agents:
            agent.l3_ippo = model  # Share same model for gradient accumulation
            agent.imitation_mode = True
        
        world.run()
        
        # Collect trajectories from all training agents
        for agent in train_agents:
            if hasattr(agent, 'flush_trajectories'):
                agent.flush_trajectories(world_step=world.current_step)
            trajs = agent.finished_episodes
            agent.finished_episodes = [] 
        
            # Flatten
            for t in trajs:
                for step in t:
                    # Step.action is target dict
                    warmup_samples.append({
                        'obs': step.obs,
                        'action': step.action,
                        'log_prob': 0.0, 'return': 0.0, 'advantage': 0.0 # Dummy
                    })
        print(f"Collected {len(warmup_samples)} warmup samples from world {i}")

    if warmup_samples:
        warmup_dataset = IPPODataset(warmup_samples)
        warmup_loader = DataLoader(warmup_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
        
        model.train()
        criterion_ce = nn.CrossEntropyLoss()
        
        for epoch in range(cfg.warmup_epochs):
            total_loss = 0
            for batch in warmup_loader:
                obs = {k: v.to(device) for k, v in batch['obs'].items()}
                actions = {k: v.to(device) for k, v in batch['actions'].items()}
                
                op_logits, time_logits, price_params, qty_logits, val = model.forward(
                    obs['history'], obs['context'], obs['time_mask']
                )
                
                loss_op = criterion_ce(op_logits, actions['op']) # op is (B,)
                
                mask_reject = (actions['op'] == 1)
                
                loss_time = torch.tensor(0.0, device=device)
                loss_price = torch.tensor(0.0, device=device)
                loss_qty = torch.tensor(0.0, device=device)
                
                if mask_reject.any():
                    loss_time = criterion_ce(time_logits[mask_reject], actions['delta_t'][mask_reject])
                    loss_qty = criterion_ce(qty_logits[mask_reject], actions['qty_bucket'][mask_reject])
                    
                    alpha, beta = price_params[mask_reject, 0], price_params[mask_reject, 1]
                    dist = torch.distributions.Beta(alpha, beta)
                    target_p = torch.clamp(actions['price_ratio'][mask_reject], 0.001, 0.999)
                    loss_price = -dist.log_prob(target_p).mean()
                
                loss = loss_op + loss_time + loss_price + loss_qty
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Warmup Epoch {epoch}: Loss {total_loss:.4f}")

    # ==================== PPO PHASE ====================
    print("=== Starting PPO Phase (ANAC-style worlds) ===")
    agent.imitation_mode = False
    
    total_updates = cfg.total_updates
    
    for update in range(total_updates):
        trajectories = []
        
        model.eval() 
        
        for _ in range(cfg.n_worlds_per_update):
            # Create world using official ANAC 2024 Standard Track configuration
            world, train_agents = create_world_anac_style(
                training_agent_type=LitaAgentHRLIPPOTrain,
                n_steps=cfg.n_steps,
                n_competitors_per_world=cfg.n_competitors_per_world,
                max_worlds_per_config=cfg.max_worlds_per_config,
                fair_assign=cfg.fair_assign,
                training_agent_params=training_agent_params,
            )
            _validate_training_agents(train_agents, require_neural=True)
            
            # Share the model weights with all training agents
            for agent in train_agents:
                agent.l3_ippo = model  # Share same model
                agent.imitation_mode = False
            
            world.run()
            
            # Collect trajectories from all training agents
            for agent in train_agents:
                if hasattr(agent, 'flush_trajectories'):
                    agent.flush_trajectories(world_step=world.current_step)
                trajectories.extend(agent.finished_episodes)
                agent.finished_episodes = [] 
            
        print(f"Update {update}: Collected {len(trajectories)} trajectories")
        if not trajectories:
            continue
            
        samples = compute_gae(trajectories, cfg.gamma, cfg.gae_lambda)
        if not samples:
            continue
            
        dataset = IPPODataset(samples)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
        
        model.train()
        
        avg_loss = 0
        total_clip_rate = 0
        num_batches = 0
        
        for _ in range(cfg.ppo_epochs):
            for batch in loader:
                obs = {k: v.to(device) for k, v in batch['obs'].items()}
                actions = {k: v.to(device) for k, v in batch['actions'].items()}
                old_log_probs = batch['old_log_probs'].to(device)
                returns = batch['returns'].to(device)
                advantages = batch['advantages'].to(device)
                policy_masks = batch['policy_masks'].to(device)
                
                # Track clip rate (policy_mask=0 means constraint-overridden)
                clip_rate = 1.0 - policy_masks.mean().item()
                total_clip_rate += clip_rate
                num_batches += 1
                
                # Normalize advantages using only policy_mask==1 samples
                valid_mask = policy_masks > 0.5
                valid_count = int(valid_mask.sum().item())
                if valid_count > 1:
                    valid_adv = advantages[valid_mask]
                    adv_mean = valid_adv.mean()
                    adv_std = valid_adv.std() + 1e-8
                    advantages = (advantages - adv_mean) / adv_std
                elif valid_count == 1:
                    # Only one valid sample, skip normalization
                    pass
                else:
                    # No valid samples, skip normalization but keep value loss
                    pass
                
                new_log_probs, entropy, new_values = model.evaluate_actions(obs, actions)
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages
                
                # Policy loss: only on policy_mask==1 samples
                policy_loss_per_sample = -torch.min(surr1, surr2)
                if policy_masks.sum() > 0:
                    policy_loss = (policy_loss_per_sample * policy_masks).sum() / policy_masks.sum()
                else:
                    policy_loss = torch.tensor(0.0, device=device)
                
                # Value loss: use all samples (reward is real)
                value_loss = 0.5 * ((new_values.squeeze() - returns) ** 2).mean()
                
                # Entropy: only on policy_mask==1 samples
                if policy_masks.sum() > 0:
                    entropy_loss = -(entropy * policy_masks).sum() / policy_masks.sum() * cfg.entropy_coef
                else:
                    entropy_loss = torch.tensor(0.0, device=device)
                
                loss = policy_loss + cfg.value_coef * value_loss + entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                
                avg_loss += loss.item()
        
        avg_clip_rate = total_clip_rate / max(num_batches, 1)
        print(f"Update {update} Complete. Avg Loss: {avg_loss / cfg.ppo_epochs:.4f}, Clip Rate: {avg_clip_rate:.2%}")
        
        if (update + 1) % cfg.log_interval == 0:
            torch.save(model.state_dict(), f"{cfg.save_dir}/model_update_{update}.pt")


def train_post_ippo_placeholder():
    raise NotImplementedError("Post-IPPO pipeline is not implemented yet")


if __name__ == "__main__":
    train_ippo()
