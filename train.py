#!/usr/bin/env python3
"""
PPO training for autoregressive ARC grid generation.

Trains a Transformer policy to generate ARC output grids cell-by-cell.

Usage:
    # Full training on all 400 tasks:
    python train.py

    # Single-task test:
    python train.py --single-task 52 --max-episodes 500
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

from arc_env.gen_env import ARCGenEnv, PHASE_H, PHASE_W, PHASE_CELL, MAX_GRID
from arc_policy import (
    ARCGridPolicy, tokenize_context, build_gen_tokens, build_gen_input_for_step,
    GEN_START_TOKEN, SIZE_BASE, PAD_TOKEN,
)


@dataclass
class EpisodeData:
    """Stores one episode's rollout data."""
    task_id: str
    ctx_tokens: list[int]
    ctx_row: list[int]
    ctx_col: list[int]
    ctx_grid: list[int]
    actions: list[int] = field(default_factory=list)
    phases: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    gen_w: int = 0
    solved: bool = False


def collect_episode(
    env: ARCGenEnv,
    policy: ARCGridPolicy,
    device: torch.device,
    task_index: int | None = None,
) -> EpisodeData:
    """Run one episode and collect trajectory data."""
    options = {"task_index": task_index} if task_index is not None else {}
    obs, info = env.reset(options=options)

    ctx_tok, ctx_row, ctx_col, ctx_grid = tokenize_context(obs)
    ep = EpisodeData(
        task_id=info["task_id"],
        ctx_tokens=ctx_tok,
        ctx_row=ctx_row,
        ctx_col=ctx_col,
        ctx_grid=ctx_grid,
    )

    policy.eval()
    policy.clear_cache()

    actions_so_far: list[int] = []
    done = False

    while not done:
        phase = obs["phase"]
        gen_w = obs["gen_w"] if obs["gen_w"] > 0 else 1

        with torch.no_grad():
            action, lp, ent, val = policy.forward_step(
                obs, actions_so_far, phase, gen_w, device,
            )

        action_int = action.item()
        obs, reward, terminated, truncated, info = env.step(action_int)
        done = terminated or truncated

        actions_so_far.append(action_int)
        ep.actions.append(action_int)
        ep.phases.append(phase)
        ep.rewards.append(reward)
        ep.log_probs.append(lp.item())
        ep.values.append(val.item())

        if phase == PHASE_W:
            ep.gen_w = action_int + 1

    ep.solved = info.get("reward_breakdown", {}).get("solved", False)
    return ep


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns for a single episode."""
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)
    last_gae = 0.0
    for t in reversed(range(T)):
        nv = values[t + 1] if t + 1 < T else 0.0
        mask = 0.0 if t == T - 1 else 1.0
        delta = rewards[t] + gamma * nv * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def ppo_update(
    policy: ARCGridPolicy,
    optimizer: optim.Optimizer,
    episodes: list[EpisodeData],
    device: torch.device,
    n_epochs: int = 4,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    gamma: float = 0.99,
    gae_lam: float = 0.95,
) -> dict:
    """Run PPO update on a batch of episodes."""
    policy.train()

    # Compute GAE for each episode
    all_advantages = []
    all_returns = []
    for ep in episodes:
        r = torch.tensor(ep.rewards, dtype=torch.float32, device=device)
        v = torch.tensor(ep.values, dtype=torch.float32, device=device)
        adv, ret = compute_gae(r, v, gamma, gae_lam)
        all_advantages.append(adv)
        all_returns.append(ret)

    # Flatten across episodes for mini-batch processing
    # But for Transformer, we need per-episode context, so process per-episode
    total_pg_loss = 0.0
    total_vf_loss = 0.0
    total_entropy = 0.0
    total_steps = 0

    for epoch in range(n_epochs):
        for i, ep in enumerate(episodes):
            adv = all_advantages[i]
            ret = all_returns[i]
            old_lp = torch.tensor(ep.log_probs, dtype=torch.float32, device=device)
            actions = torch.tensor(ep.actions, dtype=torch.long, device=device)
            phases_t = torch.tensor(ep.phases, dtype=torch.long, device=device)

            # Normalize advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            adv = adv.clamp(-4.0, 4.0)

            # Build context tensors
            ctx_tok = torch.tensor([ep.ctx_tokens], dtype=torch.long, device=device)
            ctx_row = torch.tensor([ep.ctx_row], dtype=torch.long, device=device)
            ctx_col = torch.tensor([ep.ctx_col], dtype=torch.long, device=device)
            ctx_grid = torch.tensor([ep.ctx_grid], dtype=torch.long, device=device)

            # Build decoder input: [GEN_START, action_tok_0, ..., action_tok_{n-2}]
            gen_tok_list = [GEN_START_TOKEN]
            gen_row_list = [0]
            gen_col_list = [0]
            gen_grid_list = [21]

            gen_w = ep.gen_w if ep.gen_w > 0 else 1
            for j, a in enumerate(ep.actions[:-1]):
                if j < 2:
                    gen_tok_list.append(SIZE_BASE + a)
                    gen_row_list.append(0)
                    gen_col_list.append(0)
                else:
                    cell_idx = j - 2
                    r = cell_idx // gen_w
                    c = cell_idx % gen_w
                    gen_tok_list.append(a % 10)
                    gen_row_list.append(r)
                    gen_col_list.append(c)
                gen_grid_list.append(21)

            gen_tok = torch.tensor([gen_tok_list], dtype=torch.long, device=device)
            gen_row = torch.tensor([gen_row_list], dtype=torch.long, device=device)
            gen_col = torch.tensor([gen_col_list], dtype=torch.long, device=device)
            gen_grid = torch.tensor([gen_grid_list], dtype=torch.long, device=device)

            new_lp, ent, vals = policy.evaluate_episode(
                ctx_tok, ctx_row, ctx_col, ctx_grid,
                gen_tok, gen_row, gen_col, gen_grid,
                actions.unsqueeze(0), phases_t.unsqueeze(0),
            )
            new_lp = new_lp.squeeze(0)
            ent = ent.squeeze(0)
            vals = vals.squeeze(0)

            if torch.isnan(new_lp).any():
                continue

            ratio = (new_lp - old_lp.detach()).exp().clamp(0.01, 100.0)
            s1 = ratio * adv
            s2 = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv
            pg_loss = -torch.min(s1, s2).mean()
            vf_loss = ((vals - ret.detach()) ** 2).mean()
            ent_bonus = ent.mean()

            loss = pg_loss + vf_coef * vf_loss - entropy_coef * ent_bonus
            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            total_pg_loss += pg_loss.item()
            total_vf_loss += vf_loss.item()
            total_entropy += ent_bonus.item()
            total_steps += 1

    n = max(total_steps, 1)
    return {
        "pg_loss": total_pg_loss / n,
        "vf_loss": total_vf_loss / n,
        "entropy": total_entropy / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Train ARC grid generation policy")
    parser.add_argument("--data-dir", type=str,
                        default=str(Path(__file__).parent / "ARC-AGI" / "data" / "training"))
    parser.add_argument("--single-task", type=int, default=None,
                        help="Train on a single task index (for testing)")
    parser.add_argument("--max-episodes", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Episodes per PPO update")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-ppo-epochs", type=int, default=4)
    parser.add_argument("--num-augments", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-enc-layers", type=int, default=4)
    parser.add_argument("--n-dec-layers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log every N batches")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    env = ARCGenEnv(
        data_dir=args.data_dir,
        num_augments=args.num_augments,
        curriculum=(args.single_task is None),
    )
    print(f"Task pool: {len(env)} tasks", flush=True)

    policy = ARCGridPolicy(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_enc_layers=args.n_enc_layers,
        n_dec_layers=args.n_dec_layers,
    ).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy params: {n_params:,}", flush=True)

    optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    solve_count = 0
    best_reward = -float("inf")
    total_episodes = 0
    start_time = time.time()

    n_batches = args.max_episodes // args.batch_size
    print(f"Starting training: {n_batches} batches x {args.batch_size} episodes", flush=True)

    for batch_idx in range(1, n_batches + 1):
        # Collect batch of episodes
        episodes: list[EpisodeData] = []
        batch_rewards = []
        batch_solved = 0
        batch_steps = 0

        t_collect = time.time()
        for ep_i in range(args.batch_size):
            ep = collect_episode(
                env, policy, device,
                task_index=args.single_task,
            )
            episodes.append(ep)
            batch_rewards.append(sum(ep.rewards))
            batch_steps += len(ep.actions)
            if ep.solved:
                batch_solved += 1
                solve_count += 1
            total_episodes += 1
        collect_sec = time.time() - t_collect

        # PPO update
        t_ppo = time.time()
        metrics = ppo_update(
            policy, optimizer, episodes, device,
            n_epochs=args.n_ppo_epochs,
            entropy_coef=args.entropy_coef,
            gamma=args.gamma,
        )
        ppo_sec = time.time() - t_ppo

        mean_reward = np.mean(batch_rewards)
        if mean_reward > best_reward:
            best_reward = mean_reward

        elapsed = time.time() - start_time
        task_ids = set(ep.task_id for ep in episodes)
        tasks_str = ",".join(sorted(task_ids)[:3])
        if len(task_ids) > 3:
            tasks_str += f"...+{len(task_ids)-3}"
        print(
            f"B{batch_idx:5d} | ep={total_episodes:6d} | "
            f"R={mean_reward:+6.2f} | best={best_reward:+6.2f} | "
            f"solved={batch_solved}/{args.batch_size} | total_solved={solve_count} | "
            f"steps={batch_steps/args.batch_size:.0f} | "
            f"ent={metrics['entropy']:.2f} | "
            f"col={collect_sec:.0f}s ppo={ppo_sec:.0f}s | {elapsed:.0f}s",
            flush=True,
        )

        if batch_solved > 0:
            path = os.path.join(args.checkpoint_dir,
                                f"policy_gen_ep{total_episodes}_s{solve_count}.pt")
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "episode": total_episodes,
                "solve_count": solve_count,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_enc_layers": args.n_enc_layers,
                "n_dec_layers": args.n_dec_layers,
                "single_task": args.single_task,
            }, path)
            print(f"  Saved: {path}", flush=True)

        # Curriculum update
        if env.sampler:
            for ep in episodes:
                env.sampler.update(ep.task_id, ep.solved, sum(ep.rewards))

        # Periodic stats
        if batch_idx % (args.log_interval * 10) == 0 and env.sampler:
            stats = env.sampler.stats()
            print(f"  Curriculum: {stats}", flush=True)

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "policy_gen_final.pt")
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "episode": total_episodes,
        "solve_count": solve_count,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_enc_layers": args.n_enc_layers,
        "n_dec_layers": args.n_dec_layers,
    }, final_path)
    print(f"Final: {final_path} | Solves: {solve_count}/{total_episodes}", flush=True)


if __name__ == "__main__":
    main()
