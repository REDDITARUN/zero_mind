#!/usr/bin/env python3
"""
Train an ARC agent on a SINGLE task using PPO until it can solve it.

Uses a simplified discrete action space:
  Discrete(H * W * 10) = all possible PAINT(r,c,color)

The agent learns WHAT to paint. Submission is automatic when the canvas is
correct (this mirrors how most ARC approaches work — model outputs the grid,
submission is a separate step).

Usage:
    python train_single_task.py [--task-index 52] [--max-episodes 3000]
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from arc_env.env import ARCEnv, MAX_GRID

NUM_COLORS = 11  # -1..9 → 0..10 after shift


class SimpleGridEncoder(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(NUM_COLORS, 16)
        self.conv = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, out_dim, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = out_dim

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        g = grid.long().clamp(-1, 9) + 1
        x = self.embed(g).permute(0, 3, 1, 2)
        return self.conv(x).flatten(1)


class SimplePaintPolicy(nn.Module):
    """
    Paint-only policy for proof-of-concept.

    Action space: Discrete(canvas_h * canvas_w * 10)
      Action i →  PAINT(r, c, color) where r,c,color = decode(i)

    Uses a direct one-hot encoding of canvas pixels alongside CNN features
    so the policy can distinguish canvas states on small grids.
    """

    def __init__(self, canvas_h: int, canvas_w: int, grid_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.canvas_h = canvas_h
        self.canvas_w = canvas_w
        self.num_actions = canvas_h * canvas_w * 10

        self.grid_enc = SimpleGridEncoder(out_dim=grid_dim)

        self.pair_proj = nn.Sequential(
            nn.Linear(grid_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 128),
        )

        # Direct canvas encoding: one-hot per cell → linear projection
        # This gives the policy pixel-level awareness of the canvas
        canvas_cells = canvas_h * canvas_w
        self.canvas_direct = nn.Linear(canvas_cells * NUM_COLORS, 64)

        state_dim = 128 + grid_dim + 64 + 4  # rule + test + canvas_direct + scalars
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, self.num_actions)
        self.critic = nn.Linear(hidden_dim, 1)

        self._cached_rule = None
        self._cached_test = None

    def _encode(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        B = obs["canvas"].shape[0]

        if self._cached_rule is not None and not self.training and B == 1:
            rule = self._cached_rule
            test_enc = self._cached_test
        else:
            demo_ins = obs["demo_inputs"][:, :obs["num_demos"].max()]
            demo_outs = obs["demo_outputs"][:, :obs["num_demos"].max()]
            nd = demo_ins.shape[1]
            di_flat = demo_ins.reshape(B * nd, MAX_GRID, MAX_GRID)
            do_flat = demo_outs.reshape(B * nd, MAX_GRID, MAX_GRID)
            di_enc = self.grid_enc(di_flat).reshape(B, nd, -1)
            do_enc = self.grid_enc(do_flat).reshape(B, nd, -1)
            pairs = self.pair_proj(torch.cat([di_enc, do_enc], dim=-1))
            rule = pairs.mean(dim=1)

            test_enc = self.grid_enc(obs["test_input"])

            if not self.training and B == 1:
                self._cached_rule = rule.detach()
                self._cached_test = test_enc.detach()

        # Direct one-hot encoding of the actual canvas pixels (not padded)
        canvas_raw = obs["canvas"][:, :self.canvas_h, :self.canvas_w]
        canvas_flat = (canvas_raw.long().clamp(-1, 9) + 1).reshape(B, -1)
        canvas_onehot = torch.zeros(B, self.canvas_h * self.canvas_w * NUM_COLORS,
                                    device=canvas_flat.device)
        offsets = torch.arange(canvas_flat.shape[1], device=canvas_flat.device) * NUM_COLORS
        canvas_onehot.scatter_(1, (canvas_flat + offsets.unsqueeze(0)), 1.0)
        canvas_enc = self.canvas_direct(canvas_onehot)

        scalars = torch.stack([
            obs["step"].float() / 100.0,
            obs["steps_remaining"].float() / 100.0,
            obs["attempt"].float() / 3.0,
            obs["canvas_h"].float() / 30.0,
        ], dim=-1)

        state = torch.cat([rule, test_enc, canvas_enc, scalars], dim=-1)
        return self.shared(state)

    def clear_cache(self):
        self._cached_rule = None
        self._cached_test = None

    def forward(self, obs):
        h = self._encode(obs)
        logits = self.actor(h)
        logits = torch.clamp(logits, -20.0, 20.0)
        return logits, self.critic(h).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_deterministic_action(self, obs):
        logits, _ = self.forward(obs)
        return logits.argmax(dim=-1)

    def decode_action(self, action_idx: int) -> np.ndarray:
        """Convert discrete action to env action [type, p1, p2, p3, p4, p5]."""
        color = action_idx % 10
        rc = action_idx // 10
        r = rc // self.canvas_w
        c = rc % self.canvas_w
        return np.array([0, r, c, 0, 0, color])  # PAINT


def obs_to_tensors(obs: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "demo_inputs": torch.tensor(obs["demo_inputs"], dtype=torch.long, device=device).unsqueeze(0),
        "demo_outputs": torch.tensor(obs["demo_outputs"], dtype=torch.long, device=device).unsqueeze(0),
        "num_demos": torch.tensor([obs["num_demos"]], dtype=torch.long, device=device),
        "test_input": torch.tensor(obs["test_input"], dtype=torch.long, device=device).unsqueeze(0),
        "canvas": torch.tensor(obs["canvas"], dtype=torch.long, device=device).unsqueeze(0),
        "canvas_h": torch.tensor([obs["canvas_h"]], dtype=torch.float32, device=device),
        "step": torch.tensor([obs["step"]], dtype=torch.float32, device=device),
        "steps_remaining": torch.tensor([obs["steps_remaining"]], dtype=torch.float32, device=device),
        "attempt": torch.tensor([obs["attempt"]], dtype=torch.float32, device=device),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, default=52)
    parser.add_argument("--max-episodes", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Override step limit (0 = auto)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    env = ARCEnv(
        data_dir=Path(__file__).parent / "ARC-AGI" / "data" / "training",
        num_augments=1, show_target=True, curriculum=False,
    )

    obs, info = env.reset(options={"task_index": args.task_index})
    task_id = info["task_id"]
    target = env.target
    th, tw = target.shape
    ch, cw = info["canvas_h"], info["canvas_w"]

    cells = ch * cw
    start_canvas = obs["canvas"][:ch, :cw]
    pixels_to_change = int(np.sum(start_canvas != target))

    # Step limit: just enough room to paint all wrong pixels + a few retries
    if args.max_steps > 0:
        step_limit = args.max_steps
    else:
        step_limit = max(pixels_to_change * 3, 8)

    print(f"Task: {task_id} | Target: {th}×{tw} | Canvas: {ch}×{cw}")
    print(f"Pixels to change: {pixels_to_change}/{cells} | Step limit: {step_limit}")
    print(f"Target:\n{target}")

    if (th != ch) or (tw != cw):
        print("WARNING: target size ≠ canvas. POC works best on same-size tasks.")

    policy = SimplePaintPolicy(ch, cw).to(device)
    print(f"Action space: Discrete({policy.num_actions})  ({ch}×{cw}×10 paint-only)")
    print(f"Params: {sum(p.numel() for p in policy.parameters()):,}")

    optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_reward = -float("inf")
    best_accuracy = 0.0
    solve_count = 0
    start_time = time.time()

    for ep in range(1, args.max_episodes + 1):
        obs, info = env.reset(options={"task_index": args.task_index})
        policy.eval()
        policy.clear_cache()

        obs_list, actions, log_probs, rewards, values = [], [], [], [], []
        step_count = 0
        canvas_perfect = False

        for _ in range(step_limit):
            obs_t = obs_to_tensors(obs, device)
            with torch.no_grad():
                act, lp, _, val = policy.get_action_and_value(obs_t)

            env_action = policy.decode_action(act.item())

            # Track canvas before action to detect wasted paints
            canvas_before = env.active_canvas[:ch, :cw].copy()

            obs, reward, terminated, truncated, info = env.step(env_action)
            step_count += 1

            canvas_after = env.active_canvas[:ch, :cw]
            if np.array_equal(canvas_before, canvas_after):
                reward -= 0.3  # penalize painting a pixel that's already that color

            obs_list.append(obs_t)
            actions.append(act.squeeze(0))
            log_probs.append(lp.squeeze(0))
            rewards.append(reward)
            values.append(val.squeeze(0))

            # Check if canvas is perfect → auto-submit + end episode
            cur_canvas = env.active_canvas[:ch, :cw]
            if np.array_equal(cur_canvas, target):
                canvas_perfect = True
                break

            if terminated or truncated:
                break

        # If canvas is perfect, submit to register the solve
        solved = False
        if canvas_perfect:
            submit_action = np.array([10, 0, 0, 0, 0, 0])
            _, submit_reward, _, _, info = env.step(submit_action)
            solved = info.get("reward_breakdown", {}).get("solved", False)

        # Compute final accuracy for terminal reward
        final_canvas = env.active_canvas[:ch, :cw]
        final_acc = float(np.sum(final_canvas == target)) / cells

        # Terminal reward: big bonus proportional to accuracy
        # Perfect canvas → +3.0, starting accuracy → 0, worse → negative
        start_acc = float(cells - pixels_to_change) / cells
        terminal_reward = (final_acc - start_acc) / (1.0 - start_acc + 1e-8) * 3.0
        if solved:
            terminal_reward += 2.0
        rewards[-1] += terminal_reward

        # GAE
        T = len(rewards)
        r_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        v_t = torch.stack(values)

        advantages = torch.zeros(T, device=device)
        last_gae = 0.0
        for t in reversed(range(T)):
            nv = v_t[t + 1] if t + 1 < T else torch.tensor(0.0, device=device)
            mask = 0.0 if t == T - 1 else 1.0
            delta = r_t[t] + args.gamma * nv * mask - v_t[t]
            last_gae = delta + args.gamma * 0.95 * mask * last_gae
            advantages[t] = last_gae
        returns = advantages + v_t.detach()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.clamp(-4.0, 4.0)

        # PPO update
        policy.train()
        obs_batch = {k: torch.cat([o[k] for o in obs_list]) for k in obs_list[0]}
        act_batch = torch.stack(actions)
        old_lp = torch.stack(log_probs).detach()

        for ppo_iter in range(4):
            _, new_lp, ent, new_v = policy.get_action_and_value(obs_batch, act_batch)

            if torch.isnan(new_lp).any() or torch.isnan(ent).any():
                break

            ratio = (new_lp - old_lp).exp().clamp(0.01, 100.0)
            s1 = ratio * advantages
            s2 = ratio.clamp(0.8, 1.2) * advantages
            pg_loss = -torch.min(s1, s2).mean()
            v_loss = ((new_v - returns) ** 2).mean()

            ent_mean = ent.mean()
            ent_coef = args.entropy_coef
            min_ent = math.log(policy.num_actions) * 0.25
            if ent_mean.item() < min_ent:
                ent_coef *= 3.0

            loss = pg_loss + 0.5 * v_loss - ent_coef * ent_mean
            if torch.isnan(loss):
                break

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        ep_reward = sum(rewards)
        if final_acc > best_accuracy:
            best_accuracy = final_acc
        if ep_reward > best_reward:
            best_reward = ep_reward
        if solved:
            solve_count += 1

        if ep % 20 == 0 or solved:
            elapsed = time.time() - start_time
            tag = " ★ SOLVED!" if solved else ""
            print(
                f"Ep {ep:4d} | R={ep_reward:+6.2f} | best={best_reward:+6.2f} | "
                f"steps={step_count:2d} | acc={final_acc:.2f} | solves={solve_count} | "
                f"ent={ent_mean.item():.2f} | {elapsed:.0f}s{tag}"
            )

        if solved:
            path = os.path.join(args.checkpoint_dir, f"policy_{task_id}_solve{solve_count}.pt")
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "task_id": task_id, "task_index": args.task_index,
                "episode": ep, "reward": ep_reward,
                "canvas_h": ch, "canvas_w": cw,
                "accuracy": final_acc,
            }, path)
            print(f"  Saved: {path}")
            if solve_count >= 10:
                print(f"Solved {solve_count} times — stopping.")
                break

    final = os.path.join(args.checkpoint_dir, f"policy_{task_id}_final.pt")
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "task_id": task_id, "task_index": args.task_index,
        "episode": ep, "total_solves": solve_count,
        "canvas_h": ch, "canvas_w": cw,
    }, final)
    print(f"Final: {final} | Solves: {solve_count}/{ep}")


if __name__ == "__main__":
    main()
