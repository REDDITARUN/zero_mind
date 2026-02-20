"""
Policy network for ARC-AGI RL agent.

Architecture:
  SharedGridEncoder (CNN)  →  encodes any 30×30 grid to 128-d
  DemoPairEncoder          →  pairs (enc_in ⊕ enc_out) → hint per pair
  TransformerRuleEncoder   →  self-attention across hints → 256-d rule vector
  StateFusion + ActionHeads →  rule ⊕ enc(test) ⊕ enc(canvas) ⊕ scalars → actions
  ValueHead                →  state → scalar value
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

MAX_GRID = 30
NUM_COLORS = 11  # 0-9 + padding (-1 mapped to 10)
NUM_ACTION_TYPES = 11


class GridEncoder(nn.Module):
    """Encode a 30×30 grid (values -1..9) into a fixed-size vector."""

    def __init__(self, embed_dim: int = 32, out_dim: int = 128):
        super().__init__()
        self.color_embed = nn.Embedding(NUM_COLORS, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, MAX_GRID, MAX_GRID) * 0.02)

        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, out_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = out_dim

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        grid: (B, 30, 30) int8/long, values -1..9
        returns: (B, out_dim)
        """
        g = grid.long().clamp(-1, 9) + 1  # shift -1→0, 0→1, ..., 9→10
        x = self.color_embed(g)            # (B, 30, 30, embed_dim)
        x = x.permute(0, 3, 1, 2)         # (B, embed_dim, 30, 30)
        x = x + self.pos_embed
        x = self.conv(x)                  # (B, out_dim, 1, 1)
        return x.squeeze(-1).squeeze(-1)   # (B, out_dim)


class RuleEncoder(nn.Module):
    """Encode variable-length demo pairs into a fixed-size rule vector."""

    def __init__(self, grid_dim: int = 128, rule_dim: int = 256, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.pair_proj = nn.Sequential(
            nn.Linear(grid_dim * 2, rule_dim),
            nn.ReLU(),
            nn.Linear(rule_dim, rule_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=rule_dim,
            nhead=nhead,
            dim_feedforward=rule_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.rule_dim = rule_dim

    def forward(
        self, demo_in_encs: torch.Tensor, demo_out_encs: torch.Tensor, num_demos: torch.Tensor
    ) -> torch.Tensor:
        """
        demo_in_encs:  (B, MAX_DEMOS, grid_dim)
        demo_out_encs: (B, MAX_DEMOS, grid_dim)
        num_demos:     (B,) int
        returns:       (B, rule_dim)
        """
        B, D, G = demo_in_encs.shape
        pairs = torch.cat([demo_in_encs, demo_out_encs], dim=-1)  # (B, D, grid_dim*2)
        hints = self.pair_proj(pairs)  # (B, D, rule_dim)

        mask = torch.arange(D, device=hints.device).unsqueeze(0) >= num_demos.unsqueeze(1)  # (B, D)
        hints = self.transformer(hints, src_key_padding_mask=mask)

        # Mean pool over valid demos
        mask_float = (~mask).float().unsqueeze(-1)  # (B, D, 1)
        rule = (hints * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        return rule  # (B, rule_dim)


class ARCPolicy(nn.Module):
    """Full actor-critic policy for the ARC environment."""

    def __init__(self, grid_dim: int = 128, rule_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.grid_enc = GridEncoder(embed_dim=32, out_dim=grid_dim)
        self.rule_enc = RuleEncoder(grid_dim=grid_dim, rule_dim=rule_dim)

        state_dim = rule_dim + grid_dim * 2 + 32  # rule + test + canvas + scalars
        self.scalar_embed = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
        )

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Action heads (one per multi-discrete dimension)
        self.head_action_type = nn.Linear(hidden_dim, NUM_ACTION_TYPES)
        self.head_p1 = nn.Linear(hidden_dim, MAX_GRID)
        self.head_p2 = nn.Linear(hidden_dim, MAX_GRID)
        self.head_p3 = nn.Linear(hidden_dim, MAX_GRID)
        self.head_p4 = nn.Linear(hidden_dim, MAX_GRID)
        self.head_p5 = nn.Linear(hidden_dim, 10)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self._action_heads = [
            self.head_action_type, self.head_p1, self.head_p2,
            self.head_p3, self.head_p4, self.head_p5,
        ]

    def clear_cache(self):
        """Clear cached encodings (call when task changes)."""
        self._cached_rule = None
        self._cached_test_enc = None
        self._cached_key = None

    def _encode_state(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode observation dict into a fused state vector.

        Caches demo/rule/test encodings since they don't change within an episode.
        """
        B = obs["test_input"].shape[0]
        device = obs["test_input"].device

        # Cache key: demos + test input don't change within an episode
        cache_key = (id(obs["demo_inputs"].data_ptr()), id(obs["test_input"].data_ptr()))
        use_cache = (
            hasattr(self, "_cached_rule")
            and self._cached_rule is not None
            and hasattr(self, "_cached_key")
            and self._cached_key == cache_key
            and not self.training  # skip cache during PPO update (batched)
        )

        if use_cache:
            rule = self._cached_rule
            test_enc = self._cached_test_enc
        else:
            demo_ins = obs["demo_inputs"]    # (B, 10, 30, 30)
            demo_outs = obs["demo_outputs"]  # (B, 10, 30, 30)
            all_demos = torch.cat([demo_ins, demo_outs], dim=1)  # (B, 20, 30, 30)
            all_flat = all_demos.reshape(B * 20, MAX_GRID, MAX_GRID)
            all_enc = self.grid_enc(all_flat).reshape(B, 20, -1)
            demo_in_enc = all_enc[:, :10]
            demo_out_enc = all_enc[:, 10:]

            num_demos = obs["num_demos"].long()
            rule = self.rule_enc(demo_in_enc, demo_out_enc, num_demos)
            test_enc = self.grid_enc(obs["test_input"])

            if not self.training and B == 1:
                self._cached_rule = rule.detach()
                self._cached_test_enc = test_enc.detach()
                self._cached_key = cache_key

        canvas_enc = self.grid_enc(obs["canvas"])

        scalars = torch.stack([
            obs["step"].float() / 300.0,
            obs["steps_remaining"].float() / 300.0,
            obs["attempt"].float() / 3.0,
            obs["canvas_h"].float() / 30.0,
        ], dim=-1)
        scalar_enc = self.scalar_embed(scalars)

        state = torch.cat([rule, test_enc, canvas_enc, scalar_enc], dim=-1)
        return self.shared(state)

    def get_action_and_value(
        self, obs: dict[str, torch.Tensor], action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (action, log_prob, entropy, value)
        If action is provided, computes log_prob and entropy for that action.
        """
        hidden = self._encode_state(obs)
        value = self.value_head(hidden).squeeze(-1)

        logits_list = [head(hidden) for head in self._action_heads]
        dists = [Categorical(logits=logits) for logits in logits_list]

        if action is None:
            actions = torch.stack([d.sample() for d in dists], dim=-1)
        else:
            actions = action

        log_probs = torch.stack(
            [d.log_prob(actions[:, i]) for i, d in enumerate(dists)], dim=-1
        ).sum(dim=-1)

        entropies = torch.stack([d.entropy() for d in dists], dim=-1).sum(dim=-1)

        return actions, log_probs, entropies, value

    def get_value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self._encode_state(obs)
        return self.value_head(hidden).squeeze(-1)

    def get_deterministic_action(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Greedy action (for evaluation)."""
        hidden = self._encode_state(obs)
        logits_list = [head(hidden) for head in self._action_heads]
        actions = torch.stack([logits.argmax(dim=-1) for logits in logits_list], dim=-1)
        return actions
