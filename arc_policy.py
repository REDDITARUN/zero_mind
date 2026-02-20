"""
Transformer policy for autoregressive ARC grid generation.

Architecture:
  Encoder: processes demo pairs + test input tokens (bidirectional self-attention)
  Decoder: generates H, W, cell colors (causal self-attention + cross-attention)

Token vocabulary:
  0-9  : cell colors
  10   : PAD
  11   : DEMO_IN  (separator before a demo input grid)
  12   : DEMO_OUT (separator before a demo output grid)
  13   : TEST_IN  (separator before the test input grid)
  14   : GEN_START (marks start of generation)
  15-44: SIZE_1 .. SIZE_30 (generated dimension tokens)

2D positional encoding: row_embed(r) + col_embed(c) for spatial awareness.
Grid-type embedding: identifies which grid a token belongs to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from arc_env.gen_env import MAX_GRID, MAX_DEMOS, PHASE_H, PHASE_W, PHASE_CELL

# Token IDs
PAD_TOKEN = 10
DEMO_IN_TOKEN = 11
DEMO_OUT_TOKEN = 12
TEST_IN_TOKEN = 13
GEN_START_TOKEN = 14
SIZE_BASE = 15  # SIZE_1=15, SIZE_2=16, ..., SIZE_30=44
VOCAB_SIZE = 45

# Grid type IDs (for grid_embed)
# 0-19: demo pairs (demo_in_0=0, demo_out_0=1, demo_in_1=2, ...)
# 20: test_input
# 21: generated output
NUM_GRID_TYPES = 22


def tokenize_context(obs: dict) -> tuple[list[int], list[int], list[int], list[int]]:
    """Convert demo pairs + test input into token sequences.

    Returns (tokens, row_pos, col_pos, grid_ids) as Python lists.
    """
    tokens, row_pos, col_pos, grid_ids = [], [], [], []
    num_demos = int(obs["num_demos"])

    for d in range(num_demos):
        h_in, w_in = int(obs["demo_input_sizes"][d, 0]), int(obs["demo_input_sizes"][d, 1])
        gid_in = d * 2

        tokens.append(DEMO_IN_TOKEN)
        row_pos.append(0)
        col_pos.append(0)
        grid_ids.append(gid_in)

        for r in range(h_in):
            for c in range(w_in):
                tokens.append(int(obs["demo_inputs"][d, r, c]))
                row_pos.append(r)
                col_pos.append(c)
                grid_ids.append(gid_in)

        h_out, w_out = int(obs["demo_output_sizes"][d, 0]), int(obs["demo_output_sizes"][d, 1])
        gid_out = d * 2 + 1

        tokens.append(DEMO_OUT_TOKEN)
        row_pos.append(0)
        col_pos.append(0)
        grid_ids.append(gid_out)

        for r in range(h_out):
            for c in range(w_out):
                tokens.append(int(obs["demo_outputs"][d, r, c]))
                row_pos.append(r)
                col_pos.append(c)
                grid_ids.append(gid_out)

    ti_h, ti_w = int(obs["test_input_size"][0]), int(obs["test_input_size"][1])
    tokens.append(TEST_IN_TOKEN)
    row_pos.append(0)
    col_pos.append(0)
    grid_ids.append(20)

    for r in range(ti_h):
        for c in range(ti_w):
            tokens.append(int(obs["test_input"][r, c]))
            row_pos.append(r)
            col_pos.append(c)
            grid_ids.append(20)

    return tokens, row_pos, col_pos, grid_ids


def build_gen_tokens(actions: list[int], gen_w: int) -> tuple[list[int], list[int], list[int], list[int]]:
    """Build decoder input from an action sequence.

    Decoder input = [GEN_START, tok_for_action_0, ..., tok_for_action_{n-2}]
    (standard autoregressive: input shifted right by 1 from targets)
    """
    tokens = [GEN_START_TOKEN]
    row_pos = [0]
    col_pos = [0]
    grid_ids = [21]

    for i, a in enumerate(actions[:-1]):
        if i < 2:
            tokens.append(SIZE_BASE + a)
            row_pos.append(0)
            col_pos.append(0)
        else:
            cell_idx = i - 2
            r = cell_idx // max(gen_w, 1)
            c = cell_idx % max(gen_w, 1)
            tokens.append(a % 10)
            row_pos.append(r)
            col_pos.append(c)
        grid_ids.append(21)

    return tokens, row_pos, col_pos, grid_ids


def build_gen_input_for_step(actions_so_far: list[int], gen_w: int) -> tuple[list[int], list[int], list[int], list[int]]:
    """Build decoder input for the current step (during rollout).

    Decoder input = [GEN_START] + tokens for all previous actions.
    """
    tokens = [GEN_START_TOKEN]
    row_pos = [0]
    col_pos = [0]
    grid_ids = [21]

    for i, a in enumerate(actions_so_far):
        if i < 2:
            tokens.append(SIZE_BASE + a)
            row_pos.append(0)
            col_pos.append(0)
        else:
            cell_idx = i - 2
            r = cell_idx // max(gen_w, 1)
            c = cell_idx % max(gen_w, 1)
            tokens.append(a % 10)
            row_pos.append(r)
            col_pos.append(c)
        grid_ids.append(21)

    return tokens, row_pos, col_pos, grid_ids


class ARCGridPolicy(nn.Module):
    """Encoder-decoder Transformer for autoregressive ARC grid generation."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model

        self.token_embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.row_embed = nn.Embedding(MAX_GRID + 1, d_model)
        self.col_embed = nn.Embedding(MAX_GRID + 1, d_model)
        self.grid_embed = nn.Embedding(NUM_GRID_TYPES, d_model)

        self.embed_scale = math.sqrt(d_model)
        self.embed_drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, n_dec_layers)

        self.size_head = nn.Linear(d_model, 30)
        self.color_head = nn.Linear(d_model, 10)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        self._cached_memory = None
        self._cached_memory_mask = None

    def _embed(self, tokens, row_pos, col_pos, grid_ids):
        x = self.token_embed(tokens)
        x = x + self.row_embed(row_pos) + self.col_embed(col_pos) + self.grid_embed(grid_ids)
        return self.embed_drop(x)

    def encode_context(self, ctx_tokens, ctx_row, ctx_col, ctx_grid, ctx_mask=None):
        """Encode context (demo pairs + test input). Returns memory tensor."""
        x = self._embed(ctx_tokens, ctx_row, ctx_col, ctx_grid)
        memory = self.encoder(x, src_key_padding_mask=ctx_mask)
        return memory

    def decode(self, memory, gen_tokens, gen_row, gen_col, gen_grid,
               memory_mask=None, gen_mask=None):
        """Run decoder on generation tokens with cross-attention to memory."""
        x = self._embed(gen_tokens, gen_row, gen_col, gen_grid)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            gen_tokens.size(1), device=gen_tokens.device,
        )
        h = self.decoder(x, memory, tgt_mask=tgt_mask,
                         memory_key_padding_mask=memory_mask)
        return h

    def clear_cache(self):
        self._cached_memory = None
        self._cached_memory_mask = None

    def forward_step(self, obs: dict, actions_so_far: list[int], phase: int,
                     gen_w: int, device: torch.device):
        """Single step during rollout. Caches encoder output.

        Returns: (action, log_prob, entropy, value) tensors of shape (1,).
        """
        if self._cached_memory is None:
            ctx_tok, ctx_row, ctx_col, ctx_grid = tokenize_context(obs)
            ctx_tok = torch.tensor([ctx_tok], dtype=torch.long, device=device)
            ctx_row = torch.tensor([ctx_row], dtype=torch.long, device=device)
            ctx_col = torch.tensor([ctx_col], dtype=torch.long, device=device)
            ctx_grid = torch.tensor([ctx_grid], dtype=torch.long, device=device)
            self._cached_memory = self.encode_context(ctx_tok, ctx_row, ctx_col, ctx_grid)
            self._cached_memory_mask = None

        gen_tok, gen_row, gen_col, gen_grid = build_gen_input_for_step(
            actions_so_far, gen_w,
        )
        gen_tok = torch.tensor([gen_tok], dtype=torch.long, device=device)
        gen_row = torch.tensor([gen_row], dtype=torch.long, device=device)
        gen_col = torch.tensor([gen_col], dtype=torch.long, device=device)
        gen_grid = torch.tensor([gen_grid], dtype=torch.long, device=device)

        h = self.decode(self._cached_memory, gen_tok, gen_row, gen_col, gen_grid,
                        memory_mask=self._cached_memory_mask)
        h_last = h[:, -1, :]

        if phase == PHASE_H or phase == PHASE_W:
            logits = self.size_head(h_last)
            dist = Categorical(logits=logits)
        else:
            logits = self.color_head(h_last)
            dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_head(h_last).squeeze(-1)

        return action, log_prob, entropy, value

    def evaluate_episode(
        self,
        ctx_tokens: torch.Tensor,
        ctx_row: torch.Tensor,
        ctx_col: torch.Tensor,
        ctx_grid: torch.Tensor,
        gen_tokens: torch.Tensor,
        gen_row: torch.Tensor,
        gen_col: torch.Tensor,
        gen_grid: torch.Tensor,
        actions: torch.Tensor,
        phases: torch.Tensor,
        ctx_mask: torch.Tensor | None = None,
        gen_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate all actions in an episode for PPO update.

        All inputs have batch dim (B, ...).
        Returns log_probs, entropy, values — each shape (B, T).
        """
        memory = self.encode_context(ctx_tokens, ctx_row, ctx_col, ctx_grid, ctx_mask)
        h = self.decode(memory, gen_tokens, gen_row, gen_col, gen_grid,
                        memory_mask=ctx_mask)

        B, T, D = h.shape
        size_logits = self.size_head(h)    # (B, T, 30)
        color_logits = self.color_head(h)  # (B, T, 10)

        # Select logits based on phase
        is_size = (phases < 2).unsqueeze(-1)  # (B, T, 1)

        # Pad color logits to match size logits width
        color_padded = F.pad(color_logits, (0, 20), value=-1e9)  # (B, T, 30)
        logits = torch.where(is_size, size_logits, color_padded)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_head(h).squeeze(-1)

        return log_probs, entropy, values

    def get_deterministic_action(self, obs: dict, actions_so_far: list[int],
                                 phase: int, gen_w: int, device: torch.device) -> int:
        """Return argmax action for the current step."""
        if self._cached_memory is None:
            ctx_tok, ctx_row, ctx_col, ctx_grid = tokenize_context(obs)
            ctx_tok = torch.tensor([ctx_tok], dtype=torch.long, device=device)
            ctx_row = torch.tensor([ctx_row], dtype=torch.long, device=device)
            ctx_col = torch.tensor([ctx_col], dtype=torch.long, device=device)
            ctx_grid = torch.tensor([ctx_grid], dtype=torch.long, device=device)
            self._cached_memory = self.encode_context(ctx_tok, ctx_row, ctx_col, ctx_grid)

        gen_tok, gen_row, gen_col, gen_grid_ids = build_gen_input_for_step(
            actions_so_far, gen_w,
        )
        gen_tok = torch.tensor([gen_tok], dtype=torch.long, device=device)
        gen_row = torch.tensor([gen_row], dtype=torch.long, device=device)
        gen_col = torch.tensor([gen_col], dtype=torch.long, device=device)
        gen_grid_ids = torch.tensor([gen_grid_ids], dtype=torch.long, device=device)

        h = self.decode(self._cached_memory, gen_tok, gen_row, gen_col, gen_grid_ids)
        h_last = h[:, -1, :]

        if phase == PHASE_H or phase == PHASE_W:
            return self.size_head(h_last).argmax(dim=-1).item()
        else:
            return self.color_head(h_last).argmax(dim=-1).item()
