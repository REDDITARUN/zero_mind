from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class RouterOutput:
    alpha_sft: torch.Tensor
    alpha_rl: torch.Tensor
    route_control: torch.Tensor
    novelty: torch.Tensor
    uncertainty: torch.Tensor
    routing_loss: torch.Tensor


class NoveltyRouter(nn.Module):
    """
    In-between module: controls forward compute and backward objective mixing.
    """

    def __init__(self, dim: int = 256, memory_size: int = 2048) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(dim + 2),
            nn.Linear(dim + 2, dim),
            nn.GELU(),
            nn.Linear(dim, 2),
        )
        self.memory_size = memory_size
        self.memory: Deque[torch.Tensor] = deque(maxlen=memory_size)

    def _memory_similarity(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, D]
        if len(self.memory) == 0:
            return torch.zeros((z.shape[0],), device=z.device)
        mem = torch.stack(list(self.memory), dim=0).to(z.device)
        z_norm = F.normalize(z, dim=-1)
        mem_norm = F.normalize(mem, dim=-1)
        sim = torch.matmul(z_norm, mem_norm.T).amax(dim=-1)
        return sim

    def forward(self, latent_tokens: torch.Tensor) -> RouterOutput:
        pooled = latent_tokens.mean(dim=1)  # [B, D]
        novelty = 1.0 - self._memory_similarity(pooled)
        uncertainty = latent_tokens.var(dim=1).mean(dim=-1)
        gate_in = torch.cat([pooled, novelty[:, None], uncertainty[:, None]], dim=-1)
        logits = self.gate(gate_in)
        alpha = torch.softmax(logits, dim=-1)
        alpha_sft = alpha[:, 0]
        alpha_rl = alpha[:, 1]

        # Route regularization: avoid collapse to one side.
        target = torch.full_like(alpha, 0.5)
        routing_loss = F.mse_loss(alpha, target)

        return RouterOutput(
            alpha_sft=alpha_sft,
            alpha_rl=alpha_rl,
            route_control=alpha,
            novelty=novelty,
            uncertainty=uncertainty,
            routing_loss=routing_loss,
        )

    @torch.no_grad()
    def update_memory(self, latent_tokens: torch.Tensor) -> None:
        pooled = latent_tokens.mean(dim=1).detach().cpu()
        for vec in pooled:
            self.memory.append(vec)
