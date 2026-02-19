from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn.functional as F


def pad_to_max(grid: torch.Tensor, max_hw: Tuple[int, int], pad_value: int = 0) -> torch.Tensor:
    h, w = grid.shape
    max_h, max_w = max_hw
    return F.pad(grid, (0, max_w - w, 0, max_h - h), value=pad_value)


def infer_max_hw(grids: List[torch.Tensor]) -> Tuple[int, int]:
    max_h = max(x.shape[0] for x in grids)
    max_w = max(x.shape[1] for x in grids)
    return max_h, max_w


def flatten_hw(x: torch.Tensor) -> torch.Tensor:
    # [B, H, W, C] -> [B, H*W, C]
    b, h, w, c = x.shape
    return x.reshape(b, h * w, c)


def unflatten_hw(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # [B, H*W, C] -> [B, H, W, C]
    b, _, c = x.shape
    return x.reshape(b, h, w, c)
