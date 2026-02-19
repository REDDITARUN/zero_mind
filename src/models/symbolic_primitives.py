from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch


TransformFn = Callable[[torch.Tensor], torch.Tensor]


def _identity(g: torch.Tensor) -> torch.Tensor:
    return g.clone()


def _flip_h(g: torch.Tensor) -> torch.Tensor:
    return torch.flip(g, dims=[1])


def _flip_v(g: torch.Tensor) -> torch.Tensor:
    return torch.flip(g, dims=[0])


def _rot180(g: torch.Tensor) -> torch.Tensor:
    return torch.rot90(g, k=2, dims=[0, 1])


def _translate_right(g: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(g)
    out[:, 1:] = g[:, :-1]
    return out


def _translate_down(g: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(g)
    out[1:, :] = g[:-1, :]
    return out


def primitive_bank() -> Dict[str, TransformFn]:
    return {
        "identity": _identity,
        "flip_h": _flip_h,
        "flip_v": _flip_v,
        "rot180": _rot180,
        "translate_right": _translate_right,
        "translate_down": _translate_down,
    }


@dataclass
class SymbolicResult:
    rule_name: str
    confidence: float
    grid: torch.Tensor


def _pair_accuracy(inp: torch.Tensor, out: torch.Tensor, fn: TransformFn) -> float:
    if inp.shape != out.shape:
        return 0.0
    pred = fn(inp)
    return float((pred == out).float().mean().item())


def fit_symbolic_rule(
    train_inputs: List[torch.Tensor],
    train_outputs: List[torch.Tensor],
    test_input: torch.Tensor,
) -> SymbolicResult:
    bank = primitive_bank()
    best_rule = "none"
    best_score = 0.0
    best_grid = test_input.clone()
    for name, fn in bank.items():
        scores = [_pair_accuracy(i, o, fn) for i, o in zip(train_inputs, train_outputs)]
        mean_score = sum(scores) / max(1, len(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_rule = name
            best_grid = fn(test_input)
    return SymbolicResult(rule_name=best_rule, confidence=best_score, grid=best_grid)
