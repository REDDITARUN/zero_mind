from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import random

import torch


Grid = torch.Tensor


@dataclass
class ArcEpisode:
    train_inputs: List[Grid]
    train_outputs: List[Grid]
    test_input: Grid
    test_output: Grid
    rule_name: str


def _random_grid(h: int, w: int, num_colors: int, device: torch.device) -> Grid:
    return torch.randint(low=0, high=num_colors, size=(h, w), device=device, dtype=torch.long)


def _recolor(grid: Grid, num_colors: int) -> Grid:
    perm = torch.randperm(num_colors, device=grid.device)
    return perm[grid]


def _flip_h(grid: Grid) -> Grid:
    return torch.flip(grid, dims=[1])


def _flip_v(grid: Grid) -> Grid:
    return torch.flip(grid, dims=[0])


def _rotate90(grid: Grid) -> Grid:
    return torch.rot90(grid, k=1, dims=[0, 1])


def _invert_colors(grid: Grid, num_colors: int) -> Grid:
    return (num_colors - 1) - grid


def _identity(grid: Grid) -> Grid:
    return grid.clone()


def _translate_right(grid: Grid) -> Grid:
    out = torch.zeros_like(grid)
    out[:, 1:] = grid[:, :-1]
    return out


def _translate_down(grid: Grid) -> Grid:
    out = torch.zeros_like(grid)
    out[1:, :] = grid[:-1, :]
    return out


def _rule_bank(num_colors: int) -> Dict[str, Callable[[Grid], Grid]]:
    return {
        "identity": _identity,
        "flip_h": _flip_h,
        "flip_v": _flip_v,
        "translate_right": _translate_right,
        "translate_down": _translate_down,
        "invert_colors": lambda g: _invert_colors(g, num_colors),
        "recolor": lambda g: _recolor(g, num_colors),
    }


class ArcSimulator:
    """
    Small ARC-like generator for validating unified loop mechanics.
    """

    def __init__(
        self,
        grid_min_size: int = 6,
        grid_max_size: int = 12,
        num_colors: int = 10,
        train_pairs_range: Tuple[int, int] = (2, 3),
        allowed_rules: List[str] | None = None,
        device: torch.device | None = None,
        seed: int = 0,
    ) -> None:
        self.grid_min_size = grid_min_size
        self.grid_max_size = grid_max_size
        self.num_colors = num_colors
        self.train_pairs_range = train_pairs_range
        self.allowed_rules = allowed_rules
        self.device = device or torch.device("cpu")
        self.rng = random.Random(seed)

    def sample_episode(self) -> ArcEpisode:
        rules = _rule_bank(self.num_colors)
        if self.allowed_rules:
            candidate_rules = [r for r in self.allowed_rules if r in rules]
            if not candidate_rules:
                raise ValueError("allowed_rules is set but no valid rules were provided")
        else:
            candidate_rules = list(rules.keys())
        rule_name = self.rng.choice(candidate_rules)
        rule_fn = rules[rule_name]

        k_min, k_max = self.train_pairs_range
        n_train = self.rng.randint(k_min, k_max)

        train_inputs: List[Grid] = []
        train_outputs: List[Grid] = []
        for _ in range(n_train):
            h = self.rng.randint(self.grid_min_size, self.grid_max_size)
            w = self.rng.randint(self.grid_min_size, self.grid_max_size)
            g = _random_grid(h, w, self.num_colors, self.device)
            train_inputs.append(g)
            train_outputs.append(rule_fn(g))

        h = self.rng.randint(self.grid_min_size, self.grid_max_size)
        w = self.rng.randint(self.grid_min_size, self.grid_max_size)
        test_input = _random_grid(h, w, self.num_colors, self.device)
        test_output = rule_fn(test_input)

        return ArcEpisode(
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            test_input=test_input,
            test_output=test_output,
            rule_name=rule_name,
        )
