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


class ArcSimulator:
    """
    Procedural Generator v2: Robust geometric priors.
    """

    def __init__(
        self,
        grid_min_size: int = 6,
        grid_max_size: int = 18,
        num_colors: int = 10,
        train_pairs_range: Tuple[int, int] = (2, 4),
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

    def _random_grid(self, h: int, w: int) -> Grid:
        return torch.randint(0, self.num_colors, (h, w), device=self.device, dtype=torch.long)

    def _random_rect(self, grid: Grid, color: int) -> Grid:
        h, w = grid.shape
        if h < 2 or w < 2: return grid
        x1, y1 = self.rng.randint(0, w-2), self.rng.randint(0, h-2)
        x2, y2 = self.rng.randint(x1+1, w-1), self.rng.randint(y1+1, h-1)
        out = grid.clone()
        out[y1:y2+1, x1:x2+1] = color
        return out

    def _symmetry(self, grid: Grid, axis: int) -> Grid:
        # axis=0: vertical flip (top-bottom), axis=1: horizontal flip (left-right)
        h, w = grid.shape
        out = grid.clone()
        if axis == 0:
            half = h // 2
            out[half + (h % 2):, :] = torch.flip(out[:half, :], dims=[0])
        else:
            half = w // 2
            out[:, half + (w % 2):] = torch.flip(out[:, :half], dims=[1])
        return out

    def _noise(self, grid: Grid, amount: float = 0.1) -> Grid:
        mask = torch.rand_like(grid.float()) < amount
        noise = torch.randint(0, self.num_colors, grid.shape, device=self.device)
        out = grid.clone()
        out[mask] = noise[mask]
        return out

    def _gravity(self, grid: Grid) -> Grid:
        h, w = grid.shape
        out = torch.zeros_like(grid)
        for c in range(w):
            col = grid[:, c]
            vals = col[col != 0]
            if len(vals) > 0:
                out[h-len(vals):, c] = vals
        return out

    def _object_shift(self, grid: Grid, dx: int, dy: int) -> Grid:
        out = torch.roll(grid, shifts=(dy, dx), dims=(0, 1))
        # Mask wrap-around
        if dx > 0: out[:, :dx] = 0
        elif dx < 0: out[:, dx:] = 0
        if dy > 0: out[:dy, :] = 0
        elif dy < 0: out[dy:, :] = 0
        return out

    def sample_episode(self) -> ArcEpisode:
        # Task Types
        tasks = ["denoise", "fill_rect", "symmetry_x", "symmetry_y", "gravity", "move_right", "move_down"]
        if self.allowed_rules:
            tasks = [t for t in tasks if t in self.allowed_rules]
        
        task = self.rng.choice(tasks)
        
        n_train = self.rng.randint(*self.train_pairs_range)
        train_inputs, train_outputs = [], []
        test_input, test_output = None, None

        for _ in range(n_train + 1):
            h = self.rng.randint(self.grid_min_size, self.grid_max_size)
            w = self.rng.randint(self.grid_min_size, self.grid_max_size)
            
            # Base grid generation logic depends on task
            base = torch.zeros((h, w), device=self.device, dtype=torch.long)
            
            if task == "denoise":
                # Create a pattern (e.g. random rects)
                for _ in range(3):
                    base = self._random_rect(base, self.rng.randint(1, self.num_colors-1))
                out_g = base
                in_g = self._noise(base, amount=0.15)
                
            elif task == "fill_rect":
                # Input: outlines? No, let's say input is noise, output is filled rect?
                # Simple: Input = 2 points, Output = Rect
                color = self.rng.randint(1, self.num_colors-1)
                x1, y1 = self.rng.randint(0, w-2), self.rng.randint(0, h-2)
                x2, y2 = self.rng.randint(x1+1, w-1), self.rng.randint(y1+1, h-1)
                in_g = torch.zeros_like(base)
                in_g[y1, x1] = color
                in_g[y2, x2] = color
                out_g = in_g.clone()
                out_g[y1:y2+1, x1:x2+1] = color
                
            elif task.startswith("symmetry"):
                # Input: Random pattern on left/top half
                color = self.rng.randint(1, self.num_colors-1)
                if "x" in task: # Vertical axis (left-right)
                    half = w // 2
                    in_g = torch.zeros_like(base)
                    in_g[:, :half] = torch.randint(0, self.num_colors, (h, half), device=self.device)
                    out_g = self._symmetry(in_g, axis=1)
                else:
                    half = h // 2
                    in_g = torch.zeros_like(base)
                    in_g[:half, :] = torch.randint(0, self.num_colors, (half, w), device=self.device)
                    out_g = self._symmetry(in_g, axis=0)

            elif task == "gravity":
                # Random dots
                in_g = torch.randint(0, self.num_colors, (h, w), device=self.device)
                # Sparsify
                mask = torch.rand_like(in_g.float()) < 0.2
                in_g[~mask] = 0
                out_g = self._gravity(in_g)
                
            elif task == "move_right":
                # Draw a shape, move it right
                color = self.rng.randint(1, self.num_colors-1)
                base = self._random_rect(base, color)
                in_g = base
                out_g = self._object_shift(in_g, dx=1, dy=0)
                
            elif task == "move_down":
                 color = self.rng.randint(1, self.num_colors-1)
                 base = self._random_rect(base, color)
                 in_g = base
                 out_g = self._object_shift(in_g, dx=0, dy=1)

            if len(train_inputs) < n_train:
                train_inputs.append(in_g)
                train_outputs.append(out_g)
            else:
                test_input = in_g
                test_output = out_g
                
        return ArcEpisode(train_inputs, train_outputs, test_input, test_output, task)
