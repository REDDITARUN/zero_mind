"""
Autoregressive Grid Generation Environment for ARC-AGI.

The agent generates the output grid cell by cell:
  Phase 0 (H): Choose height     → action 0-29 maps to H=1..30
  Phase 1 (W): Choose width      → action 0-29 maps to W=1..30
  Phase 2 (CELL): Choose color   → action 0-9

Episode length = 2 + H*W (deterministic, every step is meaningful).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from arc_env.augment import build_augmented_pool, load_tasks
from arc_env.curriculum import CurriculumSampler

MAX_GRID = 30
MAX_DEMOS = 10

PHASE_H = 0
PHASE_W = 1
PHASE_CELL = 2


def _pad_grid(grid: np.ndarray) -> np.ndarray:
    padded = np.full((MAX_GRID, MAX_GRID), -1, dtype=np.int8)
    h, w = grid.shape
    padded[:h, :w] = grid
    return padded


class ARCGenEnv(gym.Env):
    """Autoregressive grid generation environment."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        data_dir: str,
        num_augments: int = 1,
        curriculum: bool = True,
        eval_mode: bool = False,
        max_gen_cells: int = 900,
    ):
        super().__init__()

        if num_augments == 1:
            self._pool = load_tasks(data_dir)
            for t in self._pool:
                if "task_id" not in t:
                    t["task_id"] = "unknown"
        else:
            self._pool = build_augmented_pool(data_dir, num_augments)

        self._curriculum = curriculum
        self._eval_mode = eval_mode
        self._max_gen_cells = max_gen_cells

        self.sampler = CurriculumSampler(self._pool) if curriculum else None

        self.action_space = spaces.Discrete(30)

        self.observation_space = spaces.Dict({
            "demo_inputs": spaces.Box(-1, 9, (MAX_DEMOS, MAX_GRID, MAX_GRID), dtype=np.int8),
            "demo_outputs": spaces.Box(-1, 9, (MAX_DEMOS, MAX_GRID, MAX_GRID), dtype=np.int8),
            "demo_input_sizes": spaces.Box(0, MAX_GRID + 1, (MAX_DEMOS, 2), dtype=np.int32),
            "demo_output_sizes": spaces.Box(0, MAX_GRID + 1, (MAX_DEMOS, 2), dtype=np.int32),
            "num_demos": spaces.Discrete(MAX_DEMOS + 1),
            "test_input": spaces.Box(-1, 9, (MAX_GRID, MAX_GRID), dtype=np.int8),
            "test_input_size": spaces.Box(0, MAX_GRID + 1, (2,), dtype=np.int32),
            "phase": spaces.Discrete(3),
            "gen_h": spaces.Discrete(MAX_GRID + 1),
            "gen_w": spaces.Discrete(MAX_GRID + 1),
            "gen_grid": spaces.Box(-1, 9, (MAX_GRID, MAX_GRID), dtype=np.int8),
            "gen_pos": spaces.Box(low=0, high=MAX_GRID * MAX_GRID + 1, shape=(), dtype=np.int32),
            "current_row": spaces.Discrete(MAX_GRID + 1),
            "current_col": spaces.Discrete(MAX_GRID + 1),
        })

        # State
        self._task: dict | None = None
        self._target: np.ndarray | None = None
        self._target_h: int = 0
        self._target_w: int = 0
        self._phase: int = PHASE_H
        self._gen_h: int = 0
        self._gen_w: int = 0
        self._gen_grid: np.ndarray = np.full((MAX_GRID, MAX_GRID), -1, dtype=np.int8)
        self._gen_pos: int = 0
        self._current_row: int = 0
        self._current_col: int = 0
        self._episode_reward: float = 0.0
        self._total_steps: int = 0

    @property
    def target(self) -> np.ndarray | None:
        return self._target

    @property
    def generated_grid(self) -> np.ndarray | None:
        if self._gen_h == 0 or self._gen_w == 0:
            return None
        return self._gen_grid[:self._gen_h, :self._gen_w].copy()

    def reset(self, *, seed=None, options=None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        if self.sampler and seed is not None:
            self.sampler.reseed(seed)

        options = options or {}
        if "task_index" in options:
            self._task = self._pool[options["task_index"]]
        elif self.sampler and self._curriculum:
            self._task = self.sampler.sample()
        else:
            idx = self.np_random.integers(0, len(self._pool))
            self._task = self._pool[idx]

        test_pair = self._task["test"][0]
        self._test_input = np.array(test_pair["input"], dtype=np.int8)

        if not self._eval_mode:
            self._target = np.array(test_pair["output"], dtype=np.int8)
            self._target_h, self._target_w = self._target.shape
        else:
            self._target = None
            self._target_h, self._target_w = 0, 0

        self._phase = PHASE_H
        self._gen_h = 0
        self._gen_w = 0
        self._gen_grid = np.full((MAX_GRID, MAX_GRID), -1, dtype=np.int8)
        self._gen_pos = 0
        self._current_row = 0
        self._current_col = 0
        self._episode_reward = 0.0
        self._total_steps = 0

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        action = int(action)
        reward = 0.0
        terminated = False
        truncated = False
        breakdown = {}

        if self._phase == PHASE_H:
            self._gen_h = action + 1
            if self._target is not None:
                if self._gen_h == self._target_h:
                    reward = 1.0
                    breakdown["h_correct"] = True
                else:
                    reward = -1.5
                    breakdown["h_correct"] = False
                    breakdown["h_diff"] = abs(self._gen_h - self._target_h)
            self._phase = PHASE_W

        elif self._phase == PHASE_W:
            self._gen_w = action + 1
            if self._target is not None:
                if self._gen_w == self._target_w:
                    reward = 1.0
                    breakdown["w_correct"] = True
                else:
                    reward = -1.5
                    breakdown["w_correct"] = False
                    breakdown["w_diff"] = abs(self._gen_w - self._target_w)
            self._phase = PHASE_CELL

        elif self._phase == PHASE_CELL:
            color = action % 10
            r, c = self._current_row, self._current_col
            self._gen_grid[r, c] = color

            if self._target is not None:
                reward = self._cell_reward(r, c, color)
                breakdown["cell_reward"] = reward

            self._gen_pos += 1
            self._current_col += 1
            if self._current_col >= self._gen_w:
                self._current_col = 0
                self._current_row += 1

            total_cells = self._gen_h * self._gen_w
            if self._gen_pos >= total_cells:
                terminated = True

        self._total_steps += 1
        if self._total_steps >= self._max_gen_cells + 2:
            truncated = True

        self._episode_reward += reward

        # Terminal bonus
        if terminated or truncated:
            t_reward, t_info = self._terminal_reward()
            reward += t_reward
            self._episode_reward += t_reward
            breakdown.update(t_info)

        info = self._get_info()
        info["reward_breakdown"] = breakdown

        return self._get_obs(), float(reward), terminated, truncated, info

    def _cell_reward(self, r: int, c: int, color: int) -> float:
        """Per-cell reward based on comparison with target."""
        target_h, target_w = self._target_h, self._target_w
        total = max(target_h * target_w, 1)
        scale = 3.0 / total

        if r < target_h and c < target_w:
            target_color = int(self._target[r, c])
            if color == target_color:
                return scale
            elif target_color == 0 and color != 0:
                return -scale * 1.5  # hallucinated content
            elif target_color != 0 and color == 0:
                return -scale * 1.5  # missed content
            else:
                return -scale * 0.5  # wrong color, right structure
        else:
            return -0.05  # cell outside target dimensions

    def _terminal_reward(self) -> tuple[float, dict]:
        """End-of-episode reward."""
        if self._target is None:
            return 0.0, {"solved": False}

        gen = self._gen_grid[:self._gen_h, :self._gen_w]
        target = self._target

        size_match = (self._gen_h == self._target_h and
                      self._gen_w == self._target_w)

        if size_match:
            if np.array_equal(gen, target):
                return 5.0, {"solved": True, "accuracy": 1.0}
            accuracy = float(np.sum(gen == target)) / (self._target_h * self._target_w)
            return accuracy * 2.0, {"solved": False, "accuracy": accuracy}
        else:
            min_h = min(self._gen_h, self._target_h)
            min_w = min(self._gen_w, self._target_w)
            if min_h > 0 and min_w > 0:
                overlap = np.sum(gen[:min_h, :min_w] == target[:min_h, :min_w])
                total = self._target_h * self._target_w
                accuracy = float(overlap) / total
                return max(accuracy - 0.5, -1.0), {"solved": False, "accuracy": accuracy}
            return -2.0, {"solved": False, "accuracy": 0.0}

    def _get_obs(self) -> dict:
        demos = self._task["train"]
        num_demos = min(len(demos), MAX_DEMOS)

        demo_inputs = np.full((MAX_DEMOS, MAX_GRID, MAX_GRID), -1, dtype=np.int8)
        demo_outputs = np.full((MAX_DEMOS, MAX_GRID, MAX_GRID), -1, dtype=np.int8)
        demo_input_sizes = np.zeros((MAX_DEMOS, 2), dtype=np.int32)
        demo_output_sizes = np.zeros((MAX_DEMOS, 2), dtype=np.int32)

        for i in range(num_demos):
            inp = np.array(demos[i]["input"], dtype=np.int8)
            out = np.array(demos[i]["output"], dtype=np.int8)
            ih, iw = inp.shape
            oh, ow = out.shape
            demo_inputs[i, :ih, :iw] = inp
            demo_outputs[i, :oh, :ow] = out
            demo_input_sizes[i] = [ih, iw]
            demo_output_sizes[i] = [oh, ow]

        ti_h, ti_w = self._test_input.shape
        test_input = _pad_grid(self._test_input)

        return {
            "demo_inputs": demo_inputs,
            "demo_outputs": demo_outputs,
            "demo_input_sizes": demo_input_sizes,
            "demo_output_sizes": demo_output_sizes,
            "num_demos": num_demos,
            "test_input": test_input,
            "test_input_size": np.array([ti_h, ti_w], dtype=np.int32),
            "phase": self._phase,
            "gen_h": self._gen_h,
            "gen_w": self._gen_w,
            "gen_grid": self._gen_grid.copy(),
            "gen_pos": np.int32(self._gen_pos),
            "current_row": self._current_row,
            "current_col": self._current_col,
        }

    def _get_info(self) -> dict:
        info: dict[str, Any] = {
            "task_id": self._task["task_id"],
            "phase": self._phase,
            "gen_h": self._gen_h,
            "gen_w": self._gen_w,
            "gen_pos": self._gen_pos,
            "step": self._total_steps,
            "episode_reward": self._episode_reward,
        }
        if self._target is not None:
            info["target_h"] = self._target_h
            info["target_w"] = self._target_w
        return info

    def __len__(self) -> int:
        return len(self._pool)
