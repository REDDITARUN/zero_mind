"""
ARC-AGI Gymnasium Environment

Observation: demo pairs + test input + current canvas + metadata
Action: MultiDiscrete — (action_type, p1, p2, p3, p4, p5)
  0  PAINT(row, col, color)
  1  FILL_RECT(r1, c1, r2, c2, color)
  2  COPY_INPUT_RECT(dst_r, dst_c, src_r, src_c, _)  — copies from test input
  3  RESIZE(new_h, new_w, _, _, _)
  4  FLOOD_FILL(row, col, color, _, _)
  5  COLOR_MAP(from_color, to_color, _, _, _)
  6  ROTATE_90(_, _, _, _, _)
  7  FLIP_H(_, _, _, _, _)
  8  FLIP_V(_, _, _, _, _)
  9  COPY_CANVAS_RECT(src_r, src_c, dst_r, dst_c, _) — copies within canvas
  10 SUBMIT(_, _, _, _, _)
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arc_env.augment import build_augmented_pool
from arc_env.curriculum import CurriculumSampler
from arc_env.rewards import RewardCalculator

MAX_GRID = 30
NUM_COLORS = 10
NUM_ACTIONS = 11
MAX_DEMOS = 10


# ──────────────────────────────────────────────────────────────
# Action constants
# ──────────────────────────────────────────────────────────────
PAINT = 0
FILL_RECT = 1
COPY_INPUT_RECT = 2
RESIZE = 3
FLOOD_FILL = 4
COLOR_MAP = 5
ROTATE_90 = 6
FLIP_H = 7
FLIP_V = 8
COPY_CANVAS_RECT = 9
SUBMIT = 10

ACTION_NAMES = [
    "PAINT", "FILL_RECT", "COPY_INPUT_RECT", "RESIZE", "FLOOD_FILL",
    "COLOR_MAP", "ROTATE_90", "FLIP_H", "FLIP_V", "COPY_CANVAS_RECT", "SUBMIT",
]


def _pad_grid(grid: list[list[int]] | np.ndarray, fill: int = -1) -> np.ndarray:
    """Pad a grid to MAX_GRID×MAX_GRID, filling empty cells with `fill`."""
    if isinstance(grid, list):
        grid = np.array(grid, dtype=np.int8)
    h, w = grid.shape
    padded = np.full((MAX_GRID, MAX_GRID), fill, dtype=np.int8)
    padded[:h, :w] = grid
    return padded


def _step_limit(out_h: int, out_w: int) -> int:
    cells = out_h * out_w
    return max(30, min(cells * 3, 300))


class ARCEnv(gym.Env):
    """Gymnasium environment for ARC-AGI-1."""

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        data_dir: str | Path | None = None,
        task_pool: list[dict[str, Any]] | None = None,
        num_augments: int = 3,
        max_attempts: int = 3,
        render_mode: str | None = "ansi",
        seed: int | None = None,
        show_target: bool = False,
        curriculum: bool = True,
        eval_mode: bool = False,
    ):
        super().__init__()

        # Build task pool
        if task_pool is not None:
            self._pool = task_pool
        elif data_dir is not None:
            self._pool = build_augmented_pool(data_dir, num_augments=num_augments)
        else:
            raise ValueError("Provide data_dir or task_pool")

        self.max_attempts = max_attempts
        self.render_mode = render_mode
        self.show_target = show_target
        self._curriculum_enabled = curriculum
        self.eval_mode = eval_mode

        self.sampler = CurriculumSampler(self._pool, seed=seed)
        self.reward_calc = RewardCalculator()

        # ── Spaces ─────────────────────────────────────────────
        grid_space = spaces.Box(
            low=-1, high=9, shape=(MAX_GRID, MAX_GRID), dtype=np.int8,
        )
        self.observation_space = spaces.Dict({
            "demo_inputs": spaces.Box(
                low=-1, high=9, shape=(MAX_DEMOS, MAX_GRID, MAX_GRID), dtype=np.int8,
            ),
            "demo_outputs": spaces.Box(
                low=-1, high=9, shape=(MAX_DEMOS, MAX_GRID, MAX_GRID), dtype=np.int8,
            ),
            "num_demos": spaces.Discrete(MAX_DEMOS + 1),
            "test_input": grid_space,
            "canvas": grid_space,
            "canvas_h": spaces.Discrete(MAX_GRID + 1),
            "canvas_w": spaces.Discrete(MAX_GRID + 1),
            "step": spaces.Discrete(301),
            "steps_remaining": spaces.Discrete(301),
            "attempt": spaces.Discrete(self.max_attempts + 1),
        })

        # MultiDiscrete: [action_type, p1, p2, p3, p4, p5]
        self.action_space = spaces.MultiDiscrete(
            [NUM_ACTIONS, MAX_GRID, MAX_GRID, MAX_GRID, MAX_GRID, NUM_COLORS]
        )

        # State
        self._task: dict | None = None
        self._canvas: np.ndarray = np.zeros((MAX_GRID, MAX_GRID), dtype=np.int8)
        self._canvas_h: int = 1
        self._canvas_w: int = 1
        self._target: np.ndarray | None = None
        self._target_h: int = 0
        self._target_w: int = 0
        self._test_input: np.ndarray | None = None
        self._test_input_h: int = 0
        self._test_input_w: int = 0
        self._step: int = 0
        self._max_steps: int = 100
        self._attempt: int = 0
        self._done: bool = False
        self._episode_reward: float = 0.0
        self._task_index: int = 0
        self._last_reward_breakdown: dict = {}

    # ──────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        # Fix #2: propagate seed to curriculum sampler for reproducibility
        self.sampler.reseed(seed)

        options = options or {}

        if "task" in options:
            self._task = options["task"]
        elif "task_index" in options:
            self._task = self.sampler.sample_sequential(options["task_index"])
        elif self._curriculum_enabled:
            self._task = self.sampler.sample()
        else:
            self._task = self.sampler.sample_sequential(self._task_index)
            self._task_index += 1

        task = self._task
        test_pair = task["test"][0]

        self._test_input = np.array(test_pair["input"], dtype=np.int8)
        self._test_input_h, self._test_input_w = self._test_input.shape

        # In eval_mode the target may not exist
        if "output" in test_pair and not self.eval_mode:
            self._target = np.array(test_pair["output"], dtype=np.int8)
            self._target_h, self._target_w = self._target.shape
            self._has_target = True
        else:
            self._target = None
            self._target_h, self._target_w = 0, 0
            self._has_target = False

        # Initialise canvas to same size as test input, filled with 0
        self._canvas_h = self._test_input_h
        self._canvas_w = self._test_input_w
        self._canvas = np.zeros((MAX_GRID, MAX_GRID), dtype=np.int8)

        self._step = 0
        # Fix #3: base step limit on test *input* size, not target (no answer leakage)
        self._max_steps = _step_limit(self._test_input_h, self._test_input_w)
        self._attempt = 0
        self._done = False
        self._episode_reward = 0.0
        self._submitted_answers: list[np.ndarray] = []

        if self._has_target:
            self.reward_calc.reset(
                self._canvas[:self._canvas_h, :self._canvas_w],
                self._target,
            )

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    # ──────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────
    def step(self, action: np.ndarray | list | tuple):
        # Fix #6: RuntimeError instead of assert (survives python -O)
        if self._done:
            raise RuntimeError("Episode is done — call reset()")

        act_type = int(action[0])
        p1, p2, p3, p4, p5 = (int(action[i]) for i in range(1, 6))

        self._step += 1
        is_submit = act_type == SUBMIT

        self._execute_action(act_type, p1, p2, p3, p4, p5)

        active_canvas = self._canvas[:self._canvas_h, :self._canvas_w].copy()

        # Fix #4: in eval_mode, reward is always 0
        if self.eval_mode or not self._has_target:
            reward = 0.0
            r = {"total": 0.0, "eval_mode": True}
        else:
            r = self.reward_calc.compute(active_canvas, is_submit=is_submit)
            reward = r["total"]

        self._episode_reward += reward
        self._last_reward_breakdown = r

        terminated = False
        truncated = False

        if is_submit:
            self._submitted_answers.append(active_canvas.copy())

            if self.eval_mode or not self._has_target:
                # In eval: submit always terminates, no correctness check
                self._attempt += 1
                if self._attempt >= self.max_attempts:
                    terminated = True
                else:
                    self._canvas = np.zeros((MAX_GRID, MAX_GRID), dtype=np.int8)
                    self._canvas_h = self._test_input_h
                    self._canvas_w = self._test_input_w
            else:
                if r.get("solved", False):
                    terminated = True
                else:
                    self._attempt += 1
                    if self._attempt >= self.max_attempts:
                        terminated = True
                    else:
                        self._canvas = np.zeros((MAX_GRID, MAX_GRID), dtype=np.int8)
                        self._canvas_h = self._test_input_h
                        self._canvas_w = self._test_input_w
                        self.reward_calc.reset(
                            self._canvas[:self._canvas_h, :self._canvas_w],
                            self._target,
                        )

        if self._step >= self._max_steps and not terminated:
            truncated = True

        self._done = terminated or truncated

        if self._done and self._has_target:
            solved = r.get("solved", False)
            self.sampler.update(
                self._task["task_id"], solved, self._episode_reward,
            )

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ──────────────────────────────────────────────────────────
    # Action execution
    # ──────────────────────────────────────────────────────────
    def _execute_action(self, act: int, p1: int, p2: int, p3: int, p4: int, p5: int):
        ch, cw = self._canvas_h, self._canvas_w

        if act == PAINT:
            r, c, color = p1, p2, p5
            if 0 <= r < ch and 0 <= c < cw:
                self._canvas[r, c] = color

        elif act == FILL_RECT:
            r1, c1, r2, c2, color = p1, p2, p3, p4, p5
            r1, r2 = min(r1, r2), max(r1, r2)
            c1, c2 = min(c1, c2), max(c1, c2)
            r2, c2 = min(r2, ch - 1), min(c2, cw - 1)
            r1, c1 = max(r1, 0), max(c1, 0)
            self._canvas[r1 : r2 + 1, c1 : c2 + 1] = color

        elif act == COPY_INPUT_RECT:
            dst_r, dst_c, src_r, src_c = p1, p2, p3, p4
            ih, iw = self._test_input_h, self._test_input_w
            # Copy a block from test input starting at (src_r, src_c) into canvas at (dst_r, dst_c)
            # Infer block size from remaining space
            copy_h = min(ih - src_r, ch - dst_r)
            copy_w = min(iw - src_c, cw - dst_c)
            if copy_h > 0 and copy_w > 0 and src_r >= 0 and src_c >= 0 and dst_r >= 0 and dst_c >= 0:
                self._canvas[dst_r : dst_r + copy_h, dst_c : dst_c + copy_w] = (
                    self._test_input[src_r : src_r + copy_h, src_c : src_c + copy_w]
                )

        elif act == RESIZE:
            new_h, new_w = max(1, min(p1 + 1, MAX_GRID)), max(1, min(p2 + 1, MAX_GRID))
            old = self._canvas[:ch, :cw].copy()
            self._canvas = np.zeros((MAX_GRID, MAX_GRID), dtype=np.int8)
            copy_h, copy_w = min(ch, new_h), min(cw, new_w)
            self._canvas[:copy_h, :copy_w] = old[:copy_h, :copy_w]
            self._canvas_h = new_h
            self._canvas_w = new_w

        elif act == FLOOD_FILL:
            r, c, new_color = p1, p2, p5
            if 0 <= r < ch and 0 <= c < cw:
                old_color = int(self._canvas[r, c])
                if old_color != new_color:
                    self._bfs_fill(r, c, old_color, new_color)

        elif act == COLOR_MAP:
            from_c, to_c = p1 % NUM_COLORS, p2 % NUM_COLORS
            if from_c != to_c:
                region = self._canvas[:ch, :cw]
                region[region == from_c] = to_c

        elif act == ROTATE_90:
            region = self._canvas[:ch, :cw].copy()
            rotated = np.rot90(region, k=-1)
            self._canvas = np.zeros((MAX_GRID, MAX_GRID), dtype=np.int8)
            nh, nw = rotated.shape
            self._canvas[:nh, :nw] = rotated
            self._canvas_h, self._canvas_w = nh, nw

        elif act == FLIP_H:
            self._canvas[:ch, :cw] = self._canvas[:ch, :cw][:, ::-1]

        elif act == FLIP_V:
            self._canvas[:ch, :cw] = self._canvas[:ch, :cw][::-1, :]

        elif act == COPY_CANVAS_RECT:
            src_r, src_c, dst_r, dst_c = p1, p2, p3, p4
            # Fix #5: full bounds checking (no negative-index wrap)
            if src_r < 0 or src_c < 0 or dst_r < 0 or dst_c < 0:
                return
            if src_r >= ch or src_c >= cw or dst_r >= ch or dst_c >= cw:
                return
            copy_h = min(ch - src_r, ch - dst_r)
            copy_w = min(cw - src_c, cw - dst_c)
            if copy_h > 0 and copy_w > 0:
                block = self._canvas[src_r : src_r + copy_h, src_c : src_c + copy_w].copy()
                self._canvas[dst_r : dst_r + copy_h, dst_c : dst_c + copy_w] = block

        elif act == SUBMIT:
            pass  # handled in step()

    def _bfs_fill(self, sr: int, sc: int, old: int, new: int):
        ch, cw = self._canvas_h, self._canvas_w
        q = deque([(sr, sc)])
        visited = set()
        visited.add((sr, sc))
        while q:
            r, c = q.popleft()
            self._canvas[r, c] = new
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < ch and 0 <= nc < cw and (nr, nc) not in visited:
                    if int(self._canvas[nr, nc]) == old:
                        visited.add((nr, nc))
                        q.append((nr, nc))

    # ──────────────────────────────────────────────────────────
    # Observation / info
    # ──────────────────────────────────────────────────────────
    def _get_obs(self) -> dict:
        task = self._task
        demo_inputs = np.full((MAX_DEMOS, MAX_GRID, MAX_GRID), -1, dtype=np.int8)
        demo_outputs = np.full((MAX_DEMOS, MAX_GRID, MAX_GRID), -1, dtype=np.int8)
        # Fix #1: clamp num_demos to MAX_DEMOS
        num_demos = min(len(task["train"]), MAX_DEMOS)

        for i, pair in enumerate(task["train"][:MAX_DEMOS]):
            demo_inputs[i] = _pad_grid(pair["input"])
            demo_outputs[i] = _pad_grid(pair["output"])

        return {
            "demo_inputs": demo_inputs,
            "demo_outputs": demo_outputs,
            "num_demos": num_demos,
            "test_input": _pad_grid(self._test_input),
            "canvas": self._canvas.copy(),
            "canvas_h": self._canvas_h,
            "canvas_w": self._canvas_w,
            "step": self._step,
            "steps_remaining": self._max_steps - self._step,
            "attempt": self._attempt,
        }

    def _get_info(self) -> dict:
        info = {
            "task_id": self._task["task_id"],
            "canvas_h": self._canvas_h,
            "canvas_w": self._canvas_w,
            "step": self._step,
            "max_steps": self._max_steps,
            "attempt": self._attempt,
            "episode_reward": self._episode_reward,
            "reward_breakdown": self._last_reward_breakdown,
            "submitted_answers": [a.tolist() for a in self._submitted_answers],
        }
        # Fix #3: only expose target dimensions when show_target is on (debug)
        if self.show_target and self._has_target:
            info["target_h"] = self._target_h
            info["target_w"] = self._target_w
        return info

    # ──────────────────────────────────────────────────────────
    # Render
    # ──────────────────────────────────────────────────────────
    def render(self) -> str | None:
        from arc_env.renderer import render_state

        demo_pairs = [
            (np.array(p["input"]), np.array(p["output"]))
            for p in self._task["train"]
        ]
        target = self._target if self.show_target else None

        output = render_state(
            demo_pairs=demo_pairs,
            test_input=self._test_input,
            canvas=self._canvas[:self._canvas_h, :self._canvas_w],
            target=target,
            step=self._step,
            attempt=self._attempt,
            task_id=self._task["task_id"],
        )
        if self.render_mode == "human":
            print(output)
            return None
        return output

    @property
    def active_canvas(self) -> np.ndarray:
        return self._canvas[:self._canvas_h, :self._canvas_w].copy()

    @property
    def target(self) -> np.ndarray | None:
        if self._target is None:
            return None
        return self._target.copy()

    @property
    def submitted_answers(self) -> list[np.ndarray]:
        return [a.copy() for a in self._submitted_answers]
