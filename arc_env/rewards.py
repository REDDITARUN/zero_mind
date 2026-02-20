"""
Reward computation for the ARC-AGI RL environment.

Rewards are delta-based where possible: you get rewarded for *improving*
the canvas, not for pixels that already matched.
"""

from __future__ import annotations

import numpy as np


class RewardCalculator:
    """Computes shaped rewards for ARC grid editing."""

    def __init__(
        self,
        step_penalty: float = -0.01,
        correct_size_bonus: float = 0.5,
        wrong_size_on_submit: float = -0.5,
        pixel_delta_scale: float = 1.0,
        correct_submit: float = 3.0,
        wrong_submit: float = -2.0,
        all_colors_bonus: float = 0.2,
        correct_row_bonus: float = 0.05,
        correct_col_bonus: float = 0.05,
        perfect_canvas_bonus: float = 1.0,
    ):
        self.step_penalty = step_penalty
        self.correct_size_bonus = correct_size_bonus
        self.wrong_size_on_submit = wrong_size_on_submit
        self.pixel_delta_scale = pixel_delta_scale
        self.correct_submit = correct_submit
        self.wrong_submit = wrong_submit
        self.all_colors_bonus = all_colors_bonus
        self.correct_row_bonus = correct_row_bonus
        self.correct_col_bonus = correct_col_bonus
        self.perfect_canvas_bonus = perfect_canvas_bonus

        self._prev_pixel_score = 0.0
        self._peak_pixel_score = 0.0
        self._prev_size_correct = False
        self._prev_correct_rows = 0
        self._prev_correct_cols = 0
        self._prev_colors_correct = False
        self._hit_perfect = False

    def reset(self, canvas: np.ndarray, target: np.ndarray):
        self._target = target
        self._target_h, self._target_w = target.shape
        self._target_colors = set(np.unique(target).tolist())
        scores = self._compute_scores(canvas)
        self._prev_pixel_score = scores["pixel_score"]
        self._peak_pixel_score = scores["pixel_score"]
        self._prev_size_correct = scores["size_correct"]
        self._prev_correct_rows = scores["correct_rows"]
        self._prev_correct_cols = scores["correct_cols"]
        self._prev_colors_correct = scores["colors_correct"]
        self._hit_perfect = scores["pixel_score"] == 1.0

    def _compute_scores(self, canvas: np.ndarray) -> dict:
        th, tw = self._target_h, self._target_w
        ch, cw = canvas.shape

        size_correct = (ch == th) and (cw == tw)

        if size_correct:
            matches = (canvas == self._target).astype(np.float32)
            total = th * tw
            pixel_score = float(matches.sum()) / total if total > 0 else 0.0

            correct_rows = int(matches.all(axis=1).sum())
            correct_cols = int(matches.all(axis=0).sum())
        else:
            min_h, min_w = min(ch, th), min(cw, tw)
            if min_h > 0 and min_w > 0:
                overlap = (
                    canvas[:min_h, :min_w] == self._target[:min_h, :min_w]
                )
                total = th * tw
                pixel_score = float(overlap.sum()) / total
            else:
                pixel_score = 0.0
            correct_rows = 0
            correct_cols = 0

        canvas_colors = set(np.unique(canvas[:ch, :cw]).tolist())
        colors_correct = canvas_colors == self._target_colors

        return {
            "pixel_score": pixel_score,
            "size_correct": size_correct,
            "correct_rows": correct_rows,
            "correct_cols": correct_cols,
            "colors_correct": colors_correct,
        }

    def compute(self, canvas: np.ndarray, is_submit: bool = False) -> dict:
        """Compute reward for the current step. Returns breakdown dict."""
        scores = self._compute_scores(canvas)
        reward = self.step_penalty
        breakdown = {"step_penalty": self.step_penalty}

        # Delta pixel reward — only reward when exceeding peak (prevents oscillation farming)
        pixel_delta = scores["pixel_score"] - self._prev_pixel_score
        if scores["pixel_score"] > self._peak_pixel_score:
            pixel_reward = (scores["pixel_score"] - self._peak_pixel_score) * self.pixel_delta_scale
            self._peak_pixel_score = scores["pixel_score"]
        elif pixel_delta < 0:
            pixel_reward = pixel_delta * self.pixel_delta_scale * 0.5
        else:
            pixel_reward = 0.0
        reward += pixel_reward
        breakdown["pixel_delta"] = pixel_reward

        # Size bonus (one-time when first achieving correct size)
        if scores["size_correct"] and not self._prev_size_correct:
            reward += self.correct_size_bonus
            breakdown["size_bonus"] = self.correct_size_bonus

        # Row/col bonuses (delta)
        new_rows = scores["correct_rows"] - self._prev_correct_rows
        new_cols = scores["correct_cols"] - self._prev_correct_cols
        if new_rows > 0:
            row_r = new_rows * self.correct_row_bonus
            reward += row_r
            breakdown["row_bonus"] = row_r
        if new_cols > 0:
            col_r = new_cols * self.correct_col_bonus
            reward += col_r
            breakdown["col_bonus"] = col_r

        # Color set bonus (one-time)
        if scores["colors_correct"] and not self._prev_colors_correct:
            reward += self.all_colors_bonus
            breakdown["color_bonus"] = self.all_colors_bonus

        # Perfect canvas bonus (one-time, before submit)
        if (
            scores["pixel_score"] == 1.0
            and scores["size_correct"]
            and not self._hit_perfect
        ):
            reward += self.perfect_canvas_bonus
            breakdown["perfect_canvas"] = self.perfect_canvas_bonus
            self._hit_perfect = True

        # Submit
        if is_submit:
            ch, cw = canvas.shape
            perfect = (
                scores["size_correct"]
                and scores["pixel_score"] == 1.0
            )
            if perfect:
                reward += self.correct_submit
                breakdown["submit"] = self.correct_submit
            else:
                reward += self.wrong_submit
                breakdown["submit"] = self.wrong_submit
                if not scores["size_correct"]:
                    reward += self.wrong_size_on_submit
                    breakdown["wrong_size_penalty"] = self.wrong_size_on_submit
            breakdown["solved"] = perfect

        # Update state
        self._prev_pixel_score = scores["pixel_score"]
        self._prev_size_correct = scores["size_correct"]
        self._prev_correct_rows = scores["correct_rows"]
        self._prev_correct_cols = scores["correct_cols"]
        self._prev_colors_correct = scores["colors_correct"]

        breakdown["total"] = reward
        breakdown["pixel_accuracy"] = scores["pixel_score"]
        return breakdown
