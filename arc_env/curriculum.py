"""
Curriculum sampler: tracks mastery per task and samples harder/unsolved tasks
more frequently.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TaskRecord:
    task_id: str
    attempts: int = 0
    successes: int = 0
    total_reward: float = 0.0

    @property
    def mastery(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    @property
    def weight(self) -> float:
        """Higher weight = more likely to be sampled. Unsolved tasks get priority."""
        return 1.0 - 0.9 * self.mastery


class CurriculumSampler:
    """Weighted sampler that prioritises unsolved / low-mastery tasks."""

    def __init__(self, task_pool: list[dict[str, Any]], seed: int | None = None):
        self.task_pool = task_pool
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self.records: dict[str, TaskRecord] = {
            t["task_id"]: TaskRecord(task_id=t["task_id"]) for t in task_pool
        }
        self._index = {t["task_id"]: t for t in task_pool}

    def reseed(self, seed: int | None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def sample(self) -> dict[str, Any]:
        ids = list(self.records.keys())
        weights = np.array([self.records[i].weight for i in ids])
        weights /= weights.sum()
        chosen_id = self.rng.choice(ids, p=weights)
        return self._index[chosen_id]

    def sample_sequential(self, index: int) -> dict[str, Any]:
        return self.task_pool[index % len(self.task_pool)]

    def update(self, task_id: str, solved: bool, episode_reward: float):
        rec = self.records[task_id]
        rec.attempts += 1
        if solved:
            rec.successes += 1
        rec.total_reward += episode_reward

    def stats(self) -> dict:
        total = len(self.records)
        solved = sum(1 for r in self.records.values() if r.successes > 0)
        mastered = sum(1 for r in self.records.values() if r.mastery > 0.8)
        return {
            "total_tasks": total,
            "attempted": sum(1 for r in self.records.values() if r.attempts > 0),
            "solved_at_least_once": solved,
            "mastered_80pct": mastered,
            "avg_mastery": np.mean([r.mastery for r in self.records.values()]),
        }

    def __len__(self) -> int:
        return len(self.task_pool)
