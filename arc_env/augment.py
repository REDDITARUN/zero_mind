"""
Task augmentation for ARC-AGI: rotation and reflection variants.
400 base tasks × 8 geometric transforms = 3200 task variants.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


Grid = list[list[int]]


def _rot90(grid: Grid) -> Grid:
    a = np.array(grid)
    return np.rot90(a, k=-1).tolist()  # clockwise


def _rot180(grid: Grid) -> Grid:
    a = np.array(grid)
    return np.rot90(a, k=2).tolist()


def _rot270(grid: Grid) -> Grid:
    a = np.array(grid)
    return np.rot90(a, k=-3).tolist()  # clockwise 270 = ccw 90


def _flip_h(grid: Grid) -> Grid:
    return [row[::-1] for row in grid]


def _flip_v(grid: Grid) -> Grid:
    return grid[::-1]


TRANSFORMS: list[tuple[str, list]] = [
    ("orig", []),
    ("rot90", [_rot90]),
    ("rot180", [_rot180]),
    ("rot270", [_rot270]),
    ("flip_h", [_flip_h]),
    ("flip_v", [_flip_v]),
    ("rot90_flip_h", [_rot90, _flip_h]),
    ("rot90_flip_v", [_rot90, _flip_v]),
]


def _apply_chain(grid: Grid, fns: list) -> Grid:
    g = grid
    for fn in fns:
        g = fn(g)
    return g


def _transform_pair(pair: dict, fns: list) -> dict:
    return {
        "input": _apply_chain(pair["input"], fns),
        "output": _apply_chain(pair["output"], fns),
    }


def augment_task(task: dict, transform_name: str, fns: list) -> dict:
    new = {
        "train": [_transform_pair(p, fns) for p in task["train"]],
        "test": [_transform_pair(p, fns) for p in task["test"]],
    }
    return new


def load_tasks(data_dir: str | Path) -> list[dict[str, Any]]:
    """Load all raw ARC tasks from a directory of JSON files."""
    data_dir = Path(data_dir)
    tasks = []
    for fp in sorted(data_dir.glob("*.json")):
        with open(fp) as f:
            task = json.load(f)
        task["task_id"] = fp.stem
        tasks.append(task)
    return tasks


def build_augmented_pool(
    data_dir: str | Path,
    num_augments: int = 3,
) -> list[dict[str, Any]]:
    """
    Load tasks and produce augmented variants.

    num_augments controls how many total variants per task (including original).
      3  -> orig + rot90 + flip_h                   = 1200 tasks
      8  -> all geometric transforms                 = 3200 tasks
    """
    raw = load_tasks(data_dir)
    pool: list[dict[str, Any]] = []

    transforms_to_use = TRANSFORMS[:num_augments]

    for task in raw:
        for tname, fns in transforms_to_use:
            aug = augment_task(task, tname, fns)
            aug["task_id"] = f"{task['task_id']}_{tname}"
            aug["base_task_id"] = task["task_id"]
            pool.append(aug)

    return pool
