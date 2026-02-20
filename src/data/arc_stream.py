from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List
import json
import random

import torch

from .arc_simulator import ArcEpisode, ArcSimulator


@dataclass
class ArcStreamConfig:
    source: str = "simulator"  # simulator | arc_json | episodes_jsonl | mixed
    arc_data_dir: str = ""
    split: str = "training"
    max_steps: int = 1000
    seed: int = 0
    same_shape_only: bool = False
    require_test_output: bool = True
    max_grid_size: int = 30
    sim_grid_min_size: int = 6
    sim_grid_max_size: int = 18
    sim_num_colors: int = 10
    sim_min_train_pairs: int = 2
    sim_max_train_pairs: int = 4
    sim_allowed_rules: list[str] | None = None
    episodes_file: str = ""
    mixed_arc_ratio: float = 0.5


def _to_tensor_grid(grid: List[List[int]], device: torch.device) -> torch.Tensor:
    return torch.tensor(grid, dtype=torch.long, device=device)


def stream_from_arc_json(
    arc_data_dir: str,
    split: str,
    device: torch.device,
    seed: int,
    same_shape_only: bool = False,
    require_test_output: bool = True,
    max_grid_size: int = 30,
) -> Generator[ArcEpisode, None, None]:
    data_dir = Path(arc_data_dir) / split
    if not data_dir.exists():
        raise FileNotFoundError(f"ARC split directory not found: {data_dir}")
    task_files = sorted(data_dir.glob("*.json"))
    if not task_files:
        raise ValueError(f"No ARC task files found under: {data_dir}")
    rng = random.Random(seed)

    while True:
        rng.shuffle(task_files)
        for file in task_files:
            try:
                task = json.loads(file.read_text())
                train_pairs = task["train"]
                test_pairs = task["test"]
                # ARC typically has one test sample; if many exist, sample one.
                chosen_test = rng.choice(test_pairs)
                if require_test_output and "output" not in chosen_test:
                    continue

                train_inputs = [_to_tensor_grid(p["input"], device) for p in train_pairs]
                train_outputs = [_to_tensor_grid(p["output"], device) for p in train_pairs]
                test_input = _to_tensor_grid(chosen_test["input"], device)
                test_output = _to_tensor_grid(chosen_test["output"], device) if "output" in chosen_test else torch.zeros_like(test_input)

                if test_input.shape[0] > max_grid_size or test_input.shape[1] > max_grid_size:
                    continue
                if test_output.shape[0] > max_grid_size or test_output.shape[1] > max_grid_size:
                    continue
                if same_shape_only and test_input.shape != test_output.shape:
                    continue
                if same_shape_only:
                    if any(i.shape != o.shape for i, o in zip(train_inputs, train_outputs)):
                        continue

                yield ArcEpisode(
                    train_inputs=train_inputs,
                    train_outputs=train_outputs,
                    test_input=test_input,
                    test_output=test_output,
                    rule_name=file.stem,
                )
            except Exception:
                continue


def make_episode_stream(
    cfg: ArcStreamConfig,
    device: torch.device,
) -> Generator[ArcEpisode, None, None]:
    
    # Create simulator
    simulator = ArcSimulator(
        grid_min_size=cfg.sim_grid_min_size,
        grid_max_size=cfg.sim_grid_max_size,
        num_colors=cfg.sim_num_colors,
        train_pairs_range=(cfg.sim_min_train_pairs, cfg.sim_max_train_pairs),
        allowed_rules=cfg.sim_allowed_rules,
        device=device,
        seed=cfg.seed,
    )

    if cfg.source == "simulator":
        for _ in range(cfg.max_steps):
            yield simulator.sample_episode()
        return

    if cfg.source == "arc_json":
        steps = 0
        for ep in stream_from_arc_json(
            cfg.arc_data_dir,
            cfg.split,
            device=device,
            seed=cfg.seed,
            same_shape_only=cfg.same_shape_only,
            require_test_output=cfg.require_test_output,
            max_grid_size=cfg.max_grid_size,
        ):
            if steps >= cfg.max_steps:
                break
            steps += 1
            yield ep
        return

    if cfg.source == "mixed":
        # Mix Simulator + Real ARC JSON
        rng = random.Random(cfg.seed)
        
        # Infinite generator for ARC tasks
        arc_stream = stream_from_arc_json(
            cfg.arc_data_dir,
            cfg.split,
            device=device,
            seed=cfg.seed + 11,
            same_shape_only=cfg.same_shape_only,
            require_test_output=cfg.require_test_output,
            max_grid_size=cfg.max_grid_size,
        )
        
        arc_ratio = max(0.0, min(1.0, cfg.mixed_arc_ratio))
        
        for _ in range(cfg.max_steps):
            if rng.random() < arc_ratio:
                # Real ARC task
                try:
                    yield next(arc_stream)
                except StopIteration:
                    # Restart stream if exhausted (shouldn't happen with while True)
                    arc_stream = stream_from_arc_json(
                        cfg.arc_data_dir,
                        cfg.split,
                        device=device,
                        seed=cfg.seed + random.randint(0, 10000),
                        same_shape_only=cfg.same_shape_only,
                        require_test_output=cfg.require_test_output,
                        max_grid_size=cfg.max_grid_size,
                    )
                    yield next(arc_stream)
            else:
                # Procedural task
                yield simulator.sample_episode()
        return

    raise ValueError(f"Unknown source: {cfg.source}")
