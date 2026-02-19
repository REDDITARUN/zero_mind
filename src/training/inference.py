from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import json

import torch

from src.models.unified_model import UnifiedArcModel


@dataclass
class InferenceConfig:
    checkpoint_path: str
    arc_data_dir: str
    split: str = "evaluation"
    max_tasks: int = 10
    device: str = "cpu"


def load_model_for_inference(checkpoint_path: str, device: str = "cpu") -> UnifiedArcModel:
    payload = torch.load(checkpoint_path, map_location=device)
    cfg = payload["model_cfg"]
    model = UnifiedArcModel(**cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def infer_arc_split(cfg: InferenceConfig) -> Dict[str, float]:
    model = load_model_for_inference(cfg.checkpoint_path, device=cfg.device)
    data_dir = Path(cfg.arc_data_dir) / cfg.split
    files = sorted(data_dir.glob("*.json"))
    if not files:
        raise ValueError(f"No ARC tasks found under {data_dir}")
    solved = 0
    total = 0
    for file in files:
        task = json.loads(file.read_text())
        train_inputs = [torch.tensor(p["input"], dtype=torch.long, device=cfg.device) for p in task["train"]]
        train_outputs = [torch.tensor(p["output"], dtype=torch.long, device=cfg.device) for p in task["train"]]
        test_pairs = [p for p in task["test"] if "output" in p]
        if not test_pairs:
            continue
        total += 1
        task_ok = True
        for t in test_pairs:
            test_input = torch.tensor(t["input"], dtype=torch.long, device=cfg.device)
            test_output = torch.tensor(t["output"], dtype=torch.long, device=cfg.device)
            out = model(train_inputs, train_outputs, test_input)
            pred = out.test_logits.argmax(dim=1).squeeze(0)
            if not (pred == test_output).all().item():
                task_ok = False
                break
        solved += int(task_ok)
        if total >= cfg.max_tasks:
            break
    return {"tasks": float(total), "solved": float(solved), "solve_rate": float(solved / max(total, 1))}
