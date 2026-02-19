from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.arc_stream import ArcStreamConfig, make_episode_stream
from src.models.unified_model import UnifiedArcModel
from src.training.unified_loop import run_inference


def _to_tensor_grid(grid: list[list[int]], device: torch.device) -> torch.Tensor:
    return torch.tensor(grid, dtype=torch.long, device=device)


def _apply_transform(grid: torch.Tensor, name: str) -> torch.Tensor:
    if name == "identity":
        return grid.clone()
    if name == "flip_h":
        return torch.flip(grid, dims=[1])
    if name == "flip_v":
        return torch.flip(grid, dims=[0])
    if name == "rot180":
        return torch.rot90(grid, k=2, dims=[0, 1])
    raise ValueError(f"Unknown transform: {name}")


def _predict_with_attempt(
    model: UnifiedArcModel,
    train_inputs: list[torch.Tensor],
    train_outputs: list[torch.Tensor],
    test_input: torch.Tensor,
    attempt_name: str,
) -> torch.Tensor:
    tin = [_apply_transform(x, attempt_name) for x in train_inputs]
    tout = [_apply_transform(x, attempt_name) for x in train_outputs]
    xin = _apply_transform(test_input, attempt_name)
    out = model(tin, tout, xin)
    pred = out.test_logits.argmax(dim=1).squeeze(0)
    # All current transforms are self-inverse.
    pred = _apply_transform(pred, attempt_name)
    return pred


@torch.no_grad()
def evaluate(
    model: UnifiedArcModel,
    stream,
    max_steps: int,
    show_progress_bar: bool = True,
) -> dict[str, float]:
    solved = 0
    total = 0
    pbar = None
    if show_progress_bar:
        try:
            from tqdm import tqdm

            pbar = tqdm(total=max_steps, dynamic_ncols=True, desc="eval")
        except Exception:
            pbar = None
    for ep in stream:
        pred = run_inference(model, ep)
        total += 1
        solved += int((pred == ep.test_output).all().item())
        if pbar is not None:
            pbar.update(1)
            if total % 10 == 0:
                pbar.set_postfix({"solved%": f"{(100.0 * solved / max(1, total)):.2f}"})
        if total >= max_steps:
            break
    if pbar is not None:
        pbar.close()
    return {"episodes": float(total), "solved": float(solved), "solve_rate": float(solved / max(total, 1))}


@torch.no_grad()
def evaluate_arc_taskwise(
    model: UnifiedArcModel,
    stream_cfg: ArcStreamConfig,
    device: torch.device,
    max_tasks: int,
    pass_k: int,
    show_progress_bar: bool = True,
) -> dict[str, float]:
    data_dir = Path(stream_cfg.arc_data_dir) / stream_cfg.split
    if not data_dir.exists():
        raise FileNotFoundError(f"ARC split directory not found: {data_dir}")
    task_files = sorted(data_dir.glob("*.json"))
    if not task_files:
        raise ValueError(f"No ARC task files found under: {data_dir}")
    rng = torch.Generator().manual_seed(stream_cfg.seed)
    perm = torch.randperm(len(task_files), generator=rng).tolist()
    task_files = [task_files[i] for i in perm]

    attempts = ["identity", "flip_h", "flip_v", "rot180"][: max(1, min(pass_k, 4))]
    pass_counts = {k: 0 for k in range(1, len(attempts) + 1)}
    total_tasks = 0

    pbar = None
    if show_progress_bar:
        try:
            from tqdm import tqdm

            pbar = tqdm(total=max_tasks, dynamic_ncols=True, desc="eval(tasks)")
        except Exception:
            pbar = None

    for file in task_files:
        if total_tasks >= max_tasks:
            break
        task = json.loads(file.read_text())
        train_pairs = task["train"]
        test_pairs = [p for p in task["test"] if "output" in p]
        if not test_pairs:
            continue

        train_inputs = [_to_tensor_grid(p["input"], device) for p in train_pairs]
        train_outputs = [_to_tensor_grid(p["output"], device) for p in train_pairs]
        if stream_cfg.same_shape_only and any(i.shape != o.shape for i, o in zip(train_inputs, train_outputs)):
            continue

        first_success_attempts: list[int | None] = []
        valid_task = True
        for test in test_pairs:
            test_input = _to_tensor_grid(test["input"], device)
            test_output = _to_tensor_grid(test["output"], device)
            if test_input.shape[0] > stream_cfg.max_grid_size or test_input.shape[1] > stream_cfg.max_grid_size:
                valid_task = False
                break
            if test_output.shape[0] > stream_cfg.max_grid_size or test_output.shape[1] > stream_cfg.max_grid_size:
                valid_task = False
                break
            if stream_cfg.same_shape_only and test_input.shape != test_output.shape:
                valid_task = False
                break

            first_hit = None
            for i, attempt_name in enumerate(attempts, start=1):
                pred = _predict_with_attempt(model, train_inputs, train_outputs, test_input, attempt_name)
                if (pred == test_output).all().item():
                    first_hit = i
                    break
            first_success_attempts.append(first_hit)

        if not valid_task:
            continue

        total_tasks += 1
        for k in range(1, len(attempts) + 1):
            solved_k = all((hit is not None and hit <= k) for hit in first_success_attempts)
            pass_counts[k] += int(solved_k)

        if pbar is not None:
            pbar.update(1)
            if total_tasks % 5 == 0:
                pbar.set_postfix({f"pass@{len(attempts)}": f"{100.0 * pass_counts[len(attempts)] / max(1, total_tasks):.2f}"})

    if pbar is not None:
        pbar.close()

    out = {"tasks": float(total_tasks)}
    for k in range(1, len(attempts) + 1):
        out[f"solved_at_{k}"] = float(pass_counts[k])
        out[f"pass_at_{k}"] = float(pass_counts[k] / max(1, total_tasks))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--max_eval_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no_progress_bar", action="store_true")
    parser.add_argument("--pass_k", type=int, default=3)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ckpt_path = args.checkpoint or cfg["checkpoint_path"]
    payload = torch.load(ckpt_path, map_location=cfg["train"]["device"])
    model = UnifiedArcModel(**payload["model_cfg"])
    model.load_state_dict(payload["state_dict"])
    model.to(cfg["train"]["device"]).eval()

    stream_cfg = ArcStreamConfig(**cfg["stream"])
    stream_cfg.seed = args.seed
    stream_cfg.max_steps = args.max_eval_steps
    device = torch.device(cfg["train"]["device"])
    if stream_cfg.source == "arc_json":
        metrics = evaluate_arc_taskwise(
            model=model,
            stream_cfg=stream_cfg,
            device=device,
            max_tasks=args.max_eval_steps,
            pass_k=args.pass_k,
            show_progress_bar=not args.no_progress_bar,
        )
    else:
        stream = make_episode_stream(stream_cfg, device=device)
        metrics = evaluate(
            model,
            stream,
            max_steps=args.max_eval_steps,
            show_progress_bar=not args.no_progress_bar,
        )
    print(metrics)


if __name__ == "__main__":
    main()
