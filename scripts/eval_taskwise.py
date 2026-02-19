from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from collections import defaultdict

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.arc_stream import ArcStreamConfig, make_episode_stream
from src.models.unified_model import UnifiedArcModel


def path_from_router(alpha_sft: float) -> str:
    if alpha_sft >= 0.67:
        return "SFT"
    if alpha_sft <= 0.33:
        return "RL"
    return "HYBRID"


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


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/arc_2k_aug.yaml")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--max_eval_steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out_file", type=str, default="reports/taskwise_eval.jsonl")
    parser.add_argument("--pass_k", type=int, default=3)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ckpt_path = args.checkpoint or cfg["checkpoint_path"]
    payload = torch.load(ckpt_path, map_location=cfg["train"]["device"])
    model = UnifiedArcModel(**payload["model_cfg"])
    model.load_state_dict(payload["state_dict"])
    model.to(cfg["train"]["device"]).eval()

    stream_cfg = ArcStreamConfig(**cfg["stream"])
    stream_cfg.max_steps = args.max_eval_steps
    stream_cfg.seed = args.seed
    device = torch.device(cfg["train"]["device"])

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    by_mode = defaultdict(int)  # mode of first attempt
    solved_at = defaultdict(int)
    total = 0
    attempts = ["identity", "flip_h", "flip_v", "rot180"][: max(1, min(args.pass_k, 4))]

    if stream_cfg.source == "arc_json":
        task_files = sorted((Path(stream_cfg.arc_data_dir) / stream_cfg.split).glob("*.json"))
        rng = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(task_files), generator=rng).tolist()
        task_files = [task_files[i] for i in perm]
        for file in task_files:
            if total >= args.max_eval_steps:
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

            first_attempt_mode = "NA"
            first_attempt_sft = 0.0
            first_attempt_rl = 0.0
            first_attempt_sym = 0.0
            first_attempt_rule = "none"
            per_test_first_hit: list[int | None] = []
            valid = True
            for test in test_pairs:
                test_input = _to_tensor_grid(test["input"], device)
                test_output = _to_tensor_grid(test["output"], device)
                if stream_cfg.same_shape_only and test_input.shape != test_output.shape:
                    valid = False
                    break
                hit = None
                for k, attempt_name in enumerate(attempts, start=1):
                    tin = [_apply_transform(x, attempt_name) for x in train_inputs]
                    tout = [_apply_transform(x, attempt_name) for x in train_outputs]
                    xin = _apply_transform(test_input, attempt_name)
                    out = model(tin, tout, xin)
                    pred = out.test_logits.argmax(dim=1).squeeze(0)
                    pred = _apply_transform(pred, attempt_name)
                    if k == 1:
                        first_attempt_sft = float(out.router.alpha_sft.mean().detach().cpu())
                        first_attempt_rl = float(out.router.alpha_rl.mean().detach().cpu())
                        first_attempt_mode = path_from_router(first_attempt_sft)
                        first_attempt_sym = float(out.symbolic_confidence.detach().cpu())
                        first_attempt_rule = out.symbolic_rule
                    if (pred == test_output).all().item():
                        hit = k
                        break
                per_test_first_hit.append(hit)
            if not valid:
                continue

            total += 1
            by_mode[first_attempt_mode] += 1
            row = {
                "task_id": file.stem,
                "num_test_cases": len(test_pairs),
                "first_hit_per_test": per_test_first_hit,
                "mode_first_attempt": first_attempt_mode,
                "router_sft_first_attempt": first_attempt_sft,
                "router_rl_first_attempt": first_attempt_rl,
                "symbolic_rule_first_attempt": first_attempt_rule,
                "symbolic_confidence_first_attempt": first_attempt_sym,
            }
            for k in range(1, len(attempts) + 1):
                solved_k = int(all((h is not None and h <= k) for h in per_test_first_hit))
                row[f"solved_at_{k}"] = solved_k
                solved_at[k] += solved_k
            rows.append(row)
    else:
        stream = make_episode_stream(stream_cfg, device=device)
        for ep in stream:
            out = model(ep.train_inputs, ep.train_outputs, ep.test_input)
            pred = out.test_logits.argmax(dim=1).squeeze(0)
            exact = int((pred == ep.test_output).all().item())
            alpha_sft = float(out.router.alpha_sft.mean().detach().cpu())
            alpha_rl = float(out.router.alpha_rl.mean().detach().cpu())
            mode = path_from_router(alpha_sft)
            row = {
                "task_id": ep.rule_name,
                "exact": exact,
                "mode": mode,
                "router_sft": alpha_sft,
                "router_rl": alpha_rl,
                "symbolic_rule": out.symbolic_rule,
                "symbolic_confidence": float(out.symbolic_confidence.detach().cpu()),
            }
            rows.append(row)
            total += 1
            solved_at[1] += exact
            by_mode[mode] += 1
            if total >= args.max_eval_steps:
                break

    out_path.write_text("\n".join(json.dumps(r) for r in rows))
    summary = {
        "rows": total,
        "mode_counts": dict(by_mode),
        "out_file": str(out_path),
    }
    for k, v in sorted(solved_at.items()):
        summary[f"solved_at_{k}"] = float(v)
        summary[f"pass_at_{k}"] = float(v / max(1, total))
    print(summary)


if __name__ == "__main__":
    main()
