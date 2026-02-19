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
from src.training.unified_loop import TrainConfig, UnifiedTrainer, run_inference


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
def _evaluate_arc_tasks(
    model: UnifiedArcModel,
    cfg: ArcStreamConfig,
    device: torch.device,
    max_tasks: int,
    pass_k: int,
) -> dict[str, float]:
    data_dir = Path(cfg.arc_data_dir) / cfg.split
    files = sorted(data_dir.glob("*.json"))
    if not files:
        raise ValueError(f"No ARC tasks found under {data_dir}")
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(len(files), generator=g).tolist()
    files = [files[i] for i in perm]
    attempts = ["identity", "flip_h", "flip_v", "rot180"][: max(1, min(pass_k, 4))]
    solved_at = {k: 0 for k in range(1, len(attempts) + 1)}
    total = 0
    for file in files:
        if total >= max_tasks:
            break
        task = json.loads(file.read_text())
        train_pairs = task["train"]
        test_pairs = [p for p in task["test"] if "output" in p]
        if not test_pairs:
            continue
        train_inputs = [_to_tensor_grid(p["input"], device) for p in train_pairs]
        train_outputs = [_to_tensor_grid(p["output"], device) for p in train_pairs]
        if cfg.same_shape_only and any(i.shape != o.shape for i, o in zip(train_inputs, train_outputs)):
            continue
        first_hits: list[int | None] = []
        valid = True
        for t in test_pairs:
            ti = _to_tensor_grid(t["input"], device)
            to = _to_tensor_grid(t["output"], device)
            if cfg.same_shape_only and ti.shape != to.shape:
                valid = False
                break
            hit = None
            for k, attempt_name in enumerate(attempts, start=1):
                tin = [_apply_transform(x, attempt_name) for x in train_inputs]
                tout = [_apply_transform(x, attempt_name) for x in train_outputs]
                xin = _apply_transform(ti, attempt_name)
                out = model(tin, tout, xin)
                pred = out.test_logits.argmax(dim=1).squeeze(0)
                pred = _apply_transform(pred, attempt_name)
                if (pred == to).all().item():
                    hit = k
                    break
            first_hits.append(hit)
        if not valid:
            continue
        total += 1
        for k in range(1, len(attempts) + 1):
            solved_at[k] += int(all((h is not None and h <= k) for h in first_hits))
    out = {"eval_tasks": float(total)}
    for k in range(1, len(attempts) + 1):
        out[f"eval_pass_at_{k}"] = float(solved_at[k] / max(1, total))
        out[f"eval_solved_at_{k}"] = float(solved_at[k])
    return out


@torch.no_grad()
def evaluate_once(
    model: UnifiedArcModel,
    cfg: ArcStreamConfig,
    device: torch.device,
    max_steps: int,
    pass_k: int,
) -> dict[str, float]:
    model.eval()
    if cfg.source == "arc_json":
        return _evaluate_arc_tasks(model=model, cfg=cfg, device=device, max_tasks=max_steps, pass_k=pass_k)
    stream_cfg = ArcStreamConfig(**cfg.__dict__)
    stream_cfg.max_steps = max_steps
    stream = make_episode_stream(stream_cfg, device=device)
    solved = 0
    total = 0
    for ep in stream:
        pred = run_inference(model, ep)
        total += 1
        solved += int((pred == ep.test_output).all().item())
        if total >= max_steps:
            break
    return {"eval_steps": float(total), "eval_solved": float(solved), "eval_solve_rate": float(solved / max(1, total))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default="configs/colab_cuda.yaml")
    parser.add_argument("--eval_config", type=str, default="configs/colab_eval.yaml")
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--eval_pass_k", type=int, default=3)
    parser.add_argument("--load_checkpoint", type=str, default="")
    args = parser.parse_args()

    train_yaml = yaml.safe_load(Path(args.train_config).read_text())
    eval_yaml = yaml.safe_load(Path(args.eval_config).read_text())

    train_cfg = TrainConfig(**train_yaml["train"])
    train_stream_cfg = ArcStreamConfig(**train_yaml["stream"])
    eval_stream_cfg = ArcStreamConfig(**eval_yaml["stream"])

    device = torch.device(train_cfg.device)
    model = UnifiedArcModel(**train_yaml["model"])
    if args.load_checkpoint:
        payload = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(payload["state_dict"])
        print({"resume_from": args.load_checkpoint})
    trainer = UnifiedTrainer(model=model, cfg=train_cfg)
    train_stream = make_episode_stream(train_stream_cfg, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} params ({trainable:,} trainable)")
    print(f"LR warmup: {train_cfg.lr_warmup_steps} steps, grad_accum: {train_cfg.grad_accum_steps}")
    print(f"Loss: Correct + {train_cfg.w_draft}*draft + {train_cfg.w_aux}*aux (label_smooth={train_cfg.label_smoothing})")
    print(f"AMP: {train_cfg.use_amp} | Training {train_cfg.steps} steps on {train_cfg.device}")

    try:
        from tqdm import tqdm

        pbar = tqdm(total=train_cfg.steps, dynamic_ncols=True, desc="train+eval")
    except Exception:
        pbar = None

    logs = []
    for step, fresh_episode in enumerate(train_stream, start=1):
        episode = trainer._pick_episode(fresh_episode)
        m = trainer.train_step(episode, step=step)
        m["step"] = step
        trainer.cumulative_total += 1
        trainer.cumulative_solved += int(m["exact"] >= 1.0)
        m["cumulative_solved_pct"] = 100.0 * trainer.cumulative_solved / max(1, trainer.cumulative_total)
        trainer.recent.append(m)
        trainer._write_metric(m)
        logs.append(m)

        if pbar is not None:
            pbar.update(1)
            if step % max(1, train_cfg.log_every // 2) == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{m['total']:.3f}",
                        "px": f"{m['pixel_acc']:.3f}",
                        "drft": f"{m['draft_acc']:.3f}",
                        "exact": f"{m['exact']:.3f}",
                        "solved%": f"{m['cumulative_solved_pct']:.2f}",
                        "xprt": f"{m['active_experts']:.0f}",
                    }
                )

        if step % train_cfg.log_every == 0:
            print(
                f"step={step} loss={m['total']:.4f} correct={m['correct']:.3f} "
                f"draft={m.get('draft', 0):.3f} aux={m.get('aux', 0):.3f} "
                f"px_acc={m['pixel_acc']:.3f} drft_acc={m['draft_acc']:.3f} "
                f"exact={m['exact']:.3f} "
                f"solved%={m['cumulative_solved_pct']:.2f} xperts={m['active_experts']:.0f} lr={m['lr']:.2e}"
            )

        if step % max(1, args.eval_every) == 0:
            eval_metrics = evaluate_once(
                model=trainer.model,
                cfg=eval_stream_cfg,
                device=device,
                max_steps=args.eval_steps,
                pass_k=args.eval_pass_k,
            )
            print({"step": step, **eval_metrics})
            ckpt_path = Path(train_yaml["checkpoint_path"])
            ckpt_step = ckpt_path.with_name(f"{ckpt_path.stem}_step{step}{ckpt_path.suffix}")
            ckpt_step.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_cfg": train_yaml["model"], "state_dict": trainer.model.state_dict(), "last_metrics": m, "eval": eval_metrics},
                ckpt_step,
            )

        if step >= train_cfg.steps:
            break

    if pbar is not None:
        pbar.close()

    checkpoint_path = Path(train_yaml["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_cfg": train_yaml["model"], "state_dict": trainer.model.state_dict(), "last_metrics": logs[-1] if logs else {}},
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
