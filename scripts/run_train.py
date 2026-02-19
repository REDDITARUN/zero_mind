from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.arc_stream import ArcStreamConfig, make_episode_stream
from src.models.unified_model import UnifiedArcModel
from src.training.unified_loop import TrainConfig, UnifiedTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--load_checkpoint", type=str, default="")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    model_cfg = cfg["model"]
    train_cfg = TrainConfig(**cfg["train"])
    stream_cfg = ArcStreamConfig(**cfg["stream"])

    device = torch.device(train_cfg.device)
    model = UnifiedArcModel(**model_cfg)
    if args.load_checkpoint:
        payload = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(payload["state_dict"])
        print({"resume_from": args.load_checkpoint})
    trainer = UnifiedTrainer(model=model, cfg=train_cfg)

    print(
        {
            "training_methodology": "shared_forward_mixed_objective",
            "auto_switch": "router_controls_effective_sft_rl_weights",
            "sft_warmup_steps": train_cfg.sft_warmup_steps,
            "min_sft_weight": train_cfg.min_sft_weight,
            "sft_refresh_interval": train_cfg.sft_refresh_interval,
            "sft_refresh_span": train_cfg.sft_refresh_span,
            "sft_refresh_min": train_cfg.sft_refresh_min,
            "routing_explore_steps": train_cfg.routing_explore_steps,
            "routing_explore_floor": train_cfg.routing_explore_floor,
            "routing_entropy_bonus": train_cfg.routing_entropy_bonus,
            "adapt_structure": train_cfg.adapt_structure,
            "adapt_interval": train_cfg.adapt_interval,
            "grow_threshold": train_cfg.grow_threshold,
            "prune_threshold": train_cfg.prune_threshold,
            "hard_expert_cap": train_cfg.hard_expert_cap,
            "target_expert_cap": train_cfg.target_expert_cap,
            "prune_cooldown_steps": train_cfg.prune_cooldown_steps,
            "growth_force_until_step": train_cfg.growth_force_until_step,
            "growth_force_interval": train_cfg.growth_force_interval,
            "growth_force_exact_threshold": train_cfg.growth_force_exact_threshold,
            "expert_budget_coeff": train_cfg.expert_budget_coeff,
            "compute_penalty_coeff": train_cfg.compute_penalty_coeff,
            "reward_pixel_coeff": train_cfg.reward_pixel_coeff,
            "reward_exact_coeff": train_cfg.reward_exact_coeff,
            "metrics_jsonl_path": train_cfg.metrics_jsonl_path,
            "stream_source": stream_cfg.source,
            "max_steps": train_cfg.steps,
        }
    )

    stream = make_episode_stream(stream_cfg, device=device)
    logs = trainer.train_stream(stream)

    checkpoint_path = Path(cfg["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_cfg": model_cfg,
            "state_dict": trainer.model.state_dict(),
            "last_metrics": logs[-1] if logs else {},
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
