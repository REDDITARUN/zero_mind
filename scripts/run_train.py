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

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} params ({trainable:,} trainable)")

    if args.load_checkpoint:
        payload = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(payload["state_dict"])
        print({"resume_from": args.load_checkpoint})
    trainer = UnifiedTrainer(model=model, cfg=train_cfg)

    print(
        {
            "methodology": "HRM + Iterative MLP Decoder + TTT @ inference",
            "lr": train_cfg.lr,
            "lr_warmup_steps": train_cfg.lr_warmup_steps,
            "grad_accum_steps": train_cfg.grad_accum_steps,
            "use_amp": train_cfg.use_amp,
            "w_aux": train_cfg.w_aux,
            "label_smoothing": train_cfg.label_smoothing,
            "adapt_structure": train_cfg.adapt_structure,
            "max_steps": train_cfg.steps,
            "stream_source": stream_cfg.source,
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
