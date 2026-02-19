from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.arc_stream import ArcStreamConfig, make_episode_stream
from src.models.unified_model import UnifiedArcModel


def grid_to_ascii(grid: torch.Tensor) -> str:
    rows = []
    for r in grid.tolist():
        rows.append(" ".join(str(x) for x in r))
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--out_file", type=str, default="inference_preview.txt")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ckpt_path = args.checkpoint or cfg["checkpoint_path"]
    payload = torch.load(ckpt_path, map_location=cfg["train"]["device"])

    model = UnifiedArcModel(**payload["model_cfg"])
    model.load_state_dict(payload["state_dict"])
    model.to(cfg["train"]["device"]).eval()

    stream_cfg = ArcStreamConfig(**cfg["stream"])
    stream_cfg.max_steps = args.num_samples
    stream = make_episode_stream(stream_cfg, device=torch.device(cfg["train"]["device"]))

    lines = []
    with torch.no_grad():
        for idx, ep in enumerate(stream, start=1):
            out = model(ep.train_inputs, ep.train_outputs, ep.test_input)
            pred = out.test_logits.argmax(dim=1).squeeze(0)
            exact = bool((pred == ep.test_output).all().item())
            lines.append(f"=== SAMPLE {idx} | rule/task={ep.rule_name} | exact={exact} ===")
            lines.append("Test Input:")
            lines.append(grid_to_ascii(ep.test_input))
            lines.append("Predicted Output:")
            lines.append(grid_to_ascii(pred))
            lines.append("Target Output:")
            lines.append(grid_to_ascii(ep.test_output))
            lines.append("")

    output = "\n".join(lines)
    Path(args.out_file).write_text(output)
    print(f"Wrote visualization to {args.out_file}")


if __name__ == "__main__":
    main()
