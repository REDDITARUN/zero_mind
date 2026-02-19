from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.inference import InferenceConfig, infer_arc_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--max_tasks", type=int, default=20)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    inf_cfg = InferenceConfig(
        checkpoint_path=cfg["checkpoint_path"],
        arc_data_dir=cfg["stream"]["arc_data_dir"],
        split=args.split,
        max_tasks=args.max_tasks,
        device=cfg["train"]["device"],
    )
    metrics = infer_arc_split(inf_cfg)
    print(metrics)


if __name__ == "__main__":
    main()
