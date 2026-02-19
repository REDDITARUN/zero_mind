from __future__ import annotations

import argparse
from pathlib import Path
import json


def shape_of(grid: list[list[int]]) -> tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc_data_dir", type=str, default="References/ARC-AGI/data")
    parser.add_argument("--split", type=str, default="training")
    args = parser.parse_args()

    data_dir = Path(args.arc_data_dir) / args.split
    files = sorted(data_dir.glob("*.json"))
    total = 0
    labeled = 0
    same_shape = 0
    diff_shape = 0

    for file in files:
        task = json.loads(file.read_text())
        for pair in task.get("test", []):
            total += 1
            if "output" in pair:
                labeled += 1
                if shape_of(pair["input"]) == shape_of(pair["output"]):
                    same_shape += 1
                else:
                    diff_shape += 1

    print(
        {
            "split": args.split,
            "tasks": len(files),
            "test_pairs": total,
            "labeled_test_pairs": labeled,
            "same_shape_labeled_pairs": same_shape,
            "diff_shape_labeled_pairs": diff_shape,
            "same_shape_ratio_over_labeled": (same_shape / labeled) if labeled else 0.0,
        }
    )


if __name__ == "__main__":
    main()
