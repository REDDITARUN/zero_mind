from __future__ import annotations

import argparse
from pathlib import Path
import json
import random
from typing import Any


def permute_colors(grid: list[list[int]], mapping: list[int]) -> list[list[int]]:
    return [[mapping[v] for v in row] for row in grid]


def flip_h(grid: list[list[int]]) -> list[list[int]]:
    return [list(reversed(row)) for row in grid]


def flip_v(grid: list[list[int]]) -> list[list[int]]:
    return list(reversed(grid))


def rot180(grid: list[list[int]]) -> list[list[int]]:
    return flip_v(flip_h(grid))


def apply_transform(grid: list[list[int]], tname: str) -> list[list[int]]:
    if tname == "identity":
        return [row[:] for row in grid]
    if tname == "flip_h":
        return flip_h(grid)
    if tname == "flip_v":
        return flip_v(grid)
    if tname == "rot180":
        return rot180(grid)
    raise ValueError(f"Unknown transform: {tname}")


def shape(grid: list[list[int]]) -> tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc_data_dir", type=str, default="References/ARC-AGI/data")
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--num_tasks", type=int, default=200)
    parser.add_argument("--episodes_per_task", type=int, default=10)
    parser.add_argument("--same_shape_only", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out_file", type=str, default="data/arc_200_tasks_2k_episodes.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    files = sorted((Path(args.arc_data_dir) / args.split).glob("*.json"))
    rng.shuffle(files)
    chosen = files[: args.num_tasks]

    transforms = ["identity", "flip_h", "flip_v", "rot180"]
    out_rows: list[dict[str, Any]] = []
    for file in chosen:
        task = json.loads(file.read_text())
        train_pairs = task["train"]
        test_pairs = [p for p in task["test"] if "output" in p]
        if not test_pairs:
            continue

        for _ in range(args.episodes_per_task):
            test_pair = rng.choice(test_pairs)
            mapping = list(range(10))
            rng.shuffle(mapping)
            tname = rng.choice(transforms)

            train_inputs = []
            train_outputs = []
            keep = True
            for p in train_pairs:
                inp = apply_transform(permute_colors(p["input"], mapping), tname)
                out = apply_transform(permute_colors(p["output"], mapping), tname)
                if args.same_shape_only and shape(inp) != shape(out):
                    keep = False
                    break
                train_inputs.append(inp)
                train_outputs.append(out)
            if not keep:
                continue

            test_input = apply_transform(permute_colors(test_pair["input"], mapping), tname)
            test_output = apply_transform(permute_colors(test_pair["output"], mapping), tname)
            if args.same_shape_only and shape(test_input) != shape(test_output):
                continue

            out_rows.append(
                {
                    "rule_name": f"{file.stem}:{tname}",
                    "train_inputs": train_inputs,
                    "train_outputs": train_outputs,
                    "test_input": test_input,
                    "test_output": test_output,
                }
            )

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r) for r in out_rows))
    print({"episodes": len(out_rows), "out_file": str(out_path), "num_tasks_requested": args.num_tasks})


if __name__ == "__main__":
    main()
