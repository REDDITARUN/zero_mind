#!/usr/bin/env python3
"""
Interactive ARC environment tester.

Run:  python play_arc.py [--task-index N] [--show-target]

You'll see the task (demos, test input, your canvas) and can type actions
to manipulate the canvas, seeing rewards in real time.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from arc_env.env import (
    ARCEnv, ACTION_NAMES,
    PAINT, FILL_RECT, COPY_INPUT_RECT, RESIZE, FLOOD_FILL,
    COLOR_MAP, ROTATE_90, FLIP_H, FLIP_V, COPY_CANVAS_RECT, SUBMIT,
)
from arc_env.renderer import render_grid

HELP_TEXT = """
╔═══════════════════════════════════════════════════════════════╗
║                   ARC-AGI ENVIRONMENT                        ║
╠═══════════════════════════════════════════════════════════════╣
║  ACTIONS (type the number + params):                         ║
║                                                              ║
║   0  row col color        PAINT pixel                        ║
║   1  r1 c1 r2 c2 color   FILL_RECT                          ║
║   2  dst_r dst_c sr sc    COPY from test INPUT to canvas     ║
║   3  new_h new_w          RESIZE canvas                      ║
║   4  row col color        FLOOD_FILL (bucket fill)           ║
║   5  from_color to_color  COLOR_MAP (replace all)            ║
║   6                       ROTATE 90° clockwise               ║
║   7                       FLIP horizontal                    ║
║   8                       FLIP vertical                      ║
║   9  sr sc dr dc          COPY within CANVAS                 ║
║  10                       SUBMIT answer                      ║
║                                                              ║
║  OTHER COMMANDS:                                             ║
║   show          re-render the state                          ║
║   target        show target grid                             ║
║   info          show step/reward info                        ║
║   next          skip to next task                            ║
║   help          show this help                               ║
║   quit / q      exit                                         ║
╚═══════════════════════════════════════════════════════════════╝
"""


def parse_action(line: str) -> np.ndarray | None:
    parts = line.strip().split()
    if not parts:
        return None
    try:
        nums = [int(x) for x in parts]
    except ValueError:
        return None

    act_type = nums[0]
    if act_type < 0 or act_type >= len(ACTION_NAMES):
        print(f"  Invalid action type {act_type}. Must be 0-{len(ACTION_NAMES)-1}.")
        return None

    # Pad to 6 values: [act, p1, p2, p3, p4, p5]
    while len(nums) < 6:
        nums.append(0)

    return np.array(nums[:6], dtype=np.int32)


def print_reward_breakdown(info: dict):
    rb = info.get("reward_breakdown", {})
    if not rb:
        return
    print(f"\n  {'─' * 40}")
    print(f"  Reward Breakdown:")
    for k, v in rb.items():
        if k == "total":
            continue
        if isinstance(v, float):
            print(f"    {k:25s} = {v:+.4f}")
        else:
            print(f"    {k:25s} = {v}")
    print(f"    {'─' * 36}")
    print(f"    {'TOTAL':25s} = {rb.get('total', 0):+.4f}")
    print(f"  Episode cumulative: {info.get('episode_reward', 0):.4f}")
    print(f"  {'─' * 40}")


def main():
    parser = argparse.ArgumentParser(description="Interactive ARC-AGI environment")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to ARC training data (default: auto-detect)")
    parser.add_argument("--task-index", type=int, default=0,
                        help="Start at task index N")
    parser.add_argument("--show-target", action="store_true",
                        help="Show target grid during play")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable augmentation (use 400 raw tasks)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Auto-detect data dir
    data_dir = args.data_dir
    if data_dir is None:
        candidates = [
            Path(__file__).parent / "ARC-AGI" / "data" / "training",
            Path(__file__).parent / "ARC-AGI-1" / "data" / "training",
        ]
        for c in candidates:
            if c.exists():
                data_dir = str(c)
                break
        if data_dir is None:
            print("ERROR: Could not find ARC-AGI training data. Use --data-dir.")
            sys.exit(1)

    print(f"Loading tasks from: {data_dir}")
    num_aug = 1 if args.no_augment else 3
    env = ARCEnv(
        data_dir=data_dir,
        num_augments=num_aug,
        render_mode="human",
        show_target=args.show_target,
        seed=args.seed,
        curriculum=False,
    )
    print(f"Task pool: {len(env.sampler)} tasks")

    print(HELP_TEXT)

    def _task_header(info):
        header = f"\n  Task: {info['task_id']}  |  Max steps: {info['max_steps']}"
        if "target_h" in info:
            header += f"  |  Target: {info['target_h']}×{info['target_w']}"
        header += f"\n  Canvas: {info['canvas_h']}×{info['canvas_w']}  |  Attempt: {info['attempt']}/{env.max_attempts}"
        return header

    task_idx = args.task_index
    obs, info = env.reset(options={"task_index": task_idx})
    env.render()
    print(_task_header(info))

    while True:
        try:
            line = input("\n  action> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not line:
            continue

        if line in ("q", "quit", "exit"):
            print("Bye!")
            break

        if line == "help":
            print(HELP_TEXT)
            continue

        if line == "show":
            env.render()
            continue

        if line == "target":
            t = env.target
            if t is not None:
                print(render_grid(t, "Target"))
            else:
                print("  No target available (eval mode).")
            continue

        if line == "info":
            print(f"  Task: {info['task_id']}")
            print(f"  Step: {info['step']} / {info['max_steps']}")
            print(f"  Attempt: {info['attempt']} / {env.max_attempts}")
            print(f"  Canvas: {info['canvas_h']}×{info['canvas_w']}")
            if "target_h" in info:
                print(f"  Target: {info['target_h']}×{info['target_w']}")
            print(f"  Episode reward: {info['episode_reward']:.4f}")
            print_reward_breakdown(info)
            continue

        if line == "next":
            task_idx += 1
            obs, info = env.reset(options={"task_index": task_idx})
            env.render()
            print(_task_header(info))
            continue

        action = parse_action(line)
        if action is None:
            print("  Could not parse. Type 'help' for usage.")
            continue

        act_name = ACTION_NAMES[action[0]]
        print(f"  Executing: {act_name}({', '.join(str(x) for x in action[1:])})")

        obs, reward, terminated, truncated, info = env.step(action)

        # Show updated canvas
        print(render_grid(env.active_canvas, "Canvas"))
        print(f"\n  Reward: {reward:+.4f}")
        print_reward_breakdown(info)

        if terminated or truncated:
            if terminated:
                solved = info.get("reward_breakdown", {}).get("solved", False)
                if solved:
                    print("\n  ★★★ SOLVED! ★★★")
                else:
                    print(f"\n  ✗ FAILED (used all {env.max_attempts} attempts)")
            else:
                print(f"\n  ⏰ Out of steps! ({info['max_steps']} steps used)")

            print(f"  Episode total reward: {info['episode_reward']:.4f}")
            t = env.target
            if t is not None:
                print(render_grid(t, "Target was"))

            resp = input("\n  Press Enter for next task, or 'q' to quit: ").strip()
            if resp.lower() in ("q", "quit"):
                break
            task_idx += 1
            obs, info = env.reset(options={"task_index": task_idx})
            env.render()
            print(_task_header(info))

    env.close()
    print("\nCurriculum stats:", env.sampler.stats())


if __name__ == "__main__":
    main()
