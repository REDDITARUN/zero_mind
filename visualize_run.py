#!/usr/bin/env python3
"""
Load a checkpoint, run eval on the task, and save a GIF of the agent solving it.

The paint-only policy paints pixels, and we auto-submit when the canvas
matches the target (mirrors training behavior).

Usage:
    python visualize_run.py --checkpoint checkpoints/policy_TASKID_solve1.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from arc_env.env import ARCEnv, MAX_GRID
from train_single_task import SimplePaintPolicy, obs_to_tensors

ARC_COLORS = {
    -1: (40, 40, 40),
    0: (0, 0, 0),
    1: (0, 116, 217),
    2: (255, 65, 54),
    3: (46, 204, 64),
    4: (255, 220, 0),
    5: (170, 170, 170),
    6: (240, 18, 190),
    7: (255, 133, 27),
    8: (127, 219, 255),
    9: (135, 12, 37),
}

CELL_SIZE = 24
PADDING = 8
LABEL_HEIGHT = 20


def grid_to_image(grid: np.ndarray, label: str = "") -> Image.Image:
    h, w = grid.shape
    img_w = w * CELL_SIZE + 2
    img_h = h * CELL_SIZE + 2 + (LABEL_HEIGHT if label else 0)
    img = Image.new("RGB", (img_w, img_h), (30, 30, 30))
    draw = ImageDraw.Draw(img)

    y_off = 0
    if label:
        draw.text((4, 2), label, fill=(200, 200, 200))
        y_off = LABEL_HEIGHT

    for r in range(h):
        for c in range(w):
            color = ARC_COLORS.get(int(grid[r, c]), (40, 40, 40))
            x0 = c * CELL_SIZE + 1
            y0 = r * CELL_SIZE + 1 + y_off
            draw.rectangle([x0, y0, x0 + CELL_SIZE - 2, y0 + CELL_SIZE - 2], fill=color)

    return img


def render_frame(
    demo_pairs: list[tuple[np.ndarray, np.ndarray]],
    test_input: np.ndarray,
    canvas: np.ndarray,
    target: np.ndarray | None,
    step: int,
    reward: float,
    action_name: str,
    task_id: str,
    cumulative_reward: float,
    solved: bool = False,
) -> Image.Image:
    demo_images = []
    for i, (inp, out) in enumerate(demo_pairs[:3]):
        demo_images.append(grid_to_image(inp, f"Demo {i} In"))
        demo_images.append(grid_to_image(out, f"Demo {i} Out"))

    test_img = grid_to_image(test_input, "Test Input")
    canvas_img = grid_to_image(canvas, "Agent Canvas")
    target_img = grid_to_image(target, "Target") if target is not None else None

    demo_row_h = max((im.height for im in demo_images), default=0) + PADDING
    demo_row_w = sum(im.width for im in demo_images) + PADDING * (len(demo_images) - 1)

    bottom_images = [test_img, canvas_img]
    if target_img:
        bottom_images.append(target_img)
    bottom_row_h = max(im.height for im in bottom_images) + PADDING
    bottom_row_w = sum(im.width for im in bottom_images) + PADDING * (len(bottom_images) - 1)

    info_height = 50
    total_w = max(demo_row_w, bottom_row_w, 500) + PADDING * 2
    total_h = demo_row_h + bottom_row_h + info_height + PADDING * 3

    frame = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    draw = ImageDraw.Draw(frame)

    status_color = (46, 204, 64) if solved else (200, 200, 200)
    status_text = "SOLVED!" if solved else f"Step {step}"
    draw.text((PADDING, PADDING), f"Task: {task_id}  |  {status_text}", fill=status_color)
    draw.text((PADDING, PADDING + 16),
              f"Action: {action_name}  |  Reward: {reward:+.3f}  |  Total: {cumulative_reward:+.3f}",
              fill=(170, 170, 170))

    x = PADDING
    y = info_height + PADDING
    for im in demo_images:
        frame.paste(im, (x, y))
        x += im.width + PADDING

    x = PADDING
    y = info_height + demo_row_h + PADDING * 2
    for im in bottom_images:
        frame.paste(im, (x, y))
        x += im.width + PADDING

    return frame


def run_eval_and_capture(
    env: ARCEnv,
    policy: SimplePaintPolicy,
    device: torch.device,
    task_index: int,
    deterministic: bool = True,
    max_steps: int = 20,
) -> tuple[list[Image.Image], bool]:
    obs, info = env.reset(options={"task_index": task_index})
    task_id = info["task_id"]
    policy.clear_cache()

    ch = policy.canvas_h
    cw = policy.canvas_w

    demo_pairs = [
        (np.array(p["input"]), np.array(p["output"]))
        for p in env._task["train"]
    ]
    test_input = env._test_input
    target = env.target

    frames = []
    cumulative_reward = 0.0
    solved = False

    frames.append(render_frame(
        demo_pairs, test_input,
        env.active_canvas[:ch, :cw], target,
        step=0, reward=0.0,
        action_name="START", task_id=task_id,
        cumulative_reward=0.0,
    ))

    policy.eval()
    for step in range(1, max_steps + 1):
        obs_t = obs_to_tensors(obs, device)

        with torch.no_grad():
            if deterministic:
                act_idx = policy.get_deterministic_action(obs_t).item()
            else:
                act_idx, _, _, _ = policy.get_action_and_value(obs_t)
                act_idx = act_idx.item()

        env_action = policy.decode_action(act_idx)
        act_str = f"PAINT({env_action[1]},{env_action[2]},c={env_action[5]})"

        obs, reward, terminated, truncated, info = env.step(env_action)
        cumulative_reward += reward

        canvas = env.active_canvas[:ch, :cw]
        canvas_perfect = np.array_equal(canvas, target)

        frames.append(render_frame(
            demo_pairs, test_input,
            canvas, target,
            step=step, reward=reward,
            action_name=act_str,
            task_id=task_id,
            cumulative_reward=cumulative_reward,
            solved=False,
        ))

        if canvas_perfect:
            # Auto-submit
            submit_action = np.array([10, 0, 0, 0, 0, 0])
            _, submit_reward, _, _, submit_info = env.step(submit_action)
            solved = submit_info.get("reward_breakdown", {}).get("solved", False)
            cumulative_reward += submit_reward

            frames.append(render_frame(
                demo_pairs, test_input,
                canvas, target,
                step=step, reward=submit_reward,
                action_name="AUTO-SUBMIT",
                task_id=task_id,
                cumulative_reward=cumulative_reward,
                solved=solved,
            ))
            break

        if terminated or truncated:
            break

    # Hold last frame
    for _ in range(8):
        frames.append(frames[-1])

    return frames, solved


def main():
    parser = argparse.ArgumentParser(description="Visualize ARC agent run as GIF")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    task_id = ckpt["task_id"]
    task_index = ckpt["task_index"]
    print(f"Loaded: task {task_id} (index {task_index})")

    ch = ckpt.get("canvas_h", 3)
    cw = ckpt.get("canvas_w", 3)
    policy = SimplePaintPolicy(ch, cw).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    env = ARCEnv(
        data_dir=Path(__file__).parent / "ARC-AGI" / "data" / "training",
        num_augments=1, show_target=True, curriculum=False,
    )

    print("Running evaluation...")
    frames, solved = run_eval_and_capture(
        env, policy, device, task_index,
        deterministic=not args.stochastic,
    )

    output_path = args.output or f"eval_{task_id}.gif"
    duration_ms = 1000 // args.fps
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)

    print(f"Result: {'SOLVED' if solved else 'FAILED'} | Frames: {len(frames)} | GIF: {output_path}")


if __name__ == "__main__":
    main()
