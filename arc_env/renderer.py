"""
Terminal renderer for ARC grids. Uses ANSI colours so you can see grids
in the terminal without any GUI.
"""

from __future__ import annotations

import numpy as np

# ARC colour palette → ANSI 256-colour codes
# 0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=grey, 6=magenta, 7=orange, 8=cyan, 9=maroon
_ANSI_BG = {
    0: "\033[48;5;0m",    # black
    1: "\033[48;5;21m",   # blue
    2: "\033[48;5;196m",  # red
    3: "\033[48;5;46m",   # green
    4: "\033[48;5;226m",  # yellow
    5: "\033[48;5;244m",  # grey
    6: "\033[48;5;201m",  # magenta
    7: "\033[48;5;208m",  # orange
    8: "\033[48;5;51m",   # cyan
    9: "\033[48;5;124m",  # maroon
}
_RESET = "\033[0m"

_ANSI_FG_DARK = "\033[38;5;0m"   # black text for light backgrounds
_ANSI_FG_LIGHT = "\033[38;5;15m"  # white text for dark backgrounds

_LIGHT_BG_COLORS = {3, 4, 7, 8}


def render_grid(grid: np.ndarray | list, label: str = "") -> str:
    """Return a coloured string representation of a grid."""
    if isinstance(grid, list):
        grid = np.array(grid)

    h, w = grid.shape
    lines = []
    if label:
        lines.append(f"  {label} ({h}×{w}):")

    for r in range(h):
        cells = []
        for c in range(w):
            v = int(grid[r, c])
            bg = _ANSI_BG.get(v, "")
            fg = _ANSI_FG_DARK if v in _LIGHT_BG_COLORS else _ANSI_FG_LIGHT
            cells.append(f"{bg}{fg} {v} {_RESET}")
        lines.append("  " + "".join(cells))

    return "\n".join(lines)


def render_pair(inp: np.ndarray | list, out: np.ndarray | list, idx: int = 0) -> str:
    """Render an input→output pair side by side info."""
    parts = [
        render_grid(inp, f"Demo {idx} Input"),
        render_grid(out, f"Demo {idx} Output"),
    ]
    return "\n".join(parts)


def render_state(
    demo_pairs: list[tuple],
    test_input: np.ndarray,
    canvas: np.ndarray,
    target: np.ndarray | None = None,
    step: int = 0,
    attempt: int = 0,
    task_id: str = "",
) -> str:
    """Render the full environment state for the terminal."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"  Task: {task_id}  |  Step: {step}  |  Attempt: {attempt}")
    lines.append("=" * 60)

    for i, (inp, out) in enumerate(demo_pairs):
        lines.append(render_pair(inp, out, idx=i))
        lines.append("")

    lines.append(render_grid(test_input, "Test Input"))
    lines.append("")
    lines.append(render_grid(canvas, "Your Canvas"))

    if target is not None:
        lines.append("")
        lines.append(render_grid(target, "Target (hidden in eval)"))

    lines.append("=" * 60)
    return "\n".join(lines)
