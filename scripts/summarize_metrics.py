from __future__ import annotations

import argparse
from pathlib import Path
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_file", type=str, required=True)
    args = parser.parse_args()

    rows = [json.loads(x) for x in Path(args.metrics_file).read_text().splitlines() if x.strip()]
    if not rows:
        print({"rows": 0})
        return
    last = rows[-1]
    n = len(rows)
    tail = rows[max(0, n - 100) :]
    avg_exact = sum(r["exact"] for r in tail) / len(tail)
    avg_eff_sft = sum(r["eff_sft_weight"] for r in tail) / len(tail)
    avg_eff_rl = sum(r["eff_rl_weight"] for r in tail) / len(tail)
    print(
        {
            "rows": n,
            "last_step": last["step"],
            "last_total": last["total"],
            "last_exact": last["exact"],
            "avg_exact_last_100": avg_exact,
            "avg_eff_sft_last_100": avg_eff_sft,
            "avg_eff_rl_last_100": avg_eff_rl,
        }
    )


if __name__ == "__main__":
    main()
