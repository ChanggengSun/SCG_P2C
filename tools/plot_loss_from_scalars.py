"""Plot loss curves from an MMEngine vis_data/scalars.json file.

The file is usually JSON-lines, where each line is one logged record.
Example:

python tools/plot_loss_from_scalars.py --scalars work_dir/xxx/vis_data/scalars.json
python tools/plot_loss_from_scalars.py --work-dir work_dir_20260511_134142 --keys loss
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


PREFERRED_LOSS_KEYS = (
    "loss",
    "tracking_loss",
    "track_pair_loss",
    "regression_loss",
    "flow_loss",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read MMEngine scalars.json and save a loss-curve image."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--scalars",
        type=Path,
        help="Path to vis_data/scalars.json.",
    )
    source.add_argument(
        "--work-dir",
        type=Path,
        help="Work directory. The newest scalars.json under it will be used.",
    )
    parser.add_argument(
        "--keys",
        default=None,
        help=(
            "Comma-separated keys to draw, for example: loss,regression_loss. "
            "If omitted, all numeric keys containing 'loss' are plotted."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <scalars_dir>/loss_curve.png.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Plot title. Defaults to the scalars file parent directory.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving average window. Use 1 to plot raw values.",
    )
    return parser.parse_args()


def newest_scalars(work_dir: Path) -> Path:
    candidates = list(work_dir.rglob("scalars.json"))
    if not candidates:
        raise FileNotFoundError(f"No scalars.json found under: {work_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_json_lines(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] skip invalid json line {line_no}: {exc}")
                continue
            if isinstance(item, dict):
                records.append(item)
    if not records:
        raise RuntimeError(f"No valid records found in: {path}")
    return records


def finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def choose_loss_keys(records: Iterable[dict], requested: str | None) -> list[str]:
    if requested:
        keys = [key.strip() for key in requested.split(",") if key.strip()]
        if not keys:
            raise ValueError("--keys is empty.")
        return keys

    found: set[str] = set()
    for record in records:
        for key, value in record.items():
            if "loss" in key.lower() and finite_float(value) is not None:
                found.add(key)

    ordered = [key for key in PREFERRED_LOSS_KEYS if key in found]
    ordered.extend(sorted(found.difference(ordered)))
    if not ordered:
        raise RuntimeError("No numeric loss-like key found in scalars.json.")
    return ordered


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return values
    smoothed: list[float] = []
    running_sum = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        running_sum += value
        if len(queue) > window:
            running_sum -= queue.pop(0)
        smoothed.append(running_sum / len(queue))
    return smoothed


def collect_series(records: list[dict], key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for index, record in enumerate(records):
        y = finite_float(record.get(key))
        if y is None:
            continue
        x = finite_float(record.get("step"))
        if x is None:
            x = finite_float(record.get("iter"))
        if x is None:
            x = float(index)
        xs.append(x)
        ys.append(y)
    if not xs:
        raise RuntimeError(f"Key '{key}' has no numeric values.")
    return xs, ys


def main() -> None:
    args = parse_args()
    scalars_path = args.scalars if args.scalars else newest_scalars(args.work_dir)
    scalars_path = scalars_path.resolve()
    records = load_json_lines(scalars_path)
    keys = choose_loss_keys(records, args.keys)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    plotted = 0
    for key in keys:
        xs, ys = collect_series(records, key)
        ys = moving_average(ys, args.smooth)
        ax.plot(xs, ys, linewidth=1.6, label=key)
        plotted += 1

    title = args.title or f"Loss curves - {scalars_path.parent.parent.name}"
    if args.smooth > 1:
        title += f" (moving avg={args.smooth})"
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    if plotted > 1:
        ax.legend()
    else:
        ax.legend(loc="best")
    fig.tight_layout()

    output = args.output
    if output is None:
        output = scalars_path.parent / "loss_curve.png"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"[ok] scalars: {scalars_path}")
    print(f"[ok] keys: {', '.join(keys)}")
    print(f"[ok] output: {output}")


if __name__ == "__main__":
    main()
