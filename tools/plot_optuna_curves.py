#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _require(pkg: str, install_hint: str):
    try:
        __import__(pkg)
    except Exception as e:
        raise SystemExit(f"Missing dependency: {pkg}. Install with: {install_hint}\nError: {e}") from e


def _load_event_accumulator():
    _require("tensorboard", "python -m pip install tensorboard")
    from tensorboard.backend.event_processing import event_accumulator  # type: ignore

    return event_accumulator


def _load_matplotlib():
    _require("matplotlib", "python -m pip install matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    return plt


def _pick_tag(tags: list[str], candidates: list[str]) -> str | None:
    tags_set = set(tags)
    for c in candidates:
        if c in tags_set:
            return c
    # Fall back to substring match (e.g., "epoch_accuracy" vs "accuracy")
    for c in candidates:
        for t in tags:
            if c in t:
                return t
    return None


@dataclass(frozen=True)
class Scalars:
    steps: list[int]
    values: list[float]


def _read_scalar_series(event_accumulator_mod, logdir: Path, tag_candidates: list[str]) -> Scalars | None:
    from tensorboard.util import tensor_util  # type: ignore

    ea = event_accumulator_mod.EventAccumulator(
        str(logdir),
        size_guidance={
            event_accumulator_mod.SCALARS: 0,
            event_accumulator_mod.TENSORS: 0,
            event_accumulator_mod.HISTOGRAMS: 0,
            event_accumulator_mod.COMPRESSED_HISTOGRAMS: 0,
            event_accumulator_mod.IMAGES: 0,
            event_accumulator_mod.AUDIO: 0,
        },
    )
    ea.Reload()
    tags_obj = ea.Tags()

    scalar_tags = tags_obj.get("scalars", [])
    tensor_tags = tags_obj.get("tensors", [])

    tag = _pick_tag(scalar_tags, tag_candidates)
    if tag:
        events = ea.Scalars(tag)
        steps = [int(ev.step) for ev in events]
        values = [float(ev.value) for ev in events]
        return Scalars(steps=steps, values=values)

    tag = _pick_tag(tensor_tags, tag_candidates)
    if tag:
        events = ea.Tensors(tag)
        steps: list[int] = []
        values: list[float] = []
        for ev in events:
            steps.append(int(ev.step))
            arr = tensor_util.make_ndarray(ev.tensor_proto)
            values.append(float(arr.reshape(())))
        return Scalars(steps=steps, values=values)

    return None


def _align_by_step(a: Scalars, b: Scalars) -> tuple[list[int], list[float], list[float]]:
    a_map = {s: v for s, v in zip(a.steps, a.values)}
    b_map = {s: v for s, v in zip(b.steps, b.values)}
    steps = sorted(set(a_map) & set(b_map))
    return steps, [a_map[s] for s in steps], [b_map[s] for s in steps]


def _trial_title(trial_dir: Path) -> str:
    h5s = sorted(trial_dir.glob("*.h5"))
    if h5s:
        return h5s[0].name
    return trial_dir.name


def plot_trial(trial_dir: Path, *, out_name: str = "training_curves.png") -> Path | None:
    train_dir = trial_dir / "train"
    val_dir = trial_dir / "validation"
    if not train_dir.is_dir() or not val_dir.is_dir():
        return None

    event_accumulator_mod = _load_event_accumulator()
    plt = _load_matplotlib()

    acc_candidates = [
        "epoch_accuracy",
        "accuracy",
        "acc",
        "epoch_sparse_categorical_accuracy",
        "sparse_categorical_accuracy",
        "categorical_accuracy",
        "binary_accuracy",
    ]
    loss_candidates = [
        "epoch_loss",
        "loss",
        "cross_entropy",
        "cross_entropy_loss",
    ]

    train_acc = _read_scalar_series(event_accumulator_mod, train_dir, acc_candidates)
    val_acc = _read_scalar_series(event_accumulator_mod, val_dir, acc_candidates)
    train_loss = _read_scalar_series(event_accumulator_mod, train_dir, loss_candidates)
    val_loss = _read_scalar_series(event_accumulator_mod, val_dir, loss_candidates)
    if not (train_acc and val_acc and train_loss and val_loss):
        return None

    steps_acc, train_acc_vals, val_acc_vals = _align_by_step(train_acc, val_acc)
    steps_loss, train_loss_vals, val_loss_vals = _align_by_step(train_loss, val_loss)
    if not steps_acc or not steps_loss:
        return None

    # Convert step -> epoch index (1-based for display)
    epochs_acc = [s + 1 for s in steps_acc]
    epochs_loss = [s + 1 for s in steps_loss]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=False)
    fig.suptitle(_trial_title(trial_dir))

    ax1.plot(epochs_loss, train_loss_vals, label="train")
    ax1.plot(epochs_loss, val_loss_vals, label="validation")
    ax1.set_title(_trial_title(trial_dir))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend()

    ax2.plot(epochs_acc, train_acc_vals, label="train")
    ax2.plot(epochs_acc, val_acc_vals, label="validation")
    ax2.set_title("Classification Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    out_path = trial_dir / out_name
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _best_trial_from_trials_csv(study_dir: Path) -> str | None:
    trials_csv = study_dir / "trials.csv"
    if not trials_csv.is_file():
        return None
    with trials_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        best = None
        for row in reader:
            if row.get("state") != "COMPLETE":
                continue
            try:
                value = float(row["value"])
            except Exception:
                continue
            trial_num = row.get("number")
            if trial_num is None:
                continue
            if best is None or value > best[0]:
                best = (value, trial_num)
    if best is None:
        return None
    return f"trial_{int(best[1]):03d}"


def iter_studies(root: Path) -> list[Path]:
    if (root / "trials.csv").is_file() and any(p.name.startswith("trial_") for p in root.iterdir()):
        return [root]
    return sorted([p for p in root.rglob("optuna_*") if p.is_dir() and (p / "trials.csv").is_file()])


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Plot train/validation curves from Optuna TensorBoard logs.")
    ap.add_argument(
        "--root",
        type=str,
        default="Deep-Learning-and-Single-Cell-Phenotyping-for-Rapid-Antimicrobial-Susceptibility-Testing-main/models/classification",
        help="Classification folder or a specific optuna_* study folder.",
    )
    ap.add_argument("--best-only", action="store_true", help="Plot only the best COMPLETE trial per study.")
    ap.add_argument(
        "--out-name",
        type=str,
        default="training_curves.png",
        help="Filename to write inside each trial folder.",
    )
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser()
    if not root.exists():
        print(f"Not found: {root}", file=sys.stderr)
        return 2

    studies = iter_studies(root)
    if not studies:
        print(f"No optuna_* studies found under: {root}", file=sys.stderr)
        return 2

    total = 0
    made = 0
    for study in studies:
        if args.best_only:
            best_trial = _best_trial_from_trials_csv(study)
            trial_dirs = [study / best_trial] if best_trial else []
        else:
            trial_dirs = sorted([p for p in study.iterdir() if p.is_dir() and p.name.startswith("trial_")])

        for trial in trial_dirs:
            total += 1
            out = plot_trial(trial, out_name=args.out_name)
            if out:
                made += 1
                print(out)
            else:
                print(f"SKIP (missing scalars): {trial}", file=sys.stderr)

    print(f"Done. Plotted {made}/{total} trial(s).")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
