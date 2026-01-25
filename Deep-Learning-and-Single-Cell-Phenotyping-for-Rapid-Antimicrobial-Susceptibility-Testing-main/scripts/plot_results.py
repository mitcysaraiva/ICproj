#!/usr/bin/env python3
import argparse
import csv
import json
import os
from collections import defaultdict


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependency: matplotlib.\n"
            "Install it with:\n"
            "  python3 -m pip install matplotlib\n"
        ) from e
    return plt


def _read_history_from_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict in {path}, got {type(data).__name__}")
    return {k: list(v) for k, v in data.items()}


def _read_history_from_csv(path):
    history = defaultdict(list)
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if value is None or value == "":
                    continue
                if key == "epoch":
                    try:
                        history[key].append(int(float(value)))
                    except ValueError:
                        history[key].append(value)
                    continue
                try:
                    history[key].append(float(value))
                except ValueError:
                    history[key].append(value)
    return dict(history)


def _infer_epochs(history):
    if "epoch" in history and history["epoch"]:
        return history["epoch"]
    lengths = [len(v) for v in history.values() if isinstance(v, list)]
    n = max(lengths) if lengths else 0
    return list(range(1, n + 1))


def _get_first(history, keys):
    for key in keys:
        if key in history:
            return key, history[key]
    return None, None


def plot_training_curves(input_path, out_path=None, title=None, show=False):
    plt = _require_matplotlib()

    if input_path.endswith(".json"):
        history = _read_history_from_json(input_path)
    elif input_path.endswith(".csv"):
        history = _read_history_from_csv(input_path)
    else:
        raise SystemExit("Input must be a .json (history) or .csv (CSVLogger) file.")

    epochs = _infer_epochs(history)
    if not epochs:
        raise SystemExit(f"No epochs found in {input_path}.")

    loss_key, loss = _get_first(history, ["loss"])
    val_loss_key, val_loss = _get_first(history, ["val_loss"])

    acc_key, acc = _get_first(history, ["accuracy", "acc", "categorical_accuracy", "sparse_categorical_accuracy"])
    val_acc_key, val_acc = _get_first(history, ["val_accuracy", "val_acc"])
    test_acc_key, test_acc = _get_first(history, ["test_accuracy"])

    fig, axs = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    fig.suptitle(title or os.path.basename(input_path))

    axs[0].set_title("Loss")
    if loss is not None:
        axs[0].plot(epochs, loss, label=loss_key or "loss")
    if val_loss is not None:
        axs[0].plot(epochs, val_loss, label=val_loss_key or "val_loss")
    axs[0].set_xlabel("Epoch")
    axs[0].legend(loc="best")

    axs[1].set_title("Accuracy")
    if acc is not None:
        axs[1].plot(epochs, acc, label=acc_key or "accuracy")
    if val_acc is not None:
        axs[1].plot(epochs, val_acc, label=val_acc_key or "val_accuracy")
    if test_acc is not None:
        axs[1].plot(epochs, test_acc, label=test_acc_key or "test_accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="best")

    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.abspath(input_path)), "training_curves.png")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def _read_confidence_rows(path, score_col, label_col, row_col=None, col_col=None, fov_col=None):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader, start=2):
            try:
                score = float(r[score_col])
            except Exception:
                raise ValueError(f"Bad/missing {score_col} at {path}:{i}")
            label = r.get(label_col)
            if label is None:
                raise ValueError(f"Missing {label_col} at {path}:{i}")
            rows.append(
                {
                    "score": score,
                    "label": str(label),
                    "row": (str(r.get(row_col)) if row_col else ""),
                    "col": (str(r.get(col_col)) if col_col else ""),
                    "fov": (str(r.get(fov_col)) if fov_col else None),
                }
            )
    return rows


def plot_confidence_panels(
    input_csv,
    out_path=None,
    score_col="confidence",
    label_col="label",
    row_col=None,
    col_col=None,
    fov_col=None,
    bins=50,
    score_range=(0.0, 1.0),
    title=None,
    show=False,
):
    plt = _require_matplotlib()

    rows = _read_confidence_rows(input_csv, score_col, label_col, row_col=row_col, col_col=col_col, fov_col=fov_col)
    if not rows:
        raise SystemExit(f"No rows found in {input_csv}.")

    row_vals = sorted({r["row"] for r in rows})
    col_vals = sorted({r["col"] for r in rows})
    labels = sorted({r["label"] for r in rows})
    if len(labels) > 6:
        raise SystemExit(f"Refusing to plot {len(labels)} labels (expected a small number).")

    nrows = max(1, len(row_vals))
    ncols = max(1, len(col_vals))

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), constrained_layout=True)
    if nrows == 1 and ncols == 1:
        axs = [[axs]]
    elif nrows == 1:
        axs = [list(axs)]
    elif ncols == 1:
        axs = [[ax] for ax in axs]

    fig.suptitle(title or os.path.basename(input_csv))

    colors = ["#e76f51", "#457b9d", "#2a9d8f", "#f4a261", "#6d597a", "#8ab17d"]
    label_to_color = {label: colors[i % len(colors)] for i, label in enumerate(labels)}

    for ri, rv in enumerate(row_vals):
        for ci, cv in enumerate(col_vals):
            ax = axs[ri][ci]
            panel = [r for r in rows if r["row"] == rv and r["col"] == cv]
            if not panel:
                ax.axis("off")
                continue

            for label in labels:
                scores = [r["score"] for r in panel if r["label"] == label]
                if not scores:
                    continue
                ax.hist(
                    scores,
                    bins=bins,
                    range=score_range,
                    density=True,
                    alpha=0.35,
                    color=label_to_color[label],
                    label=label,
                )

            panel_title = cv if cv else "All"
            if rv:
                panel_title = f"{rv} — {panel_title}"
            ax.set_title(panel_title)
            ax.set_xlabel(score_col.replace("_", " ").title())
            ax.set_ylabel("Normalised frequency density")
            ax.legend(loc="upper left")

            total = len(panel)
            lines = []
            for label in labels:
                count = sum(1 for r in panel if r["label"] == label)
                if count == 0:
                    continue
                pct = 100.0 * count / total if total else 0.0
                lines.append(f"{count} ({pct:.0f}%) {label}")
            if fov_col:
                fovs = {r["fov"] for r in panel if r["fov"] is not None and r["fov"] != "" and r["fov"] != "None"}
                if fovs:
                    lines.append(f"{len(fovs)} FoVs")

            if lines:
                ax.text(
                    0.03,
                    0.97,
                    "\n".join(lines),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
                )

    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.abspath(input_csv)), "confidence_panels.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def _parse_range(s):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected min,max")
    return float(parts[0]), float(parts[1])


def main():
    parser = argparse.ArgumentParser(description="Plot training curves or confidence distributions.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_curves = sub.add_parser("curves", help="Plot loss/accuracy from Keras history.json or CSVLogger .csv")
    p_curves.add_argument("input", help="Path to history.json or metrics.csv")
    p_curves.add_argument("--out", default=None, help="Output .png path (default: next to input)")
    p_curves.add_argument("--title", default=None, help="Figure title")
    p_curves.add_argument("--show", action="store_true", help="Show the plot window (also saves)")

    p_conf = sub.add_parser("confidence", help="Plot per-class confidence histograms (like the paper figure)")
    p_conf.add_argument("input", help="CSV file containing per-item confidence scores")
    p_conf.add_argument("--out", default=None, help="Output .png path (default: next to input)")
    p_conf.add_argument("--score-col", default="confidence", help="CSV column containing confidence/probability")
    p_conf.add_argument("--label-col", default="label", help="CSV column containing the class label")
    p_conf.add_argument("--row-col", default=None, help="CSV column to facet rows (e.g. strain)")
    p_conf.add_argument("--col-col", default=None, help="CSV column to facet columns (e.g. condition)")
    p_conf.add_argument("--fov-col", default=None, help="Optional CSV column for FoV id (unique counted per panel)")
    p_conf.add_argument("--bins", type=int, default=50, help="Histogram bins")
    p_conf.add_argument("--range", type=_parse_range, default=(0.0, 1.0), help="Score range as min,max")
    p_conf.add_argument("--title", default=None, help="Figure title")
    p_conf.add_argument("--show", action="store_true", help="Show the plot window (also saves)")

    args = parser.parse_args()

    if args.cmd == "curves":
        out = plot_training_curves(args.input, out_path=args.out, title=args.title, show=args.show)
        print(out)
        return

    if args.cmd == "confidence":
        out = plot_confidence_panels(
            args.input,
            out_path=args.out,
            score_col=args.score_col,
            label_col=args.label_col,
            row_col=args.row_col,
            col_col=args.col_col,
            fov_col=args.fov_col,
            bins=args.bins,
            score_range=args.range,
            title=args.title,
            show=args.show,
        )
        print(out)
        return

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

