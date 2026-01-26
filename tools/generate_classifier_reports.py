#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

np = None  # set by _maybe_import_deps()


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(s))


def _maybe_import_deps():
    # Delay heavy imports so `--help` stays fast and missing deps give clear errors.
    try:
        import numpy as _np  # noqa: F401
        import torch  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
        import skimage.io  # noqa: F401
        import skimage  # noqa: F401
        from skimage.transform import resize  # noqa: F401
        from PIL import Image  # noqa: F401
        from torchvision import transforms  # noqa: F401
        from sklearn.metrics import (  # noqa: F401
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            roc_auc_score,
        )
    except Exception as e:
        raise RuntimeError(
            "Missing required Python packages. Install (at least) torch, torchvision, numpy, scikit-learn, "
            "matplotlib, scikit-image, and pillow in your server env."
        ) from e
    globals()["np"] = _np


def _find_repo_root_with_models(start: Path) -> Path | None:
    """
    Walk upwards from `start` looking for a repo root that contains `models/classification`.
    This makes it safe to run the script from inside `tools/` without needing `--repo-root ..`.
    """
    start = start.resolve()
    for p in (start, *start.parents):
        if (p / "models" / "classification").exists():
            return p
    return None


def _find_models_root_under(start: Path, *, max_depth: int = 4) -> Path | None:
    """
    Search downward under `start` for a `models/classification` directory.
    Helpful when the real training repo lives in a subfolder (e.g. `.../Deep-...-main/`).
    """
    start = start.resolve()
    if (start / "models" / "classification").exists():
        return start / "models" / "classification"

    start_depth = len(start.parts)
    best: Path | None = None
    best_mtime = -1.0

    for p in start.rglob("models/classification"):
        try:
            depth = len(p.parts) - start_depth
            if depth > max_depth:
                continue
            if not p.is_dir():
                continue
            mtime = p.stat().st_mtime
            if mtime > best_mtime:
                best = p
                best_mtime = mtime
        except Exception:
            continue

    return best


RUN_RE = re.compile(
    r"^(?P<prefix>optuna_)?(?P<model>[A-Za-z0-9]+)_(?P<condtag>.+?)_"
    r"(?P<splitmode>patientSplitOn|patientSplitOff)_(?P<overlap>overlap|noOverlap)_"
    r"(?P<dt>\d{8}-\d{6})(?:_pid(?P<pid>\d+))?$"
)


@dataclass(frozen=True)
class RunInfo:
    run_dir: Path
    is_optuna: bool
    model_type: str
    condtag: str
    splitmode: str
    overlap: str
    dt: str

    @property
    def sort_key(self) -> str:
        return self.dt


def _parse_run_dir(d: Path) -> RunInfo | None:
    m = RUN_RE.match(d.name)
    if not m:
        return None
    gd = m.groupdict()
    return RunInfo(
        run_dir=d,
        is_optuna=bool(gd.get("prefix")),
        model_type=str(gd["model"]),
        condtag=str(gd["condtag"]),
        splitmode=str(gd["splitmode"]),
        overlap=str(gd["overlap"]),
        dt=str(gd["dt"]),
    )


def _cond_slugs_from_condtag(condtag: str) -> list[str]:
    # Training names are formed by joining per-condition slugs with '_' and each condition here ends with '_ETOH'
    parts = [p for p in condtag.split("_") if p]
    out: list[str] = []
    buf: list[str] = []
    for p in parts:
        buf.append(p)
        if p.upper() == "ETOH":
            out.append("_".join(buf))
            buf = []
    if buf:
        # Fallback: treat whole string as one condition.
        out = [condtag]
    return out


def _discover_slug_to_condition(images_root: Path) -> dict[str, str]:
    slug_map: dict[str, str] = {}
    if not images_root.exists():
        return slug_map
    for p in images_root.iterdir():
        if not p.is_dir():
            continue
        cond = p.name
        slug_map[_slug(cond)] = cond
    return slug_map


def _infer_antibiotic_name(conditions: list[str]) -> str:
    non_wt = [c for c in conditions if not c.upper().startswith("WT")]
    if len(non_wt) == 1:
        return non_wt[0].split("+", 1)[0].split("_", 1)[0]
    if non_wt:
        return "MULTI"
    return "WT"


def _pick_latest(items: list[RunInfo]) -> RunInfo | None:
    if not items:
        return None
    return sorted(items, key=lambda r: r.sort_key)[-1]


def _find_dataset_root(data_root: Path, cond_slugs: list[str]) -> Path | None:
    if not data_root.exists():
        return None
    candidates = [p for p in data_root.glob("classifier_dataset_mg1655_*") if p.is_dir()]
    matches = [p for p in candidates if all(cs in p.name for cs in cond_slugs)]
    if not matches:
        return None
    # Prefer most recently modified.
    matches.sort(key=lambda p: p.stat().st_mtime)
    return matches[-1]


def _find_image_path(images_root: Path, annot_folder_name: str) -> Path:
    candidates = [
        images_root / annot_folder_name,
        images_root / (Path(annot_folder_name).stem + ".tif"),
        images_root / (Path(annot_folder_name).stem + ".tiff"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No image found for annot folder {annot_folder_name!r} under {images_root}")


def _match_cond(name: str, cond_ids: list[str]) -> str | None:
    matches = []
    for cond in cond_ids:
        if re.search(re.escape(cond), name) is not None:
            matches.append(cond)
    return matches[0] if len(matches) == 1 else None


def _iter_cell_crops(
    split_root: Path,
    *,
    cond_ids: list[str],
    cond_to_id: dict[str, int],
    max_per_class: int | None,
    seed: int,
    target_size: tuple[int, int, int] = (64, 64, 3),
):
    import skimage.io
    from skimage.transform import resize
    import skimage

    rng = np.random.default_rng(seed)
    images_root = split_root / "images"
    annots_root = split_root / "annots"
    if not (images_root.exists() and annots_root.exists()):
        raise FileNotFoundError(f"Expected {images_root} and {annots_root} to exist.")

    kept = {c: 0 for c in cond_ids}
    annot_dirs = [p for p in annots_root.iterdir() if p.is_dir()]
    rng.shuffle(annot_dirs)

    for annot_dir in annot_dirs:
        cond = _match_cond(annot_dir.name, cond_ids)
        if cond is None:
            continue
        if max_per_class is not None and kept[cond] >= max_per_class:
            continue

        img_path = _find_image_path(images_root, annot_dir.name)
        image = skimage.io.imread(str(img_path))
        if image.ndim == 2:
            image = image[..., None]
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] > 3:
            image = image[..., :3]

        sy, sx = image.shape[:2]

        mask_files = [p for p in annot_dir.iterdir() if p.suffix.lower() == ".bmp"]
        rng.shuffle(mask_files)

        for mask_path in mask_files:
            if max_per_class is not None and kept[cond] >= max_per_class:
                break

            mask = skimage.io.imread(str(mask_path))
            mask = np.asarray(mask)
            if mask.ndim == 3:
                mask = mask.max(axis=-1)
            mask = mask > 0
            if not mask.any():
                continue

            ys, xs = np.where(mask)
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            x1, x2 = int(xs.min()), int(xs.max()) + 1

            on_edge = (
                (y1 == 0)
                or (x1 == 0)
                or (y2 == 0)
                or (x2 == 0)
                or (y1 >= sy)
                or (x1 >= sx)
                or (y2 >= sy)
                or (x2 >= sx)
            )
            if on_edge:
                continue

            roi = mask[y1:y2, x1:x2]
            crop = image[y1:y2, x1:x2, :].copy()
            crop *= roi[..., None]

            crop = resize(crop, target_size, anti_aliasing=True)
            crop = skimage.img_as_ubyte(crop)

            yield crop, int(cond_to_id[cond])
            kept[cond] += 1

        if max_per_class is not None and all(kept[c] >= max_per_class for c in cond_ids):
            break


def _infer_head_from_state_dict(state_dict: dict) -> tuple[int, bool]:
    # Returns (num_classes, has_dropout_layer_in_head)
    # Trained head is model.classifier[1] = Sequential([Dropout?], Linear)
    w0 = state_dict.get("classifier.1.0.weight")
    w1 = state_dict.get("classifier.1.1.weight")
    if w1 is not None:
        return int(w1.shape[0]), True
    if w0 is not None:
        return int(w0.shape[0]), False
    # DenseNet: classifier = Sequential([Dropout?], Linear) OR Linear
    dw0 = state_dict.get("classifier.0.weight")
    dw1 = state_dict.get("classifier.1.weight")
    if dw1 is not None:
        return int(dw1.shape[0]), True
    if dw0 is not None:
        return int(dw0.shape[0]), False
    dw = state_dict.get("classifier.weight")
    if dw is not None:
        return int(dw.shape[0]), False
    # Fallback: find last weight tensor that looks like a classifier.
    for k, v in state_dict.items():
        if k.endswith(".weight") and hasattr(v, "shape") and len(getattr(v, "shape", ())) == 2:
            return int(v.shape[0]), False
    raise ValueError("Could not infer classifier head from checkpoint state_dict.")

def _infer_arch_from_state_dict(state_dict: dict) -> str:
    # EfficientNet uses keys like "features.0.0.weight"; DenseNet uses "features.conv0.weight".
    if any(k.startswith("features.conv0.") for k in state_dict.keys()):
        return "DenseNet121"
    return "EfficientNetB0"


def _build_model(model_type: str, num_classes: int, has_dropout: bool):
    import torch.nn as nn

    model_type = str(model_type)
    if model_type == "EfficientNetB0":
        try:
            from torchvision.models import efficientnet_b0

            model = efficientnet_b0(weights=None)
        except Exception:
            from torchvision import models as _models

            model = _models.efficientnet_b0(pretrained=False)

        in_features = model.classifier[1].in_features
        head_layers = []
        if has_dropout:
            head_layers.append(nn.Dropout(p=0.2))
        head_layers.append(nn.Linear(in_features, num_classes))
        model.classifier[1] = nn.Sequential(*head_layers)
        return model

    if model_type == "DenseNet121":
        try:
            from torchvision.models import densenet121

            model = densenet121(weights=None)
        except Exception:
            from torchvision import models as _models

            model = _models.densenet121(pretrained=False)

        in_features = model.classifier.in_features
        head_layers = []
        if has_dropout:
            head_layers.append(nn.Dropout(p=0.2))
        head_layers.append(nn.Linear(in_features, num_classes))
        model.classifier = nn.Sequential(*head_layers)
        return model

    raise ValueError(f"Unsupported model_type={model_type!r}")


def _load_checkpoint(ckpt_path: Path):
    import torch

    obj = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(obj, dict) and "model_state" in obj:
        return obj["model_state"]
    if isinstance(obj, dict):
        # Could be a raw state_dict or something similar.
        return obj
    raise TypeError(f"Unsupported checkpoint format: {ckpt_path}")


def _load_checkpoint_meta(ckpt_path: Path) -> dict:
    import torch

    obj = torch.load(str(ckpt_path), map_location="cpu")
    return obj if isinstance(obj, dict) else {}


def _read_history_csv(path: Path) -> dict[str, list[float]]:
    import csv

    out = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    if not path.exists():
        return out
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            for k in out.keys():
                try:
                    out[k].append(float(row[k]))
                except Exception:
                    out[k].append(float("nan"))
    return out


def _best_optuna_trial_dir(tune_dir: Path) -> Path | None:
    import csv

    trials_csv = tune_dir / "trials.csv"
    if not trials_csv.exists():
        return None
    best_num = None
    best_val = float("-inf")
    with trials_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                num = int(row["number"])
                val = float(row["value"])
            except Exception:
                continue
            if val > best_val:
                best_val = val
                best_num = num
    if best_num is None:
        return None
    p = tune_dir / f"trial_{best_num:03d}"
    return p if p.is_dir() else None


def _plot_learning_curves(history: dict[str, list[float]], title: str, outpath: Path):
    import matplotlib.pyplot as plt

    epochs = history.get("epoch", [])
    if not epochs:
        return
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    fig.suptitle(title)

    ax1.plot(epochs, train_loss, label="train")
    ax1.plot(epochs, val_loss, label="validation")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend(loc="best")

    ax2.plot(epochs, train_acc, label="train")
    ax2.plot(epochs, val_acc, label="validation")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_title("Classification Accuracy")
    ax2.legend(loc="best")

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _evaluate_split(
    *,
    model,
    device,
    split_root: Path,
    cond_ids: list[str],
    max_per_class: int | None,
    seed: int,
    batch_size: int,
):
    import torch
    from PIL import Image
    from torchvision import transforms

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    cond_to_id = {c: i for i, c in enumerate(cond_ids)}

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[np.ndarray] = []

    model.eval()
    batch_x = []
    batch_y = []

    def _flush():
        if not batch_x:
            return
        x = torch.stack(batch_x, dim=0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=1).astype(int).tolist()
        y_true.extend(batch_y)
        y_pred.extend(preds)
        y_prob.extend([p for p in probs])
        batch_x.clear()
        batch_y.clear()

    for crop, label in _iter_cell_crops(
        split_root,
        cond_ids=cond_ids,
        cond_to_id=cond_to_id,
        max_per_class=max_per_class,
        seed=seed,
    ):
        img = Image.fromarray(np.asarray(crop).astype(np.uint8))
        batch_x.append(tfm(img))
        batch_y.append(int(label))
        if len(batch_x) >= batch_size:
            _flush()

    _flush()
    return np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int), np.asarray(y_prob, dtype=float)


def _plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str, outpath: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True",
        xlabel="Predicted",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_train_val_accuracy(
    title: str,
    series: list[tuple[str, float | None, float | None]],
    outpath: Path,
):
    import matplotlib.pyplot as plt

    labels = [s[0] for s in series]
    train = [float(s[1]) if s[1] is not None else np.nan for s in series]
    val = [float(s[2]) if s[2] is not None else np.nan for s in series]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(x - width / 2, train, width, label="Train acc")
    ax.bar(x + width / 2, val, width, label="Val acc")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    _maybe_import_deps()

    parser = argparse.ArgumentParser(
        description="Auto-discover best study/retrain classifier checkpoints and generate metrics + plots."
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Path to repo root (the folder containing `models/` and `data/`).",
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default=None,
        help="Override models root (default: <repo-root>/models/classification).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root (default: <repo-root>/data).",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default=None,
        help="Override raw images root for condition discovery (default: <repo-root>/data/Zagajewski_Data/Data/MG1655/All_images).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/classification_eval",
        help="Where to write plots + txt summary (default: reports/classification_eval).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=2000,
        help="Cap cell crops per class per split (default: 2000). Use 0 for no cap.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size (default: 64).",
    )
    parser.add_argument(
        "--antibiotics",
        type=str,
        default=None,
        help="Comma-separated filter (e.g. CIP,COAMOX,RIF). Default: all discovered.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device string (default: cuda). Use cpu if needed.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()

    # If user runs from within `tools/` with the default `--repo-root .`, auto-correct to the real repo root.
    # Respect explicit `--models-root` (i.e., don't rewrite their intent).
    if args.models_root is None and not (repo_root / "models" / "classification").exists():
        auto_root = _find_repo_root_with_models(repo_root) or _find_repo_root_with_models(Path(__file__).resolve().parent)
        if auto_root is not None:
            repo_root = auto_root

    models_root = Path(args.models_root).expanduser().resolve() if args.models_root else (repo_root / "models" / "classification")
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else (repo_root / "data")
    images_root = (
        Path(args.images_root).expanduser().resolve()
        if args.images_root
        else (repo_root / "data" / "Zagajewski_Data" / "Data" / "MG1655" / "All_images")
    )

    # If the expected layout doesn't exist, try to auto-locate a nested training repo.
    if args.models_root is None and not models_root.exists():
        found_models_root = _find_models_root_under(repo_root, max_depth=5)
        if found_models_root is not None:
            models_root = found_models_root
            inferred_root = found_models_root.parent.parent  # .../<repo>/models/classification -> <repo>
            if args.data_root is None and (inferred_root / "data").exists():
                data_root = inferred_root / "data"
            if args.images_root is None and (inferred_root / "data" / "Zagajewski_Data" / "Data" / "MG1655" / "All_images").exists():
                images_root = inferred_root / "data" / "Zagajewski_Data" / "Data" / "MG1655" / "All_images"
    out_root = Path(args.output_dir).expanduser().resolve()
    run_out = out_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_out.mkdir(parents=True, exist_ok=True)

    max_per_class = None if int(args.max_per_class) == 0 else int(args.max_per_class)
    batch_size = int(args.batch_size)

    slug_map = _discover_slug_to_condition(images_root)
    if not slug_map:
        print(
            f"WARNING: could not discover raw condition names from images_root={images_root}. "
            "Falling back to slug-based matching (this is OK if you only trained WT+ANTIBIOTIC pairs)."
        )

    abx_filter = None
    if args.antibiotics:
        abx_filter = {a.strip().upper() for a in args.antibiotics.split(",") if a.strip()}

    # Discover runs
    if not models_root.exists():
        raise FileNotFoundError(f"models root not found: {models_root}")

    runs: list[RunInfo] = []
    for d in models_root.iterdir():
        if not d.is_dir():
            continue
        ri = _parse_run_dir(d)
        if ri is not None:
            runs.append(ri)

    if not runs:
        raise RuntimeError(f"No matching run directories found under: {models_root}")

    # Group into (antibiotic, splitmode) -> best optuna + best retrain.
    grouped: dict[tuple[str, str], dict[str, RunInfo]] = {}
    for r in runs:
        cond_slugs = _cond_slugs_from_condtag(r.condtag)
        cond_names = [slug_map.get(cs, cs.replace("_", "+")) for cs in cond_slugs]
        abx = _infer_antibiotic_name(cond_names).upper()
        if abx_filter is not None and abx not in abx_filter:
            continue

        key = (abx, r.splitmode)
        if key not in grouped:
            grouped[key] = {}
        kind = "study" if r.is_optuna else "retrain"
        prev = grouped[key].get(kind)
        if prev is None:
            grouped[key][kind] = r
        else:
            # Prefer higher val_acc if available; fallback to newer timestamp.
            def _score(run: RunInfo) -> float | None:
                ckpt = run.run_dir / ("best_study_model.pth" if run.is_optuna else "best_retrain_model.pth")
                if not ckpt.exists():
                    ckpt = run.run_dir / "best_model.pth"
                if not ckpt.exists():
                    return None
                meta = _load_checkpoint_meta(ckpt)
                try:
                    return float(meta.get("val_acc"))
                except Exception:
                    return None

            s_prev = _score(prev)
            s_new = _score(r)
            if s_prev is None and s_new is None:
                if prev.sort_key < r.sort_key:
                    grouped[key][kind] = r
            elif s_prev is None and s_new is not None:
                grouped[key][kind] = r
            elif s_prev is not None and s_new is None:
                pass
            else:
                assert s_prev is not None and s_new is not None
                if s_new > s_prev or (s_new == s_prev and prev.sort_key < r.sort_key):
                    grouped[key][kind] = r

    if not grouped:
        raise RuntimeError("No runs left after filtering.")

    import torch
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print("Using device:", device)

    report_lines: list[str] = []
    report_lines.append(f"Classifier evaluation report - generated {datetime.now().isoformat()}")
    report_lines.append(f"repo_root: {repo_root}")
    report_lines.append(f"models_root: {models_root}")
    report_lines.append(f"data_root: {data_root}")
    report_lines.append(f"images_root: {images_root}")
    report_lines.append(f"max_per_class: {max_per_class}")
    report_lines.append(f"batch_size: {batch_size}")
    report_lines.append("")

    # Stable ordering: antibiotic, then splitmode Off->On
    for (abx, splitmode) in sorted(grouped.keys(), key=lambda t: (t[0], 0 if t[1] == "patientSplitOff" else 1)):
        entry = grouped[(abx, splitmode)]

        # Determine conditions + dataset
        sample_run = entry.get("study") or entry.get("retrain")
        assert sample_run is not None
        cond_slugs = _cond_slugs_from_condtag(sample_run.condtag)
        cond_ids = [slug_map.get(cs, cs.replace("_", "+")) for cs in cond_slugs]
        dataset_root = _find_dataset_root(data_root, cond_slugs)

        report_lines.append("=" * 88)
        report_lines.append(f"ANTIBIOTIC: {abx} | split: {splitmode} | conditions: {cond_ids}")
        report_lines.append(f"dataset_root: {dataset_root if dataset_root else 'NOT FOUND'}")
        report_lines.append("")

        if dataset_root is None:
            report_lines.append(
                "ERROR: dataset not found. Create it by running training once with the same conditions "
                "(it will create `data/classifier_dataset_mg1655_...`)."
            )
            report_lines.append("")
            continue

        split_roots = {
            "train": dataset_root / "Train",
            "val": dataset_root / "Validation",
            "test": dataset_root / "Test",
        }

        # Evaluate both model artifacts if present.
        per_model_acc = []
        for kind in ("study", "retrain"):
            run = entry.get(kind)
            if run is None:
                report_lines.append(f"{kind}: MISSING")
                report_lines.append("")
                continue

            ckpt = run.run_dir / ("best_study_model.pth" if kind == "study" else "best_retrain_model.pth")
            if not ckpt.exists():
                # Fallback to best_model.pth
                ckpt = run.run_dir / "best_model.pth"
            if not ckpt.exists():
                report_lines.append(f"{kind}: checkpoint not found under {run.run_dir}")
                report_lines.append("")
                continue

            ckpt_meta = _load_checkpoint_meta(ckpt)
            state_dict = ckpt_meta["model_state"] if "model_state" in ckpt_meta else _load_checkpoint(ckpt)
            num_classes, has_dropout = _infer_head_from_state_dict(state_dict)
            model_type = str(ckpt_meta.get("arch") or run.model_type or _infer_arch_from_state_dict(state_dict))
            model = _build_model(model_type=model_type, num_classes=num_classes, has_dropout=has_dropout)
            model.load_state_dict(state_dict, strict=True)
            model.to(device)

            report_lines.append(f"{kind}: {ckpt}")
            report_lines.append(
                f"  inferred arch={model_type}, num_classes={num_classes}, head_dropout={'yes' if has_dropout else 'no'}"
            )

            # Learning curves (if present)
            hist_path = None
            if kind == "retrain":
                hp = run.run_dir / "history.csv"
                if hp.exists():
                    hist_path = hp
            else:
                best_trial = _best_optuna_trial_dir(run.run_dir)
                if best_trial is not None and (best_trial / "history.csv").exists():
                    hist_path = best_trial / "history.csv"

            if hist_path is not None:
                hist = _read_history_csv(hist_path)
                _plot_learning_curves(
                    hist,
                    title=str(run.run_dir.name),
                    outpath=run_out / abx / splitmode / kind / "learning_curves.png",
                )

            results = {}
            for split_name, split_root in split_roots.items():
                try:
                    y_true, y_pred, y_prob = _evaluate_split(
                        model=model,
                        device=device,
                        split_root=split_root,
                        cond_ids=cond_ids,
                        max_per_class=max_per_class,
                        seed=1337 if split_name == "train" else (1338 if split_name == "val" else 1339),
                        batch_size=batch_size,
                    )
                except FileNotFoundError as e:
                    results[split_name] = {"error": str(e)}
                    continue
                if y_true.size == 0:
                    results[split_name] = {"error": "no samples"}
                    continue

                acc = float(accuracy_score(y_true, y_pred))
                if num_classes == 2:
                    # Positive class: the non-WT condition if possible, else class 1.
                    pos_idx = 1
                    for i, c in enumerate(cond_ids):
                        if not c.upper().startswith("WT"):
                            pos_idx = i
                            break
                    f1 = float(f1_score(y_true, y_pred, average="binary", pos_label=pos_idx))
                    try:
                        auc = float(roc_auc_score((y_true == pos_idx).astype(int), y_prob[:, pos_idx]))
                    except Exception:
                        auc = None
                else:
                    f1 = float(f1_score(y_true, y_pred, average="macro"))
                    try:
                        auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
                    except Exception:
                        auc = None

                cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
                results[split_name] = {"acc": acc, "f1": f1, "auc": auc, "cm": cm, "n": int(y_true.size)}

                # Save confusion matrix plot for each split.
                cm_out = (
                    run_out
                    / abx
                    / splitmode
                    / kind
                    / f"confusion_{split_name}.png"
                )
                _plot_confusion_matrix(
                    cm,
                    labels=cond_ids,
                    title=f"{abx} {splitmode} {kind} - {split_name}",
                    outpath=cm_out,
                )

                if split_name == "test":
                    # Add a text classification report for hold-out test.
                    try:
                        cr = classification_report(y_true, y_pred, target_names=cond_ids, digits=4)
                    except Exception:
                        cr = classification_report(y_true, y_pred, digits=4)
                    results[split_name]["classification_report"] = cr

            # Write summary to report
            for split_name in ("train", "val", "test"):
                r = results.get(split_name, {})
                if "error" in r:
                    report_lines.append(f"  {split_name}: ERROR: {r['error']}")
                    continue
                report_lines.append(
                    f"  {split_name}: n={r['n']}, acc={r['acc']:.4f}, f1={r['f1']:.4f}"
                    + (f", auc={r['auc']:.4f}" if r.get("auc") is not None else ", auc=NA")
                )
                if split_name == "test" and "cm" in r:
                    report_lines.append("  hold-out test confusion_matrix (rows=true, cols=pred):")
                    for row in np.asarray(r["cm"]).astype(int).tolist():
                        report_lines.append("    " + " ".join(f"{v:6d}" for v in row))
                if split_name == "test" and r.get("classification_report"):
                    report_lines.append("  hold-out test classification_report:")
                    report_lines.extend(["    " + ln for ln in str(r["classification_report"]).splitlines()])

            report_lines.append("")

            per_model_acc.append(
                (
                    kind,
                    results.get("train", {}).get("acc") if "error" not in results.get("train", {}) else None,
                    results.get("val", {}).get("acc") if "error" not in results.get("val", {}) else None,
                )
            )

        # (Optional) bar summary removed; prefer per-run learning_curves.png (when history.csv exists).

    report_path = run_out / "metrics.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("Wrote:", report_path)
    print("Plots under:", run_out)
    print(
        textwrap.dedent(
            f"""
            Tip: to evaluate only a subset, use:
              python3 tools/generate_classifier_reports.py --antibiotics CIP,COAMOX --device cuda
            """
        ).strip()
    )


if __name__ == "__main__":
    main()
