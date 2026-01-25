import os
import sys
import shutil
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import warnings

# ------------------------------------------------------------------
# Torchvision compatibility helpers
# ------------------------------------------------------------------
_PIL_TRANSPOSE = getattr(Image, "Transpose", Image)

def _resolve_torch_device(cfg: dict) -> torch.device:
    raw = (
        (cfg or {}).get('device')
        or (cfg or {}).get('torch_device')
        or (cfg or {}).get('pytorch_device')
    )
    if raw is None or str(raw).strip() == '':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dev_str = str(raw).strip().lower()

    if dev_str.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Config requested device={raw!r}, but CUDA is not available in this environment.")

        idx = 0
        if ':' in dev_str:
            _, idx_str = dev_str.split(':', 1)
            idx = int(idx_str)
        n = torch.cuda.device_count()
        if idx < 0 or idx >= n:
            raise RuntimeError(
                f"Config requested device={raw!r}, but this environment reports {n} CUDA device(s)."
            )
        torch.cuda.set_device(idx)
        return torch.device(f'cuda:{idx}')

    if dev_str == 'cpu':
        return torch.device('cpu')

    if dev_str == 'mps':
        if getattr(torch.backends, 'mps', None) is None or not torch.backends.mps.is_available():
            raise RuntimeError(f"Config requested device={raw!r}, but MPS is not available in this environment.")
        return torch.device('mps')

    # Let torch validate other/rare device strings (e.g., 'xpu', 'cuda:0' variants, etc.)
    return torch.device(str(raw).strip())

try:
    RandomHorizontalFlip = transforms.RandomHorizontalFlip
except AttributeError:
    warnings.warn(
        "torchvision.transforms.RandomHorizontalFlip not found; using a fallback implementation.",
        RuntimeWarning,
    )

    class RandomHorizontalFlip:
        def __init__(self, p: float = 0.5):
            self.p = p

        def __call__(self, img):
            if random.random() >= self.p:
                return img
            if isinstance(img, torch.Tensor):
                return torch.flip(img, dims=[-1])
            return img.transpose(_PIL_TRANSPOSE.FLIP_LEFT_RIGHT)

try:
    RandomVerticalFlip = transforms.RandomVerticalFlip
except AttributeError:
    warnings.warn(
        "torchvision.transforms.RandomVerticalFlip not found; using a fallback implementation.",
        RuntimeWarning,
    )

    class RandomVerticalFlip:
        def __init__(self, p: float = 0.5):
            self.p = p

        def __call__(self, img):
            if random.random() >= self.p:
                return img
            if isinstance(img, torch.Tensor):
                return torch.flip(img, dims=[-2])
            return img.transpose(_PIL_TRANSPOSE.FLIP_TOP_BOTTOM)

# Encourage TensorFlow to grow GPU memory usage as needed rather than
# pre-allocating the full device. This makes it easier to run multiple
# training processes on the same GPU in parallel (subject to VRAM limits).
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

# Ensure legacy Keras 2 API for pipeline modules
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
try:
    import tf_keras
    sys.modules['keras'] = tf_keras
except Exception:
    pass

# Ensure we are in the repo root and the pipeline is importable.

def _find_repo_root():
    here = Path.cwd()
    candidates = [here, *here.parents]
    for name in (
        'Deep-Learning-and-Single-Cell-Phenotyping-for-Rapid-Antimicrobial-Susceptibility-Testing',
        'Deep-Learning-and-Single-Cell-Phenotyping-for-Rapid-Antimicrobial-Susceptibility-Testing-main',
    ):
        candidates.append(here / name)
    for p in candidates:
        if (p / 'pipeline' / 'helpers.py').exists():
            return p
    raise FileNotFoundError('Repo root not found (missing pipeline/helpers.py). Re-run the clone cell.')

repo_root = _find_repo_root()
os.chdir(repo_root)
print('Repo root:', repo_root)

pipeline_dir = str(repo_root / 'pipeline')
if pipeline_dir not in sys.path:
    sys.path.insert(0, pipeline_dir)

import importlib
import helpers
importlib.reload(helpers)
from datetime import datetime

from pathlib import Path

# -----------------------------
# CLI argument parsing
# -----------------------------

ARGS = None
CFG = {}
TORCH_DEVICE = None
GLOBAL_SEED = 42
if __name__ == '__main__':
    import argparse
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "PyYAML is required for configuration loading. Install it with 'pip install pyyaml'."
        ) from e

    parser = argparse.ArgumentParser(
        description='Train EfficientNetB0 classifier with Optuna tuning on MG1655 single-cell data.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/mg1655_classifier_default.yaml',
        help='Path to a YAML config file (default: configs/mg1655_classifier_default.yaml).',
    )

    ARGS = parser.parse_args()
    cfg_path = Path(ARGS.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = Path(repo_root) / cfg_path
    if not cfg_path.is_file():
        raise FileNotFoundError(f'Config file not found: {cfg_path}')
    with cfg_path.open('r') as f:
        CFG = yaml.safe_load(f) or {}

    GLOBAL_SEED = int(CFG.get('seed', 42))
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    try:
        import tensorflow as tf

        try:
            tf.random.set_seed(GLOBAL_SEED)
        except AttributeError:
            pass
    except Exception:
        # TensorFlow not available; skip TF seeding.
        pass
    try:
        torch.manual_seed(GLOBAL_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(GLOBAL_SEED)
    except Exception:
        # Torch not available or failed to seed; continue anyway.
        pass

    TORCH_DEVICE = _resolve_torch_device(CFG)
    print('PyTorch device:', TORCH_DEVICE)

# Build a real (labeled) single-cell dataset from the downloaded `Zagajewski_Data` (MG1655)
# Section 4 shows this structure exists:
#   ./data/Zagajewski_Data/Data/MG1655/All_images/<COND>/*.tif
#   ./data/Zagajewski_Data/Data/MG1655/All_segmentations/<COND>/*.tif
# We'll turn the integer-encoded segmentation masks into single-cell instance masks, then crop cells and train.

RAW_ROOT = Path('./data/Zagajewski_Data/Data/MG1655')
IMAGES_ROOT = RAW_ROOT / 'All_images'
SEGS_ROOT = RAW_ROOT / 'All_segmentations'
if not (IMAGES_ROOT.exists() and SEGS_ROOT.exists()):
    raise FileNotFoundError('MG1655 data not found under ./data/Zagajewski_Data/Data/MG1655. Run the download/unzip + Section 4 listing first.')

available_conditions = sorted({p.name for p in IMAGES_ROOT.iterdir() if p.is_dir()} & {p.name for p in SEGS_ROOT.iterdir() if p.is_dir()})
print('Available MG1655 conditions:', available_conditions)

# Pick the conditions you want to classify (two-class by default). Edit this list.
DEFAULT_COND_IDS = ['WT+ETOH', 'CIP+ETOH']
COND_IDS = CFG.get('conditions', DEFAULT_COND_IDS)
PATIENT_SPLIT = bool(CFG.get('patient_split', False))  # patient-level Train/Val/Test splits
for cond in COND_IDS:
    if cond not in available_conditions:
        raise ValueError(f'Condition {cond!r} not found. Choose from: {available_conditions}')

def _slug(s: str) -> str:
    return ''.join(ch if ch.isalnum() else '_' for ch in s)

# IMPORTANT (leakage): we split by *image* first (Train/Validation/Test), then crop cells inside each split.
# Do not re-split at the single-cell level, otherwise cells from the same microscope image can leak across splits.
TEST_SIZE = float(CFG.get('test_size', 0.10))
VAL_SIZE = float(CFG.get('val_size', 0.20))  # fraction of remaining Train+Val
DATASET_OUT = Path('./data') / (
    f"classifier_dataset_mg1655_{'__'.join(_slug(c) for c in COND_IDS)}_t{int(TEST_SIZE*100)}_v{int(VAL_SIZE*100)}"
)
OUTPUT_CLASSIFIER_DIR = CFG.get('output_dir', './models/classification')
os.makedirs(OUTPUT_CLASSIFIER_DIR, exist_ok=True)

import mask_generators
import implementations
importlib.reload(mask_generators)
importlib.reload(implementations)
from mask_generators import masks_from_integer_encoding
from implementations import TrainTestVal_split

# Ensure single-cell instance masks exist for each condition
image_sources = []
annot_sources = []
for cond in COND_IDS:
    img_dir = IMAGES_ROOT / cond
    seg_dir = SEGS_ROOT / cond
    annots_dir = seg_dir / 'annots'

    if not annots_dir.exists() or not any(annots_dir.iterdir()):
        print(f'Generating single-cell masks for {cond}...')
        masks_from_integer_encoding(mask_path=str(seg_dir), output_path=str(seg_dir), combined_convention=False)

    image_sources.append(str(img_dir))
    annot_sources.append(str(annots_dir))

# Create a standard dataset layout (Train/Validation/Test) if needed
train_split = DATASET_OUT / 'Train'
val_split = DATASET_OUT / 'Validation'
test_split = DATASET_OUT / 'Test'
if not ((train_split / 'images').exists() and (train_split / 'annots').exists()):
    print('Creating classification dataset at:', DATASET_OUT)
    TrainTestVal_split(
        data_sources=image_sources,
        annotation_sources=annot_sources,
        output_folder=str(DATASET_OUT),
        test_size=TEST_SIZE,
        validation_size=VAL_SIZE,
        seed=GLOBAL_SEED,
    )

# Sanity check: ensure image-level splits are disjoint
def _image_ids(split_root: Path):
    p = split_root / 'images'
    if not p.exists():
        return set()
    return {f.name for f in p.iterdir() if f.is_file()}

train_ids = _image_ids(train_split)
val_ids = _image_ids(val_split)
test_ids = _image_ids(test_split)
print('Image files (train/val/test):', len(train_ids), len(val_ids), len(test_ids))
overlaps = {
    'train∩val': len(train_ids & val_ids),
    'train∩test': len(train_ids & test_ids),
    'val∩test': len(val_ids & test_ids),
}
print('Overlaps (image-level splits):', overlaps)
HAS_IMAGE_LEVEL_OVERLAP = any(v > 0 for v in overlaps.values())
assert not HAS_IMAGE_LEVEL_OVERLAP, 'Image-level Train/Val/Test splits overlap; this indicates a bug in TrainTestVal_split.'

# Convert folder-of-instance-masks -> cropped single-cell images (streaming; RAM-safe)
#
# IMPORTANT: `struct_from_file(...)` loads *all* masks into RAM and can easily exceed Colab's ~12GB limit.
# This streaming approach only keeps up to `N_PER_CLASS` crops per class in memory.
import re
import gc
import skimage
import skimage.io
from skimage.transform import resize

# Optionally cap the number of cell crops per class to keep system RAM bounded.
# Set to a smaller number (e.g. 200-1000) if you still hit OOM. Set to None to use all cells (can OOM).
N_PER_CLASS = CFG.get('n_per_class', 2000)
# Keep this in sync with the model input size below.
CROP_TARGET_SIZE = (64, 64, 3)


def split_cell_sets(input=None, **kwargs):
    """
    Simple reimplementation of classification.split_cell_sets.

    Flattens the per-class cells structure into (X, y) and performs a
    stratified train/test split using sklearn.
    """
    import collections
    from sklearn.model_selection import train_test_split

    cells_struct = input or {}
    total_cells = []
    total_ids = []

    for mapping in cells_struct.get('class_id_to_name', []):
        name = mapping['name']
        cid = int(mapping['class_id'])
        cells = cells_struct.get(name, [])
        total_cells.extend(cells)
        total_ids.extend([cid] * len(cells))

    test_size = kwargs.get('test_size', 0.2)
    random_state = kwargs.get('random_state', None)

    if not total_cells:
        raise ValueError('split_cell_sets: no cells found in input.')

    X_train, X_test, y_train, y_test = train_test_split(
        total_cells,
        total_ids,
        stratify=total_ids,
        test_size=test_size,
        random_state=random_state,
    )

    # Optional: print basic class-distribution information for sanity-checking
    counts_train = collections.Counter(y_train)
    counts_test = collections.Counter(y_test)
    print('Train class counts:', dict(counts_train))
    print('Test class counts:', dict(counts_test))

    return X_train, X_test, y_train, y_test

def _find_image_path(images_root: Path, annot_folder_name: str) -> Path:
    candidates = [
        images_root / annot_folder_name,
        images_root / (Path(annot_folder_name).stem + '.tif'),
        images_root / (Path(annot_folder_name).stem + '.tiff'),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f'No image found for annot folder {annot_folder_name!r} under {images_root}')

def _match_cond(name: str, cond_ids: list[str]) -> str | None:
    matches = []
    for cond in cond_ids:
        if re.search(re.escape(cond), name) is not None:
            matches.append(cond)
    return matches[0] if len(matches) == 1 else None

def _crop_cells_streaming(dataset_root: Path, cond_ids: list[str], max_per_class: int | None, seed: int = 42):
    rng_local = np.random.default_rng(seed)
    images_root = dataset_root / 'images'
    annots_root = dataset_root / 'annots'
    if not (images_root.exists() and annots_root.exists()):
        raise FileNotFoundError(f'Expected {images_root} and {annots_root} to exist.')

    out = {'class_id_to_name': []}
    for i, cond in enumerate(cond_ids):
        out[cond] = []
        out['class_id_to_name'].append({'class_id': i, 'name': cond})

    kept = {cond: 0 for cond in cond_ids}
    removed_edge = 0
    skipped_nomatch = 0

    annot_dirs = [p for p in annots_root.iterdir() if p.is_dir()]
    rng_local.shuffle(annot_dirs)

    for annot_dir in annot_dirs:
        cond = _match_cond(annot_dir.name, cond_ids)
        if cond is None:
            skipped_nomatch += 1
            continue
        if max_per_class is not None and kept[cond] >= max_per_class:
            continue

        img_path = _find_image_path(images_root, annot_dir.name)
        image = skimage.io.imread(str(img_path))
        if image.ndim == 2:
            image = image[..., None]

        # Ensure 3 channels for ImageNet backbones (DenseNet/ResNet/VGG).
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] > 3:
            image = image[..., :3]

        sy, sx = image.shape[:2]

        mask_files = [p for p in annot_dir.iterdir() if p.suffix.lower() == '.bmp']
        rng_local.shuffle(mask_files)

        for mask_path in mask_files:
            if max_per_class is not None and kept[cond] >= max_per_class:
                break

            mask = skimage.io.imread(str(mask_path))
            mask = np.asarray(mask)
            if mask.ndim == 3:
                # Some readers return RGB; collapse to a single channel.
                mask = mask.max(axis=-1)
            # Support both 0/1 and 0/255 saved masks.
            mask = mask > 0
            if not mask.any():
                continue

            ys, xs = np.where(mask)
            y1, y2 = int(ys.min()), int(ys.max()) + 1  # y2 exclusive
            x1, x2 = int(xs.min()), int(xs.max()) + 1  # x2 exclusive

            # Mirror `helpers.remove_edge_cells()` semantics (Mask R-CNN rois are exclusive at y2/x2).
            on_edge = (y1 == 0) or (x1 == 0) or (y2 == 0) or (x2 == 0) or (y1 >= sy) or (x1 >= sx) or (y2 >= sy) or (x2 >= sx)
            if on_edge:
                removed_edge += 1
                continue

            roi = mask[y1:y2, x1:x2]
            crop = image[y1:y2, x1:x2, :].copy()
            crop *= roi[..., None]

            # Resize early so every sample is the same (small) size -> much lower RAM.
            crop = resize(crop, CROP_TARGET_SIZE, anti_aliasing=True)
            crop = skimage.img_as_ubyte(crop)
            out[cond].append(crop)
            kept[cond] += 1

        if max_per_class is not None and all(kept[c] >= max_per_class for c in cond_ids):
            break

    return out, removed_edge, skipped_nomatch

# Crop cells separately per split (prevents leakage across Train/Val/Test)
N_PER_CLASS_TRAIN = N_PER_CLASS
N_PER_CLASS_VAL = min(1000, N_PER_CLASS)  # lower by default to save RAM; set to None for all
N_PER_CLASS_TEST = None  # set to an int to cap RAM usage

cells_train, removed_train, skipped_train = _crop_cells_streaming(
    dataset_root=train_split,
    cond_ids=COND_IDS,
    max_per_class=N_PER_CLASS_TRAIN,
    seed=GLOBAL_SEED,
)
cells_val, removed_val, skipped_val = _crop_cells_streaming(
    dataset_root=val_split,
    cond_ids=COND_IDS,
    max_per_class=N_PER_CLASS_VAL,
    seed=GLOBAL_SEED + 1,
)
cells_test, removed_test, skipped_test = _crop_cells_streaming(
    dataset_root=test_split,
    cond_ids=COND_IDS,
    max_per_class=N_PER_CLASS_TEST,
    seed=GLOBAL_SEED + 2,
)

print('Removed edge cells (train/val/test):', removed_train, removed_val, removed_test)
if skipped_train or skipped_val or skipped_test:
    print('WARNING: skipped annotation folders with ambiguous/no cond match:', {'train': skipped_train, 'val': skipped_val, 'test': skipped_test})

for split_name, cells_split in [('train', cells_train), ('val', cells_val), ('test', cells_test)]:
    print('---', split_name, '---')
    for cond in COND_IDS:
        print(cond, 'cells:', len(cells_split[cond]))

def _to_xy(cells_struct):
    X, y = [], []
    for mapping in cells_struct['class_id_to_name']:
        cond = mapping['name']
        cid = int(mapping['class_id'])
        X.extend(cells_struct[cond])
        y.extend([cid] * len(cells_struct[cond]))
    return X, np.asarray(y, dtype='int32')

if PATIENT_SPLIT:
    # Current, non-leaky behaviour: Train/Val/Test are disjoint at the image level,
    # and we keep that separation when training the classifier.
    X_train, y_train = _to_xy(cells_train)
    X_val, y_val = _to_xy(cells_val)
    X_test, y_test = _to_xy(cells_test)
    print('Split sizes (image-level, per-split cropping):', {'train': len(X_train), 'val': len(X_val), 'test': len(X_test)})
    HAS_SPLIT_OVERLAP = False
else:
    # Leaky behaviour (for comparison with the original pipeline):
    # merge all cropped cells across Train/Val/Test and split at the single-cell level.
    # This ignores image-level grouping, so cells from the same image can land in
    # both train and test, which is analogous to the old implementation.
    cells_all = {'class_id_to_name': cells_train['class_id_to_name']}
    for mapping in cells_all['class_id_to_name']:
        cond = mapping['name']
        cells_all[cond] = cells_train[cond] + cells_val[cond] + cells_test[cond]

    TEST_FRACTION = TEST_SIZE
    X_train, X_test, y_train, y_test = split_cell_sets(
        input=cells_all,
        test_size=TEST_FRACTION,
        random_state=GLOBAL_SEED,
    )
    X_val, y_val = None, None
    print('Split sizes (cell-level, leaky):', {'train': len(X_train), 'test': len(X_test)})
    HAS_SPLIT_OVERLAP = True

if any(len(cells_train[c]) == 0 for c in COND_IDS):
    raise ValueError('No cells were extracted for one or more classes in TRAIN. Check masks/images alignment.')
if any(len(cells_val[c]) == 0 for c in COND_IDS):
    print('WARNING: one or more classes have 0 cells in VALIDATION (split may be too small).')
if any(len(cells_test[c]) == 0 for c in COND_IDS):
    print('WARNING: one or more classes have 0 cells in TEST (split may be too small).')

gc.collect()

# Sanity checks: if accuracy is near chance, first confirm the crops look correct.
min_cells = min(len(cells_train[c]) for c in COND_IDS)
if min_cells < 1000:
    print('WARNING: very few cells per class; accuracy will likely be low. Consider using more data (N_PER_CLASS=None) or different conditions.')

import matplotlib.pyplot as plt

rng = np.random.default_rng(GLOBAL_SEED)

def _plot_examples(label, imgs, n=6):
    if not imgs:
        return
    k = min(n, len(imgs))
    idx = rng.choice(len(imgs), size=k, replace=False)
    fig, axs = plt.subplots(1, k, figsize=(2*k, 2))
    fig.suptitle(f'Example cropped cells: {label}')
    if k == 1:
        axs = [axs]
    for ax, i in zip(axs, idx):
        img = imgs[i]
        if getattr(img, 'ndim', 0) == 3 and img.shape[-1] >= 3:
            ax.imshow(img[..., :3])
        else:
            ax.imshow(np.squeeze(img), cmap='gray')
        ax.axis('off')
    plt.show()

for cond in COND_IDS:
    _plot_examples(cond, cells_train[cond], n=6)

def _mean_intensity(imgs, n=200):
    if not imgs:
        return 0.0, 0.0
    k = min(n, len(imgs))
    idx = rng.choice(len(imgs), size=k, replace=False)
    vals = [float(np.asarray(imgs[i]).mean()) for i in idx]
    return float(np.mean(vals)), float(np.std(vals))

for cond in COND_IDS:
    mean_i, std_i = _mean_intensity(cells_train[cond])
    print(f'{cond}: mean intensity ~ {mean_i:.2f} (std {std_i:.2f})')

# `N_PER_CLASS_*` sampling happens above (during streaming collection), to avoid loading the full dataset into RAM.

# Metadata tags used for naming Optuna studies and output folders
COND_TAG = '_'.join(_slug(c) for c in COND_IDS)
SPLIT_MODE_TAG = 'patientSplitOn' if PATIENT_SPLIT else 'patientSplitOff'
OVERLAP_TAG = 'overlap' if HAS_SPLIT_OVERLAP else 'noOverlap'
SPLIT_TAG = f"{SPLIT_MODE_TAG}_{OVERLAP_TAG}"

# -----------------------------
# PyTorch model + training
# -----------------------------

MODEL_TYPE = 'EfficientNetB0'
INIT_SOURCE = 'imagenet'  # ImageNet pretrained weights
TARGET_SIZE = CROP_TARGET_SIZE
NUM_CLASSES = len(COND_IDS)
BATCH_SIZE = int(CFG.get('batch_size', 4))
EPOCHS = int(CFG.get('epochs', 100))
OPTIMIZER = str(CFG.get('optimizer', 'NAdam'))

if len(X_train) < BATCH_SIZE:
    raise ValueError(
        f'Too few training samples ({len(X_train)}) for batch_size={BATCH_SIZE}. '
        'Reduce batch_size or increase N_PER_CLASS.'
    )

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CellsDataset(Dataset):
    def __init__(self, X, y, train=True, augment=True):
        self.X = np.asarray(X)
        self.y = np.asarray(y, dtype='int64')
        self.train = train
        base = [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        if train and augment:
            self.transform = transforms.Compose(
                [
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    transforms.RandomRotation(90),
                    *base,
                ]
            )
        else:
            self.transform = transforms.Compose(base)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        img = np.asarray(img)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] > 3:
            img = img[..., :3]
        img = Image.fromarray(img.astype(np.uint8))
        x = self.transform(img)
        y = int(self.y[idx])
        return x, y


def _build_torch_model(num_classes: int, dropout_rate: float, init_source: str = 'imagenet') -> nn.Module:
    try:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if init_source == 'imagenet' else None
        model = efficientnet_b0(weights=weights)
    except Exception:
        from torchvision import models as _models

        model = _models.efficientnet_b0(pretrained=(init_source == 'imagenet'))

    in_features = model.classifier[1].in_features
    head_layers = []
    if dropout_rate > 0:
        head_layers.append(nn.Dropout(p=float(dropout_rate)))
    head_layers.append(nn.Linear(in_features, num_classes))
    model.classifier[1] = nn.Sequential(*head_layers)
    return model


def _build_optimizer(params, name: str, lr: float):
    name = str(name)
    if name == 'NAdam':
        return torch.optim.NAdam(params, lr=lr)
    if name == 'Adam':
        return torch.optim.Adam(params, lr=lr)
    if name == 'SGD+N':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    if name == 'SGD':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=False)
    raise ValueError(f'Optimizer {name!r} not supported.')


def _run_torch_training(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    num_classes: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    dropout_rate: float,
    optimizer_name: str,
    patience: int | None,
    logdir: str,
    global_seed: int,
    trial=None,
) -> float:
    os.makedirs(logdir, exist_ok=True)

    # If explicit validation not provided, split from training (leaky mode).
    if X_val is None or y_val is None:
        from sklearn.model_selection import train_test_split

        X_train_arr = np.asarray(X_train)
        y_train_arr = np.asarray(y_train, dtype='int64')
        X_train_arr, X_val_arr, y_train_arr, y_val_arr = train_test_split(
            X_train_arr,
            y_train_arr,
            test_size=0.2,
            stratify=y_train_arr,
            random_state=global_seed,
        )
    else:
        X_train_arr = np.asarray(X_train)
        y_train_arr = np.asarray(y_train, dtype='int64')
        X_val_arr = np.asarray(X_val)
        y_val_arr = np.asarray(y_val, dtype='int64')

    train_ds = CellsDataset(X_train_arr, y_train_arr, train=True, augment=True)
    val_ds = CellsDataset(X_val_arr, y_val_arr, train=False, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = TORCH_DEVICE or _resolve_torch_device(CFG)
    model = _build_torch_model(num_classes, dropout_rate, init_source=INIT_SOURCE).to(device)
    optimizer = _build_optimizer(model.parameters(), optimizer_name, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_val_acc = -1.0
    best_ckpt_path = os.path.join(logdir, 'best_model.pth')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == targets).sum().item()
            running_total += inputs.size(0)

        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss_sum += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += inputs.size(0)

        val_loss = val_loss_sum / max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        print(
            f'Epoch {epoch + 1}/{epochs} - '
            f'train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, '
            f'val_loss={val_loss:.4f}, val_acc={val_acc:.4f}',
            flush=True,
        )

        # Optuna reporting / pruning
        if trial is not None:
            trial.report(val_acc, step=epoch)
            try:
                import optuna as _optuna  # noqa: F401

                if trial.should_prune():
                    raise _optuna.TrialPruned(f'Pruned at epoch {epoch}, val_acc={val_acc}')
            except ModuleNotFoundError:
                pass

        # Early stopping + checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(
                {
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'val_acc': float(val_acc),
                },
                best_ckpt_path,
            )
        else:
            epochs_no_improve += 1

        if patience is not None and epochs_no_improve >= patience:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break

    return float(best_val_acc), best_ckpt_path


# Optuna: Bayesian optimization (TPE sampler) + pruning
optuna_cfg = CFG.get('optuna', {})
DO_OPTUNA = bool(optuna_cfg.get('enabled', True))
N_TRIALS = int(optuna_cfg.get('n_trials', 25))
EPOCHS_TUNE = int(optuna_cfg.get('epochs_tune', 40))

best_params = None
_best_study_state = {'best_val': float('-inf'), 'ckpt_path': None}

if DO_OPTUNA:
    import optuna

    tune_dt = datetime.now().strftime('%Y%m%d-%H%M%S')
    tune_name = f"optuna_{MODEL_TYPE}_{COND_TAG}_{SPLIT_TAG}_{tune_dt}_pid{os.getpid()}"
    TUNE_DIR = os.path.join(OUTPUT_CLASSIFIER_DIR, tune_name)
    os.makedirs(TUNE_DIR, exist_ok=True)
    print('Starting Optuna study. Base dir:', TUNE_DIR)

    sampler = optuna.samplers.TPESampler(seed=GLOBAL_SEED, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study_storage_url = f"sqlite:///{os.path.join(TUNE_DIR, 'optuna_study.db')}"
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name=tune_name,
        storage=study_storage_url,
        load_if_exists=True,
    )

    def objective(trial):
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.6)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 3e-3, log=True)
        patience = trial.suggest_int('patience', 3, 12)

        trial_dt = f"trial_{trial.number:03d}"
        trial_logdir = os.path.join(TUNE_DIR, trial_dt)
        os.makedirs(trial_logdir, exist_ok=True)

        score, ckpt_path = _run_torch_training(
            X_train,
            y_train,
            X_val,
            y_val,
            num_classes=NUM_CLASSES,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS_TUNE,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            optimizer_name=OPTIMIZER,
            patience=patience,
            logdir=trial_logdir,
            global_seed=GLOBAL_SEED,
            trial=trial,
        )

        if ckpt_path is not None and score is not None:
            try:
                current_val = float(score)
                if current_val > _best_study_state['best_val']:
                    _best_study_state['best_val'] = current_val
                    best_dst = os.path.join(TUNE_DIR, 'best_study_model.pth')
                    shutil.copy2(ckpt_path, best_dst)
                    _best_study_state['ckpt_path'] = best_dst
                    print(
                        f'Updated best study model: trial {trial.number}, '
                        f'val_accuracy={current_val:.4f} -> {best_dst}'
                    )
            except Exception as e:
                print('WARNING: failed to update best study model checkpoint:', e)

        return score

    study.optimize(objective, n_trials=N_TRIALS)

    # Persist Optuna study to disk for later analysis / reuse
    import pickle

    try:
        df = study.trials_dataframe()
        df.to_csv(os.path.join(TUNE_DIR, 'trials.csv'), index=False)
    except Exception as e:
        print('WARNING: failed to save Optuna trials dataframe:', e)

    try:
        with open(os.path.join(TUNE_DIR, 'study.pkl'), 'wb') as f:
            pickle.dump(study, f)
    except Exception as e:
        print('WARNING: failed to pickle Optuna study:', e)

    print('Best trial:', study.best_trial.number)
    print('Best params:', study.best_trial.params)
    if _best_study_state['best_val'] > float('-inf'):
        print(f"Best study val_accuracy across trials: {_best_study_state['best_val']:.4f}")

    best_params = study.best_trial.params
else:
    best_params = {'dropout_rate': 0.2, 'learning_rate': 3e-4, 'patience': 8}

LEARNING_RATE = float(best_params['learning_rate'])
DROPOUT = float(best_params['dropout_rate'])
PATIENCE = int(best_params['patience'])

dt_string = datetime.now().strftime('%Y%m%d-%H%M%S')
run_name = f"{MODEL_TYPE}_{COND_TAG}_{SPLIT_TAG}_{dt_string}_pid{os.getpid()}"
LOG_DIR = os.path.join(OUTPUT_CLASSIFIER_DIR, run_name)
os.makedirs(LOG_DIR, exist_ok=True)
print('Starting final classification training (PyTorch AMP). Log dir:', LOG_DIR)
print('Params:', {'lr': LEARNING_RATE, 'dropout': DROPOUT, 'patience': PATIENCE})

final_val_acc, final_ckpt = _run_torch_training(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes=NUM_CLASSES,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    dropout_rate=DROPOUT,
    optimizer_name=OPTIMIZER,
    patience=PATIENCE,
    logdir=LOG_DIR,
    global_seed=GLOBAL_SEED,
    trial=None,
)

print('Final training finished. Best val_accuracy:', final_val_acc)

# Save a stable copy of the best final (retrain) model in this LOG_DIR.
try:
    if final_ckpt is not None and os.path.isfile(final_ckpt):
        best_retrain_dst = os.path.join(LOG_DIR, 'best_retrain_model.pth')
        shutil.copy2(final_ckpt, best_retrain_dst)
        print('Saved best retrain model to:', best_retrain_dst)
    else:
        print('WARNING: no best checkpoint found in final LOG_DIR:', LOG_DIR)
except Exception as e:
    print('WARNING: failed to save best retrain model checkpoint:', e)

# Hold-out test evaluation (if test set is available)
if X_test and y_test:
    device = TORCH_DEVICE or _resolve_torch_device(CFG)
    ckpt_path = best_retrain_dst if 'best_retrain_dst' in globals() else final_ckpt
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model = _build_torch_model(NUM_CLASSES, DROPOUT, init_source=INIT_SOURCE).to(device)
        model.load_state_dict(state['model_state'])
        model.eval()

        test_ds = CellsDataset(np.asarray(X_test), np.asarray(y_test, dtype='int64'), train=False, augment=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += inputs.size(0)

        test_acc = correct / max(1, total)
        print('Hold-out test:', {'acc': float(test_acc)})
