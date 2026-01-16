from pathlib import Path

# ---- choose projection and paths
PROJECTION = "max_projection"   # or "mean_projection"
MIP_MODE   = (PROJECTION == "max_projection")

# Resolve REPO path relative to this config file: src/config.py -> snn_dev/src -> snn_dev -> ProjectRoot
# So we need to go up 3 levels.
REPO      = Path(__file__).resolve().parent.parent.parent / "borg-main"
DATA_ROOT = REPO / "data" / PROJECTION
TRAIN_JSON = DATA_ROOT / "organoid_coco_train.json"
VAL_JSON   = DATA_ROOT / "organoid_coco_val.json"
TRAIN_DIR  = DATA_ROOT / "images" / "train"
VAL_DIR    = DATA_ROOT / "images" / "val"

# ---- outputs
IMG_SIZE = 96
PATCH_CACHE_TRAIN = Path("patch_cache_train")
MASKS_DIR         = Path("processed_masks_train")

CLASSES = ["Prophase", "Metaphase", "Anaphase", "Telophase"]

# Padding fractions for each class
PAD_FRAC = {"Prophase": 0.25, "Metaphase": 0.25, "Anaphase": 0.35, "Telophase": 0.35}
DEFAULT_PAD = 0.30
