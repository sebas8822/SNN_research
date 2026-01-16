import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from skimage.filters import threshold_otsu, threshold_sauvola, apply_hysteresis_threshold
from skimage.morphology import remove_small_objects, remove_small_holes, binary_opening, binary_closing

# Add src to path to allow direct execution
sys.path.append(str(Path(__file__).parent.parent))

from src import config
from src import data_loader
from src import preprocessing
from src import segmentation

def ensure_dirs():
    config.PATCH_CACHE_TRAIN.mkdir(exist_ok=True)
    config.MASKS_DIR.mkdir(exist_ok=True)
    for c in config.CLASSES:
        (config.PATCH_CACHE_TRAIN / c).mkdir(parents=True, exist_ok=True)
        (config.MASKS_DIR / c).mkdir(parents=True, exist_ok=True)

def run_pipeline(limit: int = 0):
    print("Initializing pipeline...")
    ensure_dirs()
    
    loader = data_loader.CocoLoader()
    annotations = loader.get_annotations()
    
    if limit > 0:
        annotations = annotations[:limit]
        print(f"Limiting to first {limit} annotations.")

    print(f"Processing {len(annotations)} annotations...")
    
    miss = 0
    bad = 0
    qc_rows = []
    
    ann_to_patch = {}
    
    # ── Phase 1: Patch Extraction ────────────────────
    print("Phase 1: Extracting Patches")
    for ann in tqdm(annotations, desc="Patches"):
        cls = loader.get_category_name(ann["category_id"])
        file_name = loader.get_filename(ann["image_id"])
        
        src = loader.resolve_image_path(file_name) if file_name else None
        if src is None:
            miss += 1; continue
            
        x, y, w, h = map(float, ann.get("bbox", [0,0,0,0]))
        if w <= 1 or h <= 1: 
            bad += 1; continue

        img = Image.open(src).convert("RGB")
        img = preprocessing.rolling_shading_correction_rgb(img)
        img = preprocessing.suppress_green_ring(img, ring_rad=13)

        pad = config.PAD_FRAC.get(cls, config.DEFAULT_PAD)
        x0 = max(0, int(round(x - pad*w)))
        y0 = max(0, int(round(y - pad*h)))
        x1 = min(img.width,  int(round(x + w + pad*w)))
        y1 = min(img.height, int(round(y + h + pad*h)))
        
        if x1 <= x0 or y1 <= y0: 
            bad += 1; continue

        crop = img.crop((x0,y0,x1,y1))
        patch = preprocessing.letterbox(crop, config.IMG_SIZE)
        
        if config.MIP_MODE:
            arr = np.asarray(patch).astype(np.uint8)
            for c in range(3): 
                arr[...,c] = preprocessing.percentile_clip_u8(arr[...,c], p=99.5)
            patch = Image.fromarray(arr, "RGB")

        outp = config.PATCH_CACHE_TRAIN / cls / f"patch_{ann['id']}.png"
        patch.save(outp)
        ann_to_patch[ann["id"]] = (str(outp), cls)

    print(f"Cached: {len(ann_to_patch)} | missing: {miss} | bad bbox: {bad}")

    # ── Phase 2: Segmentation ────────────────────
    print("Phase 2: Segmentation and QC")
    for ann_id, (pth, cls) in tqdm(ann_to_patch.items(), desc="Masks"):
        patch_path = Path(pth)
        rgb = np.asarray(Image.open(patch_path).convert("RGB")).astype(np.uint8)
        
        rgbp = segmentation.preproc_rgb(rgb)
        R, G = rgbp[..., 0], rgbp[..., 1]

        # ratio + Otsu-anchored hysteresis (default)
        idx = segmentation.index_rg_smooth(R, G)
        thr = threshold_otsu(idx)
        hi, lo = min(1.0, thr + 0.06), max(0.0, thr - 0.06)
        nuc_default = apply_hysteresis_threshold(idx, lo, hi)

        if cls == "Metaphase":
            nuc_seed = remove_small_objects(nuc_default, max(40, segmentation.MIN_SIZE_NUC//2))
            nuc, det = segmentation.metaphase_nucleus_radial(idx, G, nuc_seed)
            nuc = remove_small_objects(nuc, segmentation.MIN_SIZE_NUC)
            nuc = remove_small_holes(nuc, segmentation.HOLE_SIZE)
            nuc = segmentation.watershed_keep_largest(nuc)

            cytraw = segmentation.metaphase_cyto_radial(G, det)
            ring = segmentation.cyto_ring_adaptive(nuc) if det is None else np.ones_like(nuc, bool)
            cyt = cytraw & ring
            cyt = remove_small_objects(cyt, segmentation.MIN_SIZE_CYT)
            cyt = remove_small_holes(cyt, area_threshold=segmentation.MIN_SIZE_CYT//2)

        elif cls == "Anaphase":
            nuc = remove_small_objects(nuc_default, segmentation.MIN_SIZE_NUC)
            nuc = segmentation.split_anaphase_lobes(nuc, min_dist=6)
            nuc = segmentation.watershed_keep_largest(nuc)
            th_map = threshold_sauvola(G, window_size=segmentation.LOC_BLOCK, k=segmentation.SAUV_K)
            cytraw = (G > th_map)
            ring = segmentation.cyto_ring_adaptive(nuc)
            cyt = cytraw & ring
            cyt = remove_small_objects(cyt, segmentation.MIN_SIZE_CYT)
            cyt = remove_small_holes(cyt, area_threshold=segmentation.MIN_SIZE_CYT//2)

        else:  # Prophase, Telophase
            nuc = remove_small_objects(nuc_default, segmentation.MIN_SIZE_NUC)
            nuc = remove_small_holes(nuc, segmentation.HOLE_SIZE)
            nuc = binary_opening(binary_closing(nuc))
            nuc = segmentation.watershed_keep_largest(nuc)
            th_map = threshold_sauvola(G, window_size=segmentation.LOC_BLOCK, k=segmentation.SAUV_K)
            cytraw = (G > th_map)
            ring = segmentation.cyto_ring_adaptive(nuc)
            cyt = cytraw & ring
            cyt = remove_small_objects(cyt, segmentation.MIN_SIZE_CYT)
            cyt = remove_small_holes(cyt, area_threshold=segmentation.MIN_SIZE_CYT//2)

        # Save masks
        Image.fromarray((nuc | cyt).astype(np.uint8) * 255).save(config.MASKS_DIR / cls / f"mask_{ann_id}.png")
        seg = np.zeros_like(rgb)
        seg[..., 0] = nuc.astype(np.uint8) * 255
        seg[..., 1] = cyt.astype(np.uint8) * 255
        Image.fromarray(seg).save(config.MASKS_DIR / cls / f"seg_{ann_id}.png")

        # QC metrics
        b = segmentation.BORDER_PX
        border = np.zeros_like(nuc, bool); border[:b,:]=border[-b:,:]=border[:,:b]=border[:,-b:]=True
        area = float(nuc.mean())
        snr  = float(R[nuc].mean() / (R[~nuc].std() + 1e-6)) if (~nuc).any() else 0.0
        brr  = float((nuc & border).sum()) / (nuc.size + 1e-6)
        qc_rows.append({"class": cls, "ann_id": ann_id, "area": area, "snr": snr, "border_red": brr})

    # Save QC manifest
    manifest_path = Path("kept_manifest_dev.json")
    with open(manifest_path, "w") as f:
        json.dump(qc_rows, f, indent=2)
    print(f"Saved QC manifest to {manifest_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion SNN ML Pipeline")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples for testing")
    args = parser.parse_args()
    
    run_pipeline(limit=args.limit)
