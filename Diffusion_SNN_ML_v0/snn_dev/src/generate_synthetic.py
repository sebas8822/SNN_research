import torch
from diffusers import UNet2DModel, DDPMScheduler
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src import config
from src import pipeline
from src import preprocessing
from src import segmentation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model(checkpoint_path):
    print(f"Loading model from {checkpoint_path}...")
    model = UNet2DModel.from_pretrained(checkpoint_path).to(DEVICE)
    model.eval()
    return model

def generate_and_filter(model, class_name, count=100, output_dir="synthetic_data"):
    class_idx = config.CLASSES.index(class_name)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    out_path = Path(output_dir) / class_name
    out_path.mkdir(parents=True, exist_ok=True)
    
    accepted = 0
    total_tried = 0
    
    manifest = []

    print(f"--- Generating and Filtering {class_name} ---")
    
    pbar = tqdm(total=count)
    while accepted < count:
        total_tried += 1
        # 1. Generate image
        image_pil = sample_one(model, scheduler, class_idx)
        
        # 2. Run through QC filter (Segmentation pipeline)
        is_valid, metrics = filter_image(image_pil, class_name)
        
        if is_valid:
            img_id = f"syn_{class_name}_{accepted}"
            img_file = out_path / f"{img_id}.png"
            image_pil.save(img_file)
            
            manifest.append({
                "id": img_id,
                "class": class_name,
                "file": str(img_file),
                "metrics": metrics
            })
            
            accepted += 1
            pbar.update(1)
            
    pbar.close()
    print(f"Completed {class_name}. Efficiency: {accepted/total_tried:.1%}")
    return manifest

def sample_one(model, scheduler, class_idx):
    image = torch.randn((1, 3, config.IMG_SIZE, config.IMG_SIZE)).to(DEVICE)
    labels = torch.tensor([class_idx]).to(DEVICE)
    
    scheduler.set_timesteps(1000)
    
    for t in scheduler.timesteps:
        with torch.no_grad():
            noisy_residual = model(image, t, class_labels=labels).sample
        image = scheduler.step(noisy_residual, t, image).prev_sample
        
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    return Image.fromarray((image * 255).astype("uint8"))

def filter_image(image_pil, class_name):
    """
    Uses the logic from src/pipeline.py and src/segmentation.py to verify 
    if the synthetic image is 'biologically plausible'.
    """
    rgb = np.asarray(image_pil).astype(np.uint8)
    
    # Preprocess (Denoise etc)
    rgbp = segmentation.preproc_rgb(rgb)
    R, G = rgbp[..., 0], rgbp[..., 1]
    
    # Simple check: Is there a nucleus?
    idx = segmentation.index_rg_smooth(R, G)
    from skimage.filters import threshold_otsu, apply_hysteresis_threshold
    from skimage.morphology import remove_small_objects
    
    try:
        thr = threshold_otsu(idx)
        hi, lo = min(1.0, thr + 0.06), max(0.0, thr - 0.06)
        nuc_bw = apply_hysteresis_threshold(idx, lo, hi)
        nuc_bw = remove_small_objects(nuc_bw, segmentation.MIN_SIZE_NUC)
        
        if nuc_bw.sum() < segmentation.MIN_SIZE_NUC:
            return False, None
            
        # Additional checks based on class
        if class_name == "Metaphase":
            # Check for ring?
            _, det = segmentation.metaphase_nucleus_radial(idx, G, nuc_bw)
            if det is None: return False, None
            
        elif class_name == "Anaphase":
            # Check for dual lobes?
            dist = segmentation.distance_transform_edt(nuc_bw)
            from skimage.feature import peak_local_max
            coords = peak_local_max(dist, min_distance=6, labels=nuc_bw)
            if len(coords) < 2: return False, None
            
        # If we reached here, it's valid!
        area = float(nuc_bw.mean())
        snr  = float(R[nuc_bw].mean() / (R[~nuc_bw].std() + 1e-6)) if (~nuc_bw).any() else 0.0
        
        return True, {"area": area, "snr": snr}
        
    except Exception:
        return False, None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/generate_synthetic.py <checkpoint_path>")
        sys.exit(1)
        
    checkpoint = sys.argv[1]
    model = load_trained_model(checkpoint)
    
    full_manifest = []
    # Generate for the rare classes
    for cls in ["Anaphase", "Telophase", "Metaphase"]:
        m = generate_and_filter(model, cls, count=100)
        full_manifest.extend(m)
        
    with open("synthetic_manifest.json", "w") as f:
        json.dump(full_manifest, f, indent=2)
    print("Generation complete results saved to synthetic_manifest.json")
