import numpy as np
from PIL import Image
from skimage.morphology import white_tophat, opening, disk as sk_disk
from skimage.exposure import rescale_intensity

def percentile_clip_u8(arr_u8, p=99.5):
    """
    Clips the array values at the p-th percentile and scales to 0-255 uint8.
    """
    hi = np.percentile(arr_u8, p)
    arr = np.clip(arr_u8.astype(np.float32), 0, hi) / (hi + 1e-6)
    return (arr * 255).astype(np.uint8)

def rolling_shading_correction_rgb(im: Image.Image, rad_px=27) -> Image.Image:
    """
    Applies white_tophat transform to correct shading on RGB images.
    """
    arr = np.asarray(im).astype(np.uint8)
    se = sk_disk(rad_px)
    out = np.empty_like(arr)
    for c in range(3):
        # white_tophat returns the bright spots smaller than SE (subtracts local background)
        # Note: In the notebook, it seems white_tophat was used directly. 
        # Standard shading correction usually involves dividing by the background or subtracting it.
        # Here we follow the notebook's logic: Tophat IS the corrected image (foreground).
        out[...,c] = white_tophat(arr[...,c], footprint=se)
        out[...,c] = rescale_intensity(out[...,c], in_range='image', out_range=(0,255)).astype(np.uint8)
    return Image.fromarray(out, "RGB")

def suppress_green_ring(im: Image.Image, ring_rad=13) -> Image.Image:
    """
    Suppresses the green halo artifact using morphological opening.
    """
    arr = np.asarray(im).astype(np.uint8)
    G = arr[...,1]
    G_bg   = opening(G, sk_disk(ring_rad))
    # Subtract background from G channel
    G_supp = np.clip(G.astype(np.int16) - G_bg.astype(np.int16), 0, 255).astype(np.uint8)
    arr[...,1] = G_supp
    return Image.fromarray(arr, "RGB")

def letterbox(im: Image.Image, size=96, bg=(0,0,0)) -> Image.Image:
    """
    Resizes image to fit within 'size' x 'size' box, preserving aspect ratio, and pads with 'bg'.
    """
    w, h = im.size
    s = min(size/w, size/h)
    nw, nh = int(round(w*s)), int(round(h*s))
    canvas = Image.new("RGB", (size,size), bg)
    canvas.paste(im.resize((nw,nh), Image.Resampling.LANCZOS),
                 ((size-nw)//2, (size-nh)//2))
    return canvas
