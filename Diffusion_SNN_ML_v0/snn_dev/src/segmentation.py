import numpy as np
from skimage.filters import (
    threshold_local, apply_hysteresis_threshold, gaussian,
    threshold_otsu, threshold_sauvola, sobel
)
from skimage.morphology import (
    remove_small_holes, remove_small_objects, binary_opening, binary_closing,
    h_maxima
)
from skimage.segmentation import watershed, chan_vese
from skimage.measure import label, regionprops
from skimage.exposure import equalize_adapthist
from skimage.restoration import denoise_nl_means, estimate_sigma, rolling_ball
from skimage.feature import peak_local_max
from skimage.transform import warp_polar
from scipy.ndimage import distance_transform_edt

# ── params (tunable) ────────────────────────────────────────────────────────────
TOPHAT_RAD     = 7
MIN_SIZE_NUC   = 120
MIN_SIZE_CYT   = 250
HOLE_SIZE      = 180
BORDER_PX      = 4
CROSSTALK_ALPHA= 0.20
CLAHE_CLIP     = 0.02
CLAHE_TILE     = 12
RB_RADIUS      = 24            # stronger rolling-ball to flatten G halo
LOC_BLOCK      = 19            # window for Sauvola/local ops
SAUV_K         = 0.20
GAUSS_SIGMA    = 0.8
HMIN           = 1.0

# ── preprocessing ───────────────────────────────────────────────────────────────
def preproc_rgb(rgb_u8):
    """Denoise + local contrast normalize + green background subtraction."""
    rgb = rgb_u8.astype(np.float32) / 255.0
    # estimate_sigma might return a scalar or array depending on version/ multichannel
    # We ensure we get a mean sigma
    sigma_ests = estimate_sigma(rgb, channel_axis=-1)
    sigma_est = np.mean(sigma_ests) if np.iterable(sigma_ests) else sigma_ests
    
    den = denoise_nl_means(rgb, h=1.15*sigma_est, patch_size=5, patch_distance=6,
                           fast_mode=True, channel_axis=-1, preserve_range=True)
    den = (np.clip(den, 0, 1) * 255).astype(np.uint8)

    R = (equalize_adapthist(den[..., 0], clip_limit=CLAHE_CLIP, kernel_size=CLAHE_TILE)*255).astype(np.uint8)
    G = (equalize_adapthist(den[..., 1], clip_limit=CLAHE_CLIP, kernel_size=CLAHE_TILE)*255).astype(np.uint8)
    B = den[..., 2]

    # Rolling-ball background on G, then subtract background
    bg = rolling_ball(G, radius=RB_RADIUS)
    G = (G.astype(np.float32) - bg).clip(0, 255).astype(np.uint8)
    return np.stack([R, G, B], axis=-1)

# ── ratio index on smoothed channels ────────────────────────────────────────────
def index_rg_smooth(Ru8, Gu8):
    Rf = gaussian(Ru8.astype(np.float32)/255.0, GAUSS_SIGMA)
    Gf = gaussian(Gu8.astype(np.float32)/255.0, GAUSS_SIGMA)
    Rf = np.clip(Rf - CROSSTALK_ALPHA*Gf, 0.0, 1.0)
    idx = Rf / (Rf + Gf + 1e-6)
    return idx.astype(np.float32)  # 0..1

# ── nucleus cleanup with watershed keep-largest ────────────────────────────────
def watershed_keep_largest(nuc_bw):
    if nuc_bw.sum() == 0:
        return nuc_bw
    dist = distance_transform_edt(nuc_bw)
    # suppress tiny peaks, then use maxima as markers
    markers = label(h_maxima(dist, HMIN))
    if markers.max() == 0:
        return nuc_bw
    ws = watershed(-dist, markers, mask=nuc_bw)
    labs, sizes = np.unique(ws, return_counts=True)
    if len(labs) <= 1:
        return nuc_bw
    # labs[0] is usually background (0) if mask is not full, but watershed labels start at 1 usually within mask?
    # Actually watershed with mask returns 0 for background. 
    # sizes[1:] corresponds to labels[1:] (the foreground regions).
    # We want argmax of sizes[1:].
    if len(labs) > 1 and labs[0] == 0:
         keep = labs[1:][np.argmax(sizes[1:])]
    else:
         # Fallback if label 0 is not background or single label
         keep = labs[np.argmax(sizes)]
         
    return (ws == keep)

# ── adaptive cyto ring around nucleus ──────────────────────────────────────────
def cyto_ring_adaptive(nuc_bw, rin_frac=0.25, rout_frac=0.60):
    if nuc_bw.sum() == 0:
        return np.zeros_like(nuc_bw, bool)
    area = float(nuc_bw.sum())
    r_eq = np.sqrt(area / np.pi)
    rmin = max(3, int(round(rin_frac  * r_eq)))
    rmax = max(rmin+2, int(round(rout_frac * r_eq)))
    dist_out = distance_transform_edt(~nuc_bw)
    return (dist_out >= rmin) & (dist_out <= rmax)

# ── Metaphase: detect ring (Hough) + region refinement (Chan–Vese) ─────────────

def ring_from_radial_profile(Gcorr, nuc_seed, rmin=10, rmax=None, band_frac=0.18,
                             min_prom=5.0, grad_weight=0.6):
    """
    Edge-aware ring finder:
    - Blend intensity and gradient radial profiles in polar space.
    - Return (cx, cy, R, band) or None if confidence low.
    """
    if rmax is None:
        rmax = min(Gcorr.shape)//2 - 2

    # nucleus centroid → initial center (fallback to image center)
    props = regionprops(nuc_seed.astype(np.uint8))
    if props:
        cy, cx = props[0].centroid
    else:
        cy, cx = (Gcorr.shape[0]-1)/2.0, (Gcorr.shape[1]-1)/2.0
    cy, cx = float(cy), float(cx)

    # intensity & gradient in polar
    polar_I = warp_polar(Gcorr, center=(cx, cy), radius=rmax,
                         output_shape=(360, rmax), scaling='linear', preserve_range=True)
    Gg = sobel(Gcorr)
    polar_G = warp_polar(Gg, center=(cx, cy), radius=rmax,
                         output_shape=(360, rmax), scaling='linear', preserve_range=True)

    radial_I = polar_I.mean(axis=0)
    radial_G = polar_G.mean(axis=0)

    # blend: emphasize edges but keep intensity cue
    radial = (1.0 - grad_weight)*radial_I + grad_weight*radial_G

    r0 = max(3, int(rmin)); r1 = min(rmax-1, len(radial)-1)
    prof = radial[r0:r1]

    peak_idx = int(np.argmax(prof)); peak_val = float(prof[peak_idx])
    base = float(np.median(prof))
    prom = peak_val - base
    if prom < min_prom:
        return None

    R = r0 + peak_idx
    band = max(2, int(round(band_frac * R)))
    return int(round(cx)), int(round(cy)), int(R), int(band)

def ring_angular_coverage(Gcorr, cx, cy, R, r_width=2, rel_thr=0.35):
    # Edge map in a thin annulus
    Gg = sobel(Gcorr)
    H, W = Gg.shape
    Y, X = np.ogrid[:H, :W]
    d = np.sqrt((X - cx)**2 + (Y - cy)**2)
    # ann = (d >= R - r_width) & (d <= R + r_width) # unused variable 'ann' removed
    # unwrap and compute per-angle energy
    polar = warp_polar(Gg, center=(cx, cy), radius=R + r_width,
                       output_shape=(360, R + r_width), scaling='linear', preserve_range=True)
    edge_per_angle = polar[:, max(1, R - r_width): R + r_width + 1].mean(axis=1)
    thr = edge_per_angle.max() * rel_thr
    covered = (edge_per_angle >= thr).mean()  # in [0,1]
    return float(covered)

def metaphase_nucleus_radial(idx, Gcorr, nuc_seed):
    det = ring_from_radial_profile(Gcorr, nuc_seed, rmin=10, band_frac=0.20, min_prom=5.0)
    if det is None:
        thr = threshold_otsu(idx); hi, lo = min(1.,thr+0.06), max(0.,thr-0.06)
        return apply_hysteresis_threshold(idx, lo, hi), None

    cx, cy, R, band = det  # ← unpack first
    cov = ring_angular_coverage(Gcorr, cx, cy, R, r_width=max(2, band//2), rel_thr=0.35)
    if cov < 0.45:
        thr = threshold_otsu(idx); hi, lo = min(1.,thr+0.06), max(0.,thr-0.06)
        return apply_hysteresis_threshold(idx, lo, hi), None

    H, W = idx.shape
    Y, X = np.ogrid[:H, :W]
    d = np.sqrt((X - cx)**2 + (Y - cy)**2)

    roi = d <= max(3, R - band - 1)
    # Ensure ROI is not empty before otsu
    if not roi.any():
         thr = threshold_otsu(idx); hi, lo = min(1.,thr+0.06), max(0.,thr-0.06)
         return apply_hysteresis_threshold(idx, lo, hi), None

    init_thr = threshold_otsu(idx[roi])
    init = (idx > init_thr) & roi

    cv_mask = chan_vese(idx, mu=0.0, lambda1=1, lambda2=1, tol=1e-3,
                        max_num_iter=80, dt=0.5, init_level_set=init, extended_output=False)
    nuc = cv_mask & roi
    return nuc, (cx, cy, R, band)

def metaphase_cyto_radial(Gcorr, det):
    """
    Build the cytoplasmic ring as an annulus around the detected ring.
    Uses Sauvola inside the annulus for robustness to background.
    """
    if det is None:
        th = threshold_sauvola(Gcorr, window_size=LOC_BLOCK, k=SAUV_K)
        cytraw = (Gcorr > th)
        return cytraw  # will be intersected with adaptive ring later

    cx, cy, R, band = det
    H, W = Gcorr.shape
    Y, X = np.ogrid[:H, :W]
    d = np.sqrt((X - cx)**2 + (Y - cy)**2)

    inner = max(1, R - band)
    outer = min(R + int(1.2*band), max(H, W))
    ring_mask = (d >= inner) & (d <= outer)

    th = threshold_sauvola(Gcorr, window_size=LOC_BLOCK, k=SAUV_K)
    cyto = (Gcorr > th) & ring_mask
    cyto = remove_small_objects(cyto, MIN_SIZE_CYT)
    cyto = remove_small_holes(cyto, area_threshold=MIN_SIZE_CYT//2)
    return cyto

# ── Anaphase: split two lobes with dual markers ────────────────────────────────
def split_anaphase_lobes(nuc_init, min_dist=6):
    if nuc_init.sum() == 0:
        return nuc_init
    dist = distance_transform_edt(nuc_init)
    coords = peak_local_max(dist, min_distance=min_dist, labels=nuc_init, num_peaks=2)
    markers = np.zeros_like(nuc_init, dtype=np.int32)
    for i, (r, c) in enumerate(coords[:2], start=1):
        markers[r, c] = i
    if markers.max() < 2:
        return nuc_init
    ws = watershed(-dist, markers=markers, mask=nuc_init)
    labs, sizes = np.unique(ws, return_counts=True)
    # labs[0] is background if 0
    valid_labs = labs[1:] if labs[0] == 0 else labs
    valid_sizes = sizes[1:] if labs[0] == 0 else sizes
    
    if len(valid_labs) == 0:
        return nuc_init

    order = np.argsort(valid_sizes)[::-1]
    keep_ids = [valid_labs[order[0]]] + ([valid_labs[order[1]]] if len(order) > 1 else [])
    return np.isin(ws, keep_ids)
