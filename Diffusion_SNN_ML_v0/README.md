# Diffusion SNN ML - Cell Differentiation Project

## Overview
This project focuses on machine learning analysis of organoid cell differentiation using diffusion models and Spiking Neural Networks (SNNs). The primary objective involves classifying and segmenting cells into distinct cell cycle stages: **Prophase**, **Metaphase**, **Anaphase**, and **Telophase**.

The pipeline processes raw microscopy images, extracts cell patches based on COCO annotations, generates high-quality segmentation masks using classical image processing techniques (Watershed, Chan-Vese, active contours), and prepares the data for training advanced ML models.

## Project Structure

### Notebooks
*   **`diffusion_snn_ml_cell_diff_V2.ipynb`**: The main workflow notebook. It handles:
    1.  **Setup & Config**: Defines paths, classes, and image parameters (96x96 patches).
    2.  **Data Loading**: Parses COCO JSON annotations for training and validation sets.
    3.  **Preprocessing**: Applies shading correction, green ring suppression, and letterboxing to normalize images.
    4.  **Patch Extraction**: Caches training image patches centered on cell annotations.
    5.  **Mask Generation**: Generates ground-truth segmentation masks using stage-specific algorithms (e.g., radial profile detection for Metaphase, lobe splitting for Anaphase).
    6.  **Data Curation**: Filters samples based on Quality Control (QC) metrics like Signal-to-Noise Ratio (SNR) and area.

### Key Data Files
*   **`kept_manifest.json`**: A JSON manifest listing all curated training samples. Contains metadata such as:
    *   `ann_id`: Annotation ID.
    *   `_class`: Cell cycle stage.
    *   `snr`: Signal-to-Noise Ratio.
    *   `area`: Normalized area of the nucleus.
    *   `patch_path`, `mask_path`: Relative paths to processed images.
*   **`augmentation_plan.csv`**: Defines the data augmentation strategy. Maps each sample to a `num_aug` value to balance class distribution (e.g., upsampling rarer classes like Anaphase/Telophase).
*   **`class_weights.json`**: Likely contains computed weights for loss balancing during training.

### Directories
*   **`borg-main/`**: Repository/data source containing raw images and COCO JSONs (`organoid_coco_train.json`, etc.).
*   **`patch_cache_train/`**: Stores preprocessed 96x96 RGB cell patches, organized by class.
*   **`processed_masks_train/`**: Stores generated binary masks and overlay segmentations.
*   **`qc_previews/` & `qc_hardcases/`**: Output directories for visual inspection of segmentation quality.

## Installation & Requirements
The project relies on standard Python data science and image processing libraries:
*   `numpy`
*   `matplotlib`
*   `Pillow` (PIL)
*   `scikit-image` (skimage)
*   `scipy`
*   `tqdm`

(Note: Ensure your environment supports these packages. The notebook was observed running in a `diff` conda environment.)

## Usage
1.  **Data Setup**: Ensure `borg-main/data` contains the required COCO JSONs and raw images.
2.  **Run Pipeline**: Execute the cells in `diffusion_snn_ml_cell_diff_V2.ipynb` sequentially to:
    *   Load data.
    *   Generate and cache patches.
    *   Create segmentation masks.
    *   Compute QC metrics and save the manifest.
3.  **Training**: Use the generated `kept_manifest.json` and `augmentation_plan.csv` to feed your training loop (Diffusion/SNN model).

## Methodology Highlights
*   **Preprocessing**: Advanced shading correction using rolling-ball algorithms and white-tophat transforms to handle uneven microscopy illumination.
*   **Adaptive Segmentation**:
    *   **Metaphase**: Uses radial profile analysis and Chan-Vese active contours to detect ring-like structures.
    *   **Anaphase**: Implements watershed with dual-peak markers to correctly separate splitting lobes.
    *   **General**: Hybrid thresholding (Otsu + Hysteresis) combined with morphological cleaning.
