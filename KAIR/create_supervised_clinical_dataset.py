#!/usr/bin/env python3
"""
Supervised Clinical Dataset Creation Pipeline

This script creates a supervised learning dataset from clinical CT images using:
1. ADN separation method to identify clean vs artifact images
2. InDuDoNet ODL/ASTRA synthesis to create synthetic artifacts
3. Compatible .h5 format for SCUNet-NGSWIN training

Usage:
    python create_supervised_clinical_dataset.py
    
Requirements:
    - clinical_config.yaml
    - adn_utils.py
    - build_geometry.py (optional, for artifact synthesis)
"""

import os
import sys
import yaml
import h5py
import numpy as np
import nibabel as nib
from pathlib import Path
import random
from tqdm import tqdm
from PIL import Image
import argparse
from datetime import datetime
from scipy.interpolate import interp1d

# Global cache and flags
_ODL_CACHE = None
_SIMPLIFIED_WARN_EMITTED = 0

print("🚀 Starting clinical dataset pipeline...")

# Load utilities
try:
    print("Loading imports...")
    from adn_utils import read_dir, get_connected_components, EasyDict
    print("✓ ADN utilities loaded")
    
    try:
        import odl
        from build_geometry import initialization, imaging_geo
        ODL_AVAILABLE = True
        print("✓ ODL/ASTRA available for synthesis")
    except ImportError:
        ODL_AVAILABLE = False
        print("⚠ ODL not available - will create separation only")
    
    print("✓ All imports loaded successfully")
    print()
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def _to_easydict(obj):
    """Recursively convert dict/list structures to EasyDict for dot access."""
    if isinstance(obj, dict):
        return EasyDict({k: _to_easydict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_easydict(v) for v in obj]
    return obj

def load_config(config_path="clinical_config.yaml"):
    """Load configuration from YAML file, supporting old and new schemas."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # If new schema already present
    if isinstance(cfg, dict) and all(k in cfg for k in ("paths", "adn", "processing")):
        return _to_easydict(cfg)

    # If old schema under 'clinical_config', map it to the new schema
    if isinstance(cfg, dict) and "clinical_config" in cfg:
        cc = cfg["clinical_config"]
        mapped = {
            "paths": {
                "raw_data": cc.get("raw_dir"),
                "output": cc.get("dataset_dir"),
            },
            "adn": {
                "max_hu_thresholds": cc.get("max_hu", [2000, 2500]),
                "conn_area_thresh": cc.get("connected_area", 400),
                "min_high_px": cc.get("min_high_px", 0),
            },
            "processing": {
                "target_size": cc.get("image_size", 416),
                # Force 100% train split per requirement when unspecified
                "train_test_split": 1.0 if cc.get("train_test_split") is None else float(cc.get("train_test_split")),
                "thumbnail_size": cc.get("thumbnail_size", 96),
                "seed": cc.get("seed", None),
            },
            "synth": cc.get("synth", {}),
        }
        return _to_easydict(mapped)

    # Fallback: wrap whatever is there
    return _to_easydict(cfg)

def normalize_hu_to_uint8(image_data):
    """Convert HU values to 0-255 range for visualization/storage."""
    # Clip extreme values - removed upper limit to allow metal prosthetic detection
    image_data = np.clip(image_data, -1000, 6000)  # Allow up to 6000 HU for metal detection
    # Normalize to 0-255 
    image_normalized = (image_data + 1000) / 7000.0 * 255.0  # Adjust scale for new range
    return image_normalized.astype(np.uint8)

def derive_metal_mask_uint8(image_uint8, config):
    """Derive metal mask using ADN thresholds and connected components on a uint8 image.

    Uses same logic as classification but on uint8 data for visualization.
    Converts HU thresholds to uint8 space: (HU + 1000) / 4000 * 255
    """
    H, W = image_uint8.shape
    max_hu_thresholds = getattr(config.adn, 'max_hu_thresholds', [2000, 2500])
    max_hu_high = max(max_hu_thresholds)  # 2500 HU
    
    # Convert HU threshold to uint8 space
    pix_thresh = int(round((max_hu_high + 1000.0) / 4000.0 * 255.0))
    high = image_uint8 > pix_thresh
    coords = np.column_stack(np.where(high))
    mask = np.zeros((H, W), dtype=np.uint8)
    
    if coords.size > 0:
        pts = set(map(tuple, coords.tolist()))
        comps = get_connected_components(pts)
        area_min = int(getattr(config.adn, 'conn_area_thresh', 400))
        for comp in comps:
            if len(comp) >= area_min:
                for (r, c) in comp:
                    if 0 <= r < H and 0 <= c < W:
                        mask[r, c] = 255
    return mask

def extract_middle_slice(volume_data):
    """Extract the middle slice from a 3D volume."""
    if len(volume_data.shape) == 3:
        middle_idx = volume_data.shape[2] // 2
        return volume_data[:, :, middle_idx]
    elif len(volume_data.shape) == 2:
        return volume_data
    else:
        raise ValueError(f"Unexpected volume shape: {volume_data.shape}")

def classify_image_adn(image_data, max_hu_thresholds, conn_area_thresh, min_high_px=0):
    """
    Classify image as clean or artifact using ADN method.
    
    Logic matches your reference spineweb code:
    - If max HU > 2500: check connected components
    - If max HU > 2000 but <= 2500: skip (ambiguous)  
    - If max HU <= 2000: keep as clean
    
    Args:
        image_data: 2D numpy array with HU values
        max_hu_thresholds: List of HU thresholds [2000, 2500]
        conn_area_thresh: Minimum area for connected components (400)
    
    Returns:
        'clean' or 'artifact'
    """
    max_hu_low = min(max_hu_thresholds)   # 2000
    max_hu_high = max(max_hu_thresholds)  # 2500
    image_max_hu = float(image_data.max())
    
    # Follow exact logic from your reference code
    if image_max_hu > max_hu_high:  # > 2500 HU
        # Guard: if enough super-high pixels exist, mark as artifact regardless of CC size
        try:
            mhp = int(min_high_px) if min_high_px is not None else 0
        except Exception:
            mhp = 0
        if mhp > 0 and int((image_data > max_hu_high).sum()) >= mhp:
            return 'artifact'
        # Find high intensity pixels at 2500 HU threshold
        points = np.array(np.where(image_data > max_hu_high)).T
        points = set(tuple(p) for p in points)
        components = get_connected_components(points)
        
        if len(components) > 0:
            max_area = max(len(c) for c in components)
            if max_area > conn_area_thresh:  # > 400 pixels
                return 'artifact'
        # If small components, treat as clean (continue processing)
        return 'clean'
        
    elif image_max_hu > max_hu_low:  # > 2000 but <= 2500 HU  
        return 'artifact'  # Skip ambiguous cases
        
    # <= 2000 HU maximum → definitely clean
    return 'clean'

def _get_odl_operators():
    """Cache and return (ray_trafo, FBP) operators.

    Returns:
        (ray_trafo_hh, FBPOper_hh)
    """
    global _ODL_CACHE
    if _ODL_CACHE is not None:
        return _ODL_CACHE
    init_params = initialization()
    ops = imaging_geo(init_params)
    _ODL_CACHE = ops
    return ops


def create_synthetic_metal_mask(image_shape, num_objects=2):
    """Create synthetic metal objects (circles/ellipses/rectangles) like InDuDoNet example.

    Args:
        image_shape: (H, W)
        num_objects: max number of objects to sample (1..num_objects)

    Returns:
        mask (float32) with 1.0 where synthetic metal is placed.
    """
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.float32)

    n = random.randint(1, max(1, int(num_objects)))
    for _ in range(n):
        shape_type = random.choice(['circle', 'ellipse', 'rectangle'])
        # Avoid edges
        cx = random.randint(W // 4, 3 * W // 4)
        cy = random.randint(H // 4, 3 * H // 4)

        if shape_type == 'circle':
            r = random.randint(6, 12)
            y, x = np.ogrid[:H, :W]
            part = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        elif shape_type == 'ellipse':
            a = random.randint(8, 15)
            b = random.randint(3, 8)
            y, x = np.ogrid[:H, :W]
            part = ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1
        else:
            ww = random.randint(6, 14)
            hh = random.randint(3, 8)
            x1 = max(0, cx - ww // 2)
            x2 = min(W, cx + ww // 2)
            y1 = max(0, cy - hh // 2)
            y2 = min(H, cy + hh // 2)
            part = np.zeros((H, W), dtype=bool)
            part[y1:y2, x1:x2] = True

        mask[part] = 1.0

    return mask


def interpolate_projection(proj, metal_trace):
    """Linear interpolation across metal trace (from InDuDoNet preprocessing_clinic.py)."""
    Pinterp = proj.copy()
    for i in range(Pinterp.shape[0]):
        mslice = metal_trace[i]
        pslice = Pinterp[i]

        metalpos = np.nonzero(mslice == 1)[0]
        nonmetalpos = np.nonzero(mslice == 0)[0]
        if len(nonmetalpos) > 1 and len(metalpos) > 0:
            pnonmetal = pslice[nonmetalpos]
            try:
                pslice[metalpos] = interp1d(nonmetalpos, pnonmetal)(metalpos)
            except Exception:
                pass
        Pinterp[i] = pslice
    return Pinterp


def _gaussian_kernel1d(sigma: float, radius: int) -> np.ndarray:
    """Create a 1D Gaussian kernel normalized to sum=1."""
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    k /= k.sum() if k.sum() != 0 else 1.0
    return k


def _synthesize_physics_artifact(clean_image, config):
    """Synthesize artifacts with photon starvation + scatter smearing in sinogram.

    This augments the basic forward-projection pipeline to better match real
    streaking seen in clinical images by introducing:
      - very high-µ metals (bright cores)
      - photon starvation (detector clipping) on metal-affected rays
      - wide-angle scatter smearing around the metal trace
      - realistic Poisson noise before re-log
    """
    ray_trafo_hh, FBPOper_hh = _get_odl_operators()

    H, W = clean_image.shape

    # Inverse of normalize_hu_to_uint8: HU in [-1000, 6000]
    clean_float01 = clean_image.astype(np.float32) / 255.0
    hu = clean_float01 * 7000.0 - 1000.0
    hu[hu < -1000] = -1000  # ensure lower bound

    # Convert HU to attenuation coefficient (mu): mu = HU/1000*0.192 + 0.192
    clean_mu = hu / 1000.0 * 0.192 + 0.192

    # Metal parameters
    mask_thre = 2500.0 / 1000.0 * 0.192 + 0.192
    metal_mu_value = mask_thre + 0.9  # strong metal

    # Geometric synthetic metal mask (small screws/rods)
    metal_mask = create_synthetic_metal_mask((H, W), num_objects=2)

    # Forward projections (line integrals)
    s_clean = np.asarray(ray_trafo_hh(clean_mu), dtype=np.float32)
    s_metal = np.asarray(ray_trafo_hh(metal_mu_value * metal_mask), dtype=np.float32)
    s_base = s_clean + s_metal

    # Identify metal-affected bins (boolean mask in projection domain)
    metal_trace = np.asarray(ray_trafo_hh(metal_mask), dtype=np.float32) > 0

    # === Config-driven streak enhancers ===
    synth = getattr(config, 'synth', None)
    sino_cor = s_base.copy()
    if synth is not None:
        # Widen metal trace if requested
        try:
            from scipy.ndimage import binary_dilation
            iters = int(getattr(synth, 'trace_dilate_px', 0) or 0)
            if iters > 0:
                metal_trace = binary_dilation(metal_trace, iterations=iters)
        except Exception:
            pass

        # Photon statistics (Poisson) with effective dose scaling
        if bool(getattr(synth, 'add_poisson', True)):
            I0 = float(getattr(synth, 'I0', 2e5))
            dose = float(getattr(synth, 'dose_scale', 0.2))
            I = np.clip(I0 * dose * np.exp(-sino_cor), 0.0, None)
            counts = np.random.poisson(I).astype(np.float32)
            I_noisy = np.clip(counts / max(I0 * dose, 1e-8), 1e-8, 1.0)
            sino_cor = -np.log(I_noisy)

        # Detector saturation / clipping along metal rays
        pclip = float(getattr(synth, 'clip_percentile', 0) or 0)
        if pclip > 0:
            cap = float(np.percentile(sino_cor, pclip))
            sino_cor[metal_trace] = np.minimum(sino_cor[metal_trace], cap)

        # Optional low-frequency scatter smear around metal trace
        smear_sigma = float(getattr(synth, 'smear_sigma', 0) or 0)
        smear_gain = float(getattr(synth, 'smear_gain', 0) or 0)
        if smear_sigma > 0 and smear_gain != 0:
            angles, det = sino_cor.shape
            radius = int(3 * smear_sigma)
            k = _gaussian_kernel1d(sigma=smear_sigma, radius=radius)
            smear = np.zeros_like(sino_cor, dtype=np.float32)
            for i in range(angles):
                row = metal_trace[i].astype(np.float32)
                smear[i] = np.convolve(row, k, mode='same')
            # raise intensities -> darker lines after log equivalently reduce sino
            # apply as small subtraction on line integrals where smear>0
            sino_cor = np.clip(sino_cor + (-smear_gain) * smear, 0.0, None)

    # Optional tiny interpolation preview (not used for reconstruction)
    try:
        _ = interpolate_projection(sino_cor.copy(), metal_trace.astype(np.uint8))
    except Exception:
        pass

    # Reconstruct with FBP from corrupted line integrals
    artifact_image = np.asarray(FBPOper_hh(sino_cor), dtype=np.float32)

    # Normalize to [0, 255]
    amin, amax = float(artifact_image.min()), float(artifact_image.max())
    if amax > amin:
        artifact_norm = (artifact_image - amin) / (amax - amin)
    else:
        artifact_norm = np.clip(artifact_image, 0, 1)
    out = np.clip(artifact_norm * 255.0, 0, 255).astype(np.uint8)
    return out


def create_artifact_synthesis(clean_image, config, use_physics=False):
    """
    Create synthetic artifacts using ODL/ASTRA forward projection.
    
    Args:
        clean_image: 2D numpy array (normalized to 0-255)
        config: Configuration object
    
    Returns:
        Synthetic artifact image
    """
    if not ODL_AVAILABLE:
        # Fallback: add simple noise pattern
        print("⚠ Using simple noise synthesis (ODL not available)")
        noise = np.random.normal(0, 10, clean_image.shape)
        return np.clip(clean_image + noise, 0, 255).astype(np.uint8)

    if use_physics:
        try:
            return _synthesize_physics_artifact(clean_image, config)
        except Exception as e:
            print(f"⚠ Physics synthesis failed: {e}; falling back to simplified")

    # Simplified path: add mild structured noise; print warning sparsely
    global _SIMPLIFIED_WARN_EMITTED
    if _SIMPLIFIED_WARN_EMITTED % 1000 == 0:
        print("⚠ Using simplified synthesis (ODL initialized; full physics not wired)")
    _SIMPLIFIED_WARN_EMITTED += 1
    clean_float = clean_image.astype(np.float32) / 255.0
    synthetic_artifact = clean_float + np.random.normal(0, 0.05, clean_image.shape)
    synthetic_artifact = np.clip(synthetic_artifact * 255, 0, 255).astype(np.uint8)
    return synthetic_artifact

def resize_image(image, target_size):
    """Resize image to target size maintaining aspect ratio."""
    if image.shape[0] == target_size and image.shape[1] == target_size:
        return image
    
    # Convert to PIL Image for resizing
    pil_img = Image.fromarray(image)
    pil_resized = pil_img.resize((target_size, target_size), Image.LANCZOS)
    return np.array(pil_resized)

def save_supervised_pairs(clean_images, artifact_images, output_path, split_ratio=0.8):
    """
    Save clean-artifact pairs in .h5 format compatible with SCUNet dataset loaders.
    
    Args:
        clean_images: List of clean images
        artifact_images: List of corresponding artifact images  
        output_path: Output directory path
        split_ratio: Train/test split ratio
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Split data
    num_samples = len(clean_images)
    if num_samples == 0:
        raise ValueError("No samples to save.")
    # Clamp split_ratio to [0,1]
    split_ratio = max(0.0, min(1.0, float(split_ratio)))
    train_size = int(round(num_samples * split_ratio))
    
    # Create train/test splits
    indices = list(range(num_samples))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Save training set
    train_path = os.path.join(output_path, 'train_clinical_supervised.h5')
    print(f"💾 Saving {len(train_indices)} training pairs to {train_path}")
    
    with h5py.File(train_path, 'w') as f:
        # Create datasets
        clean_data = np.array([clean_images[i] for i in train_indices])
        artifact_data = np.array([artifact_images[i] for i in train_indices])
        
        f.create_dataset('clean', data=clean_data, compression='gzip')
        f.create_dataset('artifact', data=artifact_data, compression='gzip')
        f.attrs['num_samples'] = len(train_indices)
        f.attrs['image_shape'] = clean_data.shape[1:]
    
    # Save test set (only if there are test samples)
    if len(test_indices) > 0:
        test_path = os.path.join(output_path, 'test_clinical_supervised.h5')
        print(f"💾 Saving {len(test_indices)} test pairs to {test_path}")
        with h5py.File(test_path, 'w') as f:
            clean_data = np.array([clean_images[i] for i in test_indices])
            artifact_data = np.array([artifact_images[i] for i in test_indices])
            f.create_dataset('clean', data=clean_data, compression='gzip')
            f.create_dataset('artifact', data=artifact_data, compression='gzip')
            f.attrs['num_samples'] = len(test_indices)
            f.attrs['image_shape'] = clean_data.shape[1:]
    else:
        print("ℹ No test split requested; skipping test file.")
    
    print(f"✅ Dataset saved successfully!")
    print(f"   📊 Training samples: {len(train_indices)}")
    print(f"   📊 Test samples: {len(test_indices)}")
    print(f"   📐 Image shape: {clean_data.shape[1:]}")


def _slugify(name: str) -> str:
    bad = [' ', '/', '\\', ':', ';', ',', '(', ')', '[', ']', '{', '}', '"', "'", '|', '*', '?', '!']
    out = name
    for b in bad:
        out = out.replace(b, '_')
    return out

def save_pairs_as_h5_folders(clean_images, artifact_images, src_files, output_root, subset_name='train', input_key='LI_CT', gt_key='image'):
    """
    Save each (artifact, clean) pair into its own folder as H5 files compatible with DatasetLICT/MACT:
      - <output_root>/<subset_name>/<case_id>/<slice_id>.h5  (dataset: input_key)
      - <output_root>/<subset_name>/<case_id>/gt.h5         (dataset: gt_key)

    Args:
        clean_images: list[np.ndarray HxW uint8]
        artifact_images: list[np.ndarray HxW uint8]
        src_files: list[str] original file paths for naming/manifest
        output_root: str output directory
        subset_name: str e.g., 'train'
        input_key: str dataset name for input image in h5
        gt_key: str dataset name for ground-truth in gt.h5
    """
    assert len(clean_images) == len(artifact_images) == len(src_files)
    subset_dir = os.path.join(output_root, subset_name)
    os.makedirs(subset_dir, exist_ok=True)

    manifest_path = os.path.join(output_root, f'manifest_{subset_name}.csv')
    clean_list = os.path.join(output_root, f'clean_list_{subset_name}.txt')
    artifact_list = os.path.join(output_root, f'artifact_list_{subset_name}.txt')
    with open(manifest_path, 'w') as mf, open(clean_list, 'w') as cf, open(artifact_list, 'w') as af:
        mf.write('case_id,slice_h5,gt_h5,source_path\n')
    # Group by volume path, keep per-slice index
        for i, (clean_img, art_img, src) in enumerate(zip(clean_images, artifact_images, src_files)):
            # src can be 'path::z{z}'
            if '::z' in src:
                vol_path, z_str = src.split('::z')
                z_idx = int(z_str)
            else:
                vol_path, z_idx = src, 0
            base_name = os.path.splitext(os.path.basename(vol_path))[0]
            case_id = _slugify(base_name)
            case_dir = os.path.join(subset_dir, case_id)
            os.makedirs(case_dir, exist_ok=True)
            slice_h5 = os.path.join(case_dir, f"{z_idx:04d}.h5")
            gt_h5 = os.path.join(case_dir, "gt.h5")

            # Write input slice
            with h5py.File(slice_h5, 'w') as f:
                f.create_dataset(input_key, data=art_img.astype(np.float32), compression='gzip')
            # Write gt
            with h5py.File(gt_h5, 'w') as f:
                f.create_dataset(gt_key, data=clean_img.astype(np.float32), compression='gzip')

            mf.write(f"{case_id},{slice_h5},{gt_h5},{vol_path}::z{z_idx}\n")
            cf.write(f"{vol_path}::z{z_idx}\n")
            af.write(f"{vol_path}::z{z_idx}\n")

    print(f"✅ Saved {len(clean_images)} H5 pairs to {subset_dir}")


def save_example_images(clean_images, artifact_images, src_files, examples_root, max_examples=32, raw_images=None, config=None):
    """Save multi-stage example PNGs per slice.

    For each selected slice, creates a folder <idx>_<slug>_non_metal/ with:
      - raw.png (normalized HU, pre-resize) if available
      - clean.png (resized clean, non-metal)
      - artifact.png (synthesized with metal artifacts)
      - mask.png (metal mask derived via ADN thresholds/CC on clean)
      - overlay.png (mask overlaid on clean)
    """
    os.makedirs(examples_root, exist_ok=True)
    n = min(len(clean_images), len(artifact_images), len(src_files), max_examples)
    for i in range(n):
        clean = clean_images[i]
        art = artifact_images[i]
        src = src_files[i]
        slug = _slugify(os.path.splitext(os.path.basename(src.split('::z')[0]))[0])
        out_dir = os.path.join(examples_root, f"{i:04d}_{slug}_non_metal")
        os.makedirs(out_dir, exist_ok=True)

        # Save clean and artifact
        try:
            Image.fromarray(clean).save(os.path.join(out_dir, "non_metal_clean.png"))
            Image.fromarray(art).save(os.path.join(out_dir, "synthesized_artifact.png"))
        except Exception as e:
            print(f"⚠ Failed to save clean/artifact for example {i} ({slug}): {e}")

        # Save raw if provided
        if raw_images is not None and i < len(raw_images) and raw_images[i] is not None:
            try:
                Image.fromarray(raw_images[i]).save(os.path.join(out_dir, "non_metal_original.png"))
            except Exception as e:
                print(f"⚠ Failed to save raw for example {i} ({slug}): {e}")

    # Intentionally skip saving mask/overlay for clean slices to keep only essential images
    # (mask derived from clean would be empty by design)

    print(f"🖼  Saved {n} example sets to {examples_root}")


def save_artifact_only_as_h5_folders(artifact_images, src_files, output_root, subset_name='test', input_key='LI_CT', gt_key='image'):
    """
    Save artifact-only slices into per-slice H5 folders, mirroring the supervised layout:
      - <output_root>/<subset_name>/<case_id>/<slice_id>.h5 (dataset: input_key)
      - <output_root>/<subset_name>/<case_id>/gt.h5      (dataset: gt_key)

    Note: For loader compatibility, we write gt.h5 with the same data as input. Metrics on this
    set are not meaningful; this dataset is intended for qualitative/testing only.
    """
    assert len(artifact_images) == len(src_files)
    subset_dir = os.path.join(output_root, subset_name)
    os.makedirs(subset_dir, exist_ok=True)

    manifest_path = os.path.join(output_root, f'manifest_{subset_name}.csv')
    with open(manifest_path, 'w') as mf:
        mf.write('case_id,slice_h5,gt_h5,source_path\n')

        for art_img, src in zip(artifact_images, src_files):
            # src can be 'path::z{z}'
            if '::z' in src:
                vol_path, z_str = src.split('::z')
                z_idx = int(z_str)
            else:
                vol_path, z_idx = src, 0
            base_name = os.path.splitext(os.path.basename(vol_path))[0]
            case_id = _slugify(base_name)
            case_dir = os.path.join(subset_dir, case_id)
            os.makedirs(case_dir, exist_ok=True)

            slice_h5 = os.path.join(case_dir, f"{z_idx:04d}.h5")
            gt_h5 = os.path.join(case_dir, "gt.h5")

            # Write input slice
            with h5py.File(slice_h5, 'w') as f:
                f.create_dataset(input_key, data=art_img.astype(np.float32), compression='gzip')
            # Write a loader-compatible gt.h5 (duplicate of input)
            with h5py.File(gt_h5, 'w') as f:
                f.create_dataset(gt_key, data=art_img.astype(np.float32), compression='gzip')

            mf.write(f"{case_id},{slice_h5},{gt_h5},{vol_path}::z{z_idx}\n")

    print(f"✅ Saved {len(artifact_images)} artifact-only H5 slices to {subset_dir}")


def save_artifact_only_examples(artifact_images, src_files, examples_root, max_examples=32):
    """Save PNG previews for slices classified as artifact (artifact-only dataset).

    Creates per-slice folders under examples_root with a single PNG:
      - artifact_only.png
    """
    os.makedirs(examples_root, exist_ok=True)
    n = min(len(artifact_images), len(src_files), max_examples)
    for i in range(n):
        art = artifact_images[i]
        src = src_files[i]
        base, z = (src.split('::z') + ['0'])[:2]
        slug = _slugify(os.path.splitext(os.path.basename(base))[0])
        out_dir = os.path.join(examples_root, f"{i:04d}_{slug}_artifact_only_z{z}")
        os.makedirs(out_dir, exist_ok=True)
        try:
            Image.fromarray(art).save(os.path.join(out_dir, "artifact_only.png"))
        except Exception as e:
            print(f"⚠ Failed to save artifact-only example {i} ({slug}): {e}")
    print(f"🖼  Saved {n} artifact-only example images to {examples_root}")

def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description='Create supervised clinical dataset')
    parser.add_argument('--config', default='clinical_config.yaml', help='Config file path')
    parser.add_argument('--dry-run', action='store_true', help='Run analysis only, no file creation')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum samples to process')
    parser.add_argument('--all-slices', action='store_true', help='Process all z-slices per volume')
    parser.add_argument('--no-save-examples', action='store_true', help='Do not save example PNGs')
    parser.add_argument('--examples-dir', type=str, default=None, help='Directory to save example PNGs')
    parser.add_argument('--max-examples', type=int, default=15, help='Max example pairs to save per volume (randomly sampled)')
    parser.add_argument('--examples-for-all', action='store_true', help='Save raw example for every slice (can be large)')
    parser.add_argument('--physics-synthesis', action='store_true', help='Use ODL/ASTRA physics-based synthesis (slower)')
    # Artifact-only examples controls
    parser.add_argument('--artifact-max-examples', type=int, default=None, help='Max artifact-only example PNGs to save')
    parser.add_argument('--artifact-examples-for-all', action='store_true', help='Save all artifact-only slices as PNGs')
    parser.set_defaults(physics_synthesis=True)
    args = parser.parse_args()
    
    print("=" * 50)
    print("Clinical Dataset Supervised Pipeline")
    print("=" * 50)
    
    # Load configuration
    try:
        config = load_config(args.config)
        print("✓ Configuration loaded successfully")
        print(f"📂 Raw data path: {config.paths.raw_data}")
        print(f"📂 Output path: {config.paths.output}")
        print(f"🎯 Max HU thresholds: {config.adn.max_hu_thresholds}")
        print(f"🔗 Connected area threshold: {config.adn.conn_area_thresh}")
        print(f"📐 Target image size: {config.processing.target_size}")
        print()
        # Optional deterministic synthesis
        try:
            seed = getattr(config.processing, 'seed', None)
            if seed is not None:
                random.seed(int(seed))
                np.random.seed(int(seed))
                print(f"🔒 Seeded RNG with seed={int(seed)}")
        except Exception:
            pass
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        return
    
    # Step 1: Find and classify images
    print("🔍 Step 1: Separating clean vs artifact images...")
    try:
        nii_files = read_dir(config.paths.raw_data, predicate=lambda x: x.endswith('.nii'))
        print(f"✓ Found {len(nii_files)} volume files")

        if len(nii_files) == 0:
            print("❌ No .nii files found!")
            return

        clean_images: list = []
        clean_files: list = []
        # Also collect artifact-classified slices for a separate test-only dataset
        artifact_slice_images: list = []
        artifact_slice_files: list = []
        # Keep raw (pre-resize) images for examples; collect per-volume for diversity
        raw_examples = {}
        volume_clean_slices = {}  # Track all clean slices per volume for random sampling
        artifact_files: list = []

        # Limit samples if specified
        files_to_process = nii_files[:args.max_samples] if args.max_samples else nii_files

        print(f"🔬 Processing {len(files_to_process)} files...")

        for i, file_path in enumerate(tqdm(files_to_process, desc="Classifying images")):
            try:
                # Load NIfTI volume
                nii_img = nib.load(file_path)
                volume_data = nii_img.get_fdata()

                # Decide slice indices
                if volume_data.ndim == 3 and args.all_slices:
                    z_indices = range(volume_data.shape[2])
                else:
                    if volume_data.ndim == 3:
                        z_indices = [volume_data.shape[2] // 2]
                    else:
                        z_indices = [0]

                # Process each slice
                for z in z_indices:
                    slice_data = volume_data[:, :, z] if volume_data.ndim == 3 else volume_data

                    # Classify using ADN method
                    classification = classify_image_adn(
                        slice_data,
                        config.adn.max_hu_thresholds,
                        config.adn.conn_area_thresh,
                        getattr(config.adn, 'min_high_px', 0),
                    )

                    tag = f"{file_path}::z{z}"
                    if classification == 'clean':
                        # Normalize and resize
                        normalized = normalize_hu_to_uint8(slice_data)
                        resized = resize_image(normalized, config.processing.target_size)

                        # Collect all clean slices per volume, will randomly sample later
                        idx_next = len(clean_images)
                        volume_key = os.path.basename(file_path)
                        
                        if volume_key not in volume_clean_slices:
                            volume_clean_slices[volume_key] = []
                        
                        # Store slice info for potential sampling
                        volume_clean_slices[volume_key].append({
                            'idx': idx_next,
                            'normalized': normalized,
                            'tag': tag
                        })

                        clean_images.append(resized)
                        clean_files.append(tag)
                    else:
                        artifact_files.append(tag)
                        # Keep the artifact-classified slice (preprocessed) for test-only dataset
                        normalized = normalize_hu_to_uint8(slice_data)
                        resized = resize_image(normalized, config.processing.target_size)
                        artifact_slice_images.append(resized)
                        artifact_slice_files.append(tag)

            except Exception as e:
                print(f"⚠ Error processing {file_path}: {e}")
                continue

        print(f"✅ Found {len(clean_images)} clean images out of {len(files_to_process)} processed")
        print()

        if len(clean_images) == 0:
            print("❌ No clean images found! Will save artifact-only dataset/examples if available.")
            # Save artifact-only even if no clean images
            try:
                test_only_root = os.path.join(os.path.dirname(config.paths.output), 'slincial_metal_test_Metal_artifact_only')
                os.makedirs(test_only_root, exist_ok=True)
                if artifact_slice_images:
                    print("🧪 Saving metal-artifact-only test dataset…")
                    save_artifact_only_as_h5_folders(
                        artifact_images=artifact_slice_images,
                        src_files=artifact_slice_files,
                        output_root=test_only_root,
                        subset_name='test',
                        input_key='LI_CT',
                        gt_key='image'
                    )
                    try:
                        kair_dir = os.path.dirname(__file__)
                        data_dir = os.path.join(kair_dir, 'data')
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                        examples_root = os.path.join(data_dir, 'clinical_examples_artifact_only', ts)
                        max_ex = int(args.max_examples) if args.max_examples else 32
                        save_artifact_only_examples(
                            artifact_images=artifact_slice_images,
                            src_files=artifact_slice_files,
                            examples_root=examples_root,
                            max_examples=max_ex,
                        )
                    except Exception as e:
                        print(f"⚠ Failed to save artifact-only PNG examples: {e}")
                else:
                    print("🧪 No artifact-classified slices found either.")
            finally:
                return

    except Exception as e:
        print(f"❌ Step 1 error: {e}")
        return
    
    if args.dry_run:
        print("🎉 Dry run completed!")
        print("The full pipeline would:")
        print(f"1. ✓ Process {len(clean_images)} clean images (kept)")
        print(f"   • Skipped as artifact: {len(artifact_files)}")
        print("2. ⚡ Synthesize artifacts for each clean image")
        print("3. 💾 Save each pair as per-slice H5 folder (input='LI_CT', gt='image')")
        return
    
    # Step 2: Create synthetic artifacts
    print("⚡ Step 2: Creating synthetic artifacts...")
    try:
        artifact_images = []
        for i, clean_img in enumerate(tqdm(clean_images, desc="Synthesizing artifacts")):
            try:
                synthetic_artifact = create_artifact_synthesis(clean_img, config, use_physics=args.physics_synthesis)
                artifact_images.append(synthetic_artifact)
            except Exception as e:
                print(f"⚠ Error synthesizing artifact for image {i}: {e}")
                # Use original as fallback
                artifact_images.append(clean_img)

        print(f"✅ Created {len(artifact_images)} synthetic artifact images")
        print()
        
    except Exception as e:
        print(f"❌ Step 2 error: {e}")
        return
    
    # Step 3: Save supervised dataset (per-pair H5 folders)
    print("💾 Step 3: Saving supervised pairs (folder-per-pair)...")
    try:
        # Write classification lists for traceability
        os.makedirs(config.paths.output, exist_ok=True)
        with open(os.path.join(config.paths.output, 'clean_sources.txt'), 'w') as f:
            f.write("\n".join(clean_files) + ("\n" if clean_files else ""))
        with open(os.path.join(config.paths.output, 'artifact_sources.txt'), 'w') as f:
            f.write("\n".join(artifact_files) + ("\n" if artifact_files else ""))

        # Save pairs in dataset-compatible layout
        save_pairs_as_h5_folders(
            clean_images=clean_images,
            artifact_images=artifact_images,
            src_files=clean_files,
            output_root=config.paths.output,
            subset_name='train',
            input_key='LI_CT',
            gt_key='image'
        )
        # Save example images in KAIR/data by default
        if not args.no_save_examples:
            if args.examples_dir:
                examples_root = args.examples_dir
            else:
                # Default under KAIR/data/clinical_examples/<timestamp>
                kair_dir = os.path.dirname(__file__)
                data_dir = os.path.join(kair_dir, 'data')
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                examples_root = os.path.join(data_dir, 'clinical_examples', ts)
            
            # Randomly sample examples from each volume
            if args.examples_for_all:
                # Use all slices
                for volume_key, slices in volume_clean_slices.items():
                    for slice_info in slices:
                        raw_examples[slice_info['idx']] = slice_info['normalized']
                total_examples = len(raw_examples)
            else:
                # Random sample up to max_examples per volume
                total_examples = 0
                for volume_key, slices in volume_clean_slices.items():
                    # Randomly sample up to max_examples from this volume
                    num_to_sample = min(len(slices), int(args.max_examples))
                    if num_to_sample > 0:
                        sampled_slices = random.sample(slices, num_to_sample)
                        for slice_info in sampled_slices:
                            raw_examples[slice_info['idx']] = slice_info['normalized']
                        total_examples += num_to_sample
            
            # Build aligned lists for saving
            raw_images_list = [raw_examples.get(i) for i in range(len(clean_images)) if i in raw_examples]
            clean_to_save = [clean_images[i] for i in range(len(clean_images)) if i in raw_examples]
            artifact_to_save = [artifact_images[i] for i in range(len(artifact_images)) if i in raw_examples]
            files_to_save = [clean_files[i] for i in range(len(clean_files)) if i in raw_examples]
            
            save_example_images(
                clean_images=clean_to_save,
                artifact_images=artifact_to_save,
                src_files=files_to_save,
                examples_root=examples_root,
                max_examples=total_examples,
                raw_images=raw_images_list,
                config=config,
            )
        print()
        
    except Exception as e:
        print(f"❌ Step 3 error: {e}")
        return

    # Step 4: Save artifact-only test dataset as requested
    try:
        test_only_root = os.path.join(os.path.dirname(config.paths.output), 'slincial_metal_test_Metal_artifact_only')
        os.makedirs(test_only_root, exist_ok=True)
        if artifact_slice_images:
            print("🧪 Step 4: Saving metal-artifact-only test dataset…")
            save_artifact_only_as_h5_folders(
                artifact_images=artifact_slice_images,
                src_files=artifact_slice_files,
                output_root=test_only_root,
                subset_name='test',
                input_key='LI_CT',
                gt_key='image'
            )
            # Also save PNG examples for these artifact-only slices under KAIR/data
            try:
                kair_dir = os.path.dirname(__file__)
                data_dir = os.path.join(kair_dir, 'data')
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                examples_root = os.path.join(data_dir, 'clinical_examples_artifact_only', ts)
                # Decide how many artifact-only examples to save
                if args.artifact_examples_for_all:
                    select_idx = list(range(len(artifact_slice_images)))
                else:
                    # Default higher count for artifact-only previews
                    limit = args.artifact_max_examples if args.artifact_max_examples is not None else 64
                    k = min(len(artifact_slice_images), int(limit))
                    select_idx = random.sample(range(len(artifact_slice_images)), k) if k > 0 else []
                # Build selected lists
                art_imgs_sel = [artifact_slice_images[i] for i in select_idx]
                src_files_sel = [artifact_slice_files[i] for i in select_idx]
                save_artifact_only_examples(
                    artifact_images=art_imgs_sel,
                    src_files=src_files_sel,
                    examples_root=examples_root,
                    max_examples=len(select_idx),
                )
            except Exception as e:
                print(f"⚠ Failed to save artifact-only PNG examples: {e}")
        else:
            print("🧪 Step 4: No artifact-classified slices found to save for test-only dataset.")
    except Exception as e:
        print(f"⚠ Failed to save artifact-only test dataset: {e}")
    
    print("🎉 Pipeline completed successfully!")
    print(f"📊 Final dataset statistics:")
    print(f"   • Clean images: {len(clean_images)}")
    print(f"   • Artifact images: {len(artifact_images)}")
    print(f"   • Output location: {config.paths.output}")
    print(f"   • Train folder: {os.path.join(config.paths.output, 'train')}")
    print(f"   • Pair format: <train>/<case>/0000.h5 (LI_CT) + <train>/<case>/gt.h5 (image)")
    print(f"   • Image size: {config.processing.target_size}x{config.processing.target_size}")

if __name__ == "__main__":
    main()
