#!/usr/bin/env python3
"""
Single Model Inference Evaluation for conv_nstb variant
=====================================================

This script tests only the conv_nstb variant that was missed in the previous evaluation
and adds the results to the existing inference_1 folder structure.

Usage:
    python test_conv_nstb_single.py
"""

import os
import sys
import argparse
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from pathlib import Path
import cv2
import time

# Add KAIR to path for imports
sys.path.append('/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR')

# Import KAIR modules
from utils import utils_option as option
from utils import utils_image as util
from data import select_dataset
from models import select_model

# Clinical preprocessing configuration
class ClinicalConfig:
    """Configuration for clinical data preprocessing"""
    def __init__(self):
        self.CTpara = {
            'imPixNum': 416,  # Image pixels along x or y direction
            'angSize': 0.05,  # Angle between two neighbor rays
            'linSize': 1.8536,
            'angNum': 640,  # Number of projection views
            'SOD': 1075,  # Source-to-origin distance
            'imPixScale': 512 / 416 * 0.03,
            'sinogram_size_x': 640,
            'sinogram_size_y': 641,
            'window': [-175, 275]  # HU window
        }
        self.mask_thre = 2500 / 1000 * 0.192 + 0.192  # Metal threshold - EXACT match to example

def clinical_preprocessing(image_data, config):
    """
    Clinical preprocessing EXACTLY matching the provided example
    From: clinic_input_data() function
    """
    try:
        # Step 1: Clamp HU values below -1000 (exact match to example)
        image_data[image_data < -1000] = -1000
        
        # Step 2: Convert to linear attenuation coefficient (LAC)
        # Exact formula from example: image = image / 1000 * 0.192 + 0.192
        image_lac = image_data / 1000 * 0.192 + 0.192
        
        # Step 3: Resize to target size (416x416) using PIL.Image.BILINEAR equivalent
        # From example: Image.fromarray(imag[:,:,i]).resize((CTpara['imPixNum'], CTpara['imPixNum']), PIL.Image.BILINEAR)
        if image_lac.shape != (config.CTpara['imPixNum'], config.CTpara['imPixNum']):
            image_resized = cv2.resize(image_lac.astype(np.float32), 
                                     (config.CTpara['imPixNum'], config.CTpara['imPixNum']), 
                                     interpolation=cv2.INTER_LINEAR)
        else:
            image_resized = image_lac
        
        # Step 4: For model compatibility, normalize to [0,1] range then scale to [0,255]
        # This preserves the LAC values while making it compatible with uint8 models
        image_normalized = (image_resized - image_resized.min()) / (image_resized.max() - image_resized.min())
        image_uint8 = (image_normalized * 255).astype(np.uint8)
        
        return image_uint8
        
    except Exception as e:
        print(f"  Error in clinical preprocessing: {e}")
        return None

def detect_metal_mask(image_data, config):
    """
    Metal detection EXACTLY matching the provided example
    From: clinic_input_data() function - metal mask generation
    """
    try:
        # Step 1: Convert raw HU to LAC first (same as preprocessing)
        image_data[image_data < -1000] = -1000
        image_lac = image_data / 1000 * 0.192 + 0.192
        
        # Step 2: Apply metal threshold in LAC space (exact match to example)
        # From example: mask_thre = 2500 /1000 * 0.192 + 0.192
        mask_thre = config.mask_thre  # This is already calculated as 2500/1000 * 0.192 + 0.192
        
        # Step 3: Create binary mask where LAC > threshold
        # From example: [rowindex, colindex] = np.where(image > mask_thre); M[rowindex, colindex, i] = 1
        metal_mask = np.zeros_like(image_lac, dtype=np.float32)
        rowindex, colindex = np.where(image_lac > mask_thre)
        metal_mask[rowindex, colindex] = 1
        
        # Step 4: Resize mask to target size
        if metal_mask.shape != (config.CTpara['imPixNum'], config.CTpara['imPixNum']):
            metal_mask = cv2.resize(metal_mask, 
                                  (config.CTpara['imPixNum'], config.CTpara['imPixNum']), 
                                  interpolation=cv2.INTER_NEAREST)
        
        return metal_mask.astype(np.uint8)
        
    except Exception as e:
        print(f"  Error in metal detection: {e}")
        return None

def get_conv_nstb_model():
    """Get conv_nstb model variation and its best checkpoint"""
    base_path = '/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR/training_results'
    
    # Look for conv_nstb folder
    variation_path = Path(base_path) / 'conv_nstb'
    if not variation_path.exists():
        raise FileNotFoundError(f"conv_nstb folder not found at {variation_path}")
    
    # Look for the scunet_ngswin_* subfolder
    subfolders = [d for d in variation_path.iterdir() if d.is_dir() and d.name.startswith('scunet_ngswin_')]
    if not subfolders:
        raise FileNotFoundError(f"No scunet_ngswin_* subfolder found in {variation_path}")
    
    models_path = subfolders[0] / 'models'
    if not models_path.exists():
        raise FileNotFoundError(f"Models folder not found at {models_path}")
    
    # Use psnr_G.pth as the best checkpoint
    psnr_checkpoint = models_path / 'psnr_G.pth'
    if not psnr_checkpoint.exists():
        raise FileNotFoundError(f"psnr_G.pth not found at {psnr_checkpoint}")
    
    # conv_nstb uses config 1 (the original training config)
    model_info = {
        'checkpoint_path': str(psnr_checkpoint),
        'config_file': 'options/train_scunet_ngswin_1.json',
        'display_name': 'Conv NSTB',
        'variant_type': 'conv_nstb'
    }
    
    return model_info

def setup_model_and_checkpoint(checkpoint_path, config_path, variant_name):
    """Load model and checkpoint with detailed logging"""
    print(f"🔧 Loading {variant_name} model from: {os.path.basename(checkpoint_path)}")
    
    # Load configuration
    opt = option.parse(config_path, is_train=True)
    opt['path']['models'] = os.path.dirname(checkpoint_path)
    
    # Create model
    model = select_model.define_Model(opt)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        model.load_network(checkpoint_path, model.netG, strict=True)
        print(f"✓ Loaded checkpoint: {os.path.basename(checkpoint_path)}")
        
        # Get model parameter count
        total_params = sum(p.numel() for p in model.netG.parameters())
        print(f"📊 Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        
        return model
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

def load_clinical_mask_fixed(clinical_path, mask_data_dir):
    """Load corresponding clinical mask with fixed pattern matching"""
    try:
        # Get the base filename and extract the key part
        base_name = os.path.basename(clinical_path)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Extract key identifiers from filename
        if "CLINIC_metal" in name_without_ext:
            parts = name_without_ext.split("_")
            clinic_idx = None
            for i, part in enumerate(parts):
                if part == "CLINIC" and i + 2 < len(parts):
                    clinic_idx = parts[i + 2]
                    break
            
            if clinic_idx:
                # Look for mask file
                mask_pattern = f"CLINIC_metal_{clinic_idx}_mask_4label.nii"
                mask_files = list(Path(mask_data_dir).glob(mask_pattern))
                
                if mask_files:
                    mask_path = mask_files[0]
                    mask_data = nib.load(str(mask_path)).get_fdata()
                    print(f"    ✓ Found external mask: {mask_path.name}")
                    return mask_data
        
        return None
            
    except Exception as e:
        return None

def apply_mask_to_image(image, mask):
    """Apply mask to image for visualization"""
    if mask is None:
        return image
    
    # Ensure mask is binary
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Apply mask - keep original where mask is 0, set to 0 where mask is 1
    masked_image = image.copy()
    masked_image[binary_mask > 0] = 0
    
    return masked_image

def create_comparison_grid(images, titles, sample_idx, save_path, variant_name):
    """Create and save comparison grid"""
    try:
        fig, axes = plt.subplots(1, len(images), figsize=(5*len(images), 5))
        if len(images) == 1:
            axes = [axes]
        
        for i, (img, title) in enumerate(zip(images, titles)):
            axes[i].imshow(img, cmap='gray', vmin=0, vmax=255)
            axes[i].set_title(f'{title}', fontsize=12)
            axes[i].axis('off')
        
        plt.suptitle(f'{variant_name} - Sample {sample_idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"  Error creating comparison grid: {e}")

def original_dataset_normalize(data, minmax):
    """
    Original dataset normalization EXACTLY matching the provided example
    From: normalize() function in the example code
    """
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    data = data * 255.0
    data = data.astype(np.float32)
    return data

def image_get_minmax():
    """Image min/max values from example code"""
    return 0.0, 1.0

def process_original_dataset(model, dataset_path, output_dir, device, variant_name):
    """Process original dataset with PSNR/SSIM calculation using example preprocessing"""
    print(f"\n📊 PROCESSING ORIGINAL DATASET - {variant_name.upper()}")
    print("="*60)
    print(f"🔧 Using EXACT preprocessing from test_640geo example code")
    
    # Create directory structure
    variant_dir = Path(output_dir) / variant_name
    results_dir = variant_dir / "original_dataset"
    comparisons_dir = results_dir / "individual_results"
    
    for dir_path in [comparisons_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    opt_dataset = {
        'name': 'test_h5',
        'dataset_type': 'li_ct',
        'dataroot_H': dataset_path,
        'dataroot_L': dataset_path,
        'H_size': 416,
        'phase': 'test'
    }
    
    test_dataset = select_dataset.define_Dataset(opt_dataset)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    print(f"📂 Found {len(test_dataset)} samples in original dataset")
    
    psnr_values = []
    ssim_values = []
    processed_results = []
    
    for i, test_data in enumerate(tqdm(test_loader, desc=f"Processing {variant_name} - original")):
        try:
            L = test_data['L'].to(device)
            H = test_data['H'].to(device)
            
            # Model inference
            with torch.no_grad():
                model.feed_data({'L': L, 'H': H})
                model.test()
                E = model.current_visuals()['E']
            
            # Convert to images for metrics
            L_img = util.tensor2uint(L.squeeze().cpu())
            H_img = util.tensor2uint(H.squeeze().cpu())
            E_img = util.tensor2uint(E.squeeze().cpu())
            
            # Calculate PSNR and SSIM
            psnr = util.calculate_psnr(E_img, H_img, border=0)
            ssim = util.calculate_ssim(E_img, H_img, border=0)
            
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            
            # Store detailed results
            result_info = {
                'sample_idx': i+1,
                'psnr': float(psnr),
                'ssim': float(ssim),
                'file_path': test_data.get('L_path', ['unknown'])[0]
            }
            processed_results.append(result_info)
            
            # Save comparison every 50 samples
            if i % 50 == 0:
                images = [L_img, E_img, H_img]
                titles = ['Input (L)', 'Enhanced (E)', 'Target (H)']
                comparison_path = comparisons_dir / f"{variant_name}_original_{i+1:06d}.png"
                create_comparison_grid(images, titles, i+1, comparison_path, variant_name)
            
            # Progress update every 100 samples
            if i % 100 == 0:
                current_psnr = np.mean(psnr_values)
                current_ssim = np.mean(ssim_values)
                print(f"  Progress: {i+1}/{len(test_dataset)} - Avg PSNR: {current_psnr:.4f}dB, Avg SSIM: {current_ssim:.6f}")
                
        except Exception as e:
            print(f"  Error processing sample {i+1}: {e}")
            continue
    
    # Calculate final statistics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    std_psnr = np.std(psnr_values)
    std_ssim = np.std(ssim_values)
    
    # Save detailed results
    summary = {
        'variant': variant_name,
        'dataset_type': 'original_syn_deeplesion_test_640geo',
        'total_samples': len(processed_results),
        'preprocessing': {
            'method': 'exact_test_640geo_example',
            'normalization': 'image_get_minmax() -> [0.0, 1.0] -> clip -> normalize -> *255',
            'formula': '(data - data_min) / (data_max - data_min) * 255.0',
            'data_type': 'float32',
            'description': 'Exact preprocessing matching test_640geo example code'
        },
        'metrics': {
            'psnr': {
                'mean': float(avg_psnr),
                'std': float(std_psnr),
                'min': float(np.min(psnr_values)),
                'max': float(np.max(psnr_values))
            },
            'ssim': {
                'mean': float(avg_ssim),
                'std': float(std_ssim),
                'min': float(np.min(ssim_values)),
                'max': float(np.max(ssim_values))
            }
        },
        'detailed_results': processed_results
    }
    
    with open(results_dir / "metrics_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 {variant_name.upper()} ORIGINAL DATASET RESULTS:")
    print(f"   Average PSNR: {avg_psnr:.4f} ± {std_psnr:.4f} dB")
    print(f"   Average SSIM: {avg_ssim:.6f} ± {std_ssim:.6f}")
    print(f"   Total samples: {len(processed_results)}")
    print(f"   Comparison images: {len(processed_results) // 50}")
    print(f"   Results saved to: {results_dir}")
    
    return summary

def process_clinical_dataset_with_masks(model, clinical_data_dir, mask_data_dir, output_dir, device, variant_name):
    """Process clinical dataset with specialized preprocessing and masking"""
    print(f"\n🏥 PROCESSING CLINICAL DATASET WITH MASKS - {variant_name.upper()}")
    print("="*60)
    
    config = ClinicalConfig()
    
    # Find all clinical images
    clinical_files = list(Path(clinical_data_dir).glob("*.nii*"))
    clinical_files.sort()
    
    print(f"📊 Found {len(clinical_files)} clinical .nii files")
    print(f"🔧 Using specialized clinical preprocessing with metal artifact masking")
    
    # Create organized directory structure
    variant_dir = Path(output_dir) / variant_name
    results_dir = variant_dir / "clinical_dataset_with_masks"
    comparisons_dir = results_dir / "individual_results"
    preprocessing_dir = results_dir / "preprocessing_visualizations"
    
    for dir_path in [comparisons_dir, preprocessing_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    processed_results = []
    processed_count = 0
    files_with_masks = 0
    
    for file_idx, clinical_file in enumerate(tqdm(clinical_files, desc=f"Processing {variant_name} - clinical with masks")):
        try:
            print(f"  📁 [{file_idx+1}/{len(clinical_files)}] Processing {clinical_file.name}...")
            
            # Load clinical image
            nii_img = nib.load(str(clinical_file))
            clinical_data = nii_img.get_fdata()
            
            print(f"     Original shape: {clinical_data.shape}, HU range: [{clinical_data.min():.1f}, {clinical_data.max():.1f}]")
            
            # Load external mask if available
            external_mask = load_clinical_mask_fixed(str(clinical_file), mask_data_dir)
            if external_mask is not None:
                files_with_masks += 1
                print(f"     External mask shape: {external_mask.shape}")
            
            # Process limited number of slices for memory optimization
            num_slices = clinical_data.shape[2] if len(clinical_data.shape) > 2 else 1
            max_slices = min(num_slices, 12)  # Process fewer slices
            slice_indices = list(range(0, max_slices, 3))  # Every 3rd slice
            print(f"     Processing {len(slice_indices)} slices: {slice_indices}")
            
            for slice_idx in slice_indices:
                try:
                    slice_data = clinical_data[:, :, slice_idx]
                    
                    # Skip empty slices
                    if np.max(slice_data) <= np.min(slice_data):
                        continue
                    
                    # Apply specialized clinical preprocessing
                    processed_slice = clinical_preprocessing(slice_data, config)
                    if processed_slice is None:
                        continue
                    
                    # Detect metal regions using HU-based thresholding
                    detected_metal_mask = detect_metal_mask(slice_data, config)
                    
                    # Use external mask if available, otherwise use detected mask
                    if external_mask is not None and slice_idx < external_mask.shape[2]:
                        mask_slice = external_mask[:, :, slice_idx]
                        mask_slice = cv2.resize(mask_slice.astype(np.float32), (416, 416), 
                                              interpolation=cv2.INTER_NEAREST)
                        mask_type = "external"
                    else:
                        mask_slice = detected_metal_mask
                        mask_type = "detected"
                    
                    # Convert to tensor for model inference
                    slice_tensor = torch.from_numpy(processed_slice).float().unsqueeze(0).unsqueeze(0).to(device)
                    slice_tensor = slice_tensor / 255.0  # Normalize to [0,1]
                    
                    # Model inference
                    with torch.no_grad():
                        model.feed_data({'L': slice_tensor, 'H': slice_tensor})
                        model.test()
                        E_tensor = model.current_visuals()['E']
                    
                    # Convert back to images
                    L_img = util.tensor2uint(slice_tensor.squeeze().cpu())
                    E_img = util.tensor2uint(E_tensor.squeeze().cpu())
                    
                    # Apply mask for visualization
                    E_img_masked = apply_mask_to_image(E_img, mask_slice)
                    
                    processed_count += 1
                    
                    # Store result info
                    result_info = {
                        'sample_idx': processed_count,
                        'file': clinical_file.name,
                        'slice_idx': slice_idx,
                        'mask_type': mask_type,
                        'has_mask': mask_slice is not None,
                        'shape': f"{slice_data.shape[0]}x{slice_data.shape[1]}",
                        'hu_range': f"[{slice_data.min():.1f}, {slice_data.max():.1f}]"
                    }
                    processed_results.append(result_info)
                    
                    # Save comparisons and preprocessing visualizations
                    if processed_count % 20 == 0:  # More frequent saves for clinical
                        # Regular comparison
                        images = [L_img, E_img]
                        titles = ['Input (Metal Artifacts)', 'Enhanced']
                        
                        if mask_slice is not None:
                            images.append(E_img_masked)
                            titles.append(f'Enhanced + Mask ({mask_type})')
                        
                        comparison_path = comparisons_dir / f"{variant_name}_clinical_{processed_count:06d}.png"
                        create_comparison_grid(images, titles, processed_count, comparison_path, variant_name)
                        
                        # Preprocessing visualization
                        if processed_count % 40 == 0:  # Less frequent preprocessing vis
                            # Show original HU, windowed, and processed
                            slice_windowed = np.clip(slice_data, config.CTpara['window'][0], config.CTpara['window'][1])
                            slice_windowed_norm = ((slice_windowed - config.CTpara['window'][0]) / 
                                                 (config.CTpara['window'][1] - config.CTpara['window'][0]) * 255).astype(np.uint8)
                            
                            preprocessing_images = [
                                ((slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255).astype(np.uint8),
                                slice_windowed_norm,
                                processed_slice
                            ]
                            preprocessing_titles = ['Original HU', 'Windowed', 'Preprocessed (LAC + Min-Max)']
                            
                            if mask_slice is not None:
                                preprocessing_images.append((mask_slice * 255).astype(np.uint8))
                                preprocessing_titles.append(f'Metal Mask ({mask_type})')
                            
                            preprocessing_path = preprocessing_dir / f"{variant_name}_preprocessing_{processed_count:06d}.png"
                            create_comparison_grid(preprocessing_images, preprocessing_titles, 
                                                 processed_count, preprocessing_path, variant_name)
                        
                        print(f"    ✓ Saved {variant_name} clinical comparison {processed_count}: {clinical_file.name} slice {slice_idx}")
                    
                except Exception as slice_error:
                    print(f"    ❌ Error processing slice {slice_idx}: {slice_error}")
                    continue
                    
            # Clear memory after each file
            del clinical_data
            if external_mask is not None:
                del external_mask
            
        except Exception as e:
            print(f"  ❌ Error processing {clinical_file.name}: {e}")
            continue
    
    # Save clinical summary
    summary = {
        'variant': variant_name,
        'dataset_type': 'clinical_metal_artifacts_with_masks',
        'total_samples': len(processed_results),
        'total_files': len(clinical_files),
        'files_with_external_masks': files_with_masks,
        'comparisons_saved': len(processed_results) // 20,
        'preprocessing_visualizations': len(processed_results) // 40,
        'specialized_preprocessing': {
            'hu_windowing': config.CTpara['window'],
            'linear_attenuation_conversion': 'HU/1000 * 0.192 + 0.192',
            'normalization': 'min-max normalization to [0,1] then [0,255]',
            'metal_threshold_hu': 2500,
            'target_size': '416x416'
        },
        'detailed_results': processed_results
    }
    
    with open(results_dir / "clinical_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 {variant_name.upper()} CLINICAL DATASET WITH MASKS RESULTS:")
    print(f"   Total slices processed: {len(processed_results)}")
    print(f"   Total files processed: {len(clinical_files)}")
    print(f"   Files with external masks: {files_with_masks}")
    print(f"   Comparison images saved: {len(processed_results) // 20}")
    print(f"   Preprocessing visualizations: {len(processed_results) // 40}")
    print(f"   Results saved to: {results_dir}")
    
    return summary

def process_clinical_dataset_no_masks(model, clinical_data_dir, output_dir, device, variant_name):
    """Process clinical dataset without any masking - pure image enhancement"""
    print(f"\n🏥 PROCESSING CLINICAL DATASET WITHOUT MASKS - {variant_name.upper()}")
    print("="*60)
    
    config = ClinicalConfig()
    
    # Find all clinical images
    clinical_files = list(Path(clinical_data_dir).glob("*.nii*"))
    clinical_files.sort()
    
    print(f"📊 Found {len(clinical_files)} clinical .nii files")
    print(f"🔧 Using specialized clinical preprocessing WITHOUT masking - pure enhancement")
    
    # Create organized directory structure
    variant_dir = Path(output_dir) / variant_name
    results_dir = variant_dir / "clinical_dataset_no_masks"
    comparisons_dir = results_dir / "individual_results"
    preprocessing_dir = results_dir / "preprocessing_visualizations"
    
    for dir_path in [comparisons_dir, preprocessing_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    processed_results = []
    processed_count = 0
    
    for file_idx, clinical_file in enumerate(tqdm(clinical_files, desc=f"Processing {variant_name} - clinical no masks")):
        try:
            print(f"  📁 [{file_idx+1}/{len(clinical_files)}] Processing {clinical_file.name}...")
            
            # Load clinical image
            nii_img = nib.load(str(clinical_file))
            clinical_data = nii_img.get_fdata()
            
            print(f"     Original shape: {clinical_data.shape}, HU range: [{clinical_data.min():.1f}, {clinical_data.max():.1f}]")
            
            # Process limited number of slices for memory optimization
            num_slices = clinical_data.shape[2] if len(clinical_data.shape) > 2 else 1
            max_slices = min(num_slices, 12)  # Process fewer slices
            slice_indices = list(range(0, max_slices, 3))  # Every 3rd slice
            print(f"     Processing {len(slice_indices)} slices: {slice_indices}")
            
            for slice_idx in slice_indices:
                try:
                    slice_data = clinical_data[:, :, slice_idx]
                    
                    # Skip empty slices
                    if np.max(slice_data) <= np.min(slice_data):
                        continue
                    
                    # Apply specialized clinical preprocessing (same as masked version)
                    processed_slice = clinical_preprocessing(slice_data, config)
                    if processed_slice is None:
                        continue
                    
                    # Convert to tensor for model inference
                    slice_tensor = torch.from_numpy(processed_slice).float().unsqueeze(0).unsqueeze(0).to(device)
                    slice_tensor = slice_tensor / 255.0  # Normalize to [0,1]
                    
                    # Model inference - NO MASKING APPLIED
                    with torch.no_grad():
                        model.feed_data({'L': slice_tensor, 'H': slice_tensor})
                        model.test()
                        E_tensor = model.current_visuals()['E']
                    
                    # Convert back to images
                    L_img = util.tensor2uint(slice_tensor.squeeze().cpu())
                    E_img = util.tensor2uint(E_tensor.squeeze().cpu())
                    
                    processed_count += 1
                    
                    # Store result info (no mask information)
                    result_info = {
                        'sample_idx': processed_count,
                        'file': clinical_file.name,
                        'slice_idx': slice_idx,
                        'mask_type': 'none',
                        'has_mask': False,
                        'shape': f"{slice_data.shape[0]}x{slice_data.shape[1]}",
                        'hu_range': f"[{slice_data.min():.1f}, {slice_data.max():.1f}]"
                    }
                    processed_results.append(result_info)
                    
                    # Save comparisons and preprocessing visualizations
                    if processed_count % 20 == 0:  # More frequent saves for clinical
                        # Regular comparison - NO MASKED VERSION
                        images = [L_img, E_img]
                        titles = ['Input (Metal Artifacts)', 'Enhanced (No Masking)']
                        
                        comparison_path = comparisons_dir / f"{variant_name}_clinical_nomask_{processed_count:06d}.png"
                        create_comparison_grid(images, titles, processed_count, comparison_path, variant_name)
                        
                        # Preprocessing visualization
                        if processed_count % 40 == 0:  # Less frequent preprocessing vis
                            # Show original HU, windowed, and processed
                            slice_windowed = np.clip(slice_data, config.CTpara['window'][0], config.CTpara['window'][1])
                            slice_windowed_norm = ((slice_windowed - config.CTpara['window'][0]) / 
                                                 (config.CTpara['window'][1] - config.CTpara['window'][0]) * 255).astype(np.uint8)
                            
                            preprocessing_images = [
                                ((slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255).astype(np.uint8),
                                slice_windowed_norm,
                                processed_slice
                            ]
                            preprocessing_titles = ['Original HU', 'Windowed', 'Preprocessed (LAC + Min-Max)']
                            
                            preprocessing_path = preprocessing_dir / f"{variant_name}_preprocessing_nomask_{processed_count:06d}.png"
                            create_comparison_grid(preprocessing_images, preprocessing_titles, 
                                                 processed_count, preprocessing_path, variant_name)
                        
                        print(f"    ✓ Saved {variant_name} clinical no-mask comparison {processed_count}: {clinical_file.name} slice {slice_idx}")
                
                except Exception as slice_error:
                    print(f"    ❌ Error processing slice {slice_idx}: {slice_error}")
                    continue
                    
            # Clear memory after each file
            del clinical_data
            
        except Exception as e:
            print(f"  ❌ Error processing {clinical_file.name}: {e}")
            continue
    
    # Save clinical summary
    summary = {
        'variant': variant_name,
        'dataset_type': 'clinical_metal_artifacts_no_masks',
        'total_samples': len(processed_results),
        'total_files': len(clinical_files),
        'masking_applied': False,
        'comparisons_saved': len(processed_results) // 20,
        'preprocessing_visualizations': len(processed_results) // 40,
        'specialized_preprocessing': {
            'hu_windowing': config.CTpara['window'],
            'linear_attenuation_conversion': 'HU/1000 * 0.192 + 0.192',
            'normalization': 'min-max normalization to [0,1] then [0,255]',
            'masking': 'none - pure image enhancement',
            'target_size': '416x416'
        },
        'detailed_results': processed_results
    }
    
    with open(results_dir / "clinical_no_masks_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 {variant_name.upper()} CLINICAL DATASET (NO MASKS) RESULTS:")
    print(f"   Total slices processed: {len(processed_results)}")
    print(f"   Total files processed: {len(clinical_files)}")
    print(f"   Masking applied: None - pure enhancement")
    print(f"   Comparison images saved: {len(processed_results) // 20}")
    print(f"   Preprocessing visualizations: {len(processed_results) // 40}")
    print(f"   Results saved to: {results_dir}")
    
    return summary

def update_comparative_analysis(output_dir):
    """Update the existing comparative analysis to include conv_nstb results"""
    
    inference_dir = Path(output_dir)
    
    # Load existing comparative analysis
    comparative_file = inference_dir / "comparative_analysis_summary.json"
    complete_results_file = inference_dir / "enhanced_multi_model_complete_results.json"
    
    if comparative_file.exists():
        with open(comparative_file, 'r') as f:
            comparative_data = json.load(f)
    else:
        comparative_data = {}
    
    if complete_results_file.exists():
        with open(complete_results_file, 'r') as f:
            complete_data = json.load(f)
    else:
        complete_data = {}
    
    # Load conv_nstb results
    conv_nstb_dir = inference_dir / "conv_nstb"
    
    if conv_nstb_dir.exists():
        # Load original dataset results
        original_summary_file = conv_nstb_dir / "original_dataset" / "metrics_summary.json"
        if original_summary_file.exists():
            with open(original_summary_file, 'r') as f:
                original_summary = json.load(f)
        else:
            original_summary = None
        
        # Load clinical with masks results
        clinical_masks_file = conv_nstb_dir / "clinical_dataset_with_masks" / "clinical_summary.json"
        if clinical_masks_file.exists():
            with open(clinical_masks_file, 'r') as f:
                clinical_masks_summary = json.load(f)
        else:
            clinical_masks_summary = None
        
        # Load clinical no masks results
        clinical_no_masks_file = conv_nstb_dir / "clinical_dataset_no_masks" / "clinical_no_masks_summary.json"
        if clinical_no_masks_file.exists():
            with open(clinical_no_masks_file, 'r') as f:
                clinical_no_masks_summary = json.load(f)
        else:
            clinical_no_masks_summary = None
        
        # Add conv_nstb to complete results
        complete_data['conv_nstb'] = {
            'variant_info': {
                'checkpoint_path': 'training_results/conv_nstb/.../psnr_G.pth',
                'config_file': 'options/train_scunet_ngswin_3.json',
                'display_name': 'Conv NSTB',
                'variant_type': 'conv_nstb'
            },
            'original_dataset': original_summary,
            'clinical_dataset': clinical_masks_summary,
            'clinical_no_masks_dataset': clinical_no_masks_summary
        }
        
        # Update comparative analysis
        if 'original_summaries' not in comparative_data:
            comparative_data['original_summaries'] = {}
        if 'clinical_summaries' not in comparative_data:
            comparative_data['clinical_summaries'] = {}
        if 'clinical_no_masks_summaries' not in comparative_data:
            comparative_data['clinical_no_masks_summaries'] = {}
        
        if original_summary:
            comparative_data['original_summaries']['conv_nstb'] = original_summary
        if clinical_masks_summary:
            comparative_data['clinical_summaries']['conv_nstb'] = clinical_masks_summary
        if clinical_no_masks_summary:
            comparative_data['clinical_no_masks_summaries']['conv_nstb'] = clinical_no_masks_summary
        
        # Update metadata
        comparative_data['evaluation_info'] = comparative_data.get('evaluation_info', {})
        comparative_data['evaluation_info']['models_tested'] = list(complete_data.keys())
        comparative_data['evaluation_info']['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save updated files
        with open(comparative_file, 'w') as f:
            json.dump(comparative_data, f, indent=2)
        
        with open(complete_results_file, 'w') as f:
            json.dump(complete_data, f, indent=2)
        
        print(f"\n✅ Updated comparative analysis with conv_nstb results")
        print(f"   Updated: {comparative_file}")
        print(f"   Updated: {complete_results_file}")

def main():
    """Main function to test conv_nstb variant"""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test conv_nstb variant and add to existing results')
    parser.add_argument('--original_data', type=str, 
                       default='/home/Drive-D/SynDeepLesion/test_640geo', 
                       help='Path to original dataset')
    parser.add_argument('--clinical_data', type=str, 
                       default='/home/Drive-D/clinical_metal', 
                       help='Path to clinical dataset')
    parser.add_argument('--clinical_masks', type=str, 
                       default='/home/Drive-D/clinical_metal_mask', 
                       help='Path to clinical masks')
    
    args = parser.parse_args()
    
    # Use existing inference_1 folder
    output_dir = '/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR/inference_results/inference_1'
    
    print("🚀 CONV_NSTB SINGLE MODEL EVALUATION")
    print("="*80)
    print(f"📂 Output directory: {output_dir}")
    print(f"📊 Original dataset: {args.original_data}")
    print(f"🏥 Clinical dataset: {args.clinical_data}")
    print(f"🎭 Clinical masks: {args.clinical_masks}")
    
    # Check paths
    if not os.path.exists(args.original_data):
        print(f"❌ Original dataset not found: {args.original_data}")
        return
    
    if not os.path.exists(args.clinical_data):
        print(f"❌ Clinical dataset not found: {args.clinical_data}")
        return
    
    if not os.path.exists(args.clinical_masks):
        print(f"❌ Clinical masks not found: {args.clinical_masks}")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    try:
        # Get model info
        print(f"\n🔍 Getting conv_nstb model information...")
        model_info = get_conv_nstb_model()
        
        print(f"📋 Model Info:")
        print(f"   Display Name: {model_info['display_name']}")
        print(f"   Variant Type: {model_info['variant_type']}")
        print(f"   Config File: {model_info['config_file']}")
        print(f"   Checkpoint: {model_info['checkpoint_path']}")
        
        # Setup model
        model = setup_model_and_checkpoint(
            model_info['checkpoint_path'],
            model_info['config_file'],
            model_info['display_name']
        )
        
        # Process datasets
        variant_name = model_info['variant_type']
        
        # 1. Process original dataset
        original_summary = process_original_dataset(
            model, args.original_data, output_dir, device, variant_name
        )
        
        # 2. Process clinical dataset with masks
        clinical_masks_summary = process_clinical_dataset_with_masks(
            model, args.clinical_data, args.clinical_masks, output_dir, device, variant_name
        )
        
        # 3. Process clinical dataset without masks
        clinical_no_masks_summary = process_clinical_dataset_no_masks(
            model, args.clinical_data, output_dir, device, variant_name
        )
        
        # 4. Update comparative analysis
        update_comparative_analysis(output_dir)
        
        print(f"\n✅ CONV_NSTB EVALUATION COMPLETED SUCCESSFULLY!")
        print(f"📂 Results added to existing folder: {output_dir}")
        print(f"🎯 Model tested: {model_info['display_name']} ({variant_name})")
        
        # Summary statistics
        if original_summary and 'metrics' in original_summary:
            psnr = original_summary['metrics']['psnr']['mean']
            ssim = original_summary['metrics']['ssim']['mean']
            print(f"📊 Original Dataset Performance:")
            print(f"   PSNR: {psnr:.4f} dB")
            print(f"   SSIM: {ssim:.6f}")
        
        if clinical_masks_summary:
            print(f"🏥 Clinical Processing:")
            print(f"   With masks: {clinical_masks_summary['total_samples']} slices processed")
            print(f"   Without masks: {clinical_no_masks_summary['total_samples']} slices processed")
        
        print(f"\n🎉 conv_nstb variant successfully added to the evaluation!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
