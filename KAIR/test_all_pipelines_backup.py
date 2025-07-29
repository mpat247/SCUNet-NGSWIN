#!/usr/bin/env python3
"""
Comprehensive Multi-Model Pipeline Evaluation
============================================

This script tests ALL trained model variants with exact preprocessing from test_conv_nstb_single.py:
- conv_nstb, conv_trans_nstb, trans_nstb
- Tests on 3 datasets: original (test_640geo), clinical without masks, clinical with masks
- Uses exact clinical preprocessing pipeline
- Saves results incrementally in inference_results (inference_2, inference_3, etc.)

Usage:
    python test_all_pipelines.py
"""

import os
import sys
import json
import argparse
import numpy as np
import nibabel as nib
import cv2
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# Add KAIR to path for imports
sys.path.append('/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR')

# Import KAIR modules
from utils import utils_option as option
from utils import utils_image as util
from data import select_dataset
from models import select_model

# Clinical preprocessing configuration
class ClinicalConfig:
    """Configuration for clinical data preprocessing - EXACT match to test_conv_nstb_single.py"""
    def __init__(self):
        self.CTpara = {
            'imPixNum': 416,            # Image pixels along x or y direction
            'angSize': 0.05,            # Angle between two neighbor rays
            'linSize': 1.8536,
            'angNum': 640,              # Number of projection views
            'SOD': 1075,                # Source-to-origin distance
            'imPixScale': 512 / 416 * 0.03,
            'sinogram_size_x': 640,
            'sinogram_size_y': 641,
            'window': [-175, 275]       # HU window
        }
        self.mask_thre = 2500 / 1000 * 0.192 + 0.192  # Metal threshold - EXACT match to example

def clinical_preprocessing(image_data, config):
    """
    Clinical preprocessing EXACTLY matching the provided example
    From: clinic_input_data() function in test_conv_nstb_single.py
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
    From: clinic_input_data() function - metal mask generation in test_conv_nstb_single.py
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

def original_dataset_normalize(data, minmax):
    """
    Original dataset normalization EXACTLY matching the provided example
    From: normalize() function in the example code from test_conv_nstb_single.py
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

def discover_all_trained_models():
    """Discover all trained model variants from training_results folder"""
    base_path = '/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR/training_results'
    models_info = []
    
    print("🔍 Discovering trained models...")
    
    # Check each variant folder
    variant_folders = ['conv_nstb', 'conv_trans_nstb', 'trans_nstb']
    
    for variant in variant_folders:
        variant_path = Path(base_path) / variant
        if not variant_path.exists():
            print(f"⚠️  {variant} folder not found, skipping...")
            continue
        
        # Look for the scunet_ngswin_* subfolder
        subfolders = [d for d in variant_path.iterdir() if d.is_dir() and d.name.startswith('scunet_ngswin_')]
        if not subfolders:
            print(f"⚠️  No scunet_ngswin_* subfolder found in {variant}, skipping...")
            continue
        
        models_path = subfolders[0] / 'models'
        if not models_path.exists():
            print(f"⚠️  Models folder not found for {variant}, skipping...")
            continue
        
        # Use psnr_G.pth as the best checkpoint
        psnr_checkpoint = models_path / 'psnr_G.pth'
        if not psnr_checkpoint.exists():
            print(f"⚠️  psnr_G.pth not found for {variant}, skipping...")
            continue
        
        # Determine config file based on variant
        if variant == 'conv_nstb':
            config_file = 'options/train_scunet_ngswin_1.json'
        elif variant == 'conv_trans_nstb':
            config_file = 'options/train_scunet_ngswin_2.json'
        elif variant == 'trans_nstb':
            config_file = 'options/train_scunet_ngswin_3.json'
        else:
            config_file = 'options/train_scunet_ngswin_1.json'  # fallback
        
        model_info = {
            'variant': variant,
            'checkpoint_path': str(psnr_checkpoint),
            'config_file': config_file,
            'display_name': variant.replace('_', ' ').title(),
            'subfolder_name': subfolders[0].name
        }
        
        models_info.append(model_info)
        print(f"✓ Found {variant}: {psnr_checkpoint}")
    
    print(f"📊 Total models discovered: {len(models_info)}")
    return models_info

def setup_model_and_checkpoint(model_info):
    """Load model and checkpoint with detailed logging"""
    variant = model_info['variant']
    checkpoint_path = model_info['checkpoint_path']
    config_path = model_info['config_file']
    
    print(f"🔧 Loading {variant} model from: {os.path.basename(checkpoint_path)}")
    
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

def get_next_inference_folder(base_dir):
    """Get the next available inference folder (inference_2, inference_3, etc.)"""
    base_path = Path(base_dir)
    
    # Find existing inference folders
    existing_folders = []
    for folder in base_path.glob('inference_*'):
        if folder.is_dir():
            try:
                folder_num = int(folder.name.split('_')[1])
                existing_folders.append(folder_num)
            except (IndexError, ValueError):
                continue
    
    # Get next number
    if existing_folders:
        next_num = max(existing_folders) + 1
    else:
        next_num = 1
    
    next_folder = base_path / f"inference_{next_num}"
    next_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Using output folder: {next_folder}")
    return str(next_folder)

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

def create_detailed_comparison_with_metrics(images, titles, psnr, ssim, sample_idx, save_path, variant_name):
    """Create detailed comparison with metrics overlay"""
    try:
        fig, axes = plt.subplots(2, len(images), figsize=(5*len(images), 8))
        if len(images) == 1:
            axes = axes.reshape(2, 1)
        
        # Top row: Images
        for i, (img, title) in enumerate(zip(images, titles)):
            axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=255)
            axes[0, i].set_title(f'{title}', fontsize=12)
            axes[0, i].axis('off')
        
        # Bottom row: Difference maps and metrics
        if len(images) >= 3:  # L, E, H
            L_img, E_img, H_img = images[0], images[1], images[2]
            
            # Difference between enhanced and target
            diff_E_H = np.abs(E_img.astype(float) - H_img.astype(float))
            axes[1, 0].imshow(diff_E_H, cmap='hot', vmin=0, vmax=50)
            axes[1, 0].set_title(f'|Enhanced - Target|\nMax Diff: {diff_E_H.max():.1f}', fontsize=10)
            axes[1, 0].axis('off')
            
            # Difference between input and target
            diff_L_H = np.abs(L_img.astype(float) - H_img.astype(float))
            axes[1, 1].imshow(diff_L_H, cmap='hot', vmin=0, vmax=50)
            axes[1, 1].set_title(f'|Input - Target|\nMax Diff: {diff_L_H.max():.1f}', fontsize=10)
            axes[1, 1].axis('off')
            
            # Metrics text
            axes[1, 2].text(0.1, 0.7, f'PSNR: {psnr:.4f} dB', fontsize=14, fontweight='bold')
            axes[1, 2].text(0.1, 0.5, f'SSIM: {ssim:.6f}', fontsize=14, fontweight='bold')
            axes[1, 2].text(0.1, 0.3, f'Sample: {sample_idx}', fontsize=12)
            axes[1, 2].text(0.1, 0.1, f'Model: {variant_name}', fontsize=12)
            axes[1, 2].set_xlim(0, 1)
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].axis('off')
        
        plt.suptitle(f'{variant_name} - Detailed Analysis - Sample {sample_idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"  Error creating detailed comparison: {e}")

def save_important_results(best_psnr_samples, worst_psnr_samples, best_ssim_samples, worst_ssim_samples, 
                          important_results_dir, best_samples_dir, worst_samples_dir, metrics_visualization_dir, variant_name):
    """Save the most important results and create visualizations"""
    try:
        print(f"💾 Saving important results for {variant_name}...")
        
        # Save best PSNR samples
        for i, sample in enumerate(best_psnr_samples):
            images = [sample['images']['L'], sample['images']['E'], sample['images']['H']]
            titles = [f'Input (L)', f'Enhanced (E)', f'Target (H)']
            save_path = best_samples_dir / f"best_psnr_{i+1:02d}_sample_{sample['idx']:06d}_psnr_{sample['psnr']:.4f}.png"
            create_detailed_comparison_with_metrics(images, titles, sample['psnr'], sample['ssim'], sample['idx'], save_path, variant_name)
        
        # Save worst PSNR samples
        for i, sample in enumerate(worst_psnr_samples):
            images = [sample['images']['L'], sample['images']['E'], sample['images']['H']]
            titles = [f'Input (L)', f'Enhanced (E)', f'Target (H)']
            save_path = worst_samples_dir / f"worst_psnr_{i+1:02d}_sample_{sample['idx']:06d}_psnr_{sample['psnr']:.4f}.png"
            create_detailed_comparison_with_metrics(images, titles, sample['psnr'], sample['ssim'], sample['idx'], save_path, variant_name)
        
        # Save best SSIM samples
        for i, sample in enumerate(best_ssim_samples):
            images = [sample['images']['L'], sample['images']['E'], sample['images']['H']]
            titles = [f'Input (L)', f'Enhanced (E)', f'Target (H)']
            save_path = best_samples_dir / f"best_ssim_{i+1:02d}_sample_{sample['idx']:06d}_ssim_{sample['ssim']:.6f}.png"
            create_detailed_comparison_with_metrics(images, titles, sample['psnr'], sample['ssim'], sample['idx'], save_path, variant_name)
        
        # Create metrics distribution plots
        create_metrics_visualizations([s['psnr'] for s in best_psnr_samples + worst_psnr_samples], 
                                    [s['ssim'] for s in best_ssim_samples + worst_ssim_samples],
                                    metrics_visualization_dir, variant_name)
        
        # Save important results summary
        important_summary = {
            'variant': variant_name,
            'timestamp': datetime.now().isoformat(),
            'best_psnr_samples': [
                {
                    'rank': i+1,
                    'sample_idx': s['idx'],
                    'psnr': s['psnr'],
                    'ssim': s['ssim'],
                    'file_path': s['file_path']
                } for i, s in enumerate(best_psnr_samples)
            ],
            'worst_psnr_samples': [
                {
                    'rank': i+1,
                    'sample_idx': s['idx'],
                    'psnr': s['psnr'],
                    'ssim': s['ssim'],
                    'file_path': s['file_path']
                } for i, s in enumerate(worst_psnr_samples)
            ],
            'best_ssim_samples': [
                {
                    'rank': i+1,
                    'sample_idx': s['idx'],
                    'psnr': s['psnr'],
                    'ssim': s['ssim'],
                    'file_path': s['file_path']
                } for i, s in enumerate(best_ssim_samples)
            ],
            'statistics': {
                'best_psnr_range': [best_psnr_samples[-1]['psnr'], best_psnr_samples[0]['psnr']] if best_psnr_samples else [0, 0],
                'worst_psnr_range': [worst_psnr_samples[0]['psnr'], worst_psnr_samples[-1]['psnr']] if worst_psnr_samples else [0, 0],
                'best_ssim_range': [best_ssim_samples[-1]['ssim'], best_ssim_samples[0]['ssim']] if best_ssim_samples else [0, 0],
            }
        }
        
        with open(important_results_dir / f"{variant_name}_important_results_summary.json", 'w') as f:
            json.dump(important_summary, f, indent=2)
        
        print(f"   ✓ Saved {len(best_psnr_samples)} best PSNR samples")
        print(f"   ✓ Saved {len(worst_psnr_samples)} worst PSNR samples") 
        print(f"   ✓ Saved {len(best_ssim_samples)} best SSIM samples")
        print(f"   ✓ Created metrics visualizations")
        
    except Exception as e:
        print(f"   ❌ Error saving important results: {e}")

def create_metrics_visualizations(psnr_values, ssim_values, metrics_dir, variant_name):
    """Create visualization plots for metrics analysis"""
    try:
        # PSNR distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(psnr_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f'{variant_name} - PSNR Distribution')
        plt.xlabel('PSNR (dB)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.hist(ssim_values, bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.title(f'{variant_name} - SSIM Distribution')
        plt.xlabel('SSIM')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.scatter(psnr_values, ssim_values, alpha=0.6, s=30)
        plt.title(f'{variant_name} - PSNR vs SSIM Correlation')
        plt.xlabel('PSNR (dB)')
        plt.ylabel('SSIM')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.boxplot([psnr_values], labels=['PSNR'])
        plt.title(f'{variant_name} - PSNR Box Plot')
        plt.ylabel('PSNR (dB)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(metrics_dir / f"{variant_name}_metrics_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create detailed statistics plot
        plt.figure(figsize=(10, 6))
        
        stats_text = f"""
{variant_name} - Detailed Statistics

PSNR Statistics:
Mean: {np.mean(psnr_values):.4f} dB
Std:  {np.std(psnr_values):.4f} dB
Min:  {np.min(psnr_values):.4f} dB
Max:  {np.max(psnr_values):.4f} dB

SSIM Statistics:
Mean: {np.mean(ssim_values):.6f}
Std:  {np.std(ssim_values):.6f}
Min:  {np.min(ssim_values):.6f}
Max:  {np.max(ssim_values):.6f}

Sample Count: {len(psnr_values)}
        """
        
        plt.text(0.1, 0.5, stats_text, fontsize=12, fontfamily='monospace',
                verticalalignment='center', transform=plt.gca().transAxes)
        plt.axis('off')
        plt.title(f'{variant_name} - Statistics Summary', fontsize=16, fontweight='bold')
        
        plt.savefig(metrics_dir / f"{variant_name}_statistics_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"   ❌ Error creating metrics visualizations: {e}")

def process_original_dataset(model, dataset_path, output_dir, device, variant_name):
    """Process original dataset with PSNR/SSIM calculation using exact preprocessing"""
    print(f"\n📊 PROCESSING ORIGINAL DATASET - {variant_name.upper()}")
    print("="*60)
    print(f"🔧 Using EXACT preprocessing from test_640geo example code")
    
    # Create comprehensive directory structure
    variant_dir = Path(output_dir) / variant_name
    results_dir = variant_dir / "original_dataset"
    comparisons_dir = results_dir / "individual_results"
    detailed_comparisons_dir = results_dir / "detailed_comparisons"
    important_results_dir = results_dir / f"important_results_{variant_name}"
    best_samples_dir = important_results_dir / "best_samples"
    worst_samples_dir = important_results_dir / "worst_samples"
    metrics_visualization_dir = important_results_dir / "metrics_visualizations"
    logs_dir = results_dir / "detailed_logs"
    
    for dir_path in [comparisons_dir, detailed_comparisons_dir, important_results_dir, 
                     best_samples_dir, worst_samples_dir, metrics_visualization_dir, logs_dir]:
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
    detailed_log = []
    
    # For tracking best and worst samples
    best_psnr_samples = []
    worst_psnr_samples = []
    best_ssim_samples = []
    worst_ssim_samples = []
    
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
                'file_path': test_data.get('L_path', ['unknown'])[0],
                'input_stats': {
                    'min': float(L_img.min()),
                    'max': float(L_img.max()),
                    'mean': float(L_img.mean()),
                    'std': float(L_img.std())
                },
                'enhanced_stats': {
                    'min': float(E_img.min()),
                    'max': float(E_img.max()),
                    'mean': float(E_img.mean()),
                    'std': float(E_img.std())
                },
                'target_stats': {
                    'min': float(H_img.min()),
                    'max': float(H_img.max()),
                    'mean': float(H_img.mean()),
                    'std': float(H_img.std())
                }
            }
            processed_results.append(result_info)
            
            # Track best and worst samples for detailed analysis
            sample_data = {
                'idx': i+1,
                'psnr': psnr,
                'ssim': ssim,
                'images': {'L': L_img, 'E': E_img, 'H': H_img},
                'file_path': test_data.get('L_path', ['unknown'])[0]
            }
            
            # Update best/worst tracking
            best_psnr_samples.append(sample_data)
            best_psnr_samples.sort(key=lambda x: x['psnr'], reverse=True)
            best_psnr_samples = best_psnr_samples[:10]  # Keep top 10
            
            worst_psnr_samples.append(sample_data)
            worst_psnr_samples.sort(key=lambda x: x['psnr'])
            worst_psnr_samples = worst_psnr_samples[:10]  # Keep bottom 10
            
            best_ssim_samples.append(sample_data)
            best_ssim_samples.sort(key=lambda x: x['ssim'], reverse=True)
            best_ssim_samples = best_ssim_samples[:10]  # Keep top 10
            
            worst_ssim_samples.append(sample_data)
            worst_ssim_samples.sort(key=lambda x: x['ssim'])
            worst_ssim_samples = worst_ssim_samples[:10]  # Keep bottom 10
            
            # Create detailed log entry
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'sample_idx': i+1,
                'processing_status': 'success',
                'metrics': {'psnr': psnr, 'ssim': ssim},
                'file_info': {
                    'path': test_data.get('L_path', ['unknown'])[0],
                    'tensor_shapes': {
                        'L': list(L.shape),
                        'H': list(H.shape),
                        'E': list(E.shape)
                    }
                }
            }
            detailed_log.append(log_entry)
            
            # Save comparison every 25 samples (more frequent)
            if i % 25 == 0:
                images = [L_img, E_img, H_img]
                titles = ['Input (L)', 'Enhanced (E)', 'Target (H)']
                comparison_path = comparisons_dir / f"{variant_name}_original_{i+1:06d}.png"
                create_comparison_grid(images, titles, i+1, comparison_path, variant_name)
                
                # Save detailed comparison with metrics overlay
                detailed_comparison_path = detailed_comparisons_dir / f"{variant_name}_detailed_{i+1:06d}.png"
                create_detailed_comparison_with_metrics(images, titles, psnr, ssim, i+1, detailed_comparison_path, variant_name)
            
            # Progress update every 50 samples (more frequent)
            if i % 50 == 0:
                current_psnr = np.mean(psnr_values)
                current_ssim = np.mean(ssim_values)
                print(f"  Progress: {i+1}/{len(test_dataset)} - Avg PSNR: {current_psnr:.4f}dB, Avg SSIM: {current_ssim:.6f}")
                
                # Save intermediate log
                intermediate_log_path = logs_dir / f"processing_log_{i+1:06d}.json"
                with open(intermediate_log_path, 'w') as f:
                    json.dump({
                        'processed_samples': i+1,
                        'current_metrics': {
                            'psnr': {'mean': current_psnr, 'std': np.std(psnr_values)},
                            'ssim': {'mean': current_ssim, 'std': np.std(ssim_values)}
                        },
                        'recent_samples': detailed_log[-50:] if len(detailed_log) >= 50 else detailed_log
                    }, f, indent=2)
                
        except Exception as e:
            print(f"  Error processing sample {i+1}: {e}")
            # Log error
            error_log_entry = {
                'timestamp': datetime.now().isoformat(),
                'sample_idx': i+1,
                'processing_status': 'error',
                'error_message': str(e),
                'file_info': {
                    'path': test_data.get('L_path', ['unknown'])[0] if 'L_path' in test_data else 'unknown'
                }
            }
            detailed_log.append(error_log_entry)
            continue
    
    # Calculate final statistics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    std_psnr = np.std(psnr_values)
    std_ssim = np.std(ssim_values)
    
    # Save important results (best/worst samples, visualizations)
    save_important_results(best_psnr_samples, worst_psnr_samples, best_ssim_samples, worst_ssim_samples, 
                          important_results_dir, best_samples_dir, worst_samples_dir, metrics_visualization_dir, variant_name)
    
    # Save complete detailed log
    complete_log_path = logs_dir / f"{variant_name}_complete_processing_log.json"
    with open(complete_log_path, 'w') as f:
        json.dump({
            'variant': variant_name,
            'dataset_type': 'original_syn_deeplesion_test_640geo',
            'processing_summary': {
                'total_samples_attempted': len(test_dataset),
                'successfully_processed': len(processed_results),
                'failed_samples': len(test_dataset) - len(processed_results),
                'processing_time': datetime.now().isoformat()
            },
            'final_metrics': {
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
            'detailed_processing_log': detailed_log
        }, f, indent=2)
    
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
        'important_results_summary': {
            'best_psnr_count': len(best_psnr_samples),
            'worst_psnr_count': len(worst_psnr_samples),
            'best_ssim_count': len(best_ssim_samples),
            'visualizations_created': True,
            'detailed_logs_saved': True
        },
        'output_structure': {
            'comparisons_saved': len(processed_results) // 25,
            'detailed_comparisons_saved': len(processed_results) // 25,
            'important_results_folder': f"important_results_{variant_name}",
            'logs_folder': "detailed_logs"
        },
        'detailed_results': processed_results
    }
    
    with open(results_dir / "metrics_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 {variant_name.upper()} ORIGINAL DATASET RESULTS:")
    print(f"   Average PSNR: {avg_psnr:.4f} ± {std_psnr:.4f} dB")
    print(f"   Average SSIM: {avg_ssim:.6f} ± {std_ssim:.6f}")
    print(f"   Total samples: {len(processed_results)}")
    print(f"   Comparison images: {len(processed_results) // 25}")
    print(f"   Detailed comparisons: {len(processed_results) // 25}")
    print(f"   Important results saved: {len(best_psnr_samples)} best + {len(worst_psnr_samples)} worst")
    print(f"   Complete logs saved: {complete_log_path}")
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
    important_results_dir = results_dir / f"important_results_{variant_name}_clinical_masks"
    best_enhancements_dir = important_results_dir / "best_enhancements"
    preprocessing_examples_dir = important_results_dir / "preprocessing_examples"
    mask_analysis_dir = important_results_dir / "mask_analysis"
    logs_dir = results_dir / "detailed_logs"
    
    for dir_path in [comparisons_dir, preprocessing_dir, important_results_dir, 
                     best_enhancements_dir, preprocessing_examples_dir, mask_analysis_dir, logs_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    processed_results = []
    processed_count = 0
    files_with_masks = 0
    detailed_log = []
    enhancement_quality_samples = []  # Track enhancement quality
    
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
                    
                    # Get external mask for this slice if available
                    external_slice_mask = None
                    if external_mask is not None and len(external_mask.shape) > 2:
                        if slice_idx < external_mask.shape[2]:
                            external_slice_mask = external_mask[:, :, slice_idx]
                            if external_slice_mask.shape != (config.CTpara['imPixNum'], config.CTpara['imPixNum']):
                                external_slice_mask = cv2.resize(external_slice_mask.astype(np.float32), 
                                                                (config.CTpara['imPixNum'], config.CTpara['imPixNum']), 
                                                                interpolation=cv2.INTER_NEAREST)
                    
                    # Detect metal mask using preprocessing
                    detected_mask = detect_metal_mask(slice_data, config)
                    
                    # Use external mask if available, otherwise use detected mask
                    final_mask = external_slice_mask if external_slice_mask is not None else detected_mask
                    
                    # Convert to tensor for model input
                    input_tensor = torch.from_numpy(processed_slice).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
                    
                    # Model inference
                    with torch.no_grad():
                        model.feed_data({'L': input_tensor, 'H': input_tensor})
                        model.test()
                        output_tensor = model.current_visuals()['E']
                    
                    # Convert back to image
                    enhanced_slice = util.tensor2uint(output_tensor.squeeze().cpu())
                    
                    # Apply mask to enhanced image for visualization
                    masked_enhanced = apply_mask_to_image(enhanced_slice, final_mask)
                    
                    # Calculate enhancement quality metrics
                    enhancement_diff = np.abs(enhanced_slice.astype(float) - processed_slice.astype(float))
                    enhancement_quality = {
                        'mean_enhancement': float(enhancement_diff.mean()),
                        'max_enhancement': float(enhancement_diff.max()),
                        'std_enhancement': float(enhancement_diff.std()),
                        'enhancement_range': [float(enhanced_slice.min()), float(enhanced_slice.max())]
                    }
                    
                    # Store results with detailed information
                    result_info = {
                        'file_name': clinical_file.name,
                        'slice_idx': slice_idx,
                        'processed_count': processed_count + 1,
                        'has_external_mask': external_slice_mask is not None,
                        'has_detected_mask': detected_mask is not None,
                        'original_hu_range': [float(slice_data.min()), float(slice_data.max())],
                        'processed_range': [float(processed_slice.min()), float(processed_slice.max())],
                        'enhancement_quality': enhancement_quality,
                        'mask_statistics': {
                            'external_mask_coverage': float((external_slice_mask > 0).sum() / external_slice_mask.size) if external_slice_mask is not None else 0,
                            'detected_mask_coverage': float((detected_mask > 0).sum() / detected_mask.size) if detected_mask is not None else 0
                        }
                    }
                    processed_results.append(result_info)
                    processed_count += 1
                    
                    # Track samples for important results
                    sample_data = {
                        'file_name': clinical_file.name,
                        'slice_idx': slice_idx,
                        'processed_count': processed_count,
                        'images': {
                            'original': slice_data,
                            'processed': processed_slice,
                            'enhanced': enhanced_slice,
                            'masked_enhanced': masked_enhanced
                        },
                        'masks': {
                            'external': external_slice_mask,
                            'detected': detected_mask,
                            'final': final_mask
                        },
                        'quality_metrics': enhancement_quality
                    }
                    enhancement_quality_samples.append(sample_data)
                    
                    # Keep only top enhancement samples
                    enhancement_quality_samples.sort(key=lambda x: x['quality_metrics']['mean_enhancement'], reverse=True)
                    enhancement_quality_samples = enhancement_quality_samples[:20]  # Keep top 20
                    
                    # Detailed logging
                    log_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'file_name': clinical_file.name,
                        'slice_idx': slice_idx,
                        'processed_count': processed_count,
                        'processing_status': 'success',
                        'mask_info': {
                            'external_mask_found': external_slice_mask is not None,
                            'detected_mask_created': detected_mask is not None,
                            'final_mask_used': final_mask is not None
                        },
                        'enhancement_metrics': enhancement_quality
                    }
                    detailed_log.append(log_entry)
                    
                    # Save comparison every 15 samples (more frequent)
                    if processed_count % 15 == 0:
                        images = [processed_slice, enhanced_slice, masked_enhanced]
                        titles = ['Preprocessed Input', 'Enhanced', 'Enhanced + Mask']
                        comparison_path = comparisons_dir / f"{variant_name}_clinical_masked_{processed_count:06d}.png"
                        create_comparison_grid(images, titles, processed_count, comparison_path, variant_name)
                    
                    # Save preprocessing visualization every 30 samples
                    if processed_count % 30 == 0:
                        prep_images = [slice_data, processed_slice]
                        prep_titles = ['Original HU', 'Preprocessed LAC']
                        prep_path = preprocessing_dir / f"{variant_name}_preprocessing_{processed_count:06d}.png"
                        create_comparison_grid(prep_images, prep_titles, processed_count, prep_path, variant_name)
                    
                    # Save important examples every 50 samples
                    if processed_count % 50 == 0:
                        # Save comprehensive analysis
                        analysis_images = [slice_data, processed_slice, enhanced_slice, final_mask if final_mask is not None else np.zeros_like(enhanced_slice)]
                        analysis_titles = ['Original HU', 'Preprocessed LAC', 'Enhanced', 'Applied Mask']
                        analysis_path = preprocessing_examples_dir / f"{variant_name}_analysis_{processed_count:06d}.png"
                        create_comparison_grid(analysis_images, analysis_titles, processed_count, analysis_path, variant_name)
                        
                        # Progress log
                        progress_log_path = logs_dir / f"progress_log_{processed_count:06d}.json"
                        with open(progress_log_path, 'w') as f:
                            json.dump({
                                'processed_samples': processed_count,
                                'files_processed': file_idx + 1,
                                'files_with_masks': files_with_masks,
                                'recent_enhancements': [s['quality_metrics'] for s in enhancement_quality_samples[-10:]],
                                'recent_log_entries': detailed_log[-20:]
                            }, f, indent=2)
                
                except Exception as slice_error:
                    print(f"    Error processing slice {slice_idx}: {slice_error}")
                    continue
            
        except Exception as e:
            print(f"  Error processing file {clinical_file.name}: {e}")
            continue
    
    # Save best enhancement samples to important results
    print(f"💾 Saving important clinical results for {variant_name}...")
    
    # Save top enhancement samples
    for i, sample in enumerate(enhancement_quality_samples[:10]):
        analysis_images = [
            sample['images']['original'], 
            sample['images']['processed'], 
            sample['images']['enhanced'],
            sample['masks']['final'] if sample['masks']['final'] is not None else np.zeros_like(sample['images']['enhanced'])
        ]
        analysis_titles = ['Original HU', 'Preprocessed LAC', 'Enhanced', 'Applied Mask']
        save_path = best_enhancements_dir / f"best_enhancement_{i+1:02d}_file_{sample['file_name']}_slice_{sample['slice_idx']:03d}.png"
        create_comparison_grid(analysis_images, analysis_titles, sample['processed_count'], save_path, variant_name)
    
    # Save mask analysis examples
    mask_examples = [s for s in enhancement_quality_samples if s['masks']['external'] is not None][:5]
    for i, sample in enumerate(mask_examples):
        mask_comparison = [
            sample['masks']['detected'] if sample['masks']['detected'] is not None else np.zeros_like(sample['images']['enhanced']),
            sample['masks']['external'] if sample['masks']['external'] is not None else np.zeros_like(sample['images']['enhanced']),
            sample['masks']['final'] if sample['masks']['final'] is not None else np.zeros_like(sample['images']['enhanced']),
            sample['images']['enhanced']
        ]
        mask_titles = ['Detected Mask', 'External Mask', 'Final Mask', 'Enhanced Image']
        save_path = mask_analysis_dir / f"mask_analysis_{i+1:02d}_file_{sample['file_name']}_slice_{sample['slice_idx']:03d}.png"
        create_comparison_grid(mask_comparison, mask_titles, sample['processed_count'], save_path, variant_name)
    
    # Save complete processing log
    complete_log_path = logs_dir / f"{variant_name}_clinical_masks_complete_log.json"
    with open(complete_log_path, 'w') as f:
        json.dump({
            'variant': variant_name,
            'dataset_type': 'clinical_metal_artifacts_with_masks',
            'processing_summary': {
                'total_files': len(clinical_files),
                'files_with_external_masks': files_with_masks,
                'total_slices_processed': len(processed_results),
                'processing_time': datetime.now().isoformat()
            },
            'enhancement_analysis': {
                'top_enhancements': [
                    {
                        'rank': i+1,
                        'file_name': s['file_name'],
                        'slice_idx': s['slice_idx'],
                        'enhancement_metrics': s['quality_metrics']
                    } for i, s in enumerate(enhancement_quality_samples[:10])
                ]
            },
            'detailed_processing_log': detailed_log
        }, f, indent=2)
    
    # Save clinical summary
    summary = {
        'variant': variant_name,
        'dataset_type': 'clinical_metal_artifacts_with_masks',
        'total_samples': len(processed_results),
        'total_files': len(clinical_files),
        'files_with_external_masks': files_with_masks,
        'comparisons_saved': len(processed_results) // 15,
        'preprocessing_visualizations': len(processed_results) // 30,
        'important_results_summary': {
            'best_enhancements_saved': len(enhancement_quality_samples[:10]),
            'mask_analysis_examples': len(mask_examples),
            'preprocessing_examples_saved': len(processed_results) // 50,
            'detailed_logs_saved': True
        },
        'specialized_preprocessing': {
            'hu_windowing': config.CTpara['window'],
            'linear_attenuation_conversion': 'HU/1000 * 0.192 + 0.192',
            'normalization': 'min-max normalization to [0,1] then [0,255]',
            'metal_threshold_hu': 2500,
            'target_size': '416x416'
        },
        'output_structure': {
            'individual_results': "individual_results/",
            'preprocessing_visualizations': "preprocessing_visualizations/",
            'important_results_folder': f"important_results_{variant_name}_clinical_masks",
            'logs_folder': "detailed_logs/"
        },
        'detailed_results': processed_results
    }
    
    with open(results_dir / "clinical_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 {variant_name.upper()} CLINICAL DATASET WITH MASKS RESULTS:")
    print(f"   Total slices processed: {len(processed_results)}")
    print(f"   Total files processed: {len(clinical_files)}")
    print(f"   Files with external masks: {files_with_masks}")
    print(f"   Comparison images saved: {len(processed_results) // 15}")
    print(f"   Preprocessing visualizations: {len(processed_results) // 30}")
    print(f"   Best enhancement examples: {len(enhancement_quality_samples[:10])}")
    print(f"   Mask analysis examples: {len(mask_examples)}")
    print(f"   Complete processing log: {complete_log_path}")
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
    important_results_dir = results_dir / f"important_results_{variant_name}_clinical_nomask"
    best_enhancements_dir = important_results_dir / "best_enhancements"
    enhancement_analysis_dir = important_results_dir / "enhancement_analysis"
    logs_dir = results_dir / "detailed_logs"
    
    for dir_path in [comparisons_dir, preprocessing_dir, important_results_dir, 
                     best_enhancements_dir, enhancement_analysis_dir, logs_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    processed_results = []
    processed_count = 0
    detailed_log = []
    enhancement_quality_samples = []  # Track enhancement quality
    
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
                    
                    # Apply specialized clinical preprocessing
                    processed_slice = clinical_preprocessing(slice_data, config)
                    if processed_slice is None:
                        continue
                    
                    # Convert to tensor for model input
                    input_tensor = torch.from_numpy(processed_slice).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
                    
                    # Model inference
                    with torch.no_grad():
                        model.feed_data({'L': input_tensor, 'H': input_tensor})
                        model.test()
                        output_tensor = model.current_visuals()['E']
                    
                    # Convert back to image
                    enhanced_slice = util.tensor2uint(output_tensor.squeeze().cpu())
                    
                    # Calculate enhancement quality metrics  
                    enhancement_diff = np.abs(enhanced_slice.astype(float) - processed_slice.astype(float))
                    enhancement_quality = {
                        'mean_enhancement': float(enhancement_diff.mean()),
                        'max_enhancement': float(enhancement_diff.max()),
                        'std_enhancement': float(enhancement_diff.std()),
                        'enhancement_range': [float(enhanced_slice.min()), float(enhanced_slice.max())],
                        'noise_reduction_estimate': float(np.std(processed_slice) - np.std(enhanced_slice))
                    }
                    
                    # Store results with detailed information
                    result_info = {
                        'file_name': clinical_file.name,
                        'slice_idx': slice_idx,
                        'processed_count': processed_count + 1,
                        'masking_applied': False,
                        'original_hu_range': [float(slice_data.min()), float(slice_data.max())],
                        'processed_range': [float(processed_slice.min()), float(processed_slice.max())],
                        'enhancement_quality': enhancement_quality
                    }
                    processed_results.append(result_info)
                    processed_count += 1
                    
                    # Track samples for important results
                    sample_data = {
                        'file_name': clinical_file.name,
                        'slice_idx': slice_idx,
                        'processed_count': processed_count,
                        'images': {
                            'original': slice_data,
                            'processed': processed_slice,
                            'enhanced': enhanced_slice
                        },
                        'quality_metrics': enhancement_quality
                    }
                    enhancement_quality_samples.append(sample_data)
                    
                    # Keep only top enhancement samples
                    enhancement_quality_samples.sort(key=lambda x: x['quality_metrics']['mean_enhancement'], reverse=True)
                    enhancement_quality_samples = enhancement_quality_samples[:15]  # Keep top 15
                    
                    # Detailed logging
                    log_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'file_name': clinical_file.name,
                        'slice_idx': slice_idx,
                        'processed_count': processed_count,
                        'processing_status': 'success',
                        'enhancement_metrics': enhancement_quality
                    }
                    detailed_log.append(log_entry)
                    
                    # Save comparison every 15 samples (more frequent)
                    if processed_count % 15 == 0:
                        images = [processed_slice, enhanced_slice]
                        titles = ['Preprocessed Input', 'Enhanced']
                        comparison_path = comparisons_dir / f"{variant_name}_clinical_nomask_{processed_count:06d}.png"
                        create_comparison_grid(images, titles, processed_count, comparison_path, variant_name)
                    
                    # Save preprocessing visualization every 30 samples
                    if processed_count % 30 == 0:
                        prep_images = [slice_data, processed_slice]
                        prep_titles = ['Original HU', 'Preprocessed LAC']
                        prep_path = preprocessing_dir / f"{variant_name}_preprocessing_{processed_count:06d}.png"
                        create_comparison_grid(prep_images, prep_titles, processed_count, prep_path, variant_name)
                    
                    # Save enhancement analysis every 45 samples
                    if processed_count % 45 == 0:
                        # Create enhancement difference visualization
                        enhancement_analysis = [slice_data, processed_slice, enhanced_slice, enhancement_diff]
                        analysis_titles = ['Original HU', 'Preprocessed LAC', 'Enhanced', 'Enhancement Diff']
                        analysis_path = enhancement_analysis_dir / f"{variant_name}_enhancement_analysis_{processed_count:06d}.png"
                        create_comparison_grid(enhancement_analysis, analysis_titles, processed_count, analysis_path, variant_name)
                        
                        # Progress log
                        progress_log_path = logs_dir / f"progress_log_{processed_count:06d}.json"
                        with open(progress_log_path, 'w') as f:
                            json.dump({
                                'processed_samples': processed_count,
                                'files_processed': file_idx + 1,
                                'recent_enhancements': [s['quality_metrics'] for s in enhancement_quality_samples[-10:]],
                                'recent_log_entries': detailed_log[-20:]
                            }, f, indent=2)
                
                except Exception as slice_error:
                    print(f"    Error processing slice {slice_idx}: {slice_error}")
                    continue
            
        except Exception as e:
            print(f"  Error processing file {clinical_file.name}: {e}")
            continue
    
    # Save best enhancement samples to important results
    print(f"💾 Saving important clinical no-mask results for {variant_name}...")
    
    # Save top enhancement samples
    for i, sample in enumerate(enhancement_quality_samples[:10]):
        enhancement_comparison = [
            sample['images']['original'], 
            sample['images']['processed'], 
            sample['images']['enhanced'],
            np.abs(sample['images']['enhanced'].astype(float) - sample['images']['processed'].astype(float))
        ]
        comparison_titles = ['Original HU', 'Preprocessed LAC', 'Enhanced', 'Enhancement Difference']
        save_path = best_enhancements_dir / f"best_enhancement_{i+1:02d}_file_{sample['file_name']}_slice_{sample['slice_idx']:03d}.png"
        create_comparison_grid(enhancement_comparison, comparison_titles, sample['processed_count'], save_path, variant_name)
    
    # Save complete processing log
    complete_log_path = logs_dir / f"{variant_name}_clinical_nomask_complete_log.json"
    with open(complete_log_path, 'w') as f:
        json.dump({
            'variant': variant_name,
            'dataset_type': 'clinical_metal_artifacts_no_masks',
            'processing_summary': {
                'total_files': len(clinical_files),
                'total_slices_processed': len(processed_results),
                'processing_time': datetime.now().isoformat()
            },
            'enhancement_analysis': {
                'top_enhancements': [
                    {
                        'rank': i+1,
                        'file_name': s['file_name'],
                        'slice_idx': s['slice_idx'],
                        'enhancement_metrics': s['quality_metrics']
                    } for i, s in enumerate(enhancement_quality_samples[:10])
                ]
            },
            'detailed_processing_log': detailed_log
        }, f, indent=2)
    
    # Save clinical summary
    summary = {
        'variant': variant_name,
        'dataset_type': 'clinical_metal_artifacts_no_masks',
        'total_samples': len(processed_results),
        'total_files': len(clinical_files),
        'masking_applied': False,
        'comparisons_saved': len(processed_results) // 15,
        'preprocessing_visualizations': len(processed_results) // 30,
        'important_results_summary': {
            'best_enhancements_saved': len(enhancement_quality_samples[:10]),
            'enhancement_analysis_examples': len(processed_results) // 45,
            'detailed_logs_saved': True
        },
        'specialized_preprocessing': {
            'hu_windowing': config.CTpara['window'],
            'linear_attenuation_conversion': 'HU/1000 * 0.192 + 0.192',
            'normalization': 'min-max normalization to [0,1] then [0,255]',
            'masking': 'none - pure image enhancement',
            'target_size': '416x416'
        },
        'output_structure': {
            'individual_results': "individual_results/",
            'preprocessing_visualizations': "preprocessing_visualizations/",
            'important_results_folder': f"important_results_{variant_name}_clinical_nomask",
            'logs_folder': "detailed_logs/"
        },
        'detailed_results': processed_results
    }
    
    with open(results_dir / "clinical_no_masks_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 {variant_name.upper()} CLINICAL DATASET (NO MASKS) RESULTS:")
    print(f"   Total slices processed: {len(processed_results)}")
    print(f"   Total files processed: {len(clinical_files)}")
    print(f"   Masking applied: None - pure enhancement")
    print(f"   Comparison images saved: {len(processed_results) // 15}")
    print(f"   Preprocessing visualizations: {len(processed_results) // 30}")
    print(f"   Enhancement analysis examples: {len(processed_results) // 45}")
    print(f"   Best enhancement examples: {len(enhancement_quality_samples[:10])}")
    print(f"   Complete processing log: {complete_log_path}")
    print(f"   Results saved to: {results_dir}")
    
    return summary

def create_comprehensive_analysis(all_results, output_dir):
    """Create comprehensive analysis across all models and datasets"""
    print(f"\n📊 CREATING COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # Organize results by dataset type
    original_results = {}
    clinical_masked_results = {}
    clinical_nomask_results = {}
    
    for variant, results in all_results.items():
        if 'original_dataset' in results:
            original_results[variant] = results['original_dataset']
        if 'clinical_dataset_with_masks' in results:
            clinical_masked_results[variant] = results['clinical_dataset_with_masks']
        if 'clinical_dataset_no_masks' in results:
            clinical_nomask_results[variant] = results['clinical_dataset_no_masks']
    
    # Create comprehensive summary
    comprehensive_summary = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'total_variants_tested': len(all_results),
        'variants': list(all_results.keys()),
        'datasets_evaluated': {
            'original_dataset': {
                'variants_tested': len(original_results),
                'total_samples_per_variant': original_results[list(original_results.keys())[0]]['total_samples'] if original_results else 0,
                'results': original_results
            },
            'clinical_with_masks': {
                'variants_tested': len(clinical_masked_results),
                'results': clinical_masked_results
            },
            'clinical_without_masks': {
                'variants_tested': len(clinical_nomask_results),
                'results': clinical_nomask_results
            }
        },
        'preprocessing_methods': {
            'original_dataset': 'exact_test_640geo_example - image_get_minmax() normalization',
            'clinical_datasets': 'HU clamping -> LAC conversion -> min-max normalization'
        },
        'comparison_analysis': {}
    }
    
    # Performance comparison for original dataset
    if original_results:
        print("🔍 Analyzing original dataset performance...")
        psnr_comparison = {}
        ssim_comparison = {}
        
        for variant, result in original_results.items():
            psnr_comparison[variant] = result['metrics']['psnr']['mean']
            ssim_comparison[variant] = result['metrics']['ssim']['mean']
        
        # Find best performing models
        best_psnr_variant = max(psnr_comparison, key=psnr_comparison.get)
        best_ssim_variant = max(ssim_comparison, key=ssim_comparison.get)
        
        comprehensive_summary['comparison_analysis']['original_dataset'] = {
            'psnr_ranking': dict(sorted(psnr_comparison.items(), key=lambda x: x[1], reverse=True)),
            'ssim_ranking': dict(sorted(ssim_comparison.items(), key=lambda x: x[1], reverse=True)),
            'best_psnr': {
                'variant': best_psnr_variant,
                'value': psnr_comparison[best_psnr_variant]
            },
            'best_ssim': {
                'variant': best_ssim_variant,
                'value': ssim_comparison[best_ssim_variant]
            }
        }
        
        print(f"   Best PSNR: {best_psnr_variant} ({psnr_comparison[best_psnr_variant]:.4f} dB)")
        print(f"   Best SSIM: {best_ssim_variant} ({ssim_comparison[best_ssim_variant]:.6f})")
    
    # Clinical dataset analysis
    if clinical_masked_results:
        print("🔍 Analyzing clinical dataset performance...")
        clinical_comparison = {}
        
        for variant, result in clinical_masked_results.items():
            clinical_comparison[variant] = {
                'total_samples': result['total_samples'],
                'files_processed': result['total_files']
            }
        
        comprehensive_summary['comparison_analysis']['clinical_datasets'] = {
            'samples_processed': clinical_comparison,
            'preprocessing_consistency': 'All variants use identical clinical preprocessing pipeline'
        }
    
    # Save comprehensive analysis
    analysis_file = Path(output_dir) / "comprehensive_multi_model_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(comprehensive_summary, f, indent=2)
    
    # Create summary table
    summary_table = Path(output_dir) / "results_summary_table.txt"
    with open(summary_table, 'w') as f:
        f.write("COMPREHENSIVE MULTI-MODEL EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Original dataset results
        if original_results:
            f.write("ORIGINAL DATASET (test_640geo) RESULTS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Variant':<20} {'PSNR (dB)':<12} {'SSIM':<10} {'Samples':<10}\n")
            f.write("-" * 50 + "\n")
            
            for variant in sorted(original_results.keys()):
                result = original_results[variant]
                psnr = result['metrics']['psnr']['mean']
                ssim = result['metrics']['ssim']['mean']
                samples = result['total_samples']
                f.write(f"{variant:<20} {psnr:<12.4f} {ssim:<10.6f} {samples:<10}\n")
            f.write("\n")
        
        # Clinical dataset results
        if clinical_masked_results:
            f.write("CLINICAL DATASET RESULTS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Variant':<20} {'Slices':<10} {'Files':<10} {'With Masks':<12}\n")
            f.write("-" * 50 + "\n")
            
            for variant in sorted(clinical_masked_results.keys()):
                result = clinical_masked_results[variant]
                slices = result['total_samples']
                files = result['total_files']
                masked = result['files_with_external_masks']
                f.write(f"{variant:<20} {slices:<10} {files:<10} {masked:<12}\n")
            f.write("\n")
        
        f.write(f"Evaluation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Results saved to: {output_dir}\n")
    
    print(f"📊 Comprehensive analysis saved to: {analysis_file}")
    print(f"📋 Summary table saved to: {summary_table}")
    
    return comprehensive_summary

def main():
    """Main function to test all model pipelines"""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test all model pipelines with comprehensive evaluation')
    parser.add_argument('--original_data', type=str, 
                       default='/home/Drive-D/SynDeepLesion/test_640geo', 
                       help='Path to original dataset')
    parser.add_argument('--clinical_data', type=str, 
                       default='/home/Drive-D/clinical_metal', 
                       help='Path to clinical dataset')
    parser.add_argument('--clinical_masks', type=str, 
                       default='/home/Drive-D/clinical_metal_mask', 
                       help='Path to clinical masks')
    parser.add_argument('--base_output', type=str,
                       default='/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR/inference_results',
                       help='Base output directory for inference results')
    
    args = parser.parse_args()
    
    print("🚀 COMPREHENSIVE MULTI-MODEL PIPELINE EVALUATION")
    print("="*80)
    print(f"📂 Base output directory: {args.base_output}")
    print(f"📊 Original dataset: {args.original_data}")
    print(f"🏥 Clinical dataset: {args.clinical_data}")
    print(f"🎭 Clinical masks: {args.clinical_masks}")
    
    # Check paths
    if not os.path.exists(args.original_data):
        print(f"❌ Original data path not found: {args.original_data}")
        return
    
    if not os.path.exists(args.clinical_data):
        print(f"❌ Clinical data path not found: {args.clinical_data}")
        return
    
    if not os.path.exists(args.clinical_masks):
        print(f"❌ Clinical masks path not found: {args.clinical_masks}")
        return
    
    # Get next inference folder
    output_dir = get_next_inference_folder(args.base_output)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    try:
        # Discover all trained models
        models_info = discover_all_trained_models()
        
        if not models_info:
            print("❌ No trained models found!")
            return
        
        # Store all results
        all_results = {}
        
        # Process each model
        for model_idx, model_info in enumerate(models_info):
            variant = model_info['variant']
            print(f"\n🎯 PROCESSING MODEL {model_idx+1}/{len(models_info)}: {variant.upper()}")
            print("="*80)
            
            try:
                # Load model
                model = setup_model_and_checkpoint(model_info)
                model.netG.eval()
                
                # Process all three datasets
                variant_results = {}
                
                # 1. Original dataset
                print(f"\n📊 Dataset 1/3: Original Dataset")
                original_summary = process_original_dataset(
                    model, args.original_data, output_dir, device, variant
                )
                variant_results['original_dataset'] = original_summary
                
                # 2. Clinical with masks
                print(f"\n🏥 Dataset 2/3: Clinical Dataset with Masks")
                clinical_masked_summary = process_clinical_dataset_with_masks(
                    model, args.clinical_data, args.clinical_masks, output_dir, device, variant
                )
                variant_results['clinical_dataset_with_masks'] = clinical_masked_summary
                
                # 3. Clinical without masks
                print(f"\n🏥 Dataset 3/3: Clinical Dataset without Masks")
                clinical_nomask_summary = process_clinical_dataset_no_masks(
                    model, args.clinical_data, output_dir, device, variant
                )
                variant_results['clinical_dataset_no_masks'] = clinical_nomask_summary
                
                # Store results
                all_results[variant] = variant_results
                
                print(f"\n✅ {variant.upper()} EVALUATION COMPLETED")
                print(f"   Original dataset: {original_summary['metrics']['psnr']['mean']:.4f} dB PSNR")
                print(f"   Clinical samples processed: {clinical_masked_summary['total_samples']} + {clinical_nomask_summary['total_samples']}")
                
                # Clear memory
                del model
                torch.cuda.empty_cache()
                
            except Exception as model_error:
                print(f"❌ Error processing {variant}: {model_error}")
                continue
        
        # Create comprehensive analysis
        if all_results:
            comprehensive_summary = create_comprehensive_analysis(all_results, output_dir)
            
            print(f"\n🎉 EVALUATION COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"📊 Total models evaluated: {len(all_results)}")
            print(f"📂 Results saved to: {output_dir}")
            print(f"🔍 Variants tested: {', '.join(all_results.keys())}")
            
            # Show best performers
            if 'comparison_analysis' in comprehensive_summary and 'original_dataset' in comprehensive_summary['comparison_analysis']:
                best_psnr = comprehensive_summary['comparison_analysis']['original_dataset']['best_psnr']
                best_ssim = comprehensive_summary['comparison_analysis']['original_dataset']['best_ssim']
                print(f"🏆 Best PSNR: {best_psnr['variant']} ({best_psnr['value']:.4f} dB)")
                print(f"🏆 Best SSIM: {best_ssim['variant']} ({best_ssim['value']:.6f})")
        else:
            print("❌ No results generated!")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()