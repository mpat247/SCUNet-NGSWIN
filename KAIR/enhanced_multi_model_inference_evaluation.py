#!/usr/bin/env python3
"""
Enhanced Multi-Model Inference Evaluation for SCUNet-NGSWIN with Organized Results
=================================================================================

This script performs comprehensive inference evaluation across all model variations
with specialized clinical preprocessing for metal artifact removal.
Results are saved in organized inference_results/inference_N/ folders.

Features:
- Tests ALL model variations (conv_nstb, conv_trans_nstb, trans_nstb)
- Uses psnr_G.pth (best performing) checkpoints for each variation
- Original dataset: Standard processing with PSNR/SSIM metrics
- Clinical dataset: Specialized preprocessing for metal artifact removal
- Clinical preprocessing: HU windowing → LAC conversion → min-max normalization
- Memory optimization and enhanced logging
- Comparative analysis across all models
- Organized directory structure with incremental numbering

Usage:
    python enhanced_multi_model_inference_evaluation_organized.py
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
        self.mask_thre = 2500 / 1000 * 0.192 + 0.192  # Metal threshold

def get_next_inference_folder(base_dir):
    """
    Find the next available inference folder number
    Creates inference_results/inference_N/ structure
    """
    inference_base = os.path.join(base_dir, 'inference_results')
    
    # Find existing inference folders
    existing_folders = []
    if os.path.exists(inference_base):
        for item in os.listdir(inference_base):
            if item.startswith('inference_') and os.path.isdir(os.path.join(inference_base, item)):
                try:
                    number = int(item.split('_')[1])
                    existing_folders.append(number)
                except (ValueError, IndexError):
                    continue
    
    # Get next number
    next_number = max(existing_folders) + 1 if existing_folders else 1
    
    # Create the new inference folder
    new_folder = os.path.join(inference_base, f'inference_{next_number}')
    os.makedirs(new_folder, exist_ok=True)
    
    return new_folder, next_number

def clinical_preprocessing(image_data, config):
    """
    Specialized preprocessing for clinical CT images
    Based on clinical preprocessing code with linear attenuation coefficient conversion
    """
    try:
        # Clamp extreme HU values
        image_data = np.clip(image_data, -1000, 3000)
        
        # Apply window/level settings for CT
        window_min, window_max = config.CTpara['window']
        image_windowed = np.clip(image_data, window_min, window_max)
        
        # Convert to linear attenuation coefficient
        # This is the key improvement for clinical CT data processing
        image_lac = image_windowed / 1000 * 0.192 + 0.192
        
        # Normalize to [0, 1] range using min-max normalization
        # This preserves the relative intensity relationships better
        image_normalized = (image_lac - image_lac.min()) / (image_lac.max() - image_lac.min())
        
        # Convert to 0-255 range for model input
        image_uint8 = (image_normalized * 255).astype(np.uint8)
        
        # Resize to model input size (416x416)
        if image_uint8.shape != (416, 416):
            image_resized = cv2.resize(image_uint8, (416, 416), interpolation=cv2.INTER_LINEAR)
        else:
            image_resized = image_uint8
        
        return image_resized
        
    except Exception as e:
        print(f"  Error in clinical preprocessing: {e}")
        return None

def detect_metal_mask(image_data, config):
    """
    Enhanced metal detection based on HU values
    Adapted from clinical preprocessing code
    """
    try:
        # Use direct HU threshold for metal detection
        # This is more clinically accurate than the LAC conversion
        mask_thre_hu = 2500  # Metal threshold in HU
        metal_mask = (image_data > mask_thre_hu).astype(np.uint8)
        
        # Resize mask to match model input
        if metal_mask.shape != (416, 416):
            metal_mask = cv2.resize(metal_mask, (416, 416), interpolation=cv2.INTER_NEAREST)
        
        return metal_mask
        
    except Exception as e:
        print(f"  Error in metal detection: {e}")
        return None

def get_model_variations():
    """Get all available model variations and their best checkpoints"""
    base_path = '/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR/training_results'
    variations = {}
    
    # Define model variations
    model_folders = ['conv_nstb', 'conv_trans_nstb', 'trans_nstb']
    
    for folder in model_folders:
        variation_path = Path(base_path) / folder
        if variation_path.exists():
            # Look for the scunet_ngswin_* subfolder
            subfolders = [d for d in variation_path.iterdir() if d.is_dir() and d.name.startswith('scunet_ngswin_')]
            if subfolders:
                models_path = subfolders[0] / 'models'
                if models_path.exists():
                    # Use psnr_G.pth as the best checkpoint
                    psnr_checkpoint = models_path / 'psnr_G.pth'
                    if psnr_checkpoint.exists():
                        # Determine config file based on variation
                        if 'conv_trans' in folder:
                            config_suffix = '4'  # conv_trans_nstb uses config 4
                        elif 'trans' in folder:
                            config_suffix = '2'  # trans_nstb uses config 2  
                        else:
                            config_suffix = '3'  # conv_nstb uses config 3
                        
                        variations[folder] = {
                            'checkpoint_path': str(psnr_checkpoint),
                            'config_file': f'options/train_scunet_ngswin_{config_suffix}.json',
                            'display_name': folder.replace('_', ' ').title(),
                            'variant_type': folder
                        }
    
    return variations

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
        print(f"✓ Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Set to eval mode
    model.netG.eval()
    for param in model.netG.parameters():
        param.requires_grad = False
    
    print(f"✓ {variant_name} model set to evaluation mode")
    return model, opt

def calculate_metrics(img_pred, img_gt, border=0):
    """Calculate PSNR and SSIM metrics"""
    psnr = util.calculate_psnr(img_pred, img_gt, border=border)
    ssim = util.calculate_ssim(img_pred, img_gt, border=border)
    return psnr, ssim

def create_comparison_grid(images, titles, sample_idx, output_path, variant_name, metrics=None):
    """Create side-by-side comparison grid"""
    fig, axes = plt.subplots(1, len(images), figsize=(5*len(images), 5))
    
    # Handle single image case
    if len(images) == 1:
        axes = [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')
    
    # Add metrics to title if provided
    if metrics:
        fig.suptitle(f'{variant_name} - Sample {sample_idx} - PSNR: {metrics["psnr"]:.2f}dB, SSIM: {metrics["ssim"]:.4f}', 
                     fontsize=12, fontweight='bold')
    else:
        fig.suptitle(f'{variant_name} - Sample {sample_idx}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

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
    """Apply mask to image for better visualization"""
    if mask is None:
        return image
    
    try:
        # Ensure mask and image have compatible shapes
        if mask.shape != image.shape:
            mask_resized = cv2.resize(mask.astype(np.float32), 
                                    (image.shape[1], image.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask
        
        # Create binary mask
        binary_mask = (mask_resized > 0.5).astype(np.uint8)
        
        # Apply mask
        masked_image = image.copy()
        masked_image[binary_mask == 0] = 0
        
        return masked_image
        
    except Exception as e:
        print(f"  Error applying mask: {e}")
        return image

def process_original_dataset(model, test_loader, output_dir, device, variant_name):
    """Process original SynDeepLesion dataset with ground truth"""
    print(f"\n🔬 PROCESSING ORIGINAL DATASET - {variant_name.upper()}")
    print("="*60)
    
    # Create organized directory structure
    variant_dir = Path(output_dir) / variant_name
    results_dir = variant_dir / "original_dataset"
    comparisons_dir = results_dir / "individual_results"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_list = []
    processed_count = 0
    
    for batch_idx, test_data in enumerate(tqdm(test_loader, desc=f"Processing {variant_name} - original")):
        try:
            # Move data to device
            L_tensor = test_data['L'].to(device)
            H_tensor = test_data['H'].to(device)
            
            # Inference
            with torch.no_grad():
                model.feed_data({'L': L_tensor, 'H': H_tensor})
                model.test()
                E_tensor = model.current_visuals()['E']
            
            # Convert to images
            L_img = util.tensor2uint(L_tensor.squeeze().cpu())
            H_img = util.tensor2uint(H_tensor.squeeze().cpu())
            E_img = util.tensor2uint(E_tensor.squeeze().cpu())
            
            # Calculate metrics
            psnr, ssim = calculate_metrics(E_img, H_img)
            
            # Store metrics
            metrics_list.append({
                'sample_idx': processed_count + 1,
                'psnr': float(psnr),
                'ssim': float(ssim),
                'file_path': test_data.get('L_path', ['unknown'])[0]
            })
            
            processed_count += 1
            
            # Save comparison every 50 images
            if processed_count % 50 == 0:
                images = [L_img, E_img, H_img]
                titles = ['Input (Noisy)', 'Enhanced', 'Ground Truth']
                metrics_info = {'psnr': psnr, 'ssim': ssim}
                
                comparison_path = comparisons_dir / f"comparison_{processed_count:06d}.png"
                create_comparison_grid(images, titles, processed_count, comparison_path, 
                                     variant_name, metrics_info)
            
            # Progress update every 100 images
            if processed_count % 100 == 0:
                avg_psnr = np.mean([m['psnr'] for m in metrics_list[-100:]])
                avg_ssim = np.mean([m['ssim'] for m in metrics_list[-100:]])
                print(f"  📊 Progress: {processed_count} | Avg PSNR: {avg_psnr:.2f}dB, SSIM: {avg_ssim:.4f}")
                
        except Exception as e:
            print(f"  ❌ Error processing batch {batch_idx}: {e}")
            continue
    
    # Calculate final statistics
    if metrics_list:
        avg_psnr = np.mean([m['psnr'] for m in metrics_list])
        std_psnr = np.std([m['psnr'] for m in metrics_list])
        avg_ssim = np.mean([m['ssim'] for m in metrics_list])
        std_ssim = np.std([m['ssim'] for m in metrics_list])
        
        summary = {
            'variant': variant_name,
            'dataset_type': 'original_syndeeplesion',
            'total_samples': len(metrics_list),
            'comparisons_saved': len(metrics_list) // 50,
            'metrics': {
                'psnr': {
                    'mean': float(avg_psnr),
                    'std': float(std_psnr),
                    'min': float(min(m['psnr'] for m in metrics_list)),
                    'max': float(max(m['psnr'] for m in metrics_list))
                },
                'ssim': {
                    'mean': float(avg_ssim),
                    'std': float(std_ssim),
                    'min': float(min(m['ssim'] for m in metrics_list)),
                    'max': float(max(m['ssim'] for m in metrics_list))
                }
            },
            'detailed_results': metrics_list
        }
        
        # Save detailed results
        with open(results_dir / "metrics_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 {variant_name.upper()} ORIGINAL DATASET RESULTS:")
        print(f"   Total samples: {len(metrics_list)}")
        print(f"   Average PSNR: {avg_psnr:.4f} ± {std_psnr:.4f} dB")
        print(f"   Average SSIM: {avg_ssim:.6f} ± {std_ssim:.6f}")
        print(f"   Results saved to: {results_dir}")
        
        return summary
    
    return None

def process_clinical_dataset_specialized(model, clinical_data_dir, mask_data_dir, output_dir, device, variant_name):
    """Process clinical dataset with specialized preprocessing"""
    print(f"\n🏥 PROCESSING CLINICAL DATASET - {variant_name.upper()}")
    print("="*60)
    
    config = ClinicalConfig()
    
    # Find all clinical images
    clinical_files = list(Path(clinical_data_dir).glob("*.nii*"))
    clinical_files.sort()
    
    print(f"📊 Found {len(clinical_files)} clinical .nii files")
    print(f"🔧 Using specialized clinical preprocessing for metal artifacts")
    
    # Create organized directory structure
    variant_dir = Path(output_dir) / variant_name
    results_dir = variant_dir / "clinical_dataset"
    comparisons_dir = results_dir / "individual_results"
    preprocessing_dir = results_dir / "preprocessing_visualizations"
    
    for dir_path in [comparisons_dir, preprocessing_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    processed_results = []
    processed_count = 0
    files_with_masks = 0
    
    for file_idx, clinical_file in enumerate(tqdm(clinical_files, desc=f"Processing {variant_name} - clinical")):
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
        'dataset_type': 'clinical_metal_artifacts_specialized',
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
    
    print(f"\n📊 {variant_name.upper()} CLINICAL DATASET RESULTS:")
    print(f"   Total slices processed: {len(processed_results)}")
    print(f"   Total files processed: {len(clinical_files)}")
    print(f"   Files with external masks: {files_with_masks}")
    print(f"   Comparison images saved: {len(processed_results) // 20}")
    print(f"   Preprocessing visualizations: {len(processed_results) // 40}")
    print(f"   Results saved to: {results_dir}")
    
    return summary

def create_comparative_analysis(all_results, output_dir):
    """Create comparative analysis across all model variations"""
    print(f"\n📊 CREATING COMPARATIVE ANALYSIS")
    print("="*60)
    
    # Extract original dataset metrics for comparison
    original_metrics = {}
    clinical_summaries = {}
    
    for variant, results in all_results.items():
        if results['original_dataset']:
            original_metrics[variant] = results['original_dataset']['metrics']
        clinical_summaries[variant] = results['clinical_dataset']
    
    # Create comparative plots
    if original_metrics:
        # PSNR comparison
        variants = list(original_metrics.keys())
        psnr_means = [original_metrics[v]['psnr']['mean'] for v in variants]
        psnr_stds = [original_metrics[v]['psnr']['std'] for v in variants]
        ssim_means = [original_metrics[v]['ssim']['mean'] for v in variants]
        ssim_stds = [original_metrics[v]['ssim']['std'] for v in variants]
        
        # Create comparison plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # PSNR comparison
        bars1 = ax1.bar(variants, psnr_means, yerr=psnr_stds, capsize=5, 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('PSNR Comparison Across Model Variations')
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_xlabel('Model Variation')
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars1, psnr_means, psnr_stds)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
        
        # SSIM comparison
        bars2 = ax2.bar(variants, ssim_means, yerr=ssim_stds, capsize=5,
                       color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_title('SSIM Comparison Across Model Variations')
        ax2.set_ylabel('SSIM')
        ax2.set_xlabel('Model Variation')
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars2, ssim_means, ssim_stds)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.001,
                    f'{mean:.5f}±{std:.5f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        comparison_plot_path = Path(output_dir) / 'model_comparison.png'
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved model comparison plot: {comparison_plot_path}")
    
    # Create comprehensive summary
    comparative_summary = {
        'evaluation_info': {
            'total_variants_tested': len(all_results),
            'variants': list(all_results.keys()),
            'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'specialized_clinical_preprocessing': True,
            'clinical_preprocessing_details': {
                'hu_windowing': '[-175, 275]',
                'linear_attenuation_conversion': 'HU/1000 * 0.192 + 0.192',
                'normalization': 'min-max normalization',
                'metal_threshold': '2500 HU'
            }
        },
        'original_dataset_comparison': original_metrics,
        'clinical_dataset_summaries': clinical_summaries,
        'performance_ranking': {
            'by_psnr': sorted(original_metrics.items(), 
                            key=lambda x: x[1]['psnr']['mean'], reverse=True) if original_metrics else [],
            'by_ssim': sorted(original_metrics.items(), 
                            key=lambda x: x[1]['ssim']['mean'], reverse=True) if original_metrics else []
        }
    }
    
    # Save comparative summary
    summary_path = Path(output_dir) / 'comparative_analysis_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(comparative_summary, f, indent=2)
    
    print(f"✓ Saved comparative analysis: {summary_path}")
    
    # Print performance ranking
    if original_metrics:
        print(f"\n🏆 PERFORMANCE RANKING:")
        print(f"   By PSNR:")
        for i, (variant, metrics) in enumerate(comparative_summary['performance_ranking']['by_psnr'], 1):
            print(f"   {i}. {variant}: {metrics['psnr']['mean']:.4f}±{metrics['psnr']['std']:.4f} dB")
        
        print(f"   By SSIM:")
        for i, (variant, metrics) in enumerate(comparative_summary['performance_ranking']['by_ssim'], 1):
            print(f"   {i}. {variant}: {metrics['ssim']['mean']:.6f}±{metrics['ssim']['std']:.6f}")
    
    return comparative_summary

def main():
    parser = argparse.ArgumentParser(description='Enhanced multi-model inference evaluation for SCUNet-NGSWIN with organized results')
    parser.add_argument('--clinical_data', type=str,
                       default='/home/Drive-D/clinical_metal',
                       help='Path to clinical data directory')
    parser.add_argument('--clinical_masks', type=str,
                       default='/home/Drive-D/clinical_metal_mask',
                       help='Path to clinical masks directory')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SCUNet-NGSWIN ENHANCED MULTI-MODEL INFERENCE EVALUATION")
    print("WITH ORGANIZED RESULTS STRUCTURE")
    print("="*80)
    
    # Get all model variations
    model_variations = get_model_variations()
    
    if not model_variations:
        print("❌ No model variations found in training_results!")
        return
    
    print(f"\n🎯 FOUND {len(model_variations)} MODEL VARIATIONS:")
    for variant, info in model_variations.items():
        print(f"   📍 {info['display_name']}")
        print(f"      Config: {info['config_file']}")
        print(f"      Checkpoint: {os.path.basename(info['checkpoint_path'])}")
    
    # Setup paths - Use organized directory structure
    base_dir = '/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR'
    output_dir, inference_number = get_next_inference_folder(base_dir)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 SYSTEM SETUP:")
    print(f"   Device: {device}")
    print(f"   Output directory: {output_dir}")
    print(f"   Inference run number: {inference_number}")
    print(f"   Enhanced clinical preprocessing: HU windowing → LAC conversion → min-max normalization")
    
    # Store all results for comparative analysis
    all_results = {}
    
    # Process each model variation
    for variant, info in model_variations.items():
        print(f"\n" + "="*80)
        print(f"PROCESSING MODEL VARIATION: {info['display_name'].upper()}")
        print("="*80)
        
        try:
            # Load model
            config_path = os.path.join(base_dir, info['config_file'])
            model, opt = setup_model_and_checkpoint(info['checkpoint_path'], config_path, info['display_name'])
            
            # Setup original dataset
            test_dataset_opt = opt['datasets']['test'].copy()
            test_dataset_opt['phase'] = 'test'
            test_dataset = select_dataset.define_Dataset(test_dataset_opt)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                   num_workers=1, drop_last=False, pin_memory=True)
            
            print(f"   📊 Original dataset: {len(test_dataset)} samples")
            
            # Process original dataset
            original_summary = process_original_dataset(model, test_loader, output_dir, device, variant)
            
            # Process clinical dataset with specialized preprocessing
            clinical_summary = process_clinical_dataset_specialized(
                model, args.clinical_data, args.clinical_masks, output_dir, device, variant
            )
            
            # Store results
            all_results[variant] = {
                'variant_info': info,
                'original_dataset': original_summary,
                'clinical_dataset': clinical_summary
            }
            
            print(f"✅ Completed processing {info['display_name']}")
            
            # Clear GPU memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error processing {info['display_name']}: {e}")
            all_results[variant] = {
                'variant_info': info,
                'original_dataset': None,
                'clinical_dataset': None,
                'error': str(e)
            }
            continue
    
    # Create comparative analysis
    comparative_summary = create_comparative_analysis(all_results, output_dir)
    
    # Save complete results
    complete_results = {
        'evaluation_summary': comparative_summary,
        'individual_results': all_results,
        'inference_run_info': {
            'run_number': inference_number,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'output_directory': output_dir
        }
    }
    
    final_summary_path = os.path.join(output_dir, 'enhanced_multi_model_complete_results.json')
    with open(final_summary_path, 'w') as f:
        json.dump(complete_results, f, indent=2)
    
    print(f"\n" + "="*80)
    print("🎉 ENHANCED MULTI-MODEL EVALUATION COMPLETE")
    print("="*80)
    print(f"📁 Results saved to: {output_dir}")
    print(f"📄 Complete summary: {final_summary_path}")
    print(f"🔢 Inference run number: {inference_number}")
    
    print(f"\n📋 ORGANIZED OUTPUT STRUCTURE:")
    print(f"   inference_results/")
    print(f"   └── inference_{inference_number}/")
    for variant in model_variations.keys():
        print(f"       ├── {variant}/")
        print(f"       │   ├── original_dataset/")
        print(f"       │   │   ├── individual_results/")
        print(f"       │   │   └── metrics_summary.json")
        print(f"       │   └── clinical_dataset/")
        print(f"       │       ├── individual_results/")
        print(f"       │       ├── preprocessing_visualizations/")
        print(f"       │       └── clinical_summary.json")
    print(f"       ├── model_comparison.png")
    print(f"       ├── comparative_analysis_summary.json")
    print(f"       └── enhanced_multi_model_complete_results.json")
    
    print(f"\n✅ Enhanced multi-model inference evaluation completed successfully!")
    print(f"📊 Tested {len(model_variations)} model variations with specialized clinical preprocessing")
    print(f"🔬 Clinical preprocessing: HU windowing → linear attenuation coefficient → min-max normalization")
    print(f"📂 Results organized in: inference_results/inference_{inference_number}/")

if __name__ == '__main__':
    main()
