#!/usr/bin/env python3
"""
Dataset Processing Utilities
============================
Contains functions for processing different types of datasets
"""

import os
import sys
import json
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# Add KAIR to path for imports
sys.path.append('/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR')

from utils import utils_image as util
from data import select_dataset
from .preprocessing import ClinicalConfig, clinical_preprocessing, detect_metal_mask, load_clinical_mask_fixed, apply_mask_to_image
from .visualization import create_comparison_grid, create_detailed_comparison_with_metrics, save_important_results

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
            
            # Save comparison every 25 samples
            if i % 25 == 0:
                images = [L_img, E_img, H_img]
                titles = [f'Input (L)', f'Enhanced (E)', f'Target (H)']
                save_path = comparisons_dir / f"comparison_sample_{i+1:06d}.png"
                create_comparison_grid(images, titles, i+1, save_path, variant_name)
            
            # Progress update every 50 samples
            if i % 50 == 0:
                current_avg_psnr = np.mean(psnr_values)
                current_avg_ssim = np.mean(ssim_values)
                print(f"  Progress [{i+1}/{len(test_dataset)}]: Avg PSNR: {current_avg_psnr:.4f}, Avg SSIM: {current_avg_ssim:.6f}")
                
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
    
    # Save important results (best/worst samples, visualizations) - using ALL samples for metrics
    save_important_results(best_psnr_samples, worst_psnr_samples, best_ssim_samples, worst_ssim_samples, 
                          important_results_dir, best_samples_dir, worst_samples_dir, metrics_visualization_dir, variant_name,
                          all_psnr_values=psnr_values, all_ssim_values=ssim_values)
    
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
    
    # Save detailed results summary using ALL samples for metrics visualizations
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
                'max': float(np.max(psnr_values)),
                'all_values': psnr_values  # ALL sample values for complete metrics
            },
            'ssim': {
                'mean': float(avg_ssim),
                'std': float(std_ssim),
                'min': float(np.min(ssim_values)),
                'max': float(np.max(ssim_values)),
                'all_values': ssim_values  # ALL sample values for complete metrics
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
            
            # Process each slice
            for slice_idx in slice_indices:
                slice_data = clinical_data[:, :, slice_idx]
                
                # Skip empty slices
                if np.all(slice_data == 0) or slice_data.std() < 1e-6:
                    continue
                
                # Apply clinical preprocessing
                processed_slice = clinical_preprocessing(slice_data, config)
                if processed_slice is None:
                    continue
                
                # Detect metal artifacts
                detected_mask = detect_metal_mask(slice_data, config)
                
                # Use external mask if available, otherwise use detected mask
                final_mask = None
                if external_mask is not None and slice_idx < external_mask.shape[2]:
                    final_mask = external_mask[:, :, slice_idx]
                    if final_mask.shape != (config.CTpara['imPixNum'], config.CTpara['imPixNum']):
                        import cv2
                        final_mask = cv2.resize(final_mask.astype(np.float32), 
                                              (config.CTpara['imPixNum'], config.CTpara['imPixNum']), 
                                              interpolation=cv2.INTER_NEAREST)
                elif detected_mask is not None:
                    final_mask = detected_mask
                
                # Convert to tensor for model
                processed_tensor = torch.from_numpy(processed_slice).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
                
                # Model inference
                with torch.no_grad():
                    model.feed_data({'L': processed_tensor, 'H': processed_tensor})
                    model.test()
                    enhanced_tensor = model.current_visuals()['E']
                
                # Convert back to image
                enhanced_img = util.tensor2uint(enhanced_tensor.squeeze().cpu())
                
                # Apply mask if available (for visualization)
                masked_enhanced = apply_mask_to_image(enhanced_img, final_mask) if final_mask is not None else enhanced_img
                
                processed_count += 1
                
                # Store result
                result_info = {
                    'file_name': clinical_file.name,
                    'slice_idx': slice_idx,
                    'processed_count': processed_count,
                    'has_external_mask': external_mask is not None,
                    'has_detected_mask': detected_mask is not None,
                    'enhancement_quality': {
                        'input_std': float(processed_slice.std()),
                        'enhanced_std': float(enhanced_img.std()),
                        'enhancement_ratio': float(enhanced_img.std() / processed_slice.std()) if processed_slice.std() > 0 else 1.0
                    }
                }
                processed_results.append(result_info)
                
                # Track for enhancement quality analysis
                enhancement_quality_samples.append({
                    'file_name': clinical_file.name,
                    'slice_idx': slice_idx,
                    'processed_count': processed_count,
                    'quality_metrics': result_info['enhancement_quality'],
                    'images': {
                        'original': slice_data,
                        'processed': processed_slice,
                        'enhanced': enhanced_img
                    },
                    'masks': {
                        'detected': detected_mask,
                        'external': external_mask[:, :, slice_idx] if external_mask is not None and slice_idx < external_mask.shape[2] else None,
                        'final': final_mask
                    }
                })
                
                # Save comparison every 15 slices
                if processed_count % 15 == 0:
                    comparison_images = [processed_slice, enhanced_img, masked_enhanced]
                    comparison_titles = ['Preprocessed', 'Enhanced', 'Masked Enhanced']
                    save_path = comparisons_dir / f"comparison_{processed_count:06d}_{clinical_file.stem}_slice_{slice_idx:03d}.png"
                    create_comparison_grid(comparison_images, comparison_titles, processed_count, save_path, variant_name)
                
                # Save preprocessing visualization every 30 slices
                if processed_count % 30 == 0:
                    prep_images = [slice_data, processed_slice, enhanced_img]
                    prep_titles = ['Original HU', 'Preprocessed LAC', 'Enhanced']
                    save_path = preprocessing_dir / f"preprocessing_{processed_count:06d}_{clinical_file.stem}_slice_{slice_idx:03d}.png"
                    create_comparison_grid(prep_images, prep_titles, processed_count, save_path, variant_name)
                
                # Log progress
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'file_name': clinical_file.name,
                    'slice_idx': slice_idx,
                    'processed_count': processed_count,
                    'processing_status': 'success',
                    'masks_available': {
                        'external': external_mask is not None,
                        'detected': detected_mask is not None
                    }
                }
                detailed_log.append(log_entry)
            
        except Exception as e:
            print(f"  Error processing {clinical_file.name}: {e}")
            error_log_entry = {
                'timestamp': datetime.now().isoformat(),
                'file_name': clinical_file.name,
                'processing_status': 'error',
                'error_message': str(e)
            }
            detailed_log.append(error_log_entry)
            continue
    
    # Sort enhancement samples by quality
    enhancement_quality_samples.sort(key=lambda x: x['quality_metrics']['enhancement_ratio'], reverse=True)
    
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
            
            # Process each slice
            for slice_idx in range(clinical_data.shape[2]):
                slice_data = clinical_data[:, :, slice_idx]
                
                # Skip empty slices
                if np.all(slice_data == 0) or slice_data.std() < 1e-6:
                    continue
                
                # Apply clinical preprocessing
                processed_slice = clinical_preprocessing(slice_data, config)
                if processed_slice is None:
                    continue
                
                # Convert to tensor for model
                processed_tensor = torch.from_numpy(processed_slice).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
                
                # Model inference
                with torch.no_grad():
                    model.feed_data({'L': processed_tensor, 'H': processed_tensor})
                    model.test()
                    enhanced_tensor = model.current_visuals()['E']
                
                # Convert back to image
                enhanced_img = util.tensor2uint(enhanced_tensor.squeeze().cpu())
                
                processed_count += 1
                
                # Store result
                result_info = {
                    'file_name': clinical_file.name,
                    'slice_idx': slice_idx,
                    'processed_count': processed_count,
                    'enhancement_quality': {
                        'input_std': float(processed_slice.std()),
                        'enhanced_std': float(enhanced_img.std()),
                        'enhancement_ratio': float(enhanced_img.std() / processed_slice.std()) if processed_slice.std() > 0 else 1.0
                    }
                }
                processed_results.append(result_info)
                
                # Track for enhancement quality analysis
                enhancement_quality_samples.append({
                    'file_name': clinical_file.name,
                    'slice_idx': slice_idx,
                    'processed_count': processed_count,
                    'quality_metrics': result_info['enhancement_quality'],
                    'images': {
                        'original': slice_data,
                        'processed': processed_slice,
                        'enhanced': enhanced_img
                    }
                })
                
                # Save comparison every 15 slices
                if processed_count % 15 == 0:
                    comparison_images = [processed_slice, enhanced_img]
                    comparison_titles = ['Preprocessed', 'Enhanced']
                    save_path = comparisons_dir / f"comparison_{processed_count:06d}_{clinical_file.stem}_slice_{slice_idx:03d}.png"
                    create_comparison_grid(comparison_images, comparison_titles, processed_count, save_path, variant_name)
                
                # Save preprocessing visualization every 30 slices
                if processed_count % 30 == 0:
                    prep_images = [slice_data, processed_slice, enhanced_img]
                    prep_titles = ['Original HU', 'Preprocessed LAC', 'Enhanced']
                    save_path = preprocessing_dir / f"preprocessing_{processed_count:06d}_{clinical_file.stem}_slice_{slice_idx:03d}.png"
                    create_comparison_grid(prep_images, prep_titles, processed_count, save_path, variant_name)
                
                # Log progress
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'file_name': clinical_file.name,
                    'slice_idx': slice_idx,
                    'processed_count': processed_count,
                    'processing_status': 'success'
                }
                detailed_log.append(log_entry)
            
        except Exception as e:
            print(f"  Error processing {clinical_file.name}: {e}")
            error_log_entry = {
                'timestamp': datetime.now().isoformat(),
                'file_name': clinical_file.name,
                'processing_status': 'error',
                'error_message': str(e)
            }
            detailed_log.append(error_log_entry)
            continue
    
    # Sort enhancement samples by quality
    enhancement_quality_samples.sort(key=lambda x: x['quality_metrics']['enhancement_ratio'], reverse=True)
    
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
