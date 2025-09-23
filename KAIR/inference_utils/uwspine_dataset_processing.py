#!/usr/bin/env python3
"""
UWSpine Dataset Processing Utilities
===================================
Contains functions for processing UWSpine synthetic transfer datasets
"""

import os
import sys
import json
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# Set environment variables to avoid display issues
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add KAIR to path for imports
sys.path.append('/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR')

from utils import utils_image as util
from data.select_dataset import define_Dataset

def create_simple_comparison(input_img, output_img, target_img, title, save_path):
    """Create a simple comparison without complex plotting to avoid X11 issues"""
    try:
        # Disable interactive mode completely
        plt.ioff()
        
        # Create simple side-by-side comparison using matplotlib
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=14)
        
        axes[0].imshow(input_img, cmap='gray')
        axes[0].set_title('Input (with artifacts)')
        axes[0].axis('off')
        
        axes[1].imshow(output_img, cmap='gray')
        axes[1].set_title('Enhanced Output')
        axes[1].axis('off')
        
        axes[2].imshow(target_img, cmap='gray')
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Explicitly close figure
        plt.clf()       # Clear figure
        return True
    except Exception as e:
        print(f"Warning: Could not save comparison image: {e}")
        try:
            plt.close('all')  # Close any remaining figures
        except:
            pass
        return False

def save_individual_images(input_img, output_img, target_img, img_name, output_dirs):
    """Save individual images separately to avoid plotting issues"""
    try:
        comparisons_dir, detailed_comparisons_dir = output_dirs
        
        # Save individual images
        util.imsave(input_img, str(comparisons_dir / f"{img_name}_input.png"))
        util.imsave(output_img, str(comparisons_dir / f"{img_name}_output.png"))
        if target_img is not None:
            util.imsave(target_img, str(comparisons_dir / f"{img_name}_target.png"))
        
        return True
    except Exception as e:
        print(f"Warning: Could not save individual images: {e}")
        return False

def save_simple_results(processed_results, output_dir, variant_name, avg_psnr, avg_ssim):
    """Save important results in a simple format"""
    try:
        # Sort by PSNR for best/worst analysis
        valid_results = [r for r in processed_results if r.get('psnr') is not None]
        if not valid_results:
            return
            
        sorted_by_psnr = sorted(valid_results, key=lambda x: x['psnr'], reverse=True)
        
        # Save best and worst samples info
        best_samples = sorted_by_psnr[:5]  # Top 5
        worst_samples = sorted_by_psnr[-3:]  # Bottom 3
        
        summary = {
            'variant': variant_name,
            'total_samples': len(processed_results),
            'average_psnr': avg_psnr,
            'average_ssim': avg_ssim,
            'best_samples': [{'filename': s['filename'], 'psnr': s['psnr'], 'ssim': s['ssim']} for s in best_samples],
            'worst_samples': [{'filename': s['filename'], 'psnr': s['psnr'], 'ssim': s['ssim']} for s in worst_samples]
        }
        
        summary_path = output_dir / f"important_results_summary_{variant_name}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"📋 Important results saved to: {summary_path}")
        return True
    except Exception as e:
        print(f"Warning: Could not save important results: {e}")
        return False

def process_uwspine_synthetic_transfer_dataset(model, uwspine_data_dir, output_dir, device, variant_name, metrics_only=False):
    """Process UWSpine synthetic transfer dataset
    
    Args:
        model: The trained model
        uwspine_data_dir: Path to UWSpine dataset
        output_dir: Output directory for results
        device: Computing device (cuda/cpu)
        variant_name: Model variant name
        metrics_only: If True, only compute metrics without saving images
    """
    print(f"\n🦴 PROCESSING UWSPINE SYNTHETIC TRANSFER DATASET - {variant_name.upper()}")
    print("="*60)
    print(f"📁 Dataset path: {uwspine_data_dir}")
    if metrics_only:
        print("⚡ Metrics-only mode: Skipping image generation to avoid display issues")
    
    # Create comprehensive directory structure
    variant_dir = Path(output_dir) / variant_name
    results_dir = variant_dir / "uwspine_synthetic_transfer"
    
    if not metrics_only:
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
    else:
        results_dir.mkdir(parents=True, exist_ok=True)
        comparisons_dir = None
        detailed_comparisons_dir = None
        important_results_dir = results_dir
    
    # Load dataset using synthetic_transfer dataset type
    opt_dataset = {
        'name': 'uwspine_synthetic_transfer_test',
        'dataset_type': 'synthetic_transfer',
        'dataroot_H': uwspine_data_dir,
        'phase': 'test',
        'scale': 1,
        'n_channels': 1
    }
    
    print(f"🔧 Loading UWSpine synthetic transfer dataset...")
    print(f"   Dataset type: {opt_dataset['dataset_type']}")
    print(f"   Phase: {opt_dataset['phase']}")
    print(f"   Data root: {opt_dataset['dataroot_H']}")
    
    try:
        dataset = define_Dataset(opt_dataset)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        print(f"✅ Dataset loaded successfully: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        print(f"🔍 Checking if path exists: {os.path.exists(uwspine_data_dir)}")
        return None
    
    # Process each sample
    processed_results = []
    psnr_list = []
    ssim_list = []
    
    model.netG.eval()
    
    print(f"\n🔄 Processing {len(dataset)} samples...")
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), 
                       desc=f"Processing {variant_name}")
    
    for idx, data in progress_bar:
        try:
            # Get input and target
            input_img = data['L'].to(device)  # Low quality/artifact image
            target_img = data['H'].to(device) if 'H' in data else None  # High quality/clean image
            
            # Get filename for saving
            img_name = data.get('L_path', [f'sample_{idx:04d}'])[0]
            if isinstance(img_name, str):
                img_name = os.path.basename(img_name).split('.')[0]
            else:
                img_name = f'sample_{idx:04d}'
            
            # Run inference
            with torch.no_grad():
                output_img = model.netG(input_img)
            
            # Convert to numpy for metrics and saving
            input_np = util.tensor2uint(input_img)
            output_np = util.tensor2uint(output_img)
            
            if target_img is not None:
                target_np = util.tensor2uint(target_img)
                
                # Calculate metrics
                psnr = util.calculate_psnr(output_np, target_np, border=0)
                ssim = util.calculate_ssim(output_np, target_np, border=0)
                
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                
                # Try to save comparison images, fall back to individual images if plotting fails
                comparison_saved = False
                if not metrics_only and idx % 50 == 0:  # Save every 50th image to avoid too many files
                    comparison_path = comparisons_dir / f"{img_name}_comparison.png"
                    comparison_saved = create_simple_comparison(
                        input_np, output_np, target_np,
                        f"UWSpine Sample {img_name} - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}",
                        str(comparison_path)
                    )
                    
                    if not comparison_saved:
                        # Fallback: save individual images
                        print(f"   Falling back to individual image saving for {img_name}")
                        save_individual_images(
                            input_np, output_np, target_np, img_name, 
                            (comparisons_dir, detailed_comparisons_dir)
                        )
                
                processed_results.append({
                    'filename': img_name,
                    'psnr': psnr,
                    'ssim': ssim,
                    'comparison_saved': comparison_saved,
                    'index': idx
                })
                
                progress_bar.set_postfix({
                    'PSNR': f'{psnr:.2f}',
                    'SSIM': f'{ssim:.4f}'
                })
            else:
                # No ground truth available, just save enhancement
                output_path = comparisons_dir / f"{img_name}_enhanced.png"
                util.imsave(output_np, str(output_path))
                
                processed_results.append({
                    'filename': img_name,
                    'psnr': None,
                    'ssim': None,
                    'output_path': str(output_path)
                })
            
        except Exception as e:
            print(f"❌ Error processing sample {idx}: {str(e)}")
            continue
    
    # Calculate summary statistics
    if psnr_list:
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        
        print(f"\n📊 UWSpine Synthetic Transfer Results:")
        print(f"   Average PSNR: {avg_psnr:.4f} dB")
        print(f"   Average SSIM: {avg_ssim:.6f}")
        print(f"   Processed samples: {len(processed_results)}")
        
        # Save important results
        save_simple_results(
            processed_results, important_results_dir, variant_name, avg_psnr, avg_ssim
        )
        
        # Create summary
        summary = {
            'dataset_type': 'uwspine_synthetic_transfer',
            'variant': variant_name,
            'average_psnr': avg_psnr,
            'average_ssim': avg_ssim,
            'total_samples': len(processed_results),
            'successful_samples': len([r for r in processed_results if r['psnr'] is not None]),
            'timestamp': datetime.now().isoformat(),
            'detailed_results': processed_results
        }
    else:
        print(f"\n📊 UWSpine Enhancement Results (no ground truth):")
        print(f"   Enhanced samples: {len(processed_results)}")
        
        summary = {
            'dataset_type': 'uwspine_synthetic_transfer',
            'variant': variant_name,
            'total_samples': len(processed_results),
            'enhancement_only': True,
            'timestamp': datetime.now().isoformat(),
            'detailed_results': processed_results
        }
    
    # Save summary
    with open(results_dir / "uwspine_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ UWSpine dataset processing completed!")
    print(f"📁 Results saved to: {results_dir}")
    
    return summary
