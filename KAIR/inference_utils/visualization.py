#!/usr/bin/env python3
"""
Visualization and Results Utilities
===================================
Contains functions for creating visualizations and saving results
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

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
            axes[1, 0].imshow(diff_E_H, cmap='gray', vmin=0, vmax=50)
            axes[1, 0].set_title(f'|Enhanced - Target|\nMax Diff: {diff_E_H.max():.1f}', fontsize=10)
            axes[1, 0].axis('off')
            
            # Difference between input and target
            diff_L_H = np.abs(L_img.astype(float) - H_img.astype(float))
            axes[1, 1].imshow(diff_L_H, cmap='gray', vmin=0, vmax=50)
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
                          important_results_dir, best_samples_dir, worst_samples_dir, metrics_visualization_dir, variant_name,
                          all_psnr_values=None, all_ssim_values=None):
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
        
        # Create metrics distribution plots using ALL samples if provided, otherwise use sample subset
        if all_psnr_values is not None and all_ssim_values is not None:
            print(f"   📊 Creating metrics visualizations using ALL {len(all_psnr_values)} samples")
            create_metrics_visualizations(all_psnr_values, all_ssim_values, metrics_visualization_dir, variant_name)
        else:
            print(f"   📊 Creating metrics visualizations using subset of samples")
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
    """Create visualization plots for metrics analysis (no red/orange colors)"""
    try:
        # Use blue/gray colorscheme instead of red/orange
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(psnr_values, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        plt.title(f'{variant_name} - PSNR Distribution')
        plt.xlabel('PSNR (dB)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.hist(ssim_values, bins=30, alpha=0.7, color='darkslategray', edgecolor='black')
        plt.title(f'{variant_name} - SSIM Distribution')
        plt.xlabel('SSIM')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.scatter(psnr_values, ssim_values, alpha=0.6, s=30, color='navy')
        plt.title(f'{variant_name} - PSNR vs SSIM Correlation')
        plt.xlabel('PSNR (dB)')
        plt.ylabel('SSIM')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.boxplot([psnr_values], labels=['PSNR'], patch_artist=True,
                   boxprops=dict(facecolor='lightsteelblue'))
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

def create_comprehensive_analysis(all_results, output_dir):
    """Create comprehensive analysis across all models and datasets"""
    print(f"\n📊 CREATING COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # Organize results by dataset type
    original_results = {}
    clinical_masked_results = {}
    clinical_nomask_results = {}
    clinical_artifact_only_results = {}
    
    for variant, results in all_results.items():
        if 'original_dataset' in results:
            original_results[variant] = results['original_dataset']
        if 'clinical_dataset_with_masks' in results:
            clinical_masked_results[variant] = results['clinical_dataset_with_masks']
        if 'clinical_dataset_no_masks' in results:
            clinical_nomask_results[variant] = results['clinical_dataset_no_masks']
        if 'clinical_artifact_only_dataset' in results:
            clinical_artifact_only_results[variant] = results['clinical_artifact_only_dataset']
    
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
            },
            'clinical_artifact_only': {
                'variants_tested': len(clinical_artifact_only_results),
                'results': clinical_artifact_only_results
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
            clinical_comparison[variant] = result['total_samples']
        
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
            f.write("ORIGINAL DATASET RESULTS (test_640geo):\n")
            f.write("-" * 50 + "\n")
            for variant, result in original_results.items():
                f.write(f"{variant:<15} | PSNR: {result['metrics']['psnr']['mean']:.4f} dB | SSIM: {result['metrics']['ssim']['mean']:.6f}\n")
            f.write("\n")
        
        # Clinical dataset results
        if clinical_masked_results:
            f.write("CLINICAL DATASET RESULTS (with masks):\n")
            f.write("-" * 50 + "\n")
            for variant, result in clinical_masked_results.items():
                total_files = result.get('total_files', 'N/A')
                f.write(f"{variant:<15} | Samples: {result['total_samples']:<6} | Files: {total_files:<3}\n")
            f.write("\n")
        
        # Clinical dataset results (no masks)
        if clinical_nomask_results:
            f.write("CLINICAL DATASET RESULTS (no masks):\n")
            f.write("-" * 50 + "\n")
            for variant, result in clinical_nomask_results.items():
                total_files = result.get('total_files', 'N/A')
                f.write(f"{variant:<15} | Samples: {result['total_samples']:<6} | Files: {total_files:<3}\n")
            f.write("\n")
        
        # Clinical artifact-only dataset results
        if clinical_artifact_only_results:
            f.write("CLINICAL ARTIFACT-ONLY DATASET RESULTS:\n")
            f.write("-" * 50 + "\n")
            for variant, result in clinical_artifact_only_results.items():
                total_files = result.get('total_files', result.get('total_dataset_size', 'N/A'))
                f.write(f"{variant:<15} | Samples: {result['total_samples']:<6} | Files: {total_files:<3}\n")
            f.write("\n")
        
        f.write(f"Evaluation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Results saved to: {output_dir}\n")
    
    print(f"📊 Comprehensive analysis saved to: {analysis_file}")
    print(f"📋 Summary table saved to: {summary_table}")
    
    return comprehensive_summary
