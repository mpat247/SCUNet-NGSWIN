#!/usr/bin/env python3
"""# Import organized utility modules
from inference_utils.model_utils import discover_all_trained_models, setup_model_and_checkpoint, get_next_inference_folder, get_smart_remaining_models, check_folder_completely_finished
from inference_utils.dataset_processing import process_original_dataset, process_clinical_dataset_no_masks
from inference_utils.visualization import create_comprehensive_analysisprehensive Multi-Model Pipeline Evaluation
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
import torch
import traceback
from pathlib import Path

# Add KAIR to path for imports
sys.path.append('/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR')

# Import organized utility modules
from inference_utils.model_utils import discover_all_trained_models, setup_model_and_checkpoint, get_next_inference_folder, get_smart_remaining_models, check_folder_completely_finished
from inference_utils.dataset_processing import process_original_dataset, process_clinical_dataset_no_masks
from inference_utils.visualization import create_comprehensive_analysis

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
    parser.add_argument('--base_output', type=str,
                       default='/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR/inference_results',
                       help='Base output directory for inference results')
    
    args = parser.parse_args()
    
    print("🚀 COMPREHENSIVE MULTI-MODEL PIPELINE EVALUATION")
    print("="*80)
    print(f"📂 Base output directory: {args.base_output}")
    print(f"📊 Original dataset: {args.original_data}")
    print(f"🏥 Clinical dataset: {args.clinical_data}")
    
    # Check paths
    if not os.path.exists(args.original_data):
        print(f"❌ Original data path not found: {args.original_data}")
        return
    
    if not os.path.exists(args.clinical_data):
        print(f"❌ Clinical data path not found: {args.clinical_data}")
        return
    
    # Get inference folder using smart logic
    import glob
    existing_folders = sorted(glob.glob(os.path.join(args.base_output, "inference_*")))
    
    if existing_folders:
        # Check latest folder for incomplete work
        latest_folder = existing_folders[-1]
        
        if check_folder_completely_finished(latest_folder):
            # Latest folder is completely finished, create new one
            output_dir = get_next_inference_folder(args.base_output)
            models_info = discover_all_trained_models()
            for model in models_info:
                model['resume_status'] = None
            print(f"📂 Previous work ({latest_folder}) completely finished, starting new folder: {output_dir}")
        else:
            # Found incomplete work, continue with this folder
            models_info = get_smart_remaining_models(latest_folder)
            output_dir = latest_folder
            print(f"🔄 Resuming incomplete work in: {output_dir}")
            print(f"📋 Found {len(models_info)} models that need processing")
    else:
        # No existing folders, create first one
        output_dir = get_next_inference_folder(args.base_output)
        models_info = discover_all_trained_models()
        for model in models_info:
            model['resume_status'] = None
        print(f"📂 Starting first inference folder: {output_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    try:
        if not models_info:
                print("❌ No trained models found! Check training_results directory.")
                return
        
        # Store all results
        all_results = {}
        
        # Process each model
        for model_idx, model_info in enumerate(models_info):
            variant = model_info['variant']
            resume_status = model_info.get('resume_status')
            
            print(f"\n{'='*80}")
            print(f"🎯 TESTING MODEL [{model_idx+1}/{len(models_info)}]: {variant.upper()}")
            if resume_status:
                print(f"🔄 RESUME MODE - Original: {'✅' if resume_status['original_complete'] else '❌'} | Clinical: {'✅' if resume_status['clinical_complete'] else '❌'}")
            print(f"{'='*80}")
            
            try:
                # Load model
                model = setup_model_and_checkpoint(model_info)
                model.netG.eval()  # Set to evaluation mode
                
                # Initialize results for this variant
                if variant not in all_results:
                    all_results[variant] = {}
                
                # 1. Process original dataset (test_640geo) - skip if already completed
                if not resume_status or not resume_status['original_complete']:
                    print(f"\n🔄 [{model_idx+1}/{len(models_info)}] Processing original dataset for {variant}...")
                    original_results = process_original_dataset(
                        model, args.original_data, output_dir, device, variant
                    )
                    all_results[variant]['original_dataset'] = original_results
                else:
                    print(f"\n✅ [{model_idx+1}/{len(models_info)}] Original dataset already completed for {variant}, skipping...")
                    # Load existing results
                    try:
                        results_file = Path(output_dir) / variant / "original_dataset" / "metrics_summary.json"
                        with open(results_file, 'r') as f:
                            all_results[variant]['original_dataset'] = json.load(f)
                    except Exception as e:
                        print(f"⚠️  Could not load existing original dataset results: {e}")

                # 2. Process clinical dataset without masks - skip if already completed
                if not resume_status or not resume_status['clinical_complete']:
                    print(f"\n🔄 [{model_idx+1}/{len(models_info)}] Processing clinical dataset (no masks) for {variant}...")
                    clinical_nomask_results = process_clinical_dataset_no_masks(
                        model, args.clinical_data, output_dir, device, variant
                    )
                    all_results[variant]['clinical_dataset_no_masks'] = clinical_nomask_results
                else:
                    print(f"\n✅ [{model_idx+1}/{len(models_info)}] Clinical dataset already completed for {variant}, skipping...")
                    # Load existing results
                    try:
                        results_file = Path(output_dir) / variant / "clinical_dataset_no_masks" / "clinical_no_masks_summary.json"
                        with open(results_file, 'r') as f:
                            all_results[variant]['clinical_dataset_no_masks'] = json.load(f)
                    except Exception as e:
                        print(f"⚠️  Could not load existing clinical dataset results: {e}")
                
                # Print completion status
                print(f"\n✅ COMPLETED {variant.upper()} - All datasets processed successfully!")
                if 'original_dataset' in all_results[variant]:
                    orig_results = all_results[variant]['original_dataset']
                    print(f"   📊 Original dataset: {orig_results['metrics']['psnr']['mean']:.4f} dB PSNR, {orig_results['metrics']['ssim']['mean']:.6f} SSIM")
                if 'clinical_dataset_no_masks' in all_results[variant]:
                    clin_results = all_results[variant]['clinical_dataset_no_masks']
                    print(f"   🏥 Clinical no masks: {clin_results['total_samples']} slices processed")
                
                # Clear GPU memory after each model
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Error processing {variant}: {e}")
                print(f"   Traceback: {traceback.format_exc()}")
                continue
        
        # Create comprehensive analysis
        if all_results:
            print(f"\n{'='*80}")
            print(f"📊 CREATING COMPREHENSIVE ANALYSIS")
            print(f"{'='*80}")
            
            comprehensive_analysis = create_comprehensive_analysis(all_results, output_dir)
            
            print(f"\n🎉 EVALUATION COMPLETED SUCCESSFULLY!")
            print(f"{'='*80}")
            print(f"📈 Models tested: {len(all_results)}")
            print(f"📂 Results saved to: {output_dir}")
            
            # Print final summary
            if 'original_dataset' in comprehensive_analysis['comparison_analysis']:
                original_analysis = comprehensive_analysis['comparison_analysis']['original_dataset']
                print(f"\n🏆 ORIGINAL DATASET RANKINGS:")
                for i, (variant, psnr) in enumerate(original_analysis['psnr_ranking'].items(), 1):
                    ssim = original_analysis['ssim_ranking'][variant]
                    print(f"   {i}. {variant:<15} | PSNR: {psnr:.4f} dB | SSIM: {ssim:.6f}")
            
            print(f"\n📁 Check the following folders for detailed results:")
            for variant in all_results.keys():
                print(f"   📊 {variant}: {output_dir}/{variant}/")
            print(f"   📈 Comprehensive analysis: {output_dir}/comprehensive_multi_model_analysis.json")
            
        else:
            print("❌ No results generated! All models failed to process.")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()