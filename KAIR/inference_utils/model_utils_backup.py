#!/usr/bin/env python3
"""
Model Management Utilities
==========================
Contains functions for discovering and setting up trained models
"""

import os
import sys
import json
import time
import glob
import re
from pathlib import Path
from datetime import datetime

# Add KAIR to path for imports
sys.path.append('/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR')

from utils import utils_option as option
from models import select_model

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

def discover_completed_models(output_dir):
    """Discover which models have already been completed in the output directory"""
    completed_models = set()
    
    if not os.path.exists(output_dir):
        return completed_models
    
    # Check each variant folder for completion
    variant_folders = ['conv_nstb', 'conv_trans_nstb', 'trans_nstb']
    
    for variant in variant_folders:
        variant_path = Path(output_dir) / variant
        
        # Check if original dataset is completed
        original_summary = variant_path / "original_dataset" / "metrics_summary.json"
        clinical_summary = variant_path / "clinical_dataset_no_masks" / "clinical_no_masks_summary.json"
        
        if original_summary.exists() and clinical_summary.exists():
            completed_models.add(variant)
            print(f"✓ Found completed model: {variant}")
    
    return completed_models

def get_remaining_models(output_dir):
    """Get list of models that still need to be processed"""
    all_models = discover_all_trained_models()
    completed_models = discover_completed_models(output_dir)
    
    # Filter out completed models
    remaining_models = [model for model in all_models if model['variant'] not in completed_models]
    
    if completed_models:
        print(f"🔄 Resume mode: {len(completed_models)} models already completed")
        print(f"📋 Completed models: {', '.join(completed_models)}")
    
    if remaining_models:
        remaining_variants = [m['variant'] for m in remaining_models]
        print(f"⏳ Remaining models to process: {', '.join(remaining_variants)}")
    else:
        print("✅ All models have been completed!")
    
    return remaining_models

def save_progress_checkpoint(output_dir, model_variant, dataset_type, current_file_idx, total_files, processed_count):
    """Save progress checkpoint to allow resuming if interrupted"""
    checkpoint_file = Path(output_dir) / f"progress_checkpoint_{model_variant}.json"
    
    checkpoint_data = {
        'model_variant': model_variant,
        'dataset_type': dataset_type,
        'current_file_idx': current_file_idx,
        'total_files': total_files,
        'processed_count': processed_count,
        'timestamp': time.time(),
        'timestamp_readable': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    }
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

def load_progress_checkpoint(output_dir, model_variant):
    """Load progress checkpoint to resume processing"""
    checkpoint_file = Path(output_dir) / f"progress_checkpoint_{model_variant}.json"
    
    if not checkpoint_file.exists():
        return None
    
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        return checkpoint_data
    except Exception as e:
        print(f"⚠️  Warning: Could not load checkpoint for {model_variant}: {e}")
        return None

def cleanup_progress_checkpoint(output_dir, model_variant):
    """Clean up progress checkpoint file after successful completion"""
    checkpoint_file = Path(output_dir) / f"progress_checkpoint_{model_variant}.json"
    
    if checkpoint_file.exists():
        try:
            os.remove(checkpoint_file)
            print(f"🧹 Cleaned up progress checkpoint for {model_variant}")
        except Exception as e:
            print(f"⚠️  Warning: Could not remove checkpoint file: {e}")

def check_model_completion_status(output_dir, variant):
    """Check detailed completion status of a specific model variant"""
    variant_dir = Path(output_dir) / variant
    
    # Check if both datasets are completed
    original_complete = (variant_dir / "original_dataset" / "metrics_summary.json").exists()
    clinical_complete = (variant_dir / "clinical_dataset_no_masks" / "clinical_no_masks_summary.json").exists()
    
    # Check for checkpoint
    checkpoint = load_progress_checkpoint(output_dir, variant)
    
    return {
        'variant': variant,
        'original_dataset_complete': original_complete,
        'clinical_dataset_complete': clinical_complete,
        'fully_complete': original_complete and clinical_complete,
        'has_checkpoint': checkpoint is not None,
        'checkpoint_info': checkpoint
    }

def get_smart_processing_plan(output_dir):
    """Create a smart processing plan based on current completion status"""
    all_models = discover_all_trained_models()
    
    processing_plan = {
        'completed_models': [],
        'partially_completed_models': [],
        'pending_models': [],
        'resume_info': {}
    }
    
    for model in all_models:
        variant = model['variant']
        status = check_model_completion_status(output_dir, variant)
        
        if status['fully_complete']:
            processing_plan['completed_models'].append(variant)
        elif status['original_dataset_complete'] or status['clinical_dataset_complete'] or status['has_checkpoint']:
            processing_plan['partially_completed_models'].append(variant)
            processing_plan['resume_info'][variant] = status
        else:
            processing_plan['pending_models'].append(variant)
    
    return processing_plan

def get_smart_remaining_models(output_dir):
    """
    Smart function to determine which models still need processing.
    Checks completion of both original dataset and clinical dataset for each model.
    Returns remaining work with detailed resume information.
    """
    all_models = discover_all_trained_models()
    remaining_models = []
    
    print(f"🔍 Checking completion status in: {output_dir}")
    
    for model_info in all_models:
        variant = model_info['variant']
        variant_dir = Path(output_dir) / variant
        
        # Check original dataset completion
        original_complete = False
        original_summary_path = variant_dir / "original_dataset" / "metrics_summary.json"
        if original_summary_path.exists():
            try:
                with open(original_summary_path, 'r') as f:
                    summary = json.load(f)
                if summary.get('total_samples', 0) >= 1500:  # Should have ~2000 samples
                    original_complete = True
                    print(f"  ✅ {variant}: Original dataset complete ({summary.get('total_samples', 0)} samples)")
                else:
                    print(f"  🔄 {variant}: Original dataset incomplete ({summary.get('total_samples', 0)} samples)")
            except Exception as e:
                print(f"  ❌ {variant}: Original dataset summary corrupted: {e}")
        else:
            print(f"  ⏳ {variant}: Original dataset not started")
        
        # Check clinical dataset completion
        clinical_complete = False
        clinical_summary_path = variant_dir / "clinical_dataset_no_masks" / "clinical_no_masks_summary.json"
        if clinical_summary_path.exists():
            try:
                with open(clinical_summary_path, 'r') as f:
                    summary = json.load(f)
                if summary.get('total_samples', 0) >= 40000:  # Should have ~50k+ slices
                    clinical_complete = True
                    print(f"  ✅ {variant}: Clinical dataset complete ({summary.get('total_samples', 0)} slices)")
                else:
                    print(f"  🔄 {variant}: Clinical dataset incomplete ({summary.get('total_samples', 0)} slices)")
            except Exception as e:
                print(f"  ❌ {variant}: Clinical dataset summary corrupted: {e}")
        else:
            print(f"  ⏳ {variant}: Clinical dataset not started")
        
        # Determine what needs to be done for this model
        if not original_complete or not clinical_complete:
            resume_info = {
                'original_complete': original_complete,
                'clinical_complete': clinical_complete,
                'checkpoint_available': False
            }
            
            # Check for clinical checkpoint if clinical is incomplete
            if not clinical_complete:
                checkpoint_file = Path(output_dir) / f"progress_checkpoint_{variant}.json"
                if checkpoint_file.exists():
                    resume_info['checkpoint_available'] = True
                    try:
                        with open(checkpoint_file, 'r') as f:
                            checkpoint = json.load(f)
                        resume_info['checkpoint_info'] = checkpoint
                        print(f"  📍 {variant}: Found checkpoint at file {checkpoint.get('current_file_idx', 0)}")
                    except Exception as e:
                        print(f"  ⚠️  {variant}: Checkpoint corrupted: {e}")
            
            model_info['resume_status'] = resume_info
            remaining_models.append(model_info)
        else:
            print(f"  ✅ {variant}: ALL DATASETS COMPLETE")
    
    return remaining_models

def check_folder_completely_finished(output_dir):
    """Check if all models in a folder are completely finished"""
    remaining = get_smart_remaining_models(output_dir)
    return len(remaining) == 0

def get_available_checkpoints(variant):
    """Get all available checkpoints for a model variant"""
    base_path = Path('/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR/training_results')
    variant_path = base_path / variant
    
    if not variant_path.exists():
        return {}
    
    # Look for the scunet_ngswin_* subfolder
    subfolders = [d for d in variant_path.iterdir() if d.is_dir() and d.name.startswith('scunet_ngswin_')]
    if not subfolders:
        return {}
    
    models_path = subfolders[0] / 'models'
    if not models_path.exists():
        return {}
    
    checkpoints = {
        'psnr': None,
        'latest': None,
        'iterations': []
    }
    
    # Find all checkpoint files
    checkpoint_files = list(models_path.glob('*_G.pth'))
    
    for checkpoint_file in checkpoint_files:
        name = checkpoint_file.stem  # Remove .pth extension
        
        if name == 'psnr_G':
            checkpoints['psnr'] = {
                'path': str(checkpoint_file),
                'type': 'psnr',
                'display_name': 'Best PSNR',
                'iteration': 'best',
                'date': datetime.fromtimestamp(checkpoint_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            }
        elif name.replace('_G', '').isdigit():
            iteration = int(name.replace('_G', ''))
            checkpoints['iterations'].append({
                'path': str(checkpoint_file),
                'type': 'iteration',
                'display_name': f'Iteration {iteration:,}',
                'iteration': iteration,
                'date': datetime.fromtimestamp(checkpoint_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            })
    
    # Sort iterations by number
    checkpoints['iterations'].sort(key=lambda x: x['iteration'])
    
    # Set latest checkpoint
    if checkpoints['iterations']:
        checkpoints['latest'] = checkpoints['iterations'][-1].copy()
        checkpoints['latest']['type'] = 'latest'
        checkpoints['latest']['display_name'] = f"Latest ({checkpoints['latest']['iteration']:,})"
    
    return checkpoints

def discover_all_trained_models_with_checkpoints():
    """Discover all trained model variants with their available checkpoints"""
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
        
        # Get all available checkpoints
        checkpoints = get_available_checkpoints(variant)
        
        if not (checkpoints['psnr'] or checkpoints['iterations']):
            print(f"⚠️  No valid checkpoints found for {variant}, skipping...")
            continue
        
        print(f"✓ Found {variant}: {len(checkpoints['iterations'])} checkpoints available")
        
        model_info = {
            'variant': variant,
            'display_name': variant.replace('_', ' ').title(),
            'config_path': str(subfolders[0] / 'option.json'),
            'models_dir': str(models_path),
            'checkpoints': checkpoints,
            'default_checkpoint': checkpoints['psnr'] if checkpoints['psnr'] else checkpoints['latest']
        }
        
        models_info.append(model_info)
    
    print(f"📊 Total models discovered: {len(models_info)}")
    return models_info

def setup_model_and_checkpoint_with_selection(model_info, checkpoint_selection=None):
    """Setup model with specific checkpoint selection"""
    
    # Determine which checkpoint to use
    if checkpoint_selection is None:
        # Use default (PSNR if available, otherwise latest)
        checkpoint_info = model_info['default_checkpoint']
    else:
        checkpoint_info = checkpoint_selection
    
    print(f"🔧 Loading checkpoint: {checkpoint_info['display_name']} ({checkpoint_info['date']})")
    
    # Load the configuration
    config_path = model_info['config_path']
    with open(config_path, 'r') as f:
        opt = json.load(f)
    
    # Update paths for inference
    opt['is_train'] = False
    opt['dist'] = False
    opt['path']['pretrained_netG'] = checkpoint_info['path']
    
    # Create model
    model = select_model.define_Model(opt)
    model.init_train()
    
    print(f"✅ Model {model_info['variant']} loaded successfully with {checkpoint_info['display_name']}")
    
    return model, checkpoint_info
