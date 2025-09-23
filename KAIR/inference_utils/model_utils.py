#!/usr/bin/env python3
"""
Model Management Utilities - Fixed Version
==========================================
Contains functions for discovering and setting up trained models
Compatible with your specific configuration setup
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

def get_model_variant_configs():
    """
    Get model variant configurations with their exact option file mappings
    Based on your actual configuration files, including finetuned models
    """
    return {
        'conv_nstb': {
            'path': 'conv_nstb/scunet_ngswin_conv_nstb',
            'display_name': 'Conv Nstb',
            'base_config': 'train_scunet_ngswin_1.json',
            'block_variant': 'conv_nstb',
            'task_name': 'scunet_ngswin_conv_nstb'
        },
        'conv_trans_nstb': {
            'path': 'conv_trans_nstb/scunet_ngswin_conv_trans_nstb', 
            'display_name': 'Conv Trans Nstb',
            'base_config': 'train_scunet_ngswin_4.json',
            'block_variant': 'conv_trans_nstb',
            'task_name': 'scunet_ngswin_conv_trans_nstb'
        },
        'trans_nstb': {
            'path': 'trans_nstb/scunet_ngswin_trans_nstb',
            'display_name': 'Trans Nstb',
            'base_config': 'train_scunet_ngswin_2.json',
            'block_variant': 'trans_nstb', 
            'task_name': 'scunet_ngswin_trans_nstb'
        },
        'synthetic_transfer_finetune': {
            'path': 'synthetic_transfer_finetune/scunet_ngswin_conv_trans_nstb_synthetic_transfer_finetune',
            'display_name': 'Conv Trans Nstb (UWSpine Synthetic Transfer Finetuned)',
            'base_config': 'train_scunet_ngswin_finetune_synthetic_transfer.json',
            'block_variant': 'conv_trans_nstb',
            'task_name': 'scunet_ngswin_conv_trans_nstb_synthetic_transfer_finetune'
        },
        'conv_trans_nstb_finetuned': {
            'path': 'conv_trans_nstb_finetuned/scunet_ngswin_conv_trans_nstb_finetuned',
            'display_name': 'Conv Trans Nstb (Finetuned Li_CT)',
            'base_config': 'train_scunet_ngswin_4.json',
            'block_variant': 'conv_trans_nstb',
            'task_name': 'scunet_ngswin_conv_trans_nstb_finetuned'
        },
        'conv_trans_nstb_finetuned_400k': {
            'path': 'conv_trans_nstb_finetuned/scunet_ngswin_conv_trans_nstb_finetuned_400k',
            'display_name': 'Conv Trans Nstb (Finetuned 400k Clinical)',
            'base_config': 'train_scunet_ngswin_4.json',
            'block_variant': 'conv_trans_nstb',
            'task_name': 'scunet_ngswin_conv_trans_nstb_finetuned_400k'
        }
    }

def find_most_recent_config_file(model_path):
    """
    Find the most recent configuration file for a model
    Looks in the options subdirectory for timestamped config files
    """
    options_dir = os.path.join(model_path, 'options')
    if not os.path.exists(options_dir):
        return None
    
    # Get all JSON files in options directory
    config_files = glob.glob(os.path.join(options_dir, '*.json'))
    if not config_files:
        return None
    
    # Sort by modification time (most recent first)
    config_files.sort(key=os.path.getmtime, reverse=True)
    return config_files[0]

def create_inference_config_from_template(variant, checkpoint_path):
    """
    Create inference configuration from base config template
    Uses your exact configuration files from the options directory
    """
    # Load base configuration from options directory
    options_base_dir = "/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR/options"
    model_configs = get_model_variant_configs()
    
    if variant not in model_configs:
        raise ValueError(f"Unknown model variant: {variant}")
    
    variant_config = model_configs[variant]
    base_config_file = os.path.join(options_base_dir, variant_config['base_config'])
    
    if not os.path.exists(base_config_file):
        raise FileNotFoundError(f"Base config not found: {base_config_file}")
    
    # Load base configuration
    with open(base_config_file, 'r') as f:
        content = f.read()
        # Remove comments (lines starting with //)
        lines = content.split('\n')
        filtered_lines = [line for line in lines if not line.strip().startswith('//')]
        clean_content = '\n'.join(filtered_lines)
        opt = json.loads(clean_content)
    
    # Configure for inference
    opt['is_train'] = False
    opt['dist'] = False
    
    # Set up paths correctly
    if 'path' not in opt:
        opt['path'] = {}
    opt['path']['pretrained_netG'] = checkpoint_path
    
    # Ensure correct variant configuration
    if 'netG' not in opt:
        opt['netG'] = {}
    opt['netG']['block_variant'] = variant_config['block_variant']
    opt['task'] = variant_config['task_name']
    
    print(f"🔧 Created inference config for {variant} with {variant_config['block_variant']} variant")
    return opt

def get_available_checkpoints(models_dir):
    """Get all available checkpoints for a model"""
    checkpoints = {
        'psnr': None,
        'latest': None,
        'iterations': []
    }
    
    if not os.path.exists(models_dir):
        return checkpoints
    
    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(models_dir, '*_G.pth'))
    
    for checkpoint_file in checkpoint_files:
        name = os.path.splitext(os.path.basename(checkpoint_file))[0]  # Remove .pth extension
        
        if name == 'psnr_G':
            checkpoints['psnr'] = {
                'path': checkpoint_file,
                'type': 'psnr',
                'display_name': 'Best PSNR',
                'iteration': 'best',
                'date': datetime.fromtimestamp(os.path.getmtime(checkpoint_file)).strftime('%Y-%m-%d %H:%M')
            }
        elif name.replace('_G', '').isdigit():
            iteration = int(name.replace('_G', ''))
            checkpoints['iterations'].append({
                'path': checkpoint_file,
                'type': 'iteration',
                'display_name': f'Iteration {iteration:,}',
                'iteration': iteration,
                'date': datetime.fromtimestamp(os.path.getmtime(checkpoint_file)).strftime('%Y-%m-%d %H:%M')
            })
    
    # Sort iterations by number
    checkpoints['iterations'].sort(key=lambda x: x['iteration'])
    
    # Set latest checkpoint
    if checkpoints['iterations']:
        checkpoints['latest'] = checkpoints['iterations'][-1].copy()
        checkpoints['latest']['type'] = 'latest'
        checkpoints['latest']['display_name'] = f"Latest (Iteration {checkpoints['latest']['iteration']:,})"
    
    return checkpoints

def discover_all_trained_models_with_checkpoints():
    """
    Discover all trained models and their available checkpoints
    Returns structured information about models and checkpoints
    """
    models_info = []
    base_dir = "/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR/training_results"
    model_configs = get_model_variant_configs()
    
    print("🔍 Discovering trained models...")
    
    for variant, config in model_configs.items():
        model_path = os.path.join(base_dir, config['path'])
        models_dir = os.path.join(model_path, 'models')
        
        if os.path.exists(models_dir):
            print(f"✓ Found {variant}: {model_path}")
            
            # Find most recent configuration file (or use template)
            config_file = find_most_recent_config_file(model_path)
            if config_file:
                print(f"  📄 Found config: {os.path.basename(config_file)}")
            else:
                print(f"  📄 No config found, will use template from {config['base_config']}")
                
            # Get available checkpoints
            checkpoints = get_available_checkpoints(models_dir)
            
            # Determine default checkpoint (prefer PSNR, fallback to latest)
            default_checkpoint = checkpoints['psnr'] if checkpoints['psnr'] else checkpoints['latest']
            
            if default_checkpoint:
                print(f"  🏆 Default checkpoint: {default_checkpoint['display_name']}")
                if checkpoints['iterations']:
                    print(f"  📊 Available iterations: {len(checkpoints['iterations'])} checkpoints")
                
                model_info = {
                    'variant': variant,
                    'display_name': config['display_name'],
                    'model_path': model_path,
                    'models_dir': models_dir,
                    'config_path': config_file,  # May be None, will use template
                    'variant_config': config,
                    'checkpoints': checkpoints,
                    'default_checkpoint': default_checkpoint
                }
                models_info.append(model_info)
            else:
                print(f"  ❌ No valid checkpoints found for {variant}")
        else:
            print(f"❌ Model path not found: {model_path}")
    
    print(f"📊 Total models discovered: {len(models_info)}")
    return models_info

def setup_model_and_checkpoint_with_selection(model_info, checkpoint_selection=None):
    """Setup model with specific checkpoint selection - FIXED VERSION"""
    
    variant = model_info['variant']
    
    # Determine which checkpoint to use
    if checkpoint_selection is None:
        checkpoint_info = model_info['default_checkpoint']
    else:
        checkpoint_info = checkpoint_selection
    
    print(f"🔧 Loading checkpoint: {checkpoint_info['display_name']} ({checkpoint_info['date']})")
    
    # Try to load existing config file first, fallback to template
    opt = None
    if model_info['config_path'] and os.path.exists(model_info['config_path']):
        try:
            print(f"📄 Loading existing config: {os.path.basename(model_info['config_path'])}")
            with open(model_info['config_path'], 'r') as f:
                content = f.read()
                # Remove comments (lines starting with //)
                lines = content.split('\n')
                filtered_lines = [line for line in lines if not line.strip().startswith('//')]
                clean_content = '\n'.join(filtered_lines)
                opt = json.loads(clean_content)
        except Exception as e:
            print(f"⚠️  Failed to load existing config: {e}")
            opt = None
    
    # If no existing config or failed to load, create from template
    if opt is None:
        print(f"📄 Creating config from template for {variant}")
        opt = create_inference_config_from_template(variant, checkpoint_info['path'])
    else:
        # Update existing config for inference
        opt['is_train'] = False
        opt['dist'] = False
        opt['path']['pretrained_netG'] = checkpoint_info['path']
        
        # Ensure correct variant is set
        variant_config = model_info['variant_config']
        opt['netG']['block_variant'] = variant_config['block_variant']
        opt['task'] = variant_config['task_name']
    
    # Create model
    print(f"🏗️  Creating model with {opt['netG']['block_variant']} variant...")
    model = select_model.define_Model(opt)
    model.init_train()
    
    # Get model parameter count
    total_params = sum(p.numel() for p in model.netG.parameters())
    print(f"📊 Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    print(f"✅ Model {variant} loaded successfully with {checkpoint_info['display_name']}")
    
    return model, checkpoint_info

# Legacy functions for backward compatibility
def discover_all_trained_models():
    """Legacy function - discover all trained model variants from training_results folder"""
    models_with_checkpoints = discover_all_trained_models_with_checkpoints()
    
    # Convert to legacy format
    legacy_models = []
    for model in models_with_checkpoints:
        default_checkpoint = model['default_checkpoint']
        
        # Map config based on variant for legacy compatibility
        config_mapping = {
            'conv_nstb': 'options/train_scunet_ngswin_1.json',
            'conv_trans_nstb': 'options/train_scunet_ngswin_4.json',
            'trans_nstb': 'options/train_scunet_ngswin_2.json'
        }
        
        legacy_model = {
            'variant': model['variant'],
            'checkpoint_path': default_checkpoint['path'],
            'config_file': config_mapping.get(model['variant'], 'options/train_scunet_ngswin_1.json'),
            'display_name': model['display_name'],
            'subfolder_name': os.path.basename(model['model_path'])
        }
        legacy_models.append(legacy_model)
    
    return legacy_models

def setup_model_and_checkpoint(model_info):
    """Legacy function - Load model and checkpoint with detailed logging"""
    variant = model_info['variant']
    checkpoint_path = model_info['checkpoint_path']
    config_path = model_info['config_file']
    
    print(f"🔧 Loading {variant} model from: {os.path.basename(checkpoint_path)}")
    
    try:
        # Try to use template-based loading
        opt = create_inference_config_from_template(variant, checkpoint_path)
    except Exception as e:
        print(f"⚠️  Template loading failed: {e}")
        # Fallback to original method
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

# Continue with all the other existing functions unchanged...
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
    
    # Check if all datasets are completed
    original_complete = (variant_dir / "original_dataset" / "metrics_summary.json").exists()
    clinical_complete = (variant_dir / "clinical_dataset_no_masks" / "clinical_no_masks_summary.json").exists()
    clinical_artifact_complete = (variant_dir / "clinical_artifact_only_dataset" / "clinical_artifact_only_summary.json").exists()
    
    # Check for checkpoint
    checkpoint = load_progress_checkpoint(output_dir, variant)
    
    return {
        'variant': variant,
        'original_dataset_complete': original_complete,
        'clinical_dataset_complete': clinical_complete,
        'clinical_artifact_dataset_complete': clinical_artifact_complete,
        'fully_complete': original_complete and clinical_complete and clinical_artifact_complete,
        'has_checkpoint': checkpoint is not None,
        'checkpoint_info': checkpoint
    }

def get_smart_remaining_models(output_dir):
    """
    Smart function to determine which models still need processing.
    Checks completion of both original dataset and clinical dataset for each model.
    Returns remaining work with detailed resume information.
    """
    all_models = discover_all_trained_models_with_checkpoints()
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
        # Check clinical artifact-only dataset completion
        clinical_artifact_complete = False
        clinical_artifact_summary_path = variant_dir / "clinical_artifact_only_dataset" / "clinical_artifact_only_summary.json"
        if clinical_artifact_summary_path.exists():
            try:
                with open(clinical_artifact_summary_path, 'r') as f:
                    summary = json.load(f)
                if summary.get('total_samples', 0) >= 5000:  # Should have ~7k+ slices
                    clinical_artifact_complete = True
                    print(f"  ✅ {variant}: Clinical artifact-only dataset complete ({summary.get('total_samples', 0)} slices)")
                else:
                    print(f"  🔄 {variant}: Clinical artifact-only dataset incomplete ({summary.get('total_samples', 0)} slices)")
            except Exception as e:
                print(f"  ❌ {variant}: Clinical artifact-only dataset summary corrupted: {e}")
        else:
            print(f"  ⏳ {variant}: Clinical artifact-only dataset not started")
        
        # Determine what needs to be done for this model
        if not original_complete or not clinical_complete or not clinical_artifact_complete:
            resume_info = {
                'original_complete': original_complete,
                'clinical_complete': clinical_complete,
                'clinical_artifact_complete': clinical_artifact_complete,
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
