#!/usr/bin/env python3
"""
Model Management Utilities
==========================
Contains functions for discovering and setting up trained models
"""

import os
import sys
from pathlib import Path

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
