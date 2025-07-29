#!/usr/bin/env python3
"""
Configuration and Preprocessing Utilities
=========================================
Contains clinical preprocessing configuration and functions
"""

import os
import numpy as np
import cv2
import nibabel as nib
from pathlib import Path

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
        if image_lac.shape != (config.CTpara['imPixNum'], config.CTpara['imPixNum']):
            image_resized = cv2.resize(image_lac.astype(np.float32), 
                                     (config.CTpara['imPixNum'], config.CTpara['imPixNum']), 
                                     interpolation=cv2.INTER_LINEAR)
        else:
            image_resized = image_lac
        
        # Step 4: For model compatibility, normalize to [0,1] range then scale to [0,255]
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
        mask_thre = config.mask_thre  # This is already calculated as 2500/1000 * 0.192 + 0.192
        
        # Step 3: Create binary mask where LAC > threshold
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
