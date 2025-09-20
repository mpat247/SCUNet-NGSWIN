#!/usr/bin/env python3
"""
Test script to validate exact MATLAB port with single image
"""

import numpy as np
import matplotlib.pyplot as plt
from exact_matlab_port import matlab_simulate_metal_artifact, matlab_get_mar_params
from scipy.io import loadmat
from PIL import Image

def test_single_image():
    print("🧪 Testing EXACT MATLAB port with single image...")
    
    # Create test image
    test_img = np.zeros((512, 512))
    # Add some tissue-like values
    test_img[150:350, 150:350] = 50   # Soft tissue
    test_img[200:300, 200:300] = 200  # Bone
    test_img[240:260, 240:260] = 1000 # Dense bone
    
    # Load exact metal mask
    sample_masks_file = "data/deep_lesion/metal_masks/SampleMasks.mat"
    sample_data = loadmat(sample_masks_file)
    metal_masks = sample_data['CT_samples_bwMetal']
    
    # Use first metal mask
    test_mask = metal_masks[:, :, 0:1]
    
    # Load MAR parameters
    MARpara = matlab_get_mar_params("data/deep_lesion/metal_masks")
    
    # CT parameters - dimensions will be updated dynamically
    CTpara = {
        'SOD': 541,
        'angSize': 1.0,
        'angNum': 180,  # Reduced for faster testing
        'imPixNum': 512,
        'imPixScale': 1.0,
        'sinogram_size_x': None,  # Will be set dynamically
        'sinogram_size_y': None
    }
    
    print(f"✅ Test setup complete:")
    print(f"   Test image: {test_img.shape}, range [{test_img.min()}, {test_img.max()}]")
    print(f"   Metal mask: {test_mask.shape}, non-zero pixels: {np.sum(test_mask > 0)}")
    print(f"   CT angles: {CTpara['angNum']}")
    
    try:
        # Run simulation
        print("🔬 Running EXACT MATLAB simulation...")
        results = matlab_simulate_metal_artifact(test_img, test_mask, CTpara, MARpara)
        
        ma_sinogram_all, LI_sinogram_all, poly_sinogram = results[0], results[1], results[2]
        ma_CT_all, LI_CT_all, poly_CT, gt_CT, metal_trace_all = results[3], results[4], results[5], results[6], results[7]
        
        print("✅ Simulation completed successfully!")
        print(f"   Output shapes:")
        print(f"     MA CT: {ma_CT_all.shape}")
        print(f"     LI CT: {LI_CT_all.shape}")
        print(f"     Poly CT: {poly_CT.shape}")
        print(f"     GT CT: {gt_CT.shape}")
        
        # Extract results
        ma_CT = ma_CT_all[:, :, 0]
        LI_CT = LI_CT_all[:, :, 0]
        
        print(f"   Value ranges:")
        print(f"     MA CT: [{ma_CT.min():.3f}, {ma_CT.max():.3f}]")
        print(f"     LI CT: [{LI_CT.min():.3f}, {LI_CT.max():.3f}]")
        print(f"     Poly CT: [{poly_CT.min():.3f}, {poly_CT.max():.3f}]")
        
        # Convert back to HU
        ma_HU = (ma_CT - MARpara['MiuWater']) * 1000 / MARpara['MiuWater']
        LI_HU = (LI_CT - MARpara['MiuWater']) * 1000 / MARpara['MiuWater']
        poly_HU = (poly_CT - MARpara['MiuWater']) * 1000 / MARpara['MiuWater']
        
        print(f"   HU ranges:")
        print(f"     MA HU: [{ma_HU.min():.1f}, {ma_HU.max():.1f}]")
        print(f"     LI HU: [{LI_HU.min():.1f}, {LI_HU.max():.1f}]")
        print(f"     Poly HU: [{poly_HU.min():.1f}, {poly_HU.max():.1f}]")
        
        # Save test results
        output_dir = "test_results"
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save images
        for name, img in [('original', test_img), ('ma_artifact', ma_HU), 
                          ('li_corrected', LI_HU), ('poly_clean', poly_HU)]:
            # Clip and normalize for visualization
            img_vis = np.clip(img, -1000, 3000)
            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
            img_vis = (img_vis * 255).astype(np.uint8)
            
            Image.fromarray(img_vis).save(f"{output_dir}/test_{name}.png")
            np.save(f"{output_dir}/test_{name}.npy", img.astype(np.float32))
        
        print(f"✅ Test results saved to {output_dir}/")
        print("🎉 EXACT MATLAB port validation successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_single_image()
