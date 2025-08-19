#!/usr/bin/env python3
"""
Clinical Dataset Supervised Pipeline - Simple Version
====================================
"""

print("🚀 Starting clinical dataset pipeline...")
print("Loading imports...")

import os
import os.path as path
import yaml
import numpy as np
import nibabel as nib
from tqdm import tqdm
from PIL import Image
import h5py
import random

# Import your utilities (only use what you already have)
from adn_utils import read_dir, get_connected_components, EasyDict

print("✓ All imports loaded successfully")

def main():
    """Main pipeline"""
    print("\n" + "="*50)
    print("Clinical Dataset Supervised Pipeline")
    print("="*50)
    
    config_file = path.join(path.dirname(__file__), "clinical_config.yaml")
    
    if not path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return
    
    # Load configuration
    try:
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['clinical_config']
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return
    
    print(f"📂 Raw data path: {config['raw_dir']}")
    print(f"📂 Output path: {config['dataset_dir']}")
    print(f"🎯 Max HU thresholds: {config['max_hu']}")
    print(f"🔗 Connected area threshold: {config['connected_area']}")
    print(f"📐 Target image size: {config['image_size']}")
    
    # Step 1: Find and analyze files
    print(f"\n🔍 Step 1: Looking for .nii files in {config['raw_dir']}")
    
    volume_files = read_dir(config['raw_dir'],
        predicate=lambda x: x.endswith("nii") and not x.endswith("nii.gz"), 
        recursive=True)
    
    if not volume_files:
        print("No .nii files found, trying .nii.gz files...")
        volume_files = read_dir(config['raw_dir'],
            predicate=lambda x: x.endswith("nii.gz"), 
            recursive=True)
    
    if not volume_files:
        print(f"❌ No .nii or .nii.gz files found in {config['raw_dir']}")
        return
    
    print(f"✓ Found {len(volume_files)} volume files")
    
    # Show first few files
    print("📋 First few files:")
    for i, f in enumerate(volume_files[:3]):
        print(f"   {i+1}. {f}")
    if len(volume_files) > 3:
        print(f"   ... and {len(volume_files)-3} more")
    
    # Step 2: Process first file as example
    print(f"\n🔬 Step 2: Processing first file as example...")
    first_file = volume_files[0]
    
    try:
        print(f"Loading: {first_file}")
        img = nib.load(first_file)
        volume = img.get_fdata().astype('float32')
        print(f"✓ Volume shape: {volume.shape}")
        print(f"✓ Volume range: {volume.min():.2f} to {volume.max():.2f}")
        
        # Extract middle slice if 3D
        if len(volume.shape) == 3:
            middle_slice = volume.shape[2] // 2
            image = volume[:, :, middle_slice]
            print(f"✓ Extracted slice {middle_slice} from 3D volume")
        else:
            image = volume
            
        print(f"✓ Image shape: {image.shape}")
        print(f"✓ Image range: {image.min():.2f} to {image.max():.2f}")
        
        # Apply artifact detection logic
        image_type = "no_artifact"
        
        if image.max() > config["max_hu"][1]:
            points = np.array(np.where(image > config["max_hu"][1])).T
            print(f"📊 Found {len(points)} high-intensity points (>{config['max_hu'][1]})")
            if len(points) > 0:
                points = set(tuple(p) for p in points)
                components = get_connected_components(points)
                if components:
                    max_area = max(len(c) for c in components)
                    print(f"📊 Largest connected component: {max_area} pixels")
                    if max_area > config["connected_area"]: 
                        image_type = "artifact"
                        print("🔴 Classified as ARTIFACT")
                    else:
                        print("🟡 High intensity but small area - SUSPICIOUS")
                else:
                    print("🟢 No connected components found")
        elif image.max() > config["max_hu"][0]: 
            print(f"🟡 Suspicious intensities (>{config['max_hu'][0]}) - would skip")
        else:
            print("🟢 Clean image - good for synthesis")
        
        print(f"✅ Final classification: {image_type}")
        
    except Exception as e:
        print(f"❌ Error processing {first_file}: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 Analysis completed!")
    print("This was a dry run. The full pipeline would:")
    print("1. ✓ Process all volumes and find clean images") 
    print("2. ⚡ Synthesize artifacts (if ODL available)")
    print("3. 💾 Save as supervised pairs in .h5 format")

if __name__ == "__main__":
    main()
