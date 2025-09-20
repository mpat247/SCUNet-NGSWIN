#!/usr/bin/env python3
"""
Create a 3000-slice subset from the synthetic transfer dataset
Randomly selects slices and copies them to a new subset folder
"""

import os
import shutil
import random
from pathlib import Path
import argparse

def create_dataset_subset(source_path, subset_size=3000, seed=42):
    """
    Create a subset of the synthetic transfer dataset
    
    Args:
        source_path: Path to original dataset (/home/Drive-D/UWSpine_adn)
        subset_size: Number of slices to include in subset (default: 3000)
        seed: Random seed for reproducibility (default: 42)
    """
    
    print(f"Creating dataset subset with {subset_size} slices...")
    print(f"Source path: {source_path}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Define source and target directories
    source_root = Path(source_path)
    target_root = source_root.parent / f"{source_root.name}_subset"
    
    # Source directories
    source_transfer = source_root / "train" / "synthesized_metal_transfer"
    source_clean = source_root / "train" / "no_metal"
    
    # Target directories  
    target_transfer = target_root / "train" / "synthesized_metal_transfer"
    target_clean = target_root / "train" / "no_metal"
    
    # Create target directory structure
    print(f"Creating target directory: {target_root}")
    target_transfer.mkdir(parents=True, exist_ok=True)
    target_clean.mkdir(parents=True, exist_ok=True)
    
    # Get all available slices from synthesized_metal_transfer
    print("Scanning source dataset...")
    if not source_transfer.exists():
        raise FileNotFoundError(f"Source directory not found: {source_transfer}")
    
    # Get all image files (assuming .nii, .nii.gz, .png, .jpg, etc.)
    transfer_files = []
    for ext in ['*.nii', '*.nii.gz', '*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']:
        transfer_files.extend(list(source_transfer.glob(ext)))
    
    if not transfer_files:
        # Try subdirectories (patient folders)
        for patient_dir in source_transfer.iterdir():
            if patient_dir.is_dir():
                for ext in ['*.nii', '*.nii.gz', '*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']:
                    transfer_files.extend(list(patient_dir.glob(ext)))
    
    print(f"Found {len(transfer_files)} transfer files")
    
    if len(transfer_files) == 0:
        raise ValueError("No image files found in source directory")
    
    # Randomly sample subset_size files
    if len(transfer_files) < subset_size:
        print(f"Warning: Only {len(transfer_files)} files available, using all of them")
        selected_files = transfer_files
    else:
        selected_files = random.sample(transfer_files, subset_size)
    
    print(f"Selected {len(selected_files)} files for subset")
    
    # Copy selected files
    copied_count = 0
    failed_count = 0
    
    for i, transfer_file in enumerate(selected_files):
        try:
            # Determine relative path structure
            rel_path = transfer_file.relative_to(source_transfer)
            
            # Target paths
            target_transfer_file = target_transfer / rel_path
            target_clean_file = target_clean / rel_path
            
            # Create subdirectories if needed
            target_transfer_file.parent.mkdir(parents=True, exist_ok=True)
            target_clean_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Find corresponding clean file
            clean_file = source_clean / rel_path
            
            if not clean_file.exists():
                print(f"Warning: Clean file not found for {transfer_file.name}")
                continue
            
            # Copy both files
            shutil.copy2(transfer_file, target_transfer_file)
            shutil.copy2(clean_file, target_clean_file)
            
            copied_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"Copied {i + 1}/{len(selected_files)} files...")
                
        except Exception as e:
            print(f"Failed to copy {transfer_file}: {e}")
            failed_count += 1
    
    print(f"\nDataset subset creation completed!")
    print(f"✅ Successfully copied: {copied_count} file pairs")
    print(f"❌ Failed to copy: {failed_count} file pairs")
    print(f"📁 Subset saved to: {target_root}")
    print(f"🎲 Random seed used: {seed}")
    
    # Create info file
    info_file = target_root / "subset_info.txt"
    with open(info_file, 'w') as f:
        f.write(f"Dataset Subset Information\n")
        f.write(f"========================\n")
        f.write(f"Created from: {source_path}\n")
        f.write(f"Target size: {subset_size}\n")
        f.write(f"Actual size: {copied_count}\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"Failed copies: {failed_count}\n")
    
    print(f"📄 Info saved to: {info_file}")
    
    return target_root

def main():
    parser = argparse.ArgumentParser(description='Create dataset subset for fine-tuning')
    parser.add_argument('--source', '-s', 
                       default='/home/Drive-D/UWSpine_adn',
                       help='Source dataset path (default: /home/Drive-D/UWSpine_adn)')
    parser.add_argument('--size', '-n', type=int, default=3000,
                       help='Number of slices in subset (default: 3000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    try:
        subset_path = create_dataset_subset(args.source, args.size, args.seed)
        print(f"\n🎉 Success! Dataset subset ready at: {subset_path}")
        print(f"\nTo use this subset, update your JSON config:")
        print(f'  "dataroot_H": "{subset_path}",')
        
    except Exception as e:
        print(f"❌ Error creating dataset subset: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
