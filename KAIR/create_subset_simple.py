#!/usr/bin/env python3
"""
Create 3000-slice subset from UWSpine synthetic transfer dataset
Simple script to randomly select and copy paired images
"""

import os
import shutil
import random
from pathlib import Path

def create_subset():
    print("🚀 Creating 3000-slice dataset subset...")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define paths
    source_root = Path("/home/Drive-D/UWSpine_adn")
    target_root = Path("/home/Drive-D/UWSpine_adn_subset")
    
    source_transfer = source_root / "train" / "synthesized_metal_transfer"
    source_clean = source_root / "train" / "no_metal"
    
    target_transfer = target_root / "train" / "synthesized_metal_transfer"
    target_clean = target_root / "train" / "no_metal"
    
    # Check if source exists
    if not source_transfer.exists():
        print(f"❌ Error: Source directory not found: {source_transfer}")
        return False
    
    if not source_clean.exists():
        print(f"❌ Error: Clean directory not found: {source_clean}")
        return False
    
    # Create target directories
    print(f"📁 Creating target directory: {target_root}")
    target_transfer.mkdir(parents=True, exist_ok=True)
    target_clean.mkdir(parents=True, exist_ok=True)
    
    # Get all .npy files from synthesized_metal_transfer
    print("🔍 Scanning for .npy files...")
    
    all_files = []
    # Look specifically for .npy files (the actual training data)
    all_files.extend(list(source_transfer.glob("*.npy")))
    
    print(f"📊 Found {len(all_files)} total files")
    
    if len(all_files) == 0:
        print("❌ No .npy files found! Check the source directory structure.")
        return False
    
    # Randomly select 3000 files (or all if less than 3000)
    subset_size = min(3000, len(all_files))
    selected_files = random.sample(all_files, subset_size)
    
    print(f"🎯 Selected {subset_size} files for subset")
    
    # Copy selected files
    copied_count = 0
    failed_count = 0
    
    for i, transfer_file in enumerate(selected_files):
        try:
            # Get relative path from source_transfer
            rel_path = transfer_file.relative_to(source_transfer)
            
            # Define target file paths
            target_transfer_file = target_transfer / rel_path
            target_clean_file = target_clean / rel_path
            
            # Define source clean file path
            source_clean_file = source_clean / rel_path
            
            # Create parent directories if needed
            target_transfer_file.parent.mkdir(parents=True, exist_ok=True)
            target_clean_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if corresponding clean file exists
            if not source_clean_file.exists():
                print(f"⚠️  Warning: No matching clean file for {rel_path}")
                failed_count += 1
                continue
            
            # Copy both files
            shutil.copy2(transfer_file, target_transfer_file)
            shutil.copy2(source_clean_file, target_clean_file)
            
            copied_count += 1
            
            # Progress update
            if (i + 1) % 500 == 0:
                print(f"📋 Progress: {i + 1}/{subset_size} files processed...")
                
        except Exception as e:
            print(f"❌ Failed to copy {transfer_file.name}: {e}")
            failed_count += 1
    
    # Create summary
    print(f"\n✅ Dataset subset creation completed!")
    print(f"📈 Successfully copied: {copied_count} image pairs")
    print(f"❌ Failed to copy: {failed_count} files")
    print(f"📁 Subset saved to: {target_root}")
    
    # Save info file
    info_file = target_root / "subset_info.txt"
    with open(info_file, 'w') as f:
        f.write(f"UWSpine Dataset Subset\n")
        f.write(f"=====================\n")
        f.write(f"Created from: {source_root}\n")
        f.write(f"Target subset size: 3000\n")
        f.write(f"Actual subset size: {copied_count}\n")
        f.write(f"Random seed: 42\n")
        f.write(f"Failed copies: {failed_count}\n")
        f.write(f"Creation date: {Path(__file__).stat().st_mtime}\n")
    
    print(f"📄 Info file saved: {info_file}")
    
    if copied_count > 0:
        print(f"\n🎉 SUCCESS! Your subset is ready for fine-tuning!")
        print(f"   Dataset path: {target_root}")
        print(f"   Training samples: {copied_count}")
        print(f"   Expected iterations per epoch: {copied_count // 8} (with batch_size=8)")
        return True
    else:
        print(f"\n❌ FAILED! No files were successfully copied.")
        return False

if __name__ == "__main__":
    try:
        success = create_subset()
        if success:
            print(f"\n🚀 Next step: Run fine-tuning with:")
            print(f"   python3 main_train_scunet_ngswin_1.py --opt options/train_scunet_ngswin_finetune_synthetic_transfer.json")
        else:
            print(f"\n🔧 Please check the source directory structure and try again.")
    except KeyboardInterrupt:
        print(f"\n⏹️  Operation cancelled by user.")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
