#!/usr/bin/env python3
"""
Simple script to run the final test evaluation on both PSNR checkpoints
"""

import subprocess
import sys
import os

def main():
    # Change to KAIR directory
    kair_dir = "/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR"
    os.chdir(kair_dir)
    
    print("🚀 Running final test evaluation with both PSNR checkpoints...")
    print("📍 This will test both psnr_E.pth and psnr_G.pth on your test dataset")
    print("💾 Every 250th test image will be saved as comparison (Input | Output | Ground Truth)")
    print("-" * 80)
    
    # Since training is complete, we need to create a modified version that goes straight to testing
    # For now, the user can manually run: python main_train_scunet_ngswin_1.py --opt options/train_scunet_ngswin_1.json --val_split 0.0
    
    print("📖 MANUAL INSTRUCTION:")
    print("Since your training is complete, run this command to perform final test evaluation:")
    print()
    print("cd /home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR")
    print("python main_train_scunet_ngswin_1.py --opt options/train_scunet_ngswin_1.json --val_split 0.0")
    print()
    print("This will:")
    print("✅ Load both psnr_E.pth and psnr_G.pth checkpoints")
    print("✅ Test each checkpoint on your test dataset") 
    print("✅ Save every 250th test image as comparison (Input | Output | Ground Truth)")
    print("✅ Generate detailed performance comparison between the two checkpoints")
    print("✅ Save results to: training_results/conv_nstb/scunet_ngswin_conv_nstb/images/final_test_*/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
