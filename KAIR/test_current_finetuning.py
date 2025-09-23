#!/usr/bin/env python3
"""
Test Current Fine-tuning Model with UWSpine Dataset
=================================================
"""

import os
import sys
import torch
from pathlib import Path

# Add KAIR to path
sys.path.append('/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR')

from inference_utils.model_utils import setup_model_and_checkpoint
from inference_utils.uwspine_dataset_processing import process_uwspine_synthetic_transfer_dataset

def test_current_finetuning_model():
    """Test the current fine-tuning model with UWSpine dataset"""
    
    print("🚀 TESTING CURRENT FINE-TUNING MODEL WITH UWSPINE DATASET")
    print("="*60)
    
    # Paths
    models_dir = "/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR/training_results/synthetic_transfer_finetune/scunet_ngswin_conv_trans_nstb_synthetic_transfer_finetune/models"
    uwspine_data_dir = "/home/Drive-D/UWSpine_adn"  # Your UWSpine dataset path
    output_dir = "/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR/inference_results/current_finetuning_test"
    
    # Check if we have any checkpoints
    model_files = list(Path(models_dir).glob("*_G.pth"))
    if not model_files:
        print("❌ No model checkpoints found in fine-tuning directory!")
        print(f"📁 Checked: {models_dir}")
        print("⏳ The model is probably still training. Wait for the first checkpoint to be saved.")
        return
    
    # Find latest checkpoint
    latest_checkpoint = max(model_files, key=lambda x: int(x.stem.split('_')[0]))
    iteration = latest_checkpoint.stem.split('_')[0]
    
    print(f"🔍 Found latest checkpoint: {latest_checkpoint}")
    print(f"📊 Iteration: {iteration}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Model configuration (matching your training config)
    model_config = {
        'model_type': 'scunet_ngswin',
        'netG': {
            'net_type': 'scunet_ngswin',
            'in_nc': 1,
            'out_nc': 1,
            'config': [1,1,1,1,1,1,1],
            'dim': 32,
            'drop_path_rate': 0.0,
            'input_resolution': 512,
            'block_variant': 'conv_trans_nstb'
        },
        'scale': 1
    }
    
    try:
        # Setup model
        print(f"🔧 Loading model...")
        model = setup_model_and_checkpoint(
            model_config, str(latest_checkpoint), device
        )
        
        print(f"✅ Model loaded successfully!")
        
        # Test with UWSpine dataset
        print(f"🦴 Testing with UWSpine dataset...")
        
        results = process_uwspine_synthetic_transfer_dataset(
            model, uwspine_data_dir, output_dir, device, 
            f"conv_trans_nstb_iter_{iteration}"
        )
        
        if results:
            print(f"\n🎉 Testing completed successfully!")
            print(f"📁 Results saved to: {output_dir}")
            
            if 'average_psnr' in results:
                print(f"📊 Average PSNR: {results['average_psnr']:.4f} dB")
                print(f"📊 Average SSIM: {results['average_ssim']:.6f}")
            
        else:
            print(f"❌ Testing failed!")
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_current_finetuning_model()
