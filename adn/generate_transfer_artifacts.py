#!/usr/bin/env python3
"""
ADN Neural Artifact Transfer for UWSpineCT Dataset
Generates synthesized_metal_transfer folders using trained ADN model
"""

import os
import os.path as path
import numpy as np
import torch
import yaml
import random
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

from adn.models import ADNTest
from adn.utils import read_dir
from googledrivedownloader import download_file_from_google_drive


# ADN model specs for spineweb (same as demo.py)
MODEL_SPECS = {
    "g_type": "adn",
    "d_type": "nlayer",
    "adn": {
        "input_ch": 1,
        "base_ch": 64,
        "num_down": 2,
        "num_residual": 4,
        "num_sides": 3,
        "down_norm": "instance",
        "res_norm": "instance",
        "up_norm": "layer",
        "fuse": True,
        "shared_decoder": False
    },
    "nlayer": {
        "input_nc": 1,
        "ndf": 64,
        "n_layers": 2,
        "norm_layer": "instance"
    }
}


def get_pretrained_model():
    """Download and load pre-trained ADN spineweb model"""
    model_path = "runs/spineweb/spineweb_39.pt"
    gdrive_id = "1eF-6YTJYlVa7fVMk8n9yQssAqzrhLO1T"
    
    if not path.isfile(model_path):
        print(f"📥 Downloading pre-trained ADN model...")
        os.makedirs(path.dirname(model_path), exist_ok=True)
        try:
            download_file_from_google_drive(
                file_id=gdrive_id, dest_path=model_path, unzip=False, showsize=True)
            print(f"✅ Model downloaded to {model_path}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("Please download spineweb model manually from Google Drive")
            return None
    
    return model_path


def normalize_for_adn(image):
    """Normalize CT image for ADN input (expects -1 to 1 range)"""
    # UWSpineCT uses HU values, normalize to ADN's expected range
    VALUE_RANGE = [-1000.0, 2000.0]  # HU range for spine CT
    
    # Clip values to expected range
    image = np.clip(image, VALUE_RANGE[0], VALUE_RANGE[1])
    
    # Normalize to [-1, 1]
    image = (image - VALUE_RANGE[0]) / (VALUE_RANGE[1] - VALUE_RANGE[0])
    image = image * 2.0 - 1.0
    
    return image


def denormalize_from_adn(image):
    """Convert ADN output back to HU values"""
    VALUE_RANGE = [-1000.0, 2000.0]
    
    # Convert from [-1, 1] to [0, 1]
    image = (image + 1.0) * 0.5
    
    # Scale back to HU range
    image = image * (VALUE_RANGE[1] - VALUE_RANGE[0]) + VALUE_RANGE[0]
    
    return image


def load_image_for_adn(image_file):
    """Load .npy image and prepare for ADN"""
    image = np.load(image_file)
    image = normalize_for_adn(image)
    image = torch.FloatTensor(image[np.newaxis, np.newaxis, ...])
    return image


def pair_images(no_metal_files, metal_artifact_files):
    """Create pairs between no_metal and metal_artifact images for transfer"""
    # Shuffle metal_artifact files to create diverse pairings
    metal_donors = metal_artifact_files.copy()
    random.shuffle(metal_donors)
    
    pairs = []
    for i, clean_file in enumerate(no_metal_files):
        # Cycle through metal_artifact images as donors
        donor_file = metal_donors[i % len(metal_donors)]
        pairs.append((clean_file, donor_file))
    
    return pairs


def generate_artifacts_for_split(split_name, config, model):
    """Generate transferred artifacts for train or test split"""
    base_dir = config['dataset_dir']
    
    # Get file lists
    no_metal_dir = path.join(base_dir, split_name, 'no_metal')
    metal_artifact_dir = path.join(base_dir, split_name, 'metal_artifact')
    
    no_metal_files = read_dir(no_metal_dir, predicate=lambda x: x.endswith('.npy'))
    metal_artifact_files = read_dir(metal_artifact_dir, predicate=lambda x: x.endswith('.npy'))
    
    print(f"🔍 {split_name.upper()} - Found {len(no_metal_files)} clean images, {len(metal_artifact_files)} artifact donors")
    
    # Create output directory
    output_dir = path.join(base_dir, split_name, 'synthesized_metal_transfer')
    os.makedirs(output_dir, exist_ok=True)
    
    # Pair images for artifact transfer
    image_pairs = pair_images(no_metal_files, metal_artifact_files)
    
    processed = 0
    for clean_file, donor_file in tqdm(image_pairs, desc=f"Generating {split_name} transferred artifacts"):
        try:
            # Load images
            img_clean = load_image_for_adn(clean_file)  # high quality (no artifact)
            img_donor = load_image_for_adn(donor_file)  # low quality (with artifact)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                img_clean = img_clean.cuda()
                img_donor = img_donor.cuda()
            
            # Perform artifact transfer: donor's artifacts → clean image
            with torch.no_grad():
                model.evaluate(img_donor, img_clean)
                transferred_artifact = model.pred_hl  # clean image with transferred artifacts
            
            # Convert back to numpy and HU values
            transferred_np = transferred_artifact.detach().cpu().numpy()[0, 0]
            transferred_hu = denormalize_from_adn(transferred_np)
            
            # Generate output filename (same as clean image)
            clean_basename = path.basename(clean_file)[:-4]  # remove .npy
            output_file = path.join(output_dir, clean_basename + '.npy')
            thumbnail_file = path.join(output_dir, clean_basename + '.png')
            
            # Save synthetic artifact image
            np.save(output_file, transferred_hu)
            
            # Create thumbnail for visualization
            thumbnail = (transferred_hu - transferred_hu.min()) / (transferred_hu.max() - transferred_hu.min())
            thumbnail = (thumbnail * 255).astype(np.uint8)
            Image.fromarray(thumbnail).save(thumbnail_file)
            
            processed += 1
            
        except Exception as e:
            print(f"⚠️  Error processing {path.basename(clean_file)}: {e}")
            continue
    
    print(f"✅ {split_name.upper()} - Successfully generated {processed}/{len(image_pairs)} transferred artifacts")
    return processed


def main():
    print("🎭 ADN Neural Artifact Transfer for UWSpineCT Dataset")
    print("=" * 60)
    
    # Load configuration
    config_file = "config/dataset.yaml"
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)['uwspinect']
    
    # Get pre-trained model
    model_path = get_pretrained_model()
    if not model_path:
        return
    
    # Create ADN model
    print("🤖 Loading ADN model...")
    model = ADNTest(**MODEL_SPECS)
    if torch.cuda.is_available():
        model.cuda()
        print("🚀 Using GPU acceleration")
    else:
        print("⚠️  Using CPU (will be slower)")
    
    # Load pre-trained weights
    model.resume(model_path)
    print("✅ ADN model loaded successfully")
    
    # Set random seed for reproducible pairings
    random.seed(42)
    
    # Generate artifacts for both splits
    train_count = generate_artifacts_for_split('train', config, model)
    test_count = generate_artifacts_for_split('test', config, model)
    
    print("\n🎉 Artifact Transfer Complete!")
    print(f"📊 Generated {train_count} train + {test_count} test = {train_count + test_count} total synthetic artifacts")
    print(f"📁 Results saved to: {config['dataset_dir']}")
    print("   ├── train/synthesized_metal_transfer/")
    print("   └── test/synthesized_metal_transfer/")
    print("\n🔬 Ready for SCUNet-NGSWIN training with synthetic artifact pairs!")


if __name__ == "__main__":
    main()
