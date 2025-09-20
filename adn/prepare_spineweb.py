import os
import os.path as path
import yaml
import torch
import numpy as np
import SimpleITK as sitk
import random
import shutil

from tqdm import tqdm
from PIL import Image
from adn.utils import read_dir, get_connected_components
from collections import defaultdict
from torchvision.utils import make_grid


def make_thumbnails(images):
    images = torch.tensor(np.array(images).astype(float))[:, np.newaxis, ...]
    images = (images - images.min()) / (images.max() - images.min())
    num_rows = int(len(images) ** 0.5)
    image = make_grid(
        images, nrow=images.shape[0] // num_rows, normalize=False)
    image = image.numpy().transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    return image


if __name__ == "__main__":
    config_file = "config/dataset.yaml"
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)['uwspinect']

    image_size = config['image_size']
    if type(image_size) is not list: image_size = [image_size] * 2
    thumbnail_size = config['thumbnail_size']
    if type(thumbnail_size) is not list: thumbnail_size = [thumbnail_size] * 2

    # Process train data (spine-1 through spine-5)
    print("🔄 Processing TRAINING data (spine-1 through spine-5)...")
    train_volumes = []
    for spine_folder in ['spine-1', 'spine-2', 'spine-3', 'spine-4', 'spine-5']:
        spine_path = path.join(config['raw_dir'], spine_folder)
        if path.exists(spine_path):
            patient_dirs = read_dir(spine_path, predicate=lambda x: "patient" in x, recursive=False)
            for patient_dir in patient_dirs:
                volume_files = read_dir(patient_dir,
                    predicate=lambda x: x.endswith("mhd") or x.endswith("nii.gz"), recursive=True)
                for volume_file in volume_files:
                    patient_name = path.basename(patient_dir)
                    volume_name = path.basename(volume_file).split(".")[0]
                    train_volumes.append((volume_file, patient_name, volume_name, "train"))
    
    # Process test data (spine-test-data)
    print("🔄 Processing TEST data (spine-test-data)...")
    test_volumes = []
    test_data_path = path.join(config['raw_dir'], 'spine-test-data')
    if path.exists(test_data_path):
        test_files = read_dir(test_data_path, predicate=lambda x: x.endswith("nii.gz"), recursive=False)
        for volume_file in test_files:
            volume_name = path.basename(volume_file).split(".")[0]
            test_volumes.append((volume_file, "test", volume_name, "test"))
    
    all_volumes = train_volumes + test_volumes
    print(f"🔍 Found {len(train_volumes)} train volumes + {len(test_volumes)} test volumes = {len(all_volumes)} total")

    for volume_file, patient_name, volume_name, split_type in tqdm(all_volumes, desc="Processing UWSpineCT volumes"):
        volume_obj = sitk.ReadImage(volume_file)
        volume = sitk.GetArrayFromImage(volume_obj)

        index = 0
        for image in tqdm(volume, desc=f"Processing {patient_name}_{volume_name}", leave=False):
            image_type = "no_metal"

            # Check if the image has metal artifacts
            if image.max() > config["max_hu"][1]:
                points = np.array(np.where(image > config["max_hu"][1])).T
                points = set(tuple(p) for p in points)
                components = get_connected_components(points)
                max_area = max(len(c) for c in components)

                if max_area > config["connected_area"]: 
                    image_type = "metal_artifact"
                else: 
                    continue
            elif image.max() > config["max_hu"][0]: 
                continue

            # Create folder structure: train/test -> metal_artifact/no_metal
            output_dir = path.join(config["dataset_dir"], split_type, image_type)
            if not path.isdir(output_dir): os.makedirs(output_dir)

            image = Image.fromarray(image).resize(image_size)
            image = np.array(image)

            thumbnail = (image - image.min()) / (image.max() - image.min())
            thumbnail = (thumbnail * 255).astype(np.uint8)

            image_name = "{}_{}_{:03d}".format(patient_name, volume_name, index)
            image_file = path.join(output_dir, image_name + ".npy")
            thumbnail_file = path.join(output_dir, image_name + ".png")
            
            np.save(image_file, image)
            Image.fromarray(thumbnail).save(thumbnail_file)
            index += 1

    print("✅ Processing complete! UWSpineCT data organized as:")
    print(f"📁 Train data: {path.join(config['dataset_dir'], 'train')}")
    print(f"   - Metal artifacts: {path.join(config['dataset_dir'], 'train', 'metal_artifact')}")
    print(f"   - No metal: {path.join(config['dataset_dir'], 'train', 'no_metal')}")
    print(f"📁 Test data: {path.join(config['dataset_dir'], 'test')}")
    print(f"   - Metal artifacts: {path.join(config['dataset_dir'], 'test', 'metal_artifact')}")
    print(f"   - No metal: {path.join(config['dataset_dir'], 'test', 'no_metal')}")
    print(f"� Next step: Use ADN to create synthesized_metal folders in both train/ and test/")
