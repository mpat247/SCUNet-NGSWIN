# -*- coding: utf-8 -*-
"""
Custom Dataset for Your Specific Data Format
--------------------------------------------
Adapt this template to match your dataset structure.
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset
# Import your data loading libraries (PIL, cv2, h5py, etc.)
# from PIL import Image
# import cv2
# import h5py


class DatasetCustom(Dataset):
    """
    Custom dataset for your specific data format.
    
    Expected opt structure:
    {
        "name": "your_dataset_name",
        "dataset_type": "custom",
        "dataroot_H": "/path/to/your/dataset",
        "input_format": "png",  # or "jpg", "h5", "npy", etc.
        "gt_format": "png",     # or "jpg", "h5", "npy", etc.
        "input_key": "input",   # for H5 files
        "gt_key": "gt",         # for H5 files
    }
    """
    
    def __init__(self, opt: dict):
        super().__init__()
        self.opt = opt
        self.root = opt['dataroot_H']
        self.input_format = opt.get('input_format', 'png')
        self.gt_format = opt.get('gt_format', 'png')
        
        # Collect all data pairs
        self.data_pairs = self._collect_data_pairs()
        
        print(f'Custom Dataset: {len(self.data_pairs)} pairs loaded from {self.root}')
    
    def _collect_data_pairs(self):
        """
        Collect all (input, ground_truth) file pairs.
        Adapt this method to match your data organization.
        """
        pairs = []
        
        # Example 1: Paired folders structure
        # dataset/
        # ├── input/
        # │   ├── img1.png
        # │   └── img2.png
        # └── gt/
        #     ├── img1.png
        #     └── img2.png
        
        # input_dir = os.path.join(self.root, 'input')
        # gt_dir = os.path.join(self.root, 'gt')
        # 
        # if os.path.exists(input_dir) and os.path.exists(gt_dir):
        #     for filename in os.listdir(input_dir):
        #         input_path = os.path.join(input_dir, filename)
        #         gt_path = os.path.join(gt_dir, filename)
        #         if os.path.exists(gt_path):
        #             pairs.append((input_path, gt_path))
        
        # Example 2: Single folder with naming convention
        # dataset/
        # ├── img1_input.png
        # ├── img1_gt.png
        # ├── img2_input.png
        # └── img2_gt.png
        
        # files = os.listdir(self.root)
        # input_files = [f for f in files if '_input.' in f]
        # for input_file in input_files:
        #     gt_file = input_file.replace('_input.', '_gt.')
        #     if gt_file in files:
        #         pairs.append((
        #             os.path.join(self.root, input_file),
        #             os.path.join(self.root, gt_file)
        #         ))
        
        # Example 3: Your actual implementation
        # Replace this with your specific data loading logic
        for root, dirs, files in os.walk(self.root):
            # Your data collection logic here
            pass
        
        return pairs
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx: int):
        input_path, gt_path = self.data_pairs[idx]
        
        # Load input data
        L = self._load_data(input_path, self.input_format)
        
        # Load ground truth data
        H = self._load_data(gt_path, self.gt_format)
        
        # Convert to torch tensors and ensure correct shape
        L = torch.from_numpy(L).float()
        H = torch.from_numpy(H).float()
        
        # Add channel dimension if grayscale
        if L.ndim == 2:
            L = L.unsqueeze(0)
        if H.ndim == 2:
            H = H.unsqueeze(0)
        
        # Resize to 416x416 to match training setup
        import torch.nn.functional as F
        L = L.unsqueeze(0)
        H = H.unsqueeze(0)
        L = F.interpolate(L, size=(416, 416), mode='bilinear', align_corners=False)
        H = F.interpolate(H, size=(416, 416), mode='bilinear', align_corners=False)
        L = L.squeeze(0)
        H = H.squeeze(0)
        
        return {
            'L': L,
            'H': H,
            'L_path': input_path,
            'H_path': gt_path
        }
    
    def _load_data(self, path: str, format: str):
        """Load data based on format."""
        if format in ['png', 'jpg', 'jpeg']:
            # For image files
            from PIL import Image
            img = Image.open(path).convert('L')  # Convert to grayscale
            return np.array(img)
        
        elif format == 'h5':
            # For H5 files
            import h5py
            with h5py.File(path, 'r') as f:
                key = self.opt.get('input_key', 'data')
                return f[key][:]
        
        elif format == 'npy':
            # For numpy files
            return np.load(path)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
