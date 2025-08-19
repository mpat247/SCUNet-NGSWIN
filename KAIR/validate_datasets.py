#!/usr/bin/env python3
"""
Comprehensive Dataset Validation Script
======================================

Validates all SCUNet-NGSWIN datasets to ensure proper loading, preprocessing, 
and data integrity for training and testing.
"""

import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import h5py
from pathlib import Path
import traceback

# Add KAIR to path
sys.path.append('/home/grad/mppatel/Documents/Project/SCUNet-NGSWIN/KAIR')

from data.dataset_scunet_ngswin import DatasetLICT, DatasetMACT, DatasetClinicalTrain, DatasetNIINCT
from data.select_dataset import define_Dataset

class DatasetValidator:
    """Comprehensive dataset validation"""
    
    def __init__(self):
        self.results = {}
        
    def validate_basic_properties(self, dataset, name):
        """Validate basic dataset properties"""
        print(f"\n=== VALIDATING {name.upper()} ===")
        results = {
            'name': name,
            'length': len(dataset),
            'samples_checked': 0,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            # Test first sample
            if len(dataset) > 0:
                sample = dataset[0]
                results['sample_keys'] = list(sample.keys())
                results['L_shape'] = sample['L'].shape
                results['H_shape'] = sample['H'].shape
                results['L_dtype'] = str(sample['L'].dtype)
                results['H_dtype'] = str(sample['H'].dtype)
                results['L_range'] = [float(sample['L'].min()), float(sample['L'].max())]
                results['H_range'] = [float(sample['H'].min()), float(sample['H'].max())]
                
                print(f"✓ Dataset length: {len(dataset)}")
                print(f"✓ Sample keys: {sample.keys()}")
                print(f"✓ L shape: {sample['L'].shape}, dtype: {sample['L'].dtype}")
                print(f"✓ H shape: {sample['H'].shape}, dtype: {sample['H'].dtype}")
                print(f"✓ L range: [{sample['L'].min():.4f}, {sample['L'].max():.4f}]")
                print(f"✓ H range: [{sample['H'].min():.4f}, {sample['H'].max():.4f}]")
                
                # Check for common issues
                if sample['L'].shape != torch.Size([1, 416, 416]):
                    results['warnings'].append(f"L shape {sample['L'].shape} != expected [1, 416, 416]")
                if sample['H'].shape != torch.Size([1, 416, 416]):
                    results['warnings'].append(f"H shape {sample['H'].shape} != expected [1, 416, 416]")
                if sample['L'].min() < 0 or sample['L'].max() > 1:
                    results['warnings'].append(f"L range [{sample['L'].min():.4f}, {sample['L'].max():.4f}] outside [0,1]")
                if sample['H'].min() < 0 or sample['H'].max() > 1:
                    results['warnings'].append(f"H range [{sample['H'].min():.4f}, {sample['H'].max():.4f}] outside [0,1]")
                if torch.equal(sample['L'], sample['H']):
                    results['warnings'].append("L and H are identical (no learning signal)")
                    
            else:
                results['errors'].append("Dataset is empty")
                
        except Exception as e:
            results['errors'].append(f"Basic validation failed: {str(e)}")
            
        self.results[name] = results
        return results
        
    def validate_multiple_samples(self, dataset, name, num_samples=20):
        """Check multiple samples for consistency"""
        print(f"\n--- Checking {num_samples} samples for consistency ---")
        results = self.results[name]
        
        if len(dataset) == 0:
            return
            
        # Check random samples
        np.random.seed(42)
        max_samples = min(num_samples, len(dataset))
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        
        L_ranges = []
        H_ranges = []
        shape_issues = 0
        range_issues = 0
        identical_count = 0
        
        for idx in indices:
            try:
                sample = dataset[idx]
                L, H = sample['L'], sample['H']
                
                L_ranges.append([float(L.min()), float(L.max())])
                H_ranges.append([float(H.min()), float(H.max())])
                
                # Count issues
                if L.shape != torch.Size([1, 416, 416]) or H.shape != torch.Size([1, 416, 416]):
                    shape_issues += 1
                if L.min() < 0 or L.max() > 1 or H.min() < 0 or H.max() > 1:
                    range_issues += 1
                if torch.equal(L, H):
                    identical_count += 1
                    
                results['samples_checked'] += 1
                
            except Exception as e:
                results['errors'].append(f"Sample {idx} failed: {str(e)}")
        
        # Statistics
        if L_ranges:
            L_mins, L_maxs = zip(*L_ranges)
            H_mins, H_maxs = zip(*H_ranges)
            
            results['stats'] = {
                'L_min_range': [min(L_mins), max(L_mins)],
                'L_max_range': [min(L_maxs), max(L_maxs)],
                'H_min_range': [min(H_mins), max(H_mins)],
                'H_max_range': [min(H_maxs), max(H_maxs)],
                'shape_issues': shape_issues,
                'range_issues': range_issues,
                'identical_count': identical_count
            }
            
            print(f"✓ Checked {len(indices)} samples")
            print(f"✓ L value range: [{min(L_mins):.4f}, {max(L_maxs):.4f}]")
            print(f"✓ H value range: [{min(H_mins):.4f}, {max(H_maxs):.4f}]")
            
            if shape_issues > 0:
                results['warnings'].append(f"{shape_issues} samples with wrong shapes")
            if range_issues > 0:
                results['warnings'].append(f"{range_issues} samples with values outside [0,1]")
            if identical_count > 0:
                results['warnings'].append(f"{identical_count} samples with identical L/H")
    
    def validate_dataloader(self, dataset, name, batch_size=2):
        """Test DataLoader integration"""
        print(f"\n--- Testing DataLoader integration ---")
        results = self.results[name]
        
        try:
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=0,
                drop_last=True
            )
            
            # Test one batch
            for batch in dataloader:
                results['batch_L_shape'] = list(batch['L'].shape)
                results['batch_H_shape'] = list(batch['H'].shape)
                results['batch_size_actual'] = batch['L'].shape[0]
                
                print(f"✓ DataLoader batch shape: L={batch['L'].shape}, H={batch['H'].shape}")
                break
                
        except Exception as e:
            results['errors'].append(f"DataLoader failed: {str(e)}")
    
    def validate_h5_files(self, dataset_path, name):
        """Validate raw H5 files"""
        print(f"\n--- Checking raw H5 files ---")
        results = self.results.get(name, {})
        
        if not os.path.exists(dataset_path):
            results['errors'] = results.get('errors', []) + [f"Dataset path does not exist: {dataset_path}"]
            return
            
        h5_files = []
        gt_files = []
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.h5'):
                    full_path = os.path.join(root, file)
                    if file.startswith('gt'):
                        gt_files.append(full_path)
                    else:
                        h5_files.append(full_path)
        
        print(f"✓ Found {len(h5_files)} slice H5 files")
        print(f"✓ Found {len(gt_files)} GT H5 files")
        
        # Check a few files
        if h5_files:
            try:
                with h5py.File(h5_files[0], 'r') as f:
                    keys = list(f.keys())
                    print(f"✓ Sample slice file keys: {keys}")
                    
            except Exception as e:
                results['errors'] = results.get('errors', []) + [f"H5 file read error: {str(e)}"]
        
        if gt_files:
            try:
                with h5py.File(gt_files[0], 'r') as f:
                    keys = list(f.keys())
                    print(f"✓ Sample GT file keys: {keys}")
                    
            except Exception as e:
                results['errors'] = results.get('errors', []) + [f"GT file read error: {str(e)}"]
    
    def print_summary(self):
        """Print validation summary"""
        print(f"\n{'='*60}")
        print("DATASET VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        for name, results in self.results.items():
            print(f"\n{name.upper()}:")
            print(f"  Length: {results.get('length', 'N/A')}")
            print(f"  Samples checked: {results.get('samples_checked', 0)}")
            
            if results.get('errors'):
                print(f"  ❌ ERRORS ({len(results['errors'])}):")
                for error in results['errors']:
                    print(f"    - {error}")
            else:
                print(f"  ✅ No errors")
            
            if results.get('warnings'):
                print(f"  ⚠️  WARNINGS ({len(results['warnings'])}):")
                for warning in results['warnings']:
                    print(f"    - {warning}")
            else:
                print(f"  ✅ No warnings")
            
            # Print key stats
            if 'L_range' in results:
                print(f"  📊 L range: [{results['L_range'][0]:.4f}, {results['L_range'][1]:.4f}]")
                print(f"  📊 H range: [{results['H_range'][0]:.4f}, {results['H_range'][1]:.4f}]")

def main():
    """Run complete dataset validation"""
    validator = DatasetValidator()
    
    # Dataset configurations
    datasets = [
        {
            'name': 'clinical_supervised',
            'class': DatasetClinicalTrain,
            'config': {'dataroot_H': '/home/Drive-D/clinical_metal_supervised/train'},
            'validate_h5': True
        },
        {
            'name': 'syndeeplesion_train', 
            'class': DatasetLICT,
            'config': {'dataroot_H': '/home/Drive-D/SynDeepLesion/train_640geo'},
            'validate_h5': True
        },
        {
            'name': 'syndeeplesion_test',
            'class': DatasetLICT, 
            'config': {'dataroot_H': '/home/Drive-D/SynDeepLesion/test_640geo'},
            'validate_h5': False
        },
        {
            'name': 'clinical_nifti',
            'class': DatasetNIINCT,
            'config': {'dataroot_H': '/home/Drive-D/clinical_metal'},
            'validate_h5': False
        }
    ]
    
    print("Starting comprehensive dataset validation...")
    
    for dataset_info in datasets:
        try:
            # Create dataset
            dataset = dataset_info['class'](dataset_info['config'])
            
            # Run validations
            validator.validate_basic_properties(dataset, dataset_info['name'])
            validator.validate_multiple_samples(dataset, dataset_info['name'])
            validator.validate_dataloader(dataset, dataset_info['name'])
            
            if dataset_info.get('validate_h5'):
                validator.validate_h5_files(dataset_info['config']['dataroot_H'], dataset_info['name'])
                
        except Exception as e:
            print(f"❌ Failed to validate {dataset_info['name']}: {str(e)}")
            traceback.print_exc()
    
    # Test validation split
    print(f"\n{'='*60}")
    print("TESTING VALIDATION SPLIT")
    print(f"{'='*60}")
    
    try:
        clinical_dataset = DatasetClinicalTrain({'dataroot_H': '/home/Drive-D/clinical_metal_supervised/train'})
        val_split = 0.1
        total_size = len(clinical_dataset)
        val_size = int(val_split * total_size)
        train_size = total_size - val_size
        
        indices = list(range(total_size))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_dataset = Subset(clinical_dataset, indices[val_size:])
        val_dataset = Subset(clinical_dataset, indices[:val_size])
        
        print(f"✅ Total: {total_size}, Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        # Test both splits
        train_sample = train_dataset[0]
        val_sample = val_dataset[0]
        print(f"✅ Train sample shape: {train_sample['L'].shape}")
        print(f"✅ Val sample shape: {val_sample['L'].shape}")
        
    except Exception as e:
        print(f"❌ Validation split test failed: {str(e)}")
    
    # Print final summary
    validator.print_summary()
    
    print(f"\n{'='*60}")
    print("VALIDATION COMPLETE!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
