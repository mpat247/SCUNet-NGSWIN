# -*- coding: utf-8 -*-
"""
Dataset  ➜  SynDeepLesion  ➜  SCUNet-NGswin (CT-only)
-----------------------------------------------------
Supports two CT modes:
  • LI_CT → image  (direct)
  • MA_CT → image  (direct)

Each subclass finds all numeric .h5 slice files under the root
(excluding gt.h5) and, for each:
  • loads L from that slice’s `input_key`
  • loads H from the sibling gt.h5’s `gt_key`
  • skips any slice-folder missing a gt.h5 (with a warning)
"""
import os
import re
import h5py
import logging
import torch
import torch.nn.functional as F
import nibabel as nib
from datetime import datetime
from torch.utils.data import Dataset

# ───────────────────────  logging  ─────────────────────────────────────────
_log_dir = os.path.join(os.path.dirname(__file__), 'data_loading_logs')
os.makedirs(_log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(_log_dir, f'run_{datetime.now():%Y%m%d_%H%M%S}.txt'),
    level=logging.DEBUG,
    format='%(asctime)s  %(levelname)-8s  %(message)s'
)

class BaseCTDataset(Dataset):
    """
    Base CT dataset:
      1) Walks `dataroot_H` for all .h5 files except those starting with 'gt'
      2) Filters out any slice-folder without a sibling gt.h5
      3) For each remaining slice.h5:
         - L ← slice.h5[self.input_key]
         - H ← slice-folder/gt.h5[self.gt_key]
    """
    def __init__(self, opt: dict, input_key: str, gt_key: str):
        super().__init__()
        self.root      = opt['dataroot_H']
        self.input_key = input_key
        self.gt_key    = gt_key
        self.filelist  = opt.get('filelist', None)

        logging.info(
            f'Init {self.__class__.__name__} | root={self.root} | '
            f'input="{self.input_key}" → gt="{self.gt_key}"'
        )

        # 1) collect all numeric slice files (exclude any named starting with 'gt')
        if self.filelist:
            with open(self.filelist, 'r') as f:
                rels = [l.strip() for l in f if l.strip()]
            all_slices = [
                rel if os.path.isabs(rel) else os.path.join(self.root, rel)
                for rel in rels
            ]
        else:
            all_slices = []
            for dirpath, _, fns in os.walk(self.root):
                for fn in fns:
                    # Option A: skip any gt*.h5 file
                    if fn.endswith('.h5') and not fn.lower().startswith('gt'):
                        all_slices.append(os.path.join(dirpath, fn))
                    # Option B: only include purely numeric names
                    # if re.fullmatch(r'\d+\.h5', fn):
                    #     all_slices.append(os.path.join(dirpath, fn))

        all_slices.sort()
        assert all_slices, f'No slice .h5 files found under {self.root}'
        logging.info(f'  ➜  {len(all_slices)} slice-files found')

        # 2) filter only those with a same-folder gt.h5
        valid, skipped = [], []
        for in_path in all_slices:
            gt_path = os.path.join(os.path.dirname(in_path), 'gt.h5')
            if os.path.exists(gt_path):
                valid.append(in_path)
            else:
                skipped.append(in_path)

        if skipped:
            logging.warning(
                f"Skipping {len(skipped)} slice(s) with no sibling gt.h5. "
                f"Examples:\n  " + "\n  ".join(skipped[:5]) +
                ("" if len(skipped) <= 5 else "\n  …")
            )

        assert valid, f'No valid slice/GT pairs under {self.root}'
        self.input_files = valid
        logging.info(f'  ➜  {len(self.input_files)} valid slices retained')

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx: int):
        in_path = self.input_files[idx]
        gt_path = os.path.join(os.path.dirname(in_path), 'gt.h5')

        # by construction, gt_path exists
        with h5py.File(in_path, 'r') as f_in, h5py.File(gt_path, 'r') as f_gt:
            # sanity‐check keys
            if self.input_key not in f_in:
                raise KeyError(
                    f"Key '{self.input_key}' not in {in_path}. "
                    f"Available: {list(f_in.keys())}"
                )
            if self.gt_key not in f_gt:
                raise KeyError(
                    f"Key '{self.gt_key}' not in {gt_path}. "
                    f"Available: {list(f_gt.keys())}"
                )

            L_np = f_in[self.input_key][:]
            H_np = f_gt[self.gt_key][:]

        # to torch.Tensor, add channel dim and normalize to [0,1]
        # H5 stores float32 in [0,255] (written from uint8). Normalize here for model/metrics.
        L = torch.from_numpy(L_np).unsqueeze(0).float().div(255.0)
        H = torch.from_numpy(H_np).unsqueeze(0).float().div(255.0)

        # ─── RESIZE to 416×416 to match training resolution ───
        L = L.unsqueeze(0)
        H = H.unsqueeze(0)
        L = F.interpolate(L, size=(416, 416), mode='bilinear', align_corners=False)
        H = F.interpolate(H, size=(416, 416), mode='bilinear', align_corners=False)
        L = L.squeeze(0)
        H = H.squeeze(0)

        return {'L': L, 'H': H, 'L_path': in_path, 'H_path': gt_path}

    def __repr__(self):
        return (
            f'{self.__class__.__name__}'
            f'(n={len(self)}, input="{self.input_key}", gt="{self.gt_key}")'
        )

class DatasetLICT(BaseCTDataset):
    """Low‐energy CT → full‐energy CT"""
    def __init__(self, opt):
        super().__init__(opt, input_key='LI_CT', gt_key='image')

class DatasetMACT(BaseCTDataset):
    """Medium‐energy CT → full‐energy CT"""
    def __init__(self, opt):
        super().__init__(opt, input_key='ma_CT', gt_key='image')

class DatasetNIINCT(Dataset):
    """Clinical Metal CT dataset: loads single .nii files for inference (no GT available)."""
    def __init__(self, opt):
        super().__init__()
        self.root = opt['dataroot_H']
        self.input_files = []
        
        # Find all .nii files in the folder
        for dirpath, _, fns in os.walk(self.root):
            for fn in fns:
                if fn.endswith('.nii') and not fn.endswith('.nii.gz'):
                    self.input_files.append(os.path.join(dirpath, fn))
        
        self.input_files.sort()
        assert self.input_files, f'No .nii files found under {self.root}'
        logging.info(f'  ➜  {len(self.input_files)} .nii files found (inference-only mode)')

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        in_path = self.input_files[idx]
        
        # Load .nii file
        L_np = nib.load(in_path).get_fdata().astype('float32')
        
        # Handle 3D volumes by extracting middle slice
        if len(L_np.shape) == 3:
            # For 3D volume [H, W, D], take middle slice along z-axis
            middle_slice = L_np.shape[2] // 2
            L_np = L_np[:, :, middle_slice]
            logging.info(f"Extracted slice {middle_slice} from 3D volume {L_np.shape}")
        
        # Use same image as both input and "GT" (for pipeline compatibility)
        # Note: PSNR/SSIM will compare enhanced vs original (not meaningful, but won't crash)
        H_np = L_np.copy()
        
        # Normalize to [0, 1] range
        L_np = (L_np - L_np.min()) / (L_np.max() - L_np.min() + 1e-8)
        H_np = (H_np - H_np.min()) / (H_np.max() - H_np.min() + 1e-8)
        
        # Add channel dim and convert to torch
        L = torch.from_numpy(L_np).unsqueeze(0).float()
        H = torch.from_numpy(H_np).unsqueeze(0).float()
        
        # Resize to 416x416 (same as .h5 preprocessing)
        L = L.unsqueeze(0)
        H = H.unsqueeze(0)
        L = F.interpolate(L, size=(416, 416), mode='bilinear', align_corners=False)
        H = F.interpolate(H, size=(416, 416), mode='bilinear', align_corners=False)
        L = L.squeeze(0)
        H = H.squeeze(0)
        
        return {'L': L, 'H': H, 'L_path': in_path, 'H_path': in_path}

# ───────────────────────  New: Clinical training pipeline  ───────────────
class DatasetClinicalTrain(BaseCTDataset):
        """Supervised clinical CT training dataset (our per-slice H5 layout).

        Expects folders like:
            <dataroot>/<case>/<slice_id>.h5  (dataset: 'LI_CT')
            <dataroot>/<case>/gt.h5          (dataset: 'image')
        """
        def __init__(self, opt):
                super().__init__(opt, input_key='LI_CT', gt_key='image')
