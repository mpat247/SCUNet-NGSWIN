import os
import h5py
import numpy as np
import torch
import torch.utils.data as data
import logging
from datetime import datetime

# Create a logging directory and file
log_dir = os.path.join(os.path.dirname(__file__), 'data_loading_logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'run_1_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class DatasetSCUNetNGSWIN(data.Dataset):
    """
    SynDeepLesion sinogram interpolation dataset for SCUNet-NGswin.
    - Reads full-view sinogram (gt.h5) from each slice directory.
    - Generates a linearly-interpolated sparse-view input by sampling every Nth view
      (N = sparse_factor) and filling in the missing views by linear interpolation.
    """

    def __init__(self, opt):
        super(DatasetSCUNetNGSWIN, self).__init__()
        self.opt = opt
        self.phase = opt.get('phase', 'train')
        self.root = opt['dataroot_H']
        self.sparse_factor = opt.get('sparse_factor', 10)

        logging.info(f"Initializing DatasetSCUNetNGSWIN with root: {self.root} and sparse_factor: {self.sparse_factor}")

        # collect all slice directories
        self.slice_dirs = []
        logging.info(f"Checking directory: {self.root}")
        print(f"=== Debugging Directory Structure for {self.root} ===")
        for case in sorted(os.listdir(self.root)):
            case_dir = os.path.join(self.root, case)
            logging.debug(f"Processing case directory: {case_dir}")
            if not os.path.isdir(case_dir):
                logging.warning(f"Skipping invalid case directory: {case_dir}")
                continue
            for vol in sorted(os.listdir(case_dir)):
                vol_dir = os.path.join(case_dir, vol)
                logging.debug(f"Processing volume directory: {vol_dir}")
                if not os.path.isdir(vol_dir):
                    logging.warning(f"Skipping invalid volume directory: {vol_dir}")
                    continue
                for sl in sorted(os.listdir(vol_dir)):
                    sl_path = os.path.join(vol_dir, sl)
                    if os.path.isfile(sl_path) and sl.endswith('.h5'):
                        logging.debug(f"Found .h5 file: {sl_path}")
                        self.slice_dirs.append(sl_path)
                        logging.info(f"Valid .h5 file added: {sl_path}")
                    elif os.path.isdir(sl_path):
                        logging.debug(f"Found directory: {sl_path}")
                        gt_path = os.path.join(sl_path, 'gt.h5')
                        if os.path.exists(gt_path):
                            self.slice_dirs.append(sl_path)
                            logging.info(f"Valid slice directory added: {sl_path}")
                        else:
                            logging.warning(f"Missing gt.h5 in slice directory: {sl_path}")
                    else:
                        logging.debug(f"Skipping non-h5 file: {sl_path}, Type: {os.path.splitext(sl_path)[1]}")
        logging.info(f"Total valid slice directories or files found: {len(self.slice_dirs)}")
        print(f"=== End of Debugging Directory Structure ===")

        assert self.slice_dirs, f'No valid slices found under {self.root}'

    def __len__(self):
        logging.debug(f"Dataset length requested: {len(self.slice_dirs)}")
        return len(self.slice_dirs)

    def __getitem__(self, idx):
        slice_dir = self.slice_dirs[idx]
        gt_path = os.path.join(slice_dir, 'gt.h5')
        logging.debug(f"Loading data for slice directory: {slice_dir}")

        # --- load ground truth sinogram ---
        with h5py.File(gt_path, 'r') as f:
            logging.debug(f"Reading gt.h5 file: {gt_path}")
            H = f['sinogram'][:]  # shape (V, H, W)

        V = H.shape[0]
        logging.debug(f"Sinogram shape: {H.shape}, Total views: {V}")

        input_idxs = list(range(0, V, self.sparse_factor))
        if input_idxs[-1] != V - 1:
            input_idxs.append(V - 1)
        logging.debug(f"Input indices for sparse views: {input_idxs}")

        L = np.zeros_like(H)
        for v in range(V):
            if v in input_idxs:
                L[v] = H[v]
            else:
                low = max(i for i in input_idxs if i < v)
                high = min(i for i in input_idxs if i > v)
                w = (v - low) / (high - low)
                L[v] = (1 - w) * H[low] + w * H[high]
        logging.debug(f"Sparse-view sinogram generated for slice: {slice_dir}")

        L = torch.from_numpy(L).unsqueeze(0).float()
        H = torch.from_numpy(H).unsqueeze(0).float()

        logging.info(f"Data successfully loaded for slice: {slice_dir}")
        return {
            'L': L,
            'H': H,
            'L_path': slice_dir,
            'H_path': gt_path
        }
