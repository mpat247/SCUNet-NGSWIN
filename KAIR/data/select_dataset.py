# -*- coding: utf-8 -*-
"""
Dataset selector for KAIR
-------------------------
• Turns the `dataset_type` string (declared in the option-JSON) into an
  actual Python class and instantiates it.
• The lookup table below makes aliases explicit and keeps the code clean.
• Only one line to change when you add a new dataset.

Author: adapted by <you> from the original KAIR helper.
"""

import importlib


# ──────────────────────────  ALIAS TABLE  ─────────────────────────────────
#  key (str)        →  (module_path, class_name)
#
#  *You can list multiple aliases pointing to the same class.*
DATASET_MAP = {
    # low-quality input only
    'l'                       : ('data.dataset_l',           'DatasetL'),
    'low-quality'             : ('data.dataset_l',           'DatasetL'),
    'input-only'              : ('data.dataset_l',           'DatasetL'),

    # denoising families
    'dncnn'                   : ('data.dataset_dncnn',       'DatasetDnCNN'),
    'denoising'               : ('data.dataset_dncnn',       'DatasetDnCNN'),
    'dnpatch'                 : ('data.dataset_dnpatch',     'DatasetDnPatch'),
    'ffdnet'                  : ('data.dataset_ffdnet',      'DatasetFFDNet'),
    'denoising-noiselevel'    : ('data.dataset_ffdnet',      'DatasetFFDNet'),
    'fdncnn'                  : ('data.dataset_fdncnn',      'DatasetFDnCNN'),
    'denoising-noiselevelmap' : ('data.dataset_fdncnn',      'DatasetFDnCNN'),

    # super-resolution
    'sr'                      : ('data.dataset_sr',          'DatasetSR'),
    'super-resolution'        : ('data.dataset_sr',          'DatasetSR'),
    'srmd'                    : ('data.dataset_srmd',        'DatasetSRMD'),
    'dpsr'                    : ('data.dataset_dpsr',        'DatasetDPSR'),
    'dnsr'                    : ('data.dataset_dpsr',        'DatasetDPSR'),
    'usrnet'                  : ('data.dataset_usrnet',      'DatasetUSRNet'),
    'usrgan'                  : ('data.dataset_usrnet',      'DatasetUSRNet'),
    'bsrnet'                  : ('data.dataset_blindsr',     'DatasetBlindSR'),
    'bsrgan'                  : ('data.dataset_blindsr',     'DatasetBlindSR'),
    'blindsr'                 : ('data.dataset_blindsr',     'DatasetBlindSR'),

    # JPEG deblocking
    'jpeg'                    : ('data.dataset_jpeg',        'DatasetJPEG'),

    # video restoration / VFI
    'videorecurrenttraindataset'                : ('data.dataset_video_train', 'VideoRecurrentTrainDataset'),
    'videorecurrenttrainnonblinddenoisingdataset': ('data.dataset_video_train', 'VideoRecurrentTrainNonblindDenoisingDataset'),
    'videorecurrenttrainvimeodataset'           : ('data.dataset_video_train', 'VideoRecurrentTrainVimeoDataset'),
    'videorecurrenttrainvimeovfidataset'        : ('data.dataset_video_train', 'VideoRecurrentTrainVimeoVFIDataset'),
    'videorecurrenttestdataset'                 : ('data.dataset_video_test',  'VideoRecurrentTestDataset'),
    'singlevideorecurrenttestdataset'           : ('data.dataset_video_test',  'SingleVideoRecurrentTestDataset'),
    'videotestvimeo90kdataset'                  : ('data.dataset_video_test',  'VideoTestVimeo90KDataset'),
    'vfi_davis'                                 : ('data.dataset_video_test',  'VFI_DAVIS'),
    'vfi_ucf101'                                : ('data.dataset_video_test',  'VFI_UCF101'),
    'vfi_vid4'                                  : ('data.dataset_video_test',  'VFI_Vid4'),

    # ─── DeepLesion CT datasets ─────────────────────────────────────────────
    # only CT images: low‐energy (LI_CT) and medium‐energy (ma_CT)
    'li_ct'                  : ('data.dataset_scunet_ngswin', 'DatasetLICT'),
    'ma_ct'                  : ('data.dataset_scunet_ngswin', 'DatasetMACT'),
    'nii_ct'                 : ('data.dataset_scunet_ngswin', 'DatasetNIINCT'),
    'clinical_supervised'    : ('data.dataset_scunet_ngswin', 'DatasetLICT'),  # Your new supervised clinical dataset
    'clinical_training'      : ('data.dataset_scunet_ngswin', 'DatasetClinicalTrain'),

    # ─── Custom datasets ─────────────────────────────────────────────────────
    'custom'                 : ('data.dataset_custom',        'DatasetCustom'),

    # plain folders
    'plain'                  : ('data.dataset_plain',         'DatasetPlain'),
    'plainpatch'             : ('data.dataset_plainpatch',    'DatasetPlainPatch'),
}


# ────────────────────────── helper: dynamic import ────────────────────────
def _import_module_class(module_path: str, class_name: str):
    module = importlib.import_module(module_path)
    try:
        cls = getattr(module, class_name)
    except AttributeError as e:
        raise ImportError(f'Class "{class_name}" not found in module "{module_path}"') from e
    return cls


# ────────────────────────── public API ────────────────────────────────────
def define_Dataset(dataset_opt: dict):
    """
    dataset_opt must contain at least:
        { "name": "...", "dataset_type": "li_ct" | "ma_ct" | ... }

    Returns an instantiated PyTorch Dataset.
    """
    dataset_type = dataset_opt['dataset_type'].lower()
    if dataset_type not in DATASET_MAP:
        raise NotImplementedError(f'Dataset type "{dataset_type}" is not registered.')

    module_path, class_name = DATASET_MAP[dataset_type]
    DatasetClass = _import_module_class(module_path, class_name)

    dataset = DatasetClass(dataset_opt)
    print(f'Dataset [{dataset.__class__.__name__} – {dataset_opt["name"]}] is created.')
    return dataset
