# KAIR/models/model_scunet_ngswin.py
# Minimal trainer wrapper for SCUNet-NGSwin
from collections import OrderedDict

from models.model_plain import ModelPlain     # the generic pixel-loss trainer
from models.network_scunet_ngswin import SCUNet_NGSwin   # <- rename here if you changed class name


class ModelSCUNetNGSwin(ModelPlain):
    """
    Thin wrapper so KAIR's training loop can use SCUNet-NGSwin.
    Inherits everything (optimiser, schedulers, logging, etc.) from ModelPlain.
    """

    def __init__(self, opt):
        super().__init__(opt)     # this already builds netG via define_G(), etc.
        # Nothing else is needed unless you want custom losses.

    # ---- OPTIONAL: keep plain L1/L2 loss from ModelPlain ----
    # If you'd like special losses, override define_loss() here.


# Convenience alias (so “scunet_ngswin” still prints nicely)
Model = ModelSCUNetNGSwin
