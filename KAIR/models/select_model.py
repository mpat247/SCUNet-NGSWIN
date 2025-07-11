# KAIR/models/select_model.py
"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""

def define_Model(opt):
    """
    Instantiate the training model based on opt['model'].
    For 'scunet_ngswin', unpacks opt['netG'] into SCUNet constructor.
    """
    model_name = opt['model']  # e.g. "plain", "gan", "scunet_ngswin", etc.

    if model_name == 'plain':
        from models.model_plain import ModelPlain as M
        m = M(opt)

    elif model_name == 'plain2':
        from models.model_plain2 import ModelPlain2 as M
        m = M(opt)

    elif model_name == 'plain4':
        from models.model_plain4 import ModelPlain4 as M
        m = M(opt)

    elif model_name == 'gan':
        from models.model_gan import ModelGAN as M
        m = M(opt)

    elif model_name == 'vrt':
        from models.model_vrt import ModelVRT as M
        m = M(opt)

    elif model_name == 'scunet_ngswin':
        from models.model_scunet_ngswin import ModelSCUNetNGSwin as M
        m = M(opt)

    else:
        raise NotImplementedError(f"Model [{model_name}] is not defined.")

    print(f'Training model [{m.__class__.__name__}] is created.')
    return m
