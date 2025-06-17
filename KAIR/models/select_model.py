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
        from models.network_scunet_ngswin import SCUNet
        cfg = opt.get('netG', {})
        # Unpack exactly what SCUNet.__init__ expects:
        m = SCUNet(
            in_nc=cfg.get('in_nc', 1),
            config=cfg.get('config', [2, 2, 2, 2, 2, 2, 2]),
            dim=cfg.get('dim', 64),
            drop_path_rate=cfg.get('drop_path_rate', 0.0),
            input_resolution=cfg.get('input_resolution', 256),
            block_variant=cfg.get('block_variant', 'conv')
        )

    else:
        raise NotImplementedError(f"Model [{model_name}] is not defined.")

    print(f'Training model [{m.__class__.__name__}] is created.')
    return m
