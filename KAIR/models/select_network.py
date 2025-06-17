# /KAIR/models/select_network.py
import functools
import torch
from torch.nn import init


"""
# --------------------------------------------
# select the network of G, D and F
# --------------------------------------------
"""


# --------------------------------------------
# Generator, netG, G
# --------------------------------------------
def define_G(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']


    # ----------------------------------------
    # denoising task
    # ----------------------------------------

    # ----------------------------------------
    # DnCNN
    # ----------------------------------------
    if net_type == 'dncnn':
        from models.network_dncnn import DnCNN as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],  # total number of conv layers
                   act_mode=opt_net['act_mode'])

    # ----------------------------------------
    # Flexible DnCNN
    # ----------------------------------------
    elif net_type == 'fdncnn':
        from models.network_dncnn import FDnCNN as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],  # total number of conv layers
                   act_mode=opt_net['act_mode'])

    # ----------------------------------------
    # FFDNet
    # ----------------------------------------
    elif net_type == 'ffdnet':
        from models.network_ffdnet import FFDNet as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   act_mode=opt_net['act_mode'])

    # ----------------------------------------
    # super-resolution task
    # ----------------------------------------

    # ----------------------------------------
    # SRMD
    # ----------------------------------------
    elif net_type == 'srmd':
        from models.network_srmd import SRMD as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   upscale=opt_net['scale'],
                   act_mode=opt_net['act_mode'],
                   upsample_mode=opt_net['upsample_mode'])

    # ----------------------------------------
    # super-resolver prior of DPSR
    # ----------------------------------------
    elif net_type == 'dpsr':
        from models.network_dpsr import MSRResNet_prior as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   upscale=opt_net['scale'],
                   act_mode=opt_net['act_mode'],
                   upsample_mode=opt_net['upsample_mode'])

    # ----------------------------------------
    # modified SRResNet v0.0
    # ----------------------------------------
    elif net_type == 'msrresnet0':
        from models.network_msrresnet import MSRResNet0 as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   upscale=opt_net['scale'],
                   act_mode=opt_net['act_mode'],
                   upsample_mode=opt_net['upsample_mode'])

    # ----------------------------------------
    # modified SRResNet v0.1
    # ----------------------------------------
    elif net_type == 'msrresnet1':
        from models.network_msrresnet import MSRResNet1 as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   upscale=opt_net['scale'],
                   act_mode=opt_net['act_mode'],
                   upsample_mode=opt_net['upsample_mode'])

    # ----------------------------------------
    # RRDB
    # ----------------------------------------
    elif net_type == 'rrdb':
        from models.network_rrdb import RRDB as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   gc=opt_net['gc'],
                   upscale=opt_net['scale'],
                   act_mode=opt_net['act_mode'],
                   upsample_mode=opt_net['upsample_mode'])

    # ----------------------------------------
    # RRDBNet
    # ----------------------------------------
    elif net_type == 'rrdbnet':
        from models.network_rrdbnet import RRDBNet as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nf=opt_net['nf'],
                   nb=opt_net['nb'],
                   gc=opt_net['gc'],
                   sf=opt_net['scale'])

    # ----------------------------------------
    # IMDN
    # ----------------------------------------
    elif net_type == 'imdn':
        from models.network_imdn import IMDN as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   upscale=opt_net['scale'],
                   act_mode=opt_net['act_mode'],
                   upsample_mode=opt_net['upsample_mode'])

    # ----------------------------------------
    # USRNet
    # ----------------------------------------
    elif net_type == 'usrnet':
        from models.network_usrnet import USRNet as net
        netG = net(n_iter=opt_net['n_iter'],
                   h_nc=opt_net['h_nc'],
                   in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   act_mode=opt_net['act_mode'],
                   downsample_mode=opt_net['downsample_mode'],
                   upsample_mode=opt_net['upsample_mode']
                   )

    # ----------------------------------------
    # Deep Residual U-Net (drunet)
    # ----------------------------------------
    elif net_type == 'drunet':
        from models.network_unet import UNetRes as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   act_mode=opt_net['act_mode'],
                   downsample_mode=opt_net['downsample_mode'],
                   upsample_mode=opt_net['upsample_mode'],
                   bias=opt_net['bias'])

    # ----------------------------------------
    # SwinIR
    # ----------------------------------------
    elif net_type == 'swinir':
        from models.network_swinir import SwinIR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])

    # ----------------------------------------
    # VRT
    # ----------------------------------------
    elif net_type == 'vrt':
        from models.network_vrt import VRT as net
        netG = net(upscale=opt_net['upscale'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   depths=opt_net['depths'],
                   indep_reconsts=opt_net['indep_reconsts'],
                   embed_dims=opt_net['embed_dims'],
                   num_heads=opt_net['num_heads'],
                   spynet_path=opt_net['spynet_path'],
                   pa_frames=opt_net['pa_frames'],
                   deformable_groups=opt_net['deformable_groups'],
                   nonblind_denoising=opt_net['nonblind_denoising'],
                   use_checkpoint_attn=opt_net['use_checkpoint_attn'],
                   use_checkpoint_ffn=opt_net['use_checkpoint_ffn'],
                   no_checkpoint_attn_blocks=opt_net['no_checkpoint_attn_blocks'],
                   no_checkpoint_ffn_blocks=opt_net['no_checkpoint_ffn_blocks'])

    # ----------------------------------------
    # RVRT
    # ----------------------------------------
    elif net_type == 'rvrt':
        from models.network_rvrt import RVRT as net
        netG = net(upscale=opt_net['upscale'],
                   clip_size=opt_net['clip_size'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   num_blocks=opt_net['num_blocks'],
                   depths=opt_net['depths'],
                   embed_dims=opt_net['embed_dims'],
                   num_heads=opt_net['num_heads'],
                   inputconv_groups=opt_net['inputconv_groups'],
                   spynet_path=opt_net['spynet_path'],
                   deformable_groups=opt_net['deformable_groups'],
                   attention_heads=opt_net['attention_heads'],
                   attention_window=opt_net['attention_window'],
                   nonblind_denoising=opt_net['nonblind_denoising'],
                   use_checkpoint_attn=opt_net['use_checkpoint_attn'],
                   use_checkpoint_ffn=opt_net['use_checkpoint_ffn'],
                   no_checkpoint_attn_blocks=opt_net['no_checkpoint_attn_blocks'],
                   no_checkpoint_ffn_blocks=opt_net['no_checkpoint_ffn_blocks'],
                   cpu_cache_length=opt_net['cpu_cache_length'])

    # ----------------------------------------
    # SCUNet-NGswin
    # ----------------------------------------
    elif net_type == 'scunet_ngswin':
        from models.network_scunet_ngswin import SCUNet as net
        netG = net(
            in_nc=opt_net.get('in_nc', 1),
            config=opt_net.get('config', [2,2,2,2,2,2,2]),
            dim=opt_net.get('nc', 64),
            drop_path_rate=opt_net.get('drop_path_rate', 0.0),
            input_resolution=opt_net.get('input_resolution', 256),
            block_variant=opt_net.get('block_variant', 'conv')      # <-- added here
        )

    # ----------------------------------------
    # others
    # ----------------------------------------
    else:
        raise NotImplementedError('netG [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if opt['is_train']:
        init_weights(netG,
                     init_type=opt_net['init_type'],
                     init_bn_type=opt_net['init_bn_type'],
                     gain=opt_net['init_gain'])

    return netG


# --------------------------------------------
# Discriminator, netD, D
# --------------------------------------------
def define_D(opt):
    opt_net = opt['netD']
    net_type = opt_net['net_type']

    if net_type == 'discriminator_vgg_96':
        from models.network_discriminator import Discriminator_VGG_96 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    elif net_type == 'discriminator_vgg_128':
        from models.network_discriminator import Discriminator_VGG_128 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    elif net_type == 'discriminator_vgg_192':
        from models.network_discriminator import Discriminator_VGG_192 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    elif net_type == 'discriminator_vgg_128_SN':
        from models.network_discriminator import Discriminator_VGG_128_SN as discriminator
        netD = discriminator()

    elif net_type == 'discriminator_patchgan':
        from models.network_discriminator import Discriminator_PatchGAN as discriminator
        netD = discriminator(input_nc=opt_net['in_nc'],
                             ndf=opt_net['base_nc'],
                             n_layers=opt_net['n_layers'],
                             norm_type=opt_net['norm_type'])

    elif net_type == 'discriminator_unet':
        from models.network_discriminator import Discriminator_UNet as discriminator
        netD = discriminator(input_nc=opt_net['in_nc'],
                             ndf=opt_net['base_nc'])

    else:
        raise NotImplementedError('netD [{:s}] is not found.'.format(net_type))

    init_weights(netD,
                 init_type=opt_net['init_type'],
                 init_bn_type=opt_net['init_bn_type'],
                 gain=opt_net['init_gain'])
    return netD


# --------------------------------------------
# VGGfeature, netF, F
# --------------------------------------------
def define_F(opt, use_bn=False):
    device = torch.device('cuda' if opt['gpu_ids'] else 'cpu')
    from models.network_feature import VGGFeatureExtractor
    feature_layer = 49 if use_bn else 34
    netF = VGGFeatureExtractor(feature_layer=feature_layer,
                               use_bn=use_bn,
                               use_input_norm=True,
                               device=device)
    netF.eval()
    return netF


"""
# --------------------------------------------
# weights initialization
# --------------------------------------------
"""
def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            # ... same as before ...
            pass
        elif classname.find('BatchNorm2d') != -1:
            # ... same as before ...
            pass

    if init_type not in ['default', 'none']:
        print(f'Initialization method [{init_type} + {init_bn_type}], gain is [{gain:.2f}]')
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network definition!')
