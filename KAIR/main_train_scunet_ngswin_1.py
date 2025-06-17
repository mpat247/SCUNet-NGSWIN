# Relative path: KAIR/main_train_scunet_ngswin_1.py
#
# training code for SCUNet‐NGSWIN
# --------------------------------------------
# adapted from KAIR/main_train_psnr.py
# --------------------------------------------
import os
import os.path
import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option

from data.select_dataset import define_Dataset
from models.select_model import define_Model

def main(json_path='options/train_scunet_ngswin_1.json'):
    # ----------------------------------------
    # Step 1: prepare options
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path,
                        help='Path to option JSON file.')
    parser.add_argument('--block_variant', type=str,
                        choices=['conv','conv_nstb','trans_nstb','conv_trans_nstb'],
                        help='Which SCUNet block to train')
    args = parser.parse_args()

    # parse JSON, inject chosen block if provided
    opt = option.parse(args.opt, is_train=True)
    if args.block_variant:
        opt['netG']['block_variant'] = args.block_variant

    # ----------------------------------------
    # create output folders
    # ----------------------------------------
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # resume from last checkpoints if any
    # ----------------------------------------
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimG, init_path_optimG = option.find_last_checkpoint(
        opt['path']['models'], net_type='optimizerG'
    )
    opt['path']['pretrained_optimizerG'] = init_path_optimG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimG)

    border = opt.get('scale', 0)

    # ----------------------------------------
    # save opt for reproducibility
    # ----------------------------------------
    option.save(opt)
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # ----------------------------------------
    # set seeds
    # ----------------------------------------
    seed = opt['train']['manual_seed'] or random.randint(1, 10000)
    print(f'Random seed: {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ----------------------------------------
    # Step 2: create dataloaders
    # ----------------------------------------
    for phase, ds_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(ds_opt)
            shuffle = ds_opt['dataloader_shuffle']
            train_loader = DataLoader(
                train_set,
                batch_size=ds_opt['dataloader_batch_size'],
                shuffle=shuffle,
                num_workers=ds_opt['dataloader_num_workers'],
                drop_last=True,
                pin_memory=True,
            )
            logger.info(f'Number of train images: {len(train_set):,d}, '
                        f'iters per epoch: {math.ceil(len(train_set)/ds_opt["dataloader_batch_size"]):,d}')
        elif phase == 'test':
            test_set = define_Dataset(ds_opt)
            test_loader = DataLoader(
                test_set,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                drop_last=False,
                pin_memory=True,
            )
        else:
            raise NotImplementedError(f"Phase [{phase}] is not recognized.")

    # ----------------------------------------
    # Step 3: initialize model
    # ----------------------------------------
    model = define_Model(opt)
    model.init_train()
    logger.info(model.info_network())
    logger.info(model.info_params())

    # ----------------------------------------
    # Step 4: main training loop
    # ----------------------------------------
    for epoch in range(1, 10**9):
        for train_data in train_loader:
            current_step += 1

            # 1) update lr
            model.update_learning_rate(current_step)
            # 2) feed
            model.feed_data(train_data)
            # 3) optimize
            model.optimize_parameters(current_step)

            # 4) logging training loss & metrics
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()
                msg = f'<epoch:{epoch:3d}, iter:{current_step:8,d}, lr:{model.current_learning_rate():.3e}> '
                for k, v in logs.items():
                    msg += f'{k}: {v:.3e} '
                logger.info(msg)

            # 5) save model
            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # 6) validation (PSNR + SSIM)
            if current_step % opt['train']['checkpoint_test'] == 0:
                avg_psnr, avg_ssim = 0.0, 0.0
                idx = 0
                for test_data in test_loader:
                    idx += 1
                    img_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, _ = os.path.splitext(img_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(test_data)
                    model.test()
                    visuals = model.current_visuals()

                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    # save output
                    save_path = os.path.join(img_dir, f'{img_name}_{current_step}.png')
                    util.imsave(E_img, save_path)

                    # compute PSNR & SSIM
                    psnr = util.calculate_psnr(E_img, H_img, border=border)
                    ssim = util.calculate_ssim(E_img, H_img, border=border)
                    logger.info(f'{idx:->4d}--> {img_name_ext:<12s} | PSNR: {psnr:.2f}dB | SSIM: {ssim:.4f}')

                    avg_psnr += psnr
                    avg_ssim += ssim

                avg_psnr /= idx
                avg_ssim /= idx
                logger.info(
                    f'<epoch:{epoch:3d}, iter:{current_step:8,d}, '
                    f'Avg PSNR: {avg_psnr:.2f}dB, Avg SSIM: {avg_ssim:.4f}\n'
                )

if __name__ == '__main__':
    main()
