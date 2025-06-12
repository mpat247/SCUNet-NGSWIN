import os.path
import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

'''
# --------------------------------------------
# training code for SCUNet-NGswin (PSNR oriented)
# --------------------------------------------
'''

def main(json_path='options/train_scunet_ngswin.json'):
    # Step--1: prepare options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', action='store_true', help='Use distributed training')
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)
    opt['dist'] = args.dist

    # distributed settings
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    # create necessary directories
    if opt['rank'] == 0:
        util.mkdirs((p for k, p in opt['path'].items() if 'pretrained' not in k))

    # resume training paths
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    init_iter_optG, init_path_optG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    opt['path']['pretrained_optimizerG'] = init_path_optG
    current_step = max(init_iter_G, init_iter_E, init_iter_optG)
    border = opt['scale']

    # save options
    if opt['rank'] == 0:
        option.save(opt)
    opt = option.dict_to_nonedict(opt)

    # configure logger
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if opt['rank'] == 0:
        logger.info(f'Random seed: {seed}')
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    # create dataloaders
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info(f'Number of train images: {len(train_set)}, iters: {train_size}')
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], seed=seed)
                train_loader = DataLoader(
                    train_set,
                    batch_size=dataset_opt['dataloader_batch_size'] // opt['world_size'],
                    sampler=train_sampler,
                    num_workers=dataset_opt['dataloader_num_workers'] // opt['world_size'],
                    drop_last=True,
                    pin_memory=True,
                )
            else:
                train_loader = DataLoader(
                    train_set,
                    batch_size=dataset_opt['dataloader_batch_size'],
                    shuffle=dataset_opt['dataloader_shuffle'],
                    num_workers=dataset_opt['dataloader_num_workers'],
                    drop_last=True,
                    pin_memory=True,
                )
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError(f'Phase [{phase}] is not recognized.')

    # define model
    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network()); logger.info(model.info_params())

    # training loop
    if opt['rank'] == 0:
        logger.info('Start training...')
    for epoch in range(1000000):  # keep running
        if opt['dist']:
            train_sampler.set_epoch(epoch + seed)
        for i, train_data in enumerate(train_loader):
            current_step += 1
            # update learning rate
            model.update_learning_rate(current_step)
            # feed data and optimize
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            # print logs
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items(): message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)
            # save model
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.'); model.save(current_step)
            # test and report PSNR
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
                avg_psnr = 0.0; idx = 0
                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, _ = os.path.splitext(image_name_ext)
                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)
                    model.feed_data(test_data); model.test()
                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E']); H_img = util.tensor2uint(visuals['H'])
                    save_path = os.path.join(img_dir, f'{img_name}_{current_step}.png')
                    util.imsave(E_img, save_path)
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)
                    logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))
                    avg_psnr += current_psnr
                avg_psnr /= idx
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR: {:<.2f}dB'.format(epoch, current_step, avg_psnr))

if __name__ == '__main__':
    main()
