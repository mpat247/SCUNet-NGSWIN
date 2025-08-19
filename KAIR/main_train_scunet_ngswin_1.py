# -*- coding: utf-8 -*-
"""
Relative path: KAIR/main_train_scunet_ngswin_1.py

Training script for SCUNet-NGSWIN with optional pipeline override
and PSNR-based checkpointing, plus per-epoch training/validation sample counting.
"""
import os
import os.path
import math
import argparse
import random
import numpy as np
import logging
import sys
from torch.utils.data import DataLoader, Subset
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option

from data.select_dataset import define_Dataset
from models.select_model import define_Model

def main(json_path='options/train_scunet_ngswin_1.json'):
    # Clear CUDA cache to prevent OOM errors    
    torch.cuda.empty_cache()

    # ----------------------------------------
    # Step 1: parse args & setup device
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt', type=str, default=json_path,
        help='Path to option JSON file.')
    parser.add_argument(
        '--block_variant', type=str,
        choices=['conv','conv_nstb','trans_nstb','conv_trans_nstb'],
        help='Which SCUNet block to train')
    parser.add_argument(
        '--pipeline', type=str,
        choices=['li_ct','ma_ct'],
        help='Override dataset_type in the JSON (li_ct or ma_ct)')
    parser.add_argument(
        '--resume', type=str, choices=['latest','psnr'], default='latest',
        help='Which checkpoint to resume from: "latest" or "psnr"')
    parser.add_argument(
        '--val_split', type=float, default=0.1,
        help='Validation split ratio from training data (default: 0.1 = 10%)')
    parser.add_argument(
        '--test', action='store_true',
        help='Skip training and run only final test evaluation on existing checkpoints')
    parser.add_argument(
        '--testset', type=str, default='test',
        help='Which test dataset to use: "test" (default) or "test_2"')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ----------------------------------------
    # Step 2: load & tweak raw options
    # ----------------------------------------
    raw_opt = option.parse(args.opt, is_train=True)
    if args.block_variant:
        raw_opt['netG']['block_variant'] = args.block_variant
    if args.pipeline:
        for phase in ('train', 'test'):
            if phase in raw_opt.get('datasets', {}):
                ds = raw_opt['datasets'][phase]
                ds['dataset_type'] = args.pipeline
                if 'name' in ds:
                    ds['name'] = f"{ds['name']}_{args.pipeline}"

    # ----------------------------------------
    # create all output dirs
    # ----------------------------------------
    util.mkdirs(
        (path for key, path in raw_opt['path'].items()
         if 'pretrained' not in key)
    )

    # ----------------------------------------
    # resume from checkpoints if present
    # ----------------------------------------
    models_dir = raw_opt['path']['models']
    if args.resume == 'psnr':
        raw_opt['path']['pretrained_netG']       = os.path.join(models_dir, 'psnr_G.pth')
        raw_opt['path']['pretrained_netE']       = os.path.join(models_dir, 'psnr_E.pth')
        raw_opt['path']['pretrained_optimizerG'] = os.path.join(models_dir, 'optimizerG_psnr.pth')
        init_iter_G = init_iter_E = init_iter_O = 0
        print("Resuming from PSNR checkpoint")
    else:
        init_iter_G, init_path_G = option.find_last_checkpoint(
            models_dir, net_type='G'
        )
        init_iter_E, init_path_E = option.find_last_checkpoint(
            models_dir, net_type='E'
        )
        init_iter_O, init_path_O = option.find_last_checkpoint(
            models_dir, net_type='optimizerG'
        )
        raw_opt['path']['pretrained_netG']       = init_path_G
        raw_opt['path']['pretrained_netE']       = init_path_E
        raw_opt['path']['pretrained_optimizerG'] = init_path_O
        init_iter_G = init_iter_G or 0
        init_iter_E = init_iter_E or 0
        init_iter_O = init_iter_O or 0
        print(f"Resuming from latest checkpoint at iter {max(init_iter_G, init_iter_E, init_iter_O)}")

    current_step = max(init_iter_G, init_iter_E, init_iter_O)
    border = raw_opt.get('scale', 0)

    # ----------------------------------------
    # save raw_opt JSON, then convert to non-edict
    # ----------------------------------------
    option.save(raw_opt)
    opt = option.dict_to_nonedict(raw_opt)
    opt['device'] = device  # inject device

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(
        logger_name,
        os.path.join(opt['path']['log'], logger_name + '.log')
    )
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))
    logger.info(f"Using device: {device}")

    # ----------------------------------------
    # set random seeds
    # ----------------------------------------
    seed = opt['train'].get('manual_seed', None) or random.randint(1, 10000)
    print(f"Random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # ----------------------------------------
    # Step 3: create dataloaders with validation split
    # ----------------------------------------
    for phase, ds_opt in opt['datasets'].items():
        if phase == 'train':
            # Load full training dataset
            full_train_set = define_Dataset(ds_opt)
            total_samples = len(full_train_set)
            
            # Create train/validation split
            if args.val_split > 0.0:
                # Calculate split sizes
                val_size = int(total_samples * args.val_split)
                train_size = total_samples - val_size
                
                # Create indices for reproducible split
                indices = list(range(total_samples))
                random.shuffle(indices)  # Use the same seed as set above
                
                train_indices = indices[:train_size]
                val_indices = indices[train_size:]
                
                # Create subset datasets
                train_set = Subset(full_train_set, train_indices)
                val_set = Subset(full_train_set, val_indices)
                
                # Create data loaders
                train_loader = DataLoader(
                    train_set,
                    batch_size=ds_opt['dataloader_batch_size'],
                    shuffle=ds_opt.get('dataloader_shuffle', True),
                    num_workers=ds_opt.get('dataloader_num_workers', 0),
                    drop_last=True,
                    pin_memory=(device.type == 'cuda'),
                )
                
                val_loader = DataLoader(
                    val_set,
                    batch_size=1,
                    shuffle=False,
                    num_workers=1,
                    drop_last=False,
                    pin_memory=(device.type == 'cuda'),
                )
                
                logger.info(
                    f'Total samples: {total_samples:,d} | '
                    f'Train: {len(train_set):,d} ({(1-args.val_split)*100:.1f}%) | '
                    f'Validation: {len(val_set):,d} ({args.val_split*100:.1f}%)'
                )
                logger.info(
                    f'Training iters per epoch: '
                    f'{math.ceil(len(train_set)/ds_opt["dataloader_batch_size"]):,d}'
                )
                
            else:
                # No validation split - use all training data
                train_set = full_train_set
                val_set = None
                val_loader = None
                
                train_loader = DataLoader(
                    train_set,
                    batch_size=ds_opt['dataloader_batch_size'],
                    shuffle=ds_opt.get('dataloader_shuffle', True),
                    num_workers=ds_opt.get('dataloader_num_workers', 0),
                    drop_last=True,
                    pin_memory=(device.type == 'cuda'),
                )
                
                logger.info(
                    f'Number of train samples: {len(train_set):,d}, '
                    f'iters per epoch: '
                    f'{math.ceil(len(train_set)/ds_opt["dataloader_batch_size"]):,d}'
                )
                logger.info('No validation split - using all training data for training')
                
        elif phase == 'test' or phase == 'test_2' or phase == 'test_3':
            if phase == args.testset:
                test_set = define_Dataset(ds_opt)
                test_loader = DataLoader(
                    test_set,
                    batch_size=1,
                    shuffle=False,
                    num_workers=1,
                    drop_last=False,
                    pin_memory=(device.type == 'cuda'),
                )
                logger.info(f'Test set samples: {len(test_set):,d}')
        else:
            raise NotImplementedError(f"Phase [{phase}] is not recognized.")

    # ----------------------------------------
    # Step 4: initialize model
    # ----------------------------------------
    model = define_Model(opt)
    model.init_train()
    logger.info(model.info_network())
    logger.info(model.info_params())

    # ----------------------------------------
    # TensorBoard setup
    # ----------------------------------------
    # Get block variant from JSON config or command line argument
    block_variant = args.block_variant or opt.get('netG', {}).get('block_variant', 'default')
    pipeline = args.pipeline or opt.get('datasets', {}).get('train', {}).get('dataset_type', 'default')
    
    # Include validation split information in experiment name
    val_split_str = f"_val{args.val_split:.1f}" if args.val_split > 0.0 else "_noval"
    experiment_name = f"SCUNet_{block_variant}_{pipeline}{val_split_str}"
    tb_log_dir = os.path.join(opt['path']['log'], 'tensorboard', experiment_name)
    writer = SummaryWriter(log_dir=tb_log_dir)
    logger.info(f"TensorBoard logging to: {tb_log_dir}")
    logger.info(f"Experiment name: {experiment_name}")
    logger.info(f"Validation split: {args.val_split:.1%} of training data")

    # ----------------------------------------
    # Early stopping setup (by PSNR) - COMMENTED OUT
    # ----------------------------------------
    # patience = opt['train'].get('early_stopping_patience', None)
    # if patience is not None:
    #     best_psnr = -float('inf')
    #     no_improve_count = 0
    #     if args.val_split > 0.0:
    #         logger.info(f"Early stopping enabled with patience={patience} (using validation PSNR)")
    #     else:
    #         logger.info(f"Early stopping disabled - no validation set available. Training will run for full 100 epochs.")
    #         patience = None  # Disable early stopping when no validation set
    
    # Early stopping disabled - training will run for full 100 epochs
    logger.info("Early stopping is disabled. Training will run for full 100 epochs.")

    # ----------------------------------------
    # Check if test-only mode is requested
    # ----------------------------------------
    if args.test:
        logger.info("="*80)
        logger.info("TEST-ONLY MODE: Skipping training, running final test evaluation")
        logger.info("="*80)
        # Jump directly to final test evaluation
        # Set current_step to a reasonable value for logging
        current_step = max(init_iter_G, init_iter_E, init_iter_O)
        epoch = 100  # Pretend we completed training
        # Skip to final test evaluation section
        # (We'll add a function or jump to that section)
        
        # Load test dataset
        if 'test' in opt['datasets']:
            test_dataset_opt = opt['datasets'][args.testset]
            test_set = define_Dataset(test_dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                   shuffle=False, num_workers=1,
                                   drop_last=False, pin_memory=True)
            logger.info(f'Number of test images: {len(test_set)}')
        else:
            logger.error(f"No test dataset configuration found for {args.testset} in JSON!")
            return
            
        # Jump to final test evaluation
        goto_final_test_evaluation = True
    else:
        goto_final_test_evaluation = False
        logger.info("Starting training...")

    # ----------------------------------------
    # Step 5: main training loop
    # ----------------------------------------
    if not goto_final_test_evaluation:
        # Calculate starting epoch based on current checkpoint iteration
        ds_opt_train = opt['datasets']['train']
        batch_size = ds_opt_train['dataloader_batch_size']
        train_dataset_size = len(train_set)
        iters_per_epoch = math.ceil(train_dataset_size / batch_size)
        start_epoch = max(1, (current_step // iters_per_epoch) + 1)
        
        logger.info(f'Training dataset size: {train_dataset_size:,d}')
        logger.info(f'Iterations per epoch: {iters_per_epoch:,d}')
        logger.info(f'Current step: {current_step:,d}')
        logger.info(f'Starting from epoch: {start_epoch}')
        
        for epoch in range(start_epoch, 101):
            # ---- train ----
            train_count = 0
            for train_data in tqdm(train_loader,
                                    desc=f"Epoch {epoch} [Train]",
                                    leave=False):
                batch_size = train_data['L'].size(0)
                train_count += batch_size
                current_step += 1

                # 1) update lr
                model.update_learning_rate(current_step)
                # 2) feed data
                model.feed_data(train_data)
                # 3) optimize
                model.optimize_parameters(current_step)

                # 4) logging
                if current_step % opt['train']['checkpoint_print'] == 0:
                    logs = model.current_log()
                    msg = (
                        f'<epoch:{epoch:3d}, iter:{current_step:8,d}, '
                        f'lr:{model.current_learning_rate():.3e}> '
                    )
                    for k, v in logs.items():
                        msg += f'{k}: {v:.3e} '
                    logger.info(msg)

                    # TensorBoard logging for training metrics
                    current_lr = model.current_learning_rate()
                    writer.add_scalar('Training/Learning_Rate', current_lr, current_step)
                    for k, v in logs.items():
                        writer.add_scalar(f'Training/{k}', v, current_step)
                    
                    # Flush logs and TensorBoard for immediate updates
                    for handler in logger.handlers:
                        if hasattr(handler, 'flush'):
                            handler.flush()
                    writer.flush()

                # 5) save regular checkpoint
                if current_step % opt['train']['checkpoint_save'] == 0:
                    logger.info('Saving the model.')
                    model.save(current_step)

                # 6) validation evaluation
                if current_step % opt['train']['checkpoint_test'] == 0:
                    # ---- Validation on training split (if available) ----
                    if val_loader is not None:
                        val_count = 0
                        avg_val_psnr, avg_val_ssim = 0.0, 0.0
                        val_idx = 0
                        
                        model.netG.eval()
                        with torch.no_grad():
                            for val_data in tqdm(val_loader,
                                               desc=f"Iter {current_step} [Val]",
                                               leave=False):
                                val_idx += 1
                                val_count += val_data['L'].size(0)
                                model.feed_data(val_data)
                                model.test()
                                visuals = model.current_visuals()

                                E_img = util.tensor2uint(visuals['E'])
                                H_img = util.tensor2uint(visuals['H'])

                                psnr = util.calculate_psnr(E_img, H_img, border=border)
                                ssim = util.calculate_ssim(E_img, H_img, border=border)
                                avg_val_psnr += psnr
                                avg_val_ssim += ssim

                        avg_val_psnr /= val_idx
                        avg_val_ssim /= val_idx
                        logger.info(
                            f'<epoch:{epoch:3d}, iter:{current_step:8,d}, '
                            f'Val PSNR: {avg_val_psnr:.2f}dB, Val SSIM: {avg_val_ssim:.4f}'
                        )

                        # TensorBoard logging for validation metrics
                        writer.add_scalar('Validation/PSNR', avg_val_psnr, current_step)
                        writer.add_scalar('Validation/SSIM', avg_val_ssim, current_step)
                        writer.add_scalar('Validation/Epoch', epoch, current_step)
                        
                        # Log training vs validation comparison for overfitting detection
                        if logs:
                            for k, v in logs.items():
                                if 'loss' in k.lower():
                                    writer.add_scalars('Loss_Comparison', 
                                                     {'Training': v, 'Validation_PSNR_Inverse': 50.0 - avg_val_psnr}, 
                                                     current_step)

                        # report validation counts
                        logger.info(f'Training samples this epoch: {train_count}')
                        logger.info(f'Validation samples this check: {val_count}')
                        
                        model.netG.train()  # Return to training mode

                        # Use validation PSNR for early stopping and checkpoint saving
                        current_psnr = avg_val_psnr
                        metric_type = "Validation"
                        
                    else:
                        # No validation set available - we'll use test set for early stopping at the end
                        # For now, just save the model and continue
                        logger.info(f'Training samples this epoch: {train_count}')
                        logger.info('No validation set - skipping validation evaluation')
                        
                        model.netG.train()  # Ensure we're in training mode
                        current_psnr = None
                        metric_type = None

                    # 7) save PSNR checkpoint (overwrite previous)
                    logger.info('Saving PSNR checkpoint.')
                    model.save('psnr')
                    
                    # Flush logs and TensorBoard for immediate updates
                    for handler in logger.handlers:
                        if hasattr(handler, 'flush'):
                            handler.flush()
                    writer.flush()

                # ---- early stopping check (COMMENTED OUT) ----
                # if patience is not None and current_psnr is not None:
                #     if current_psnr > best_psnr:
                #         best_psnr = current_psnr
                #         no_improve_count = 0
                #         logger.info(f"New best {metric_type} PSNR: {best_psnr:.4f}")
                #         # Log best PSNR updates
                #         writer.add_scalar(f'{metric_type}/Best_PSNR', best_psnr, current_step)
                #         writer.add_scalar('Early_Stopping/No_Improve_Count', no_improve_count, current_step)
                #     else:
                #         no_improve_count += 1
                #         logger.info(
                #             f"{metric_type} PSNR did not improve "
                #             f"({no_improve_count}/{patience})"
                #         )
                #         writer.add_scalar('Early_Stopping/No_Improve_Count', no_improve_count, current_step)
                #         if no_improve_count >= patience:
                #             logger.info(
                #                 f"Early stopping triggered: "
                #                 f"no {metric_type} PSNR improvement for {patience} validations."
                #             )
                #             writer.close()  # Close TensorBoard writer before exiting
                #             return

    # ----------------------------------------
    # Final Test Evaluation (after all epochs complete or in test-only mode)
    # ----------------------------------------
    # If test-only, test_loader is already defined above. If not, define it here.
    if not args.test:
        if 'test' in opt['datasets']:
            test_dataset_opt = opt['datasets']['test']
            test_set = define_Dataset(test_dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                   shuffle=False, num_workers=1,
                                   drop_last=False, pin_memory=True)
            logger.info(f'Number of test images: {len(test_set)}')
        else:
            logger.error("No test dataset configuration found in JSON!")
            return

    logger.info("="*80)
    logger.info("FINAL TEST EVALUATION - After 100 epochs completed")
    logger.info("="*80)
    
    # Load the best PSNR checkpoint for final evaluation
    logger.info("Loading best PSNR checkpoint for final test evaluation...")
    
    # Test both PSNR checkpoints if enabled
    test_both = opt['train'].get('test_both_checkpoints', False)
    save_every_nth = opt['train'].get('test_save_every_nth', 250)
    
    if test_both:
        # Test both psnr_E.pth and psnr_G.pth
        checkpoints_to_test = [
            ('psnr_E.pth', 'PSNR_E'),
            ('psnr_G.pth', 'PSNR_G')
        ]
    else:
        # Default: only test psnr_G.pth
        checkpoints_to_test = [('psnr_G.pth', 'PSNR_G')]
    
    final_results = {}
    
    for checkpoint_file, checkpoint_name in checkpoints_to_test:
        logger.info("-"*60)
        logger.info(f"Testing checkpoint: {checkpoint_name} ({checkpoint_file})")
        logger.info("-"*60)
        
        # Load the specific checkpoint
        checkpoint_path = os.path.join(models_dir, checkpoint_file)
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            # Fix argument order for load_network: (load_path, network, ...)
            model.load_network(checkpoint_path, model.netG, strict=True)
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}, using current model")
        
        # Test evaluation for this checkpoint
        test_count = 0
        avg_test_psnr, avg_test_ssim = 0.0, 0.0
        test_idx = 0
        
        # Create directory for this checkpoint's test images
        testset_suffix = f"_{args.testset}" if args.testset != 'test' else ''
        checkpoint_img_dir = os.path.join(opt['path']['images'], f'final_test_{checkpoint_name.lower()}{testset_suffix}')
        util.mkdir(checkpoint_img_dir)
        
        model.netG.eval()
        with torch.no_grad():
            for test_data in tqdm(test_loader,
                                  desc=f"Testing {checkpoint_name}",
                                  leave=True):
                test_idx += 1
                test_count += test_data['L'].size(0)
                model.feed_data(test_data)
                model.test()
                visuals = model.current_visuals()

                E_img = util.tensor2uint(visuals['E'])
                L_img = util.tensor2uint(visuals['L'])
                H_img = util.tensor2uint(visuals['H'])

                img_name_ext = os.path.basename(test_data['L_path'][0])
                img_name, _  = os.path.splitext(img_name_ext)
                
                # Detect if this is inference-only mode (clinical data without GT)
                is_inference_only = args.testset == 'test_2'
                
                if not is_inference_only:
                    # Normal mode: calculate PSNR/SSIM with real GT
                    psnr = util.calculate_psnr(E_img, H_img, border=border)
                    ssim = util.calculate_ssim(E_img, H_img, border=border)
                    avg_test_psnr += psnr
                    avg_test_ssim += ssim
                else:
                    # Inference-only mode: skip meaningful PSNR/SSIM calculation
                    psnr = None
                    ssim = None
                
                # Save every nth image with detailed before/after naming
                if test_idx % save_every_nth == 0:
                    if is_inference_only:
                        # Clinical inference mode: save individual before/after + comparison
                        # Save original (before)
                        original_path = os.path.join(checkpoint_img_dir, f'{img_name}_original.png')
                        util.imsave(L_img, original_path)
                        
                        # Save enhanced (after)
                        enhanced_path = os.path.join(checkpoint_img_dir, f'{img_name}_enhanced.png')
                        util.imsave(E_img, enhanced_path)
                        
                        # Save side-by-side comparison
                        comparison_img = np.concatenate([L_img, E_img], axis=1)
                        comparison_path = os.path.join(checkpoint_img_dir, f'{img_name}_comparison.png')
                        util.imsave(comparison_img, comparison_path)
                        
                        logger.info(f'Saved clinical test {test_idx}: {img_name}')
                        logger.info(f'  Original: {original_path}')
                        logger.info(f'  Enhanced: {enhanced_path}')
                        logger.info(f'  Comparison: {comparison_path}')
                    else:
                        # Normal test mode: save L|E|H comparison
                        comparison_img = np.concatenate([L_img, E_img, H_img], axis=1)
                        save_path = os.path.join(checkpoint_img_dir, 
                                               f'test_{test_idx:06d}_{checkpoint_name}_{img_name}.png')
                        util.imsave(comparison_img, save_path)
                        logger.info(f'Saved test image {test_idx}: {save_path} | PSNR: {psnr:.2f}dB | SSIM: {ssim:.4f}')

        # Calculate averages only for non-inference mode
        if test_idx > 0 and not (args.testset == 'test_2'):
            avg_test_psnr /= test_idx
            avg_test_ssim /= test_idx
        else:
            # Inference-only mode: no meaningful PSNR/SSIM
            avg_test_psnr = None
            avg_test_ssim = None
        
        # Store results for this checkpoint
        final_results[checkpoint_name] = {
            'psnr': avg_test_psnr,
            'ssim': avg_test_ssim,
            'samples': test_count,
            'images_saved': test_idx // save_every_nth,
            'inference_only': args.testset == 'test_2'
        }
        
        logger.info("-"*60)
        if args.testset == 'test_2':
            logger.info(f'{checkpoint_name} CLINICAL INFERENCE RESULTS:')
            logger.info(f'Processed {test_count} clinical images')
            logger.info(f'Enhanced images saved with before/after comparison')
        else:
            logger.info(f'{checkpoint_name} RESULTS:')
            logger.info(f'Avg PSNR: {avg_test_psnr:.4f}dB, Avg SSIM: {avg_test_ssim:.6f}')
        logger.info(f'Images saved: {test_idx // save_every_nth} (every {save_every_nth}th)')
        logger.info(f'Test images directory: {checkpoint_img_dir}')
        logger.info("-"*60)

        # TensorBoard logging for this checkpoint
        if avg_test_psnr is not None:
            writer.add_scalar(f'Final_Test_{checkpoint_name}/PSNR', avg_test_psnr, current_step)
            writer.add_scalar(f'Final_Test_{checkpoint_name}/SSIM', avg_test_ssim, current_step)
    
    # Final summary
    logger.info("="*80)
    logger.info("FINAL TEST SUMMARY - ALL CHECKPOINTS")
    logger.info("="*80)
    
    for checkpoint_name, results in final_results.items():
        if results.get('inference_only', False):
            # Clinical inference mode - no PSNR/SSIM available
            logger.info(
                f'{checkpoint_name}: Clinical Inference Mode - '
                f'Processed {results["samples"]} images, '
                f'Saved {results["images_saved"]} enhanced images'
            )
        else:
            # Normal mode with PSNR/SSIM metrics
            logger.info(
                f'{checkpoint_name}: PSNR={results["psnr"]:.4f}dB, '
                f'SSIM={results["ssim"]:.6f}, Samples={results["samples"]}, '
                f'Images_Saved={results["images_saved"]}'
            )
    
    # Compare checkpoints if testing both
    if len(final_results) > 1:
        psnr_e_result = final_results.get('PSNR_E', {})
        psnr_g_result = final_results.get('PSNR_G', {})
        
        # Check if this is clinical inference mode (no metrics to compare)
        if (psnr_e_result.get('inference_only', False) or 
            psnr_g_result.get('inference_only', False)):
            logger.info("-"*60)
            logger.info("CLINICAL INFERENCE MODE:")
            logger.info("Both checkpoints processed clinical data successfully.")
            logger.info("Enhanced images saved for visual comparison.")
            logger.info("-"*60)
        elif psnr_e_result and psnr_g_result:
            psnr_diff = psnr_g_result['psnr'] - psnr_e_result['psnr']
            ssim_diff = psnr_g_result['ssim'] - psnr_e_result['ssim']
            
            logger.info("-"*60)
            logger.info("CHECKPOINT COMPARISON:")
            logger.info(f"PSNR_G vs PSNR_E: {psnr_diff:+.4f}dB difference")
            logger.info(f"SSIM_G vs SSIM_E: {ssim_diff:+.6f} difference")
            
            better_checkpoint = "PSNR_G" if psnr_diff > 0 else "PSNR_E"
            logger.info(f"Better checkpoint: {better_checkpoint}")
            logger.info("-"*60)
    
    logger.info(f'Total test samples evaluated: {test_count}')
    logger.info("="*80)

    # TensorBoard logging for final test metrics (using last checkpoint's results)
    last_checkpoint_results = list(final_results.values())[-1] if final_results else {}
    if last_checkpoint_results and not last_checkpoint_results.get('inference_only', False):
        # Only log metrics if not in clinical inference mode
        writer.add_scalar('Final_Test/PSNR', last_checkpoint_results['psnr'], current_step)
        writer.add_scalar('Final_Test/SSIM', last_checkpoint_results['ssim'], current_step)
        
        # Also add to Test/ namespace for consistency
        writer.add_scalar('Test/Final_PSNR', last_checkpoint_results['psnr'], current_step)
        writer.add_scalar('Test/Final_SSIM', last_checkpoint_results['ssim'], current_step)
    elif last_checkpoint_results and last_checkpoint_results.get('inference_only', False):
        # For clinical inference mode, log number of processed samples
        writer.add_scalar('Final_Test/Clinical_Samples_Processed', last_checkpoint_results['samples'], current_step)
        writer.add_scalar('Final_Test/Clinical_Images_Saved', last_checkpoint_results['images_saved'], current_step)

    # Close TensorBoard writer at the end of training
    writer.close()
    logger.info("Training completed. TensorBoard logs saved.")
    testset_suffix = f"_{args.testset}" if args.testset != 'test' else ''
    logger.info(f"Final test images saved to: {os.path.join(opt['path']['images'], f'final_test_*{testset_suffix}')}")


if __name__ == '__main__':
    import os
    import sys
    # Fix: allow running from any directory, always resolve script dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    sys.path.insert(0, script_dir)
    main()
