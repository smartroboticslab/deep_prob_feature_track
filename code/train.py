"""
The training script for Deep Probabilistic Feature-metric Tracking,

# SPDX-FileCopyrightText: 2021 Binbin Xu
# SPDX-License-Identifier: BSD-3-Clause

@author: Zhaoyang Lv 
@date: March 2019
"""

import os, sys, argparse, time
import evaluate as eval_utils
import models.LeastSquareTracking as ICtracking
import models.criterions as criterions
import models.geometry as geometry
import train_utils
import config
from data.dataloader import load_data
from Logger import log_git_revisions_hash

import torch
import torch.nn as nn
import torch.utils.data as data

from timers import Timers
from tqdm import tqdm


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()
        #self.next_batch = None

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            for k, val in enumerate(self.next_batch):
                if torch.is_tensor(val):
                    self.next_batch[k] = self.next_batch[k].cuda(non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        next_batch = self.next_batch
        self.preload()
        return next_batch

def create_train_eval_loaders(options, eval_type, keyframes, 
    total_batch_size = 8, 
    trajectory  = ''):
    """ create the evaluation loader at different keyframes set-up
    """
    eval_loaders = {}

    for kf in keyframes:
        if options.image_resize is not None:
            np_loader = load_data(options.dataset, [kf], eval_type, trajectory,
                                  image_resize=options.image_resize,
                                  options=options)
        else:
            np_loader = load_data(options.dataset, [kf], eval_type, trajectory, options=options)
        eval_loaders['{:}_keyframe_{:}'.format(trajectory, kf)] = data.DataLoader(np_loader, 
            pin_memory=True,
            batch_size = int(total_batch_size),
            shuffle = False, num_workers = options.cpu_workers)
    
    return eval_loaders

def train_one_epoch(options, dataloader, net, optim, epoch, logger, objectives,
    known_mask=False, timers=None):

    net.train()

    # prefetcher = data_prefetcher(dataloader)
    
    progress = tqdm(dataloader, ncols=100,
        desc = 'train deeper inverse compositional algorithm #epoch{:}'.format(epoch),
        total= len(dataloader))

    epoch_len = len(dataloader)

    if timers is None: timers_iter = Timers()

    if timers: timers.tic('one iteration')
    else: timers_iter.tic('one iteration')

    for batch_idx, batch in enumerate(progress):

    # batch = prefetcher.next()
    # batch_idx = 0
    # with tqdm(total=epoch_len) as pbar:
    #     while batch is not None:
    #         batch_idx += 1
    #         if batch_idx >= epoch_len:
    #             break
            
            iteration = epoch*epoch_len + batch_idx
            display_dict = {}

            optim.zero_grad()

            if timers: timers.tic('forward step')

            if known_mask: # for dataset that with mask or need mask
                color0, color1, depth0, depth1, Rt, K, obj_mask0, obj_mask1 = \
                    train_utils.check_cuda(batch[:8])
            else:
                color0, color1, depth0, depth1, Rt, K = \
                    train_utils.check_cuda(batch[:6])
                obj_mask0, obj_mask1 = None, None

            # Bypass lazy way to bypass invalid pixels. 
            invalid_mask = (depth0 == depth0.min()) | (depth0 == depth0.max())
            if obj_mask0 is not None:
                invalid_mask = ~obj_mask0 | invalid_mask

            if options.train_uncer_prop:
                if options.obj_only:
                    Rs, ts, sigma_ksi = net.forward(color0, color1, depth0, depth1, K,
                                                    obj_mask0=obj_mask0, obj_mask1=obj_mask1,
                                                    logger=logger,
                                                    vis=options.vis_feat,
                                                    iteration=iteration)[:3]
                else:
                    Rs, ts, sigma_ksi = net.forward(color0, color1, depth0, depth1, K,
                                                    logger=logger,
                                                    vis=options.vis_feat,
                                                    iteration=iteration)[:3]
            else:
                if options.obj_only:
                    Rs, ts = net.forward(color0, color1, depth0, depth1, K,
                                        obj_mask0=obj_mask0, obj_mask1=obj_mask1,
                                        logger=logger,
                                        vis=options.vis_feat,
                                        iteration=iteration)[:2]
                else:
                    Rs, ts = net.forward(color0, color1, depth0, depth1, K,
                                        logger=logger,
                                        vis=options.vis_feat,
                                        iteration=iteration)[:2]

            if timers: timers.toc('forward step')
            if timers: timers.tic('calculate loss')

            R_gt, t_gt = Rt[:,:3,:3], Rt[:,:3,3]

            # assert(flow_loss) # the only loss used for training
            # we want to compute epe anyway
            flow_loss = criterions.compute_RT_EPE_loss

            epes3d = flow_loss(Rs, ts, R_gt, t_gt, depth0, K, invalid=invalid_mask).mean() * 1e2
            if 'EPE3D' in objectives:
                loss = epes3d
            elif 'RPE' in objectives:
                angle_error, trans_error = criterions.compute_RPE_loss(Rs, ts, R_gt, t_gt)
                loss = angle_error + trans_error
            elif 'URPE' in objectives:
                assert options.train_uncer_prop
                loss = criterions.compute_RPE_uncertainty(Rs, ts, R_gt, t_gt, sigma_ksi)
            elif 'UEPE' in objectives:
                loss = criterions.compute_RT_EPE_uncertainty_loss(Rs, ts, R_gt, t_gt, depth0, K, sigma_ksi=sigma_ksi, uncertainty_type=options.uncertainty, invalid=invalid_mask)

            display_dict['train_epes3d'] = epes3d.item()
            display_dict['train_loss'] = loss.item()

            if timers: timers.toc('calculate loss')
            if timers: timers.tic('backward')

            loss.backward()

            if timers: timers.toc('backward')
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            # if options.uncertainty == 'gaussian':
            #     torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optim.step()

            lr = train_utils.get_learning_rate(optim)
            display_dict['lr'] = lr

            if timers:
                timers.toc('one iteration')
                batch_time = timers.get_avg('one iteration')
                timers.tic('one iteration')
            else:
                timers_iter.toc('one iteration')
                batch_time = timers_iter.get_avg('one iteration')
                timers_iter.tic('one iteration')

            logger.write_to_tensorboard(display_dict, iteration)
            logger.write_to_terminal(display_dict, epoch, batch_idx, epoch_len, batch_time, is_train=True)
            
            # batch = prefetcher.next()
            # pbar.update(1)

def train(options):

    if options.time:
        timers = Timers()
    else:
        timers = None

    total_batch_size = options.batch_per_gpu *  torch.cuda.device_count()

    checkpoint = train_utils.load_checkpoint_train(options)

    keyframes = [int(x) for x in options.keyframes.split(',')]
    if options.image_resize is not None:
        train_loader = load_data(options.dataset, keyframes, load_type='train',
                                 image_resize=options.image_resize, options=options)
    else:
        train_loader = load_data(options.dataset, keyframes, load_type = 'train', options=options)
    train_loader = data.DataLoader(train_loader,
        batch_size = total_batch_size,
        pin_memory=True,
        shuffle = True, num_workers = options.cpu_workers)
    if options.dataset in ['BundleFusion', 'TUM_RGBD', 'ScanNet']:
        obj_has_mask = False
    else:
        obj_has_mask = True

    eval_loaders = create_train_eval_loaders(options, 'validation', keyframes, total_batch_size)

    logfile_name = '_'.join([
        options.prefix, # the current test version
        # options.network,
        options.encoder_name,
        options.mestimator,
        options.solver,
        options.dataset,
        'obj', str(options.obj_only),
        'uCh', str(options.uncertainty_channel),
        options.uncertainty,
        'rmT', str(options.remove_tru_sigma),
        # options.direction,
        'fCh', str(options.feature_channel),
        options.feature_extract,
        'iP', options.init_pose,
        'mH', options.multi_hypo,
        # 'resInput', str(options.res_input),
        # 'initScale', str(options.scale_init_pose),
        # 'uncer_prop', str(options.train_uncer_prop),
        'wICP', str(options.combine_ICP),
        's', options.scaler,
        'lr', str(options.lr),
        'batch', str(total_batch_size),
        # 'kf', options.keyframes
        ])

    print("Initialize and train the Deep Trust Region Network")
    net = ICtracking.LeastSquareTracking(
        encoder_name    = options.encoder_name,
        uncertainty_type= options.uncertainty,
        direction       = options.direction,
        max_iter_per_pyr= options.max_iter_per_pyr,
        mEst_type       = options.mestimator,
        solver_type     = options.solver,
        options         = options,
        tr_samples      = options.tr_samples,
        add_init_noise  = options.add_init_pose_noise,
        no_weight_sharing = options.no_weight_sharing,
        timers          = timers)

    if options.no_weight_sharing:
        logfile_name += '_no_weight_sharing'
    logger = train_utils.initialize_logger(options, logfile_name)
    log_git_revisions_hash(logger.log_dir)
    with open(os.path.join(logger.log_dir,'commandline_args.txt'), 'w') as f:
        f.write('\n'.join(sys.argv[1:]))

    if options.checkpoint:
        net.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.is_available():
        net.cuda()

    net.train()

    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs for training!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        net = nn.DataParallel(net)

    train_objective = [options.loss]  # ['EPE3D'] # Note: we don't use RPE for training
    eval_objectives = ['EPE3D', 'RPE']

    num_params = train_utils.count_parameters(net)

    if num_params < 1:
        print('There is no learnable parameters in this baseline.')
        print('No training. Only one iteration of evaluation')
        no_training = True
    else:
        print('There is a total of {:} learnabled parameters'.format(num_params))
        no_training = False
        optim = train_utils.create_optim(options, net)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,
            milestones=options.lr_decay_epochs,
            gamma=options.lr_decay_ratio)

    freq = options.save_checkpoint_freq
    for epoch in range(options.start_epoch, options.epochs):

        if epoch % freq == 0:
            checkpoint_name = 'checkpoint_epoch{:d}.pth.tar'.format(epoch)
            print('save {:}'.format(checkpoint_name))
            state_info = {'epoch': epoch, 'num_param': num_params}
            logger.save_checkpoint(net, state_info, filename=checkpoint_name)

        if options.no_val is False:
            for k, loader in eval_loaders.items():

                eval_name = '{:}_{:}'.format(options.dataset, k)

                eval_info = eval_utils.evaluate_trust_region(
                    loader, net, eval_objectives, 
                    known_mask  = obj_has_mask, 
                    eval_name   = eval_name,
                    timers      = timers,
                    logger=logger,
                    obj_only=options.obj_only,
                    epoch=epoch,
                    tracker='learning_based',
                )

                display_dict = {"{:}_epe3d".format(eval_name): eval_info['epes'].mean(), 
                    "{:}_rpe_angular".format(eval_name): eval_info['angular_error'].mean(), 
                    "{:}_rpe_translation".format(eval_name): eval_info['translation_error'].mean()}

                logger.write_to_tensorboard(display_dict, epoch)

        if no_training: break

        train_one_epoch(options, train_loader, net, optim, epoch, logger,
            train_objective, known_mask=obj_has_mask, timers=timers)

        scheduler.step()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training the network')

    config.add_basics_config(parser)
    config.add_train_basics_config(parser)
    config.add_train_optim_config(parser)
    config.add_train_log_config(parser)
    config.add_train_loss_config(parser)
    config.add_tracking_config(parser)

    options = parser.parse_args()

    options.start_epoch = 0

    print('---------------------------------------')
    print_options = vars(options)
    for key in print_options.keys():
        print(key+': '+str(print_options[key]))
    print('---------------------------------------')
    
    # torch.backends.cudnn.benchmark = True
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    print('Start training...')
    train(options)
