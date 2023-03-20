""" 
Evaluate the convergence basin of the learned tracker
# SPDX-FileCopyrightText: 2021 Binbin Xu
# SPDX-License-Identifier: BSD-3-Clause
"""

import os, sys, argparse, pickle
import os.path as osp
import numpy as np
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from tools.rgbd_odometry import RGBDOdometry
from tools.ICP import ICP_Odometry

import torch
import torch.utils.data as data
import torchvision.utils as torch_utils
import torch.nn as nn
# import models.LeastSquareTracking as ICtracking
from experiments.select_method import select_method
from evaluate import create_eval_loaders
from Logger import check_directory
from models.geometry import batch_mat2euler, batch_euler2mat
import models.criterions as criterions
import train_utils
import config

from data.dataloader import load_data
from timers import Timers
from tqdm import tqdm


# wrap the function needed to compute the residuals
def compute_residuals(net, color0, color1, depth0, depth1, K, level_i=0,
                      obj_mask0=None, obj_mask1=None, pose_I=None,
                      logger=None, obj_only=False, tracker='learning_based'):
    with torch.no_grad():
        if tracker == 'learning_based':
            if obj_only:
                output = net.compute_residual(color0, color1, depth0, depth1, K, level_i=level_i,
                                              obj_mask0=obj_mask0, obj_mask1=obj_mask1,
                                              logger=logger, poseI=pose_I)
            else:
                output = net.compute_residual(color0, color1, depth0, depth1, K, level_i=level_i,
                                              logger=logger, poseI=pose_I)
        elif options.tracker in ['ColorICP', 'ICP', 'RGBD']:
            if obj_only:
                output = net.batch_track(color0, depth0, color1, depth1, K,
                                         batch_objmask0=obj_mask0, batch_objmask1=obj_mask1)
            else:
                output = net.batch_track(color0, depth0, color1, depth1, K)
        else:
            raise NotImplementedError("unsupported test tracker: check argument of --tracker again")
    return output


# wrap the function needed to estimate the pose
def estimate_pose(net, color0, color1, depth0, depth1, K,
                  obj_mask0=None, obj_mask1=None,
                  logger=None, obj_only=False, tracker='learning_based'):
    with torch.no_grad():
        if tracker == 'learning_based':
            if obj_only:
                output = net.forward(color0, color1, depth0, depth1, K,
                                     obj_mask0=obj_mask0, obj_mask1=obj_mask1,
                                     logger=logger, iteration=iter)
            else:
                output = net.forward(color0, color1, depth0, depth1, K,
                                     logger=logger, iteration=iter)
        elif options.tracker in ['ColorICP', 'ICP', 'RGBD']:
            if obj_only:
                output = net.batch_track(color0, depth0, color1, depth1, K,
                                         batch_objmask0=obj_mask0, batch_objmask1=obj_mask1)
            else:
                output = net.batch_track(color0, depth0, color1, depth1, K)
        else:
            raise NotImplementedError("unsupported test tracker: check argument of --tracker again")
        # R, t = output
    return output

def evaluate_2d_convergence_basin(dataloader, net, objectives, eval_name='', obj_only=False, level_i=None,
                                  known_mask = False, timers = None, logger=None, tracker='learning_based',
                                  trans_pert_range=(-0.1, 0.1), pert_samples=31,
                                  ):
    """ evaluate the trust-region method given the two-frame pose estimation
    :param the pytorch dataloader
    :param the network
    :param the evaluation objective names, e.g. RPE, EPE3D
    :param True if ground mask if known
    :param (optional) timing each step
    """

    progress = tqdm(dataloader, ncols=100,
        desc = 'evaluate deeper inverse compositional algorithm {:}'.format(eval_name),
        total= len(dataloader))

    if tracker == 'learning_based':
        net.eval()

    total_frames = len(dataloader.dataset)
    if level_i is None:
        level_i = [0]
        level_num = 1
    else:
        level_num = len(level_i)

    outputs = {
        'color0': np.zeros((total_frames, 120, 160, 3)),
        'color1': np.zeros((total_frames, 120, 160, 3)),
        'R_est': np.zeros((total_frames, 3, 3)),
        't_est': np.zeros((total_frames, 3)),
        'pose_per_x': np.zeros((level_num, pert_samples, pert_samples, total_frames)),
        'pose_per_y': np.zeros((level_num, pert_samples, pert_samples, total_frames)),
        'avg_loss': np.zeros((level_num, pert_samples, pert_samples, total_frames)),
        'pose_spe_x': np.zeros((3, total_frames)),
        'pose_spe_y': np.zeros((3, total_frames)),
        'loss_spe': np.zeros((3, total_frames)),
        'R_gt': np.zeros((total_frames, 3)),
        't_gt': np.zeros((total_frames, 3)),
        'names': eval_name,
    }
    flow_loss, rpe_loss = None, None
    if 'EPE3D' in objectives:
        flow_loss = criterions.compute_RT_EPE_loss
        outputs['epes'] = np.zeros(total_frames)
    if 'RPE' in objectives:
        rpe_loss = criterions.compute_RPE_loss
        outputs['angular_error'] = np.zeros(total_frames)
        outputs['translation_error'] = np.zeros(total_frames)

    count_base = 0

    if timers: timers.tic('one iteration')

    for idx, batch in enumerate(progress):
        if known_mask: # for dataset that with mask or need mask
            color0, color1, depth0, depth1, Rt, K, obj_mask0, obj_mask1 = \
                train_utils.check_cuda(batch[:8])
        else:
            color0, color1, depth0, depth1, Rt, K = \
                train_utils.check_cuda(batch[:6])
            obj_mask0, obj_mask1 = None, None
        B, _, H, W = depth0.shape

        outputs['color0'][count_base:count_base + B] = color0.permute(0,2,3,1).cpu().numpy()
        outputs['color1'][count_base:count_base + B] = color1.permute(0,2,3,1).cpu().numpy()

        R_gt, t_gt = Rt[:,:3,:3], Rt[:,:3,3]
        R_gt_angle_tensor = torch.stack(batch_mat2euler(R_gt), dim=1)

        est_pose = estimate_pose(net, color0, color1, depth0, depth1, K,
                                 obj_mask0=obj_mask0, obj_mask1=obj_mask1,
                                 logger=logger)
        est_R, est_t = est_pose
        outputs['R_est'][count_base:count_base + B] = est_R.cpu().numpy()
        outputs['t_est'][count_base:count_base + B] = est_t.cpu().numpy()
        outputs['R_gt'][count_base:count_base + B] = R_gt_angle_tensor.cpu().numpy()
        outputs['t_gt'][count_base:count_base + B] = t_gt.cpu().numpy()

        eye_id = 0
        pred_id = 1
        opt_id = 2

        trs_per_1d = np.linspace(start=trans_pert_range[0], stop=trans_pert_range[1], num=pert_samples)
        # freeze rotation and z-translation and uniform sampling on x-y plane
        for eps_i_x in range(pert_samples):  # x-translation
            for eps_i_y in range(pert_samples):  # y-translation
                trs_per_x = trs_per_1d[eps_i_x]
                trs_per_y = trs_per_1d[eps_i_y]
                trs_per = torch.tensor([trs_per_x, trs_per_y, 0]).unsqueeze(dim=0).type_as(t_gt)
                # trs_p = t_gt.clone()
                trs_p = t_gt + trs_per
                pose_I = [R_gt, trs_p]

                for level in level_i:
                    output = compute_residuals(net, color0, color1, depth0, depth1, K, level_i=level,
                                               obj_mask0=obj_mask0, obj_mask1=obj_mask1,
                                               logger=logger, pose_I=pose_I)
                    avg_residual = output[1]
                    outputs['avg_loss'][level, eps_i_x, eps_i_y, count_base:count_base+B] = avg_residual.cpu().numpy()
                    outputs['pose_per_x'][level, eps_i_x, eps_i_y, count_base:count_base+B] = trs_per_x
                    outputs['pose_per_y'][level, eps_i_x, eps_i_y, count_base:count_base+B] = trs_per_y


        # identity initialization
        trs_per_x = 0. - t_gt[:, 0]
        trs_per_y = 0. - t_gt[:, 1]

        outputs['loss_spe'][eye_id, count_base:count_base + B] = None
        outputs['pose_spe_x'][eye_id, count_base:count_base + B] = trs_per_x.cpu().numpy()
        outputs['pose_spe_y'][eye_id, count_base:count_base + B] = trs_per_y.cpu().numpy()

        # predicted init pose
        output = compute_residuals(net, color0, color1, depth0, depth1, K,
                                   obj_mask0=obj_mask0, obj_mask1=obj_mask1,
                                   logger=logger)
        pred_I, avg_residual = output
        pred_t = pred_I[1]
        if pred_t.dim() == 3: pred_t = pred_t.squeeze()
        trs_per = pred_t - t_gt
        trs_per_x = trs_per[:, 0]
        trs_per_y = trs_per[:, 1]
        outputs['loss_spe'][pred_id, count_base:count_base + B] = avg_residual.cpu().numpy()
        outputs['pose_spe_x'][pred_id, count_base:count_base + B] = trs_per_x.cpu().numpy()
        outputs['pose_spe_y'][pred_id, count_base:count_base + B] = trs_per_y.cpu().numpy()

        # optimized pose
        output = compute_residuals(net, color0, color1, depth0, depth1, K,
                                   obj_mask0=obj_mask0, obj_mask1=obj_mask1,
                                   logger=logger, pose_I=est_pose)
        avg_residual = output[1]
        trs_diff = est_t - t_gt
        trs_per_x = trs_diff[:, 0]
        trs_per_y = trs_diff[:, 1]
        outputs['loss_spe'][opt_id, count_base:count_base + B] = avg_residual.cpu().numpy()
        outputs['pose_spe_x'][opt_id, count_base:count_base + B] = trs_per_x.cpu().numpy()
        outputs['pose_spe_y'][opt_id, count_base:count_base + B] = trs_per_y.cpu().numpy()


        if timers: timers.tic('evaluate')
        if rpe_loss: # evaluate the relative pose error
            angle_error, trans_error = rpe_loss(est_R, est_t, R_gt, t_gt)
            outputs['angular_error'][count_base:count_base+B] = angle_error.cpu().numpy()
            outputs['translation_error'][count_base:count_base+B] = trans_error.cpu().numpy()

        if flow_loss:# evaluate the end-point-error loss 3D
            invalid_mask = (depth0 == depth0.min()) | (depth0 == depth0.max())
            if obj_mask0 is not None:
                invalid_mask = ~obj_mask0 | invalid_mask

            epes3d = flow_loss(est_R, est_t, R_gt, t_gt, depth0, K, invalid=invalid_mask)
            outputs['epes'][count_base:count_base+B] = epes3d.cpu().numpy()


        count_base += B

        if timers: timers.toc('evaluate')
        if timers: timers.toc('one iteration')
        if timers: timers.tic('one iteration')

    # if timers: timers.print()

    return outputs


def evaluate_convergence_basin(dataloader, net, objectives, eval_name='', obj_only=False, level_i=None,
                               known_mask = False, timers = None, logger=None, tracker='learning_based',
                               trans_pert_range=(-0.1, 0.1), rot_pert_range=(-0.1, 0.1), pert_samples=31,
                               eval_each_level=True, eval_opt=True, eval_init=True, eval_pred=True,
                               ):
    """ evaluate the trust-region method given the two-frame pose estimation
    :param the pytorch dataloader
    :param the network
    :param the evaluation objective names, e.g. RPE, EPE3D
    :param True if ground mask if known
    :param (optional) timing each step
    """

    progress = tqdm(dataloader, ncols=100,
        desc = 'evaluate deeper inverse compositional algorithm {:}'.format(eval_name),
        total= len(dataloader))

    if tracker == 'learning_based':
        net.eval()

    total_frames = len(dataloader.dataset)
    if level_i is None:
        level_i = [0]
        level_num = 1
    else:
        level_num = len(level_i)

    translation_list = ['tran-x', 'tran-y', 'tran-z']
    rotation_list = ['yaw', 'pitch', 'roll']
    outputs = {
        'color0': np.zeros((total_frames, 120, 160, 3)),
        'color1': np.zeros((total_frames, 120, 160, 3)),
        'R_est': np.zeros((total_frames, 3, 3)),
        't_est': np.zeros((total_frames, 3)),
        'pose_per': np.zeros((6, pert_samples+3, total_frames)),
        'avg_loss': np.zeros((6, pert_samples+3, total_frames)),
        'coarse_pose_per': np.zeros((level_num-1, 6, pert_samples, total_frames)),
        'coarse_avg_loss': np.zeros((level_num-1, 6, pert_samples, total_frames)),
        'R_gt': np.zeros((total_frames, 3)),
        't_gt': np.zeros((total_frames, 3)),
        'names': eval_name,
    }
    flow_loss, rpe_loss = None, None
    if 'EPE3D' in objectives:
        flow_loss = criterions.compute_RT_EPE_loss
        outputs['epes'] = np.zeros(total_frames)
    if 'RPE' in objectives:
        rpe_loss = criterions.compute_RPE_loss
        outputs['angular_error'] = np.zeros(total_frames)
        outputs['translation_error'] = np.zeros(total_frames)

    count_base = 0

    if timers: timers.tic('one iteration')

    for idx, batch in enumerate(progress):
        if known_mask: # for dataset that with mask or need mask
            color0, color1, depth0, depth1, Rt, K, obj_mask0, obj_mask1 = \
                train_utils.check_cuda(batch[:8])
        else:
            color0, color1, depth0, depth1, Rt, K = \
                train_utils.check_cuda(batch[:6])
            obj_mask0, obj_mask1 = None, None
        B, _, H, W = depth0.shape

        outputs['color0'][count_base:count_base + B] = color0.permute(0,2,3,1).cpu().numpy()
        outputs['color1'][count_base:count_base + B] = color1.permute(0,2,3,1).cpu().numpy()

        R_gt, t_gt = Rt[:,:3,:3], Rt[:,:3,3]
        R_gt_angle_tensor = torch.stack(batch_mat2euler(R_gt), dim=1)

        est_pose = estimate_pose(net, color0, color1, depth0, depth1, K,
                                 obj_mask0=obj_mask0, obj_mask1=obj_mask1,
                                 logger=logger)
        est_R, est_t = est_pose
        est_R_angle_tensor = torch.stack(batch_mat2euler(est_R), dim=1)
        outputs['R_est'][count_base:count_base + B] = est_R.cpu().numpy()
        outputs['t_est'][count_base:count_base + B] = est_t.cpu().numpy()
        outputs['R_gt'][count_base:count_base + B] = R_gt_angle_tensor.cpu().numpy()
        outputs['t_gt'][count_base:count_base + B] = t_gt.cpu().numpy()

        eye_id = pert_samples
        pred_id = pert_samples + 1
        opt_id = pert_samples + 2

        # generate perturbation candidates
        rot_pert = np.linspace(start=rot_pert_range[0], stop=rot_pert_range[1], num=pert_samples)
        for i in range(len(rotation_list)):
            # uniform sampling
            for eps_i in range(pert_samples):
                rot_vec = R_gt_angle_tensor.clone()
                rot_vec[:, i] = R_gt_angle_tensor[:, i] + rot_pert[eps_i]
                R_pert_vec = torch.split(rot_vec, 1, dim=1)
                R_pert = batch_euler2mat(R_pert_vec[0], R_pert_vec[1], R_pert_vec[2]).squeeze()
                pose_I = [R_pert, t_gt]

                for level in level_i:
                    if level == 0:
                        output = compute_residuals(net, color0, color1, depth0, depth1, K, level_i=0,
                                                   obj_mask0=obj_mask0, obj_mask1=obj_mask1,
                                                   logger=logger, pose_I=pose_I)
                        avg_residual = output[1]
                        outputs['avg_loss'][i, eps_i, count_base:count_base + B] = avg_residual.cpu().numpy()
                        outputs['pose_per'][i, eps_i, count_base:count_base + B] = rot_pert[eps_i]
                    else:
                        output = compute_residuals(net, color0, color1, depth0, depth1, K, level_i=level,
                                                   obj_mask0=obj_mask0, obj_mask1=obj_mask1,
                                                   logger=logger, pose_I=pose_I)
                        avg_residual = output[1]
                        outputs['coarse_pose_per'][level-1, i, eps_i, count_base:count_base + B] = rot_pert[eps_i]
                        outputs['coarse_avg_loss'][level-1, i, eps_i, count_base:count_base + B] = avg_residual.cpu().numpy()

            # identity initialization
            rot_vec = R_gt_angle_tensor.clone()
            rot_diff = 0. - R_gt_angle_tensor[:, i]
            rot_vec[:, i] = 0.
            R_pert_vec = torch.split(rot_vec, 1, dim=1)
            R_pert = batch_euler2mat(R_pert_vec[0], R_pert_vec[1], R_pert_vec[2]).squeeze()
            pose_I = [R_pert, t_gt]
            output = compute_residuals(net, color0, color1, depth0, depth1, K,
                                       obj_mask0=obj_mask0, obj_mask1=obj_mask1,
                                       logger=logger, pose_I=pose_I)
            avg_residual = output[1]
            outputs['avg_loss'][i, eye_id, count_base:count_base + B] = avg_residual.cpu().numpy()
            outputs['pose_per'][i, eye_id, count_base:count_base + B] = rot_diff.cpu().numpy()

            # predicted init pose
            output = compute_residuals(net, color0, color1, depth0, depth1, K,
                                       obj_mask0=obj_mask0, obj_mask1=obj_mask1,
                                       logger=logger)
            pred_I, avg_residual = output
            pred_R = pred_I[0]
            pred_R_vec = torch.stack(batch_mat2euler(pred_R), dim=1)
            rot_diff = pred_R_vec[:, i] - R_gt_angle_tensor[:, i]
            outputs['avg_loss'][i, pred_id, count_base:count_base + B] = avg_residual.cpu().numpy()
            outputs['pose_per'][i, pred_id, count_base:count_base + B] = rot_diff.cpu().numpy()

            # optimized pose
            output = compute_residuals(net, color0, color1, depth0, depth1, K,
                                       obj_mask0=obj_mask0, obj_mask1=obj_mask1,
                                       logger=logger, pose_I=est_pose)
            avg_residual = output[1]
            rot_diff = est_R_angle_tensor[:, i] - R_gt_angle_tensor[:, i]
            outputs['avg_loss'][i, opt_id, count_base:count_base + B] = avg_residual.cpu().numpy()
            outputs['pose_per'][i, opt_id, count_base:count_base + B] = rot_diff.cpu().numpy()

        trs_per = np.linspace(start=trans_pert_range[0], stop=trans_pert_range[1], num=pert_samples)
        for i in range(len(translation_list)):
            # uniform sampling
            for eps_i in range(pert_samples):
                trs_p = t_gt.clone()
                trs_p[:, i] = t_gt[:, i] + trs_per[eps_i]
                pose_I = [R_gt, trs_p]

                for level in level_i:
                    if level == 0:
                        output = compute_residuals(net, color0, color1, depth0, depth1, K, level_i=0,
                                                   obj_mask0=obj_mask0, obj_mask1=obj_mask1,
                                                   logger=logger, pose_I=pose_I)
                        avg_residual = output[1]
                        outputs['avg_loss'][i+3, eps_i, count_base:count_base+B] = avg_residual.cpu().numpy()
                        outputs['pose_per'][i+3, eps_i, count_base:count_base+B] = trs_per[eps_i]
                    else:
                        output = compute_residuals(net, color0, color1, depth0, depth1, K, level_i=level,
                                                   obj_mask0=obj_mask0, obj_mask1=obj_mask1,
                                                   logger=logger, pose_I=pose_I)
                        avg_residual = output[1]
                        outputs['coarse_pose_per'][level-1, i+3, eps_i, count_base:count_base + B] = trs_per[eps_i]
                        outputs['coarse_avg_loss'][level-1, i+3, eps_i, count_base:count_base + B] = avg_residual.cpu().numpy()

            # identity initialization
            trs_p = t_gt.clone()
            trs_diff = 0. - t_gt[:, i]
            trs_p[:, i] = 0.
            pose_I = [R_gt, trs_p]
            output = compute_residuals(net, color0, color1, depth0, depth1, K,
                                       obj_mask0=obj_mask0, obj_mask1=obj_mask1,
                                       logger=logger, pose_I=pose_I)
            avg_residual = output[1]
            outputs['avg_loss'][i + 3, eye_id, count_base:count_base + B] = avg_residual.cpu().numpy()
            outputs['pose_per'][i + 3, eye_id, count_base:count_base + B] = trs_diff.cpu().numpy()

            # predicted init pose
            output = compute_residuals(net, color0, color1, depth0, depth1, K,
                                       obj_mask0=obj_mask0, obj_mask1=obj_mask1,
                                       logger=logger)
            pred_I, avg_residual = output
            pred_t = pred_I[1]
            if pred_t.dim() == 3: pred_t = pred_t.squeeze()
            trs_diff = pred_t[:, i] - t_gt[:, i]
            outputs['avg_loss'][i + 3, pred_id, count_base:count_base + B] = avg_residual.cpu().numpy()
            outputs['pose_per'][i + 3, pred_id, count_base:count_base + B] = trs_diff.cpu().numpy()

            # optimized pose
            output = compute_residuals(net, color0, color1, depth0, depth1, K,
                                       obj_mask0=obj_mask0, obj_mask1=obj_mask1,
                                       logger=logger, pose_I=est_pose)
            avg_residual = output[1]
            trs_diff = est_t[:, i] - t_gt[:, i]
            outputs['avg_loss'][i + 3, opt_id, count_base:count_base + B] = avg_residual.cpu().numpy()
            outputs['pose_per'][i + 3, opt_id, count_base:count_base + B] = trs_diff.cpu().numpy()

        if timers: timers.tic('evaluate')
        if rpe_loss: # evaluate the relative pose error
            angle_error, trans_error = rpe_loss(est_R, est_t, R_gt, t_gt)
            outputs['angular_error'][count_base:count_base+B] = angle_error.cpu().numpy()
            outputs['translation_error'][count_base:count_base+B] = trans_error.cpu().numpy()

        if flow_loss:# evaluate the end-point-error loss 3D
            invalid_mask = (depth0 == depth0.min()) | (depth0 == depth0.max())
            if obj_mask0 is not None:
                invalid_mask = ~obj_mask0 | invalid_mask

            epes3d = flow_loss(est_R, est_t, R_gt, t_gt, depth0, K, invalid=invalid_mask)
            outputs['epes'][count_base:count_base+B] = epes3d.cpu().numpy()


        count_base += B

        if timers: timers.toc('evaluate')
        if timers: timers.toc('one iteration')
        if timers: timers.tic('one iteration')

    # if timers: timers.print()

    return outputs


def visualize_2d_convergence_basin(info, levels, output_fig_dir,
                                pose_color, pose_linestyle,
                                level_color, level_alpha):
    """ =============================================================== """
    """     Evaluation of the convergence basin of X and Y translation  """
    """ =============================================================== """

    dataset_size = info['avg_loss'].shape[-1]
    for f in range(dataset_size):
        # f = 6
        fig = plt.figure(figsize=(18, 7))

        # for_demo = True
        color0 = info['color0'][f]
        color1 = info['color1'][f]
        #
        axes = fig.add_subplot(2,5,1)
        axes.imshow(color0)
        axes.set_title('color 0')
        axes = fig.add_subplot(2,5,6)
        axes.imshow(color1)
        axes.set_title('color 1')

        """ =============================================================== """
        """              visualization of all levels                        """
        """ =============================================================== """
        # for level in levels:
        #     pert_x = info['pose_per_x'][level, :, :, f]
        #     pert_y = info['pose_per_y'][level, :, :, f]
        #     pert_loss = info['avg_loss'][level, :, :, f]
        #     eye_pose = [info['pose_spe_x'][0, f], info['pose_spe_y'][0, f]]
        #     pred_pose = [info['pose_spe_x'][1, f], info['pose_spe_y'][1, f]]
        #     opt_pose = [info['pose_spe_x'][2, f], info['pose_spe_y'][2, f]]
        #
        #     # 2D visualization
        #     ax_2d = fig.add_subplot(2, 5, 1 + level + 1 + 5,)
        #     # ax_2d.imshow(pert_loss, cmap='jet', interpolation='nearest', alpha=0.5)
        #     ax_2d.contourf(pert_x, pert_y, pert_loss, cmap='jet')
        #     ax_2d.scatter(eye_pose[0], eye_pose[1], color=pose_color['eye'], label='identity', marker='o',)
        #     ax_2d.scatter(pred_pose[0], pred_pose[1], color=pose_color['predict'], label='predict', marker='o',)
        #     ax_2d.scatter(opt_pose[0], opt_pose[1], color=pose_color['opt'], label='opt', marker='o',)
        #     ax_2d.set_xlabel('translation-X')
        #     ax_2d.set_ylabel('translation-Y')
        #     ax_2d.legend(loc='upper left')
        #     ax_2d.set_title("level_" + str(level))
        #
        #
        #     # # 3D visualization: Plot the surface.
        #     # ax = fig.gca(projection='3d')
        #     ax = fig.add_subplot(2, 5, 1 + level + 1, projection='3d')   # level + 1 + 1 * (level//2+1)
        #     surf = ax.plot_surface(pert_x, pert_y, pert_loss, cmap=cm.jet, #cmap=cm.YlGnBu,
        #                            linewidth=0, antialiased=False)
        #     loss_min = np.min(pert_loss)
        #     loss_max = np.max(pert_loss)
        #     cset = ax.contourf(pert_x, pert_y, pert_loss, zdir='z', offset=loss_min, cmap=cm.gray, alpha=0.8)
        #
        #     # denote special pose position
        #     ax.plot([eye_pose[0], eye_pose[0]], [eye_pose[1], eye_pose[1]], [loss_min, loss_max],
        #             color=pose_color['eye'], linestyle=pose_linestyle['eye'], label='identity')
        #     # ax.scatter(eye_pose[0], eye_pose[1], loss_min, color=pose_color['eye'], marker='o')
        #     ax.plot([pred_pose[0], pred_pose[0]], [pred_pose[1], pred_pose[1]], [loss_min, loss_max],
        #             color=pose_color['predict'], linestyle=pose_linestyle['predict'], label='predict')
        #     ax.plot([opt_pose[0], opt_pose[0]], [opt_pose[1], opt_pose[1]], [loss_min, loss_max],
        #             color=pose_color['opt'], linestyle=pose_linestyle['opt'], label='opt')
        #
        #     ax.set_xlabel('translation-X')
        #     ax.set_ylabel('translation-Y')
        #     ax.set_zlabel('cost')
        #     ax.grid(True)
        #     ax.legend(loc='upper left')
        #     ax.set_title("level_"+str(level))
        # fig.tight_layout()
        # plt.show()
        # fig.show()
        #
        # img_save = osp.join(output_fig_dir, str(f) + '.png')
        # check_directory(img_save)
        # fig.savefig(img_save, dpi=300)

        """ =============================================================== """
        """              only visualize the coarsest level                  """
        """              better run through command instead of gui          """
        """ =============================================================== """
        img_save = osp.join(output_fig_dir, 'color0_'+str(f) + '.png')
        cv2.imwrite(img_save, cv2.cvtColor((color0*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        print(img_save)
        img_save = osp.join(output_fig_dir, 'color1_'+str(f) + '.png')
        cv2.imwrite(img_save, cv2.cvtColor((color1*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        ax_2d = plt.axes()
        pert_x = info['pose_per_x'][3, :, :, f]
        pert_y = info['pose_per_y'][3, :, :, f]
        pert_loss = info['avg_loss'][3, :, :, f]
        ax_2d.contourf(pert_x, pert_y, pert_loss, cmap='jet')
        # eye_pose = [info['pose_spe_x'][0, f], info['pose_spe_y'][0, f]]
        # pred_pose = [info['pose_spe_x'][1, f], info['pose_spe_y'][1, f]]
        # opt_pose = [info['pose_spe_x'][2, f], info['pose_spe_y'][2, f]]
        # ax_2d.scatter(eye_pose[0], eye_pose[1], color=pose_color['eye'], label='identity', marker='o', )
        # ax_2d.scatter(pred_pose[0], pred_pose[1], color=pose_color['predict'], label='predict', marker='o', )
        # ax_2d.scatter(opt_pose[0], opt_pose[1], color=pose_color['opt'], label='opt', marker='o', )
        ax_2d.set_xlabel('translation-X')
        ax_2d.set_ylabel('translation-Y')
        # ax_2d.legend(loc='upper left')
        plt.show()
        fig.show()

        img_save = osp.join(output_fig_dir, '2d_'+str(f) + '.png')
        check_directory(img_save)
        # fig.savefig(img_save, dpi=300)
        fig.clf()
        # plt.close(fig)
        # plt.close("all")

        # 3D
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(pert_x, pert_y, pert_loss, cmap=cm.jet,  # cmap=cm.YlGnBu,
                               linewidth=0, antialiased=False)
        loss_min = np.min(pert_loss)
        loss_max = np.max(pert_loss)
        cset = ax.contourf(pert_x, pert_y, pert_loss, zdir='z', offset=loss_min, cmap=cm.gray, alpha=0.8)

        # denote special pose position
        # ax.plot([eye_pose[0], eye_pose[0]], [eye_pose[1], eye_pose[1]], [loss_min, loss_max],
        #         color=pose_color['eye'], linestyle=pose_linestyle['eye'], label='identity')
        # ax.scatter(eye_pose[0], eye_pose[1], loss_min, color=pose_color['eye'], marker='o')
        # ax.plot([pred_pose[0], pred_pose[0]], [pred_pose[1], pred_pose[1]], [loss_min, loss_max],
        #         color=pose_color['predict'], linestyle=pose_linestyle['predict'], label='predict')
        # ax.plot([opt_pose[0], opt_pose[0]], [opt_pose[1], opt_pose[1]], [loss_min, loss_max],
        #         color=pose_color['opt'], linestyle=pose_linestyle['opt'], label='opt')

        ax.set_xlabel('translation-X')
        ax.set_ylabel('translation-Y')
        ax.set_zlabel('cost')
        ax.grid(True)
        # ax.legend(loc='upper left')
        img_save = osp.join(output_fig_dir, '3d_' + str(f) + '.png')
        check_directory(img_save)
        # fig.savefig(img_save, dpi=300)
        plt.show()
        fig.show()
        fig.clf()
        plt.close(fig)
        plt.close("all")

        plt.pause(3)
        fig.clf()
        plt.close(fig)
        plt.close("all")


def visualize_convergence_basin(info, levels, output_fig_dir,
                                pose_color, pose_linestyle,
                                level_color, level_alpha):
    """ =============================================================== """
    """             Evaluation of the convergence basin                 """
    """ =============================================================== """

    pose_label_list = ['yaw: (rad)', 'pitch: (rad)', 'roll: (rad)', 'trans: x (cm)', 'trans: y (cm)', 'trans: z (cm)']
    dataset_size = info['pose_per'].shape[2]
    for f in range(dataset_size):
        fig = plt.figure(figsize=(12, 6))

        for_demo = True
        color0 = info['color0'][f]
        color1 = info['color1'][f]

        axes = fig.add_subplot(241)
        axes.imshow(color0)
        axes.set_title('color 0')
        axes = fig.add_subplot(245)
        axes.imshow(color1)
        axes.set_title('color 1')

        for i in range(6):
            subplot_id = 240 + i + 1 + 1 * (i//3+1)
            axes = fig.add_subplot(subplot_id)
            # title = pose_list[i]
            if i >= 3:
                pert_list = info['pose_per'][i, :, f] * 100  # cm
            else:
                pert_list = info['pose_per'][i, :, f]  # radius
            fea_cost_list = info['avg_loss'][i, :-3, f]
            axes.plot(pert_list[:-3], fea_cost_list, '-b', label='level_0')

            # draw lowest point
            eye_pose = pert_list[-3]
            pred_pose = pert_list[-2]
            opt_pose = pert_list[-1]

            if not (np.abs(eye_pose) > np.abs(pred_pose) >= np.abs(opt_pose)):
                for_demo = False

            axes.axvline(x=eye_pose, color=pose_color['eye'], linestyle=pose_linestyle['eye'], alpha=0.5)
            axes.axvline(x=pred_pose, color=pose_color['predict'], linestyle=pose_linestyle['predict'], alpha=0.5)
            axes.axvline(x=opt_pose, color=pose_color['opt'], linestyle=pose_linestyle['opt'], alpha=0.5)

            # also visualization of coarse feature pyramid levels
            if len(levels) > 1:
                for level in levels:
                    if level > 0:
                        if i >= 3:
                            sample_pose = info['coarse_pose_per'][level - 1, i, :, f] * 100  # cm
                        else:
                            sample_pose = info['coarse_pose_per'][level - 1, i, :, f]  # radius
                        sample_cost = info['coarse_avg_loss'][level - 1, i, :, f]
                        axes.plot(sample_pose, sample_cost, level_color[level], alpha=level_alpha[level],
                                  label='level_' + str(level))

            axes.legend(loc='upper left')
            axes.set_xlabel(pose_label_list[i])
            axes.set_ylabel('Cost')
            # axes.set_title()
            axes.grid(True)

        # if info_in_img:
        #     fig.suptitle(str(info_in_img))

        # center text
        caption = 'Red solid: identity initial pose; Magenta dashed: predicted initial pose; Green dash-dot: optimized final pose'
        # plt.title(.5, .05, caption, ha='center', wrap=True, horizontalalignment='center', fontsize=12)
        # plt.title(caption)
        fig.text(.001, .005, caption)
        # fig.tight_layout()

        # if img_save:
        plt.tight_layout()
        # fig.show()

        img_save = osp.join(output_fig_dir, str(f) + '.png')
        check_directory(img_save)
        fig.savefig(img_save, dpi=300)
        if for_demo:
            img_save2 = osp.join(output_fig_dir, 'for_demo', str(f) + '.png')
            check_directory(img_save2)
            fig.savefig(img_save2, dpi=300)
            print('===> Demo fig saved')
        fig.clf()
        plt.close(fig)
        plt.close("all")


def test_convegence_basin(options, method_list):
    if options.time:
        timers = Timers()
    else:
        timers = None

    print('Evaluate test performance with the (deep) direct method.')

    total_batch_size = options.batch_per_gpu * torch.cuda.device_count()

    keyframes = [int(x) for x in options.keyframes.split(',')]
    if options.dataset in ['BundleFusion', 'TUM_RGBD']:
        obj_has_mask = False
    else:
        obj_has_mask = True

    eval_loaders = create_eval_loaders(options, options.eval_set,
                                       keyframes, total_batch_size, options.trajectory)

    # per-level evaluation
    levels = [0,1,2,3]
    level_color = ['-b', '--y', '-.c', ':k']
    level_alpha = [1.0, 0.8, 0.8, 0.8]

    # identity, predicted, optimised
    pose_color = {'eye': 'r',
                  'predict': 'm',
                  'opt': 'g'}
    pose_linestyle = {'eye': '-',
                      'predict': '--',
                      'opt': '-.'}

    # load comparison methods
    for method_name in method_list:
        method_list[method_name] = select_method(method_name, options)

    eval_objectives = ['EPE3D', 'RPE', 'convergence_basin']   # optimized_pose, identity_pose

    output_prefix = '_'.join([
        options.network,
        options.encoder_name,
        options.mestimator,
        options.solver,
        'iter', str(options.max_iter_per_pyr),
    ])

    rot_range = {
        '1': (-0.15, 0.15),
        '2': (-0.20, 0.20),
        '4': (-0.25, 0.25),
        '8': (-0.30, 0.30),
    }
    trs_range = {
        '1': (-0.15, 0.15),
        '2': (-0.20, 0.20),
        '4': (-0.25, 0.25),
        '8': (-0.30, 0.30),
    }

    # evaluate convergence basin & results per trajectory per key-frame
    checkpoint_name = options.checkpoint.replace('.pth.tar', '')
    for method in method_list:
        print("======> Evaluate method:", method)
        outputs = {}
        if options.trajectory == '':
            output_dir_method = osp.join(checkpoint_name, method)
        else:
            output_dir_method = osp.join(checkpoint_name, options.trajectory, method)
        for k, loader in eval_loaders.items():

            output_dir = osp.join(output_dir_method, k)
            output_fig_dir = output_dir + '_convergence_basin'
            output_pkl = output_dir + '.pkl'
            check_directory(output_pkl)

            traj_name, kf = k.split('_keyframe_')

            debug = True
            if debug:
                # for debuging
                print("==> Currently using", method)
                # initial
                # f=45
                # pkl_file = '/media/binbin/code/SLAM/DeeperInverseCompositionalAlgorithm/logs/TUM_RGBD/convergence basin/full_cb2/test/feature/rgbd_dataset_freiburg1_desk_keyframe_8.pkl'
                # icp
                # pkl_file = '/media/binbin/code/SLAM/DeeperInverseCompositionalAlgorithm/logs/TUM_RGBD/convergence basin/full_cb2/test/ICP/rgbd_dataset_freiburg1_desk_keyframe_8.pkl'
                # rgb
                # pkl_file = '/media/binbin/code/SLAM/DeeperInverseCompositionalAlgorithm/logs/TUM_RGBD/convergence basin/full_cb2/test/RGB/rgbd_dataset_freiburg1_desk_keyframe_8.pkl'
                # feature
                pkl_file = '/media/binbin/code/SLAM/DeeperInverseCompositionalAlgorithm/logs/TUM_RGBD/convergence basin/full_cb2/test/feature/rgbd_dataset_freiburg1_desk_keyframe_8.pkl'
                with open(pkl_file, 'rb') as pkl_file:
                    info = pickle.load(pkl_file)
            else:
                # compute convergence basin if file is not exsiting or required to re-compute it
                compute_cb = options.reset_cb or not osp.isfile(output_pkl)
                if compute_cb:
                    if options.cb_dimension == '1D':
                        info = evaluate_convergence_basin(loader, method_list[method],
                                                          eval_objectives,
                                                          eval_name=method,
                                                          trans_pert_range=trs_range[kf],
                                                          rot_pert_range=rot_range[kf],
                                                          known_mask=obj_has_mask,
                                                          obj_only=options.obj_only,
                                                          tracker=options.tracker,
                                                          level_i=levels,
                                                          pert_samples=options.pert_samples,
                                                          )
                    elif options.cb_dimension == '2D':
                        info = evaluate_2d_convergence_basin(loader, method_list[method],
                                                             eval_objectives,
                                                             eval_name=method,
                                                             trans_pert_range=trs_range[kf],
                                                             known_mask=obj_has_mask,
                                                             obj_only=options.obj_only,
                                                             tracker=options.tracker,
                                                             level_i=levels,
                                                             pert_samples=options.pert_samples,
                                                             )
                    else:
                        raise NotImplementedError()
                    # dump per-frame results info
                    with open(output_pkl, 'wb') as output:
                        info = info
                        pickle.dump(info, output)
                else:
                    with open(output_pkl, 'rb') as pkl_file:
                        info = pickle.load(pkl_file)

            # collect results
            outputs[k] = pd.Series([info['epes'].mean(),
                                    info['angular_error'].mean(),
                                    info['translation_error'].mean(),
                                    info['epes'].shape[0],
                                    int(kf),
                                    traj_name,
                                    ],
                                   index=['3D EPE',
                                          'axis error',
                                          'trans error',
                                          'total frames',
                                          'keyframe',
                                          'trajectory',
                                          ])
            print(outputs[k])

            # draw convergence cost
            if options.save_img:
                if options.cb_dimension == '1D':
                    visualize_convergence_basin(info, levels, output_fig_dir,
                                                pose_color, pose_linestyle,
                                                level_color, level_alpha)
                elif options.cb_dimension == '2D':
                    visualize_2d_convergence_basin(info, levels, output_fig_dir,
                                                   pose_color, pose_linestyle,
                                                   level_color, level_alpha)
                else:
                    raise NotImplementedError()

        """ =============================================================== """
        """             Generate the final evaluation results                   """
        """ =============================================================== """

        outputs_pd = pd.DataFrame(outputs).T
        outputs_pd['3D EPE'] *= 100  # convert to cm
        outputs_pd['axis error'] *= (180 / np.pi)  # convert to degree
        outputs_pd['trans error'] *= 100  # convert to cm

        print(outputs_pd)

        stats_dict = {}
        for kf in keyframes:
            kf_outputs = outputs_pd[outputs_pd['keyframe'] == kf]

            stats_dict['mean values of trajectories keyframe {:}'.format(kf)] = pd.Series(
                [kf_outputs['3D EPE'].mean(),
                 kf_outputs['axis error'].mean(),
                 kf_outputs['trans error'].mean(), kf],
                index=['3D EPE', 'axis error', 'trans error', 'keyframe'])

            total_frames = kf_outputs['total frames'].sum()
            stats_dict['mean values of frames keyframe {:}'.format(kf)] = pd.Series(
                [(kf_outputs['3D EPE'] * kf_outputs['total frames']).sum() / total_frames,
                 (kf_outputs['axis error'] * kf_outputs['total frames']).sum() / total_frames,
                 (kf_outputs['trans error'] * kf_outputs['total frames']).sum() / total_frames, kf],
                index=['3D EPE', 'axis error', 'trans error', 'keyframe'])

        stats_pd = pd.DataFrame(stats_dict).T
        print(stats_pd)

        final_pd = outputs_pd.append(stats_pd, sort=False)
        final_pd.to_csv('{:}.csv'.format(output_dir_method))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the network")
    config.add_basics_config(parser)
    config.add_test_basics_config(parser)
    config.add_tracking_config(parser)
    config.add_cb_config(parser)
    options = parser.parse_args()

    print('---------------------------------------')

    method_list = {
        'DeepIC': None,
        'RGB': None,
        'ICP': None,
        'RGB+ICP': None,
        'feature': None,
        'feature_icp': None,
    }
    outputs = test_convegence_basin(options, method_list)
