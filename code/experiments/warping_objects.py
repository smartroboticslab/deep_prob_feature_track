""" 
Experiments to warp objects for visualization
# SPDX-FileCopyrightText: 2021 Binbin Xu
# SPDX-License-Identifier: BSD-3-Clause
"""

import argparse, pickle
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as data

import config
import models.geometry as geometry
from Logger import check_directory
from train_utils import check_cuda
from data.dataloader import load_data, MOVING_OBJECTS_3D
from experiments.select_method import select_method
from tools import display

def compute_pose(dataloader, tracker, k, tracker_name, use_gt_pose):
    count_base = 0
    total_frames = len(dataloader.dataset)
    progress = tqdm(dataloader, ncols=100,
        # desc = 'evaluate deeper inverse compositional algorithm {:}'.format(eval_name),
        total= len(dataloader))
    outputs = {
        'R_est': np.zeros((total_frames, 3, 3)),
        't_est': np.zeros((total_frames, 3)),
        'seq_idx': np.zeros((total_frames)),
        'frame0': np.zeros((total_frames)),
        'frame1': np.zeros((total_frames)),
    }
    # computing pose
    for idx, batch in enumerate(progress):
        color0, color1, depth0, depth1, transform, K, mask0, mask1, names = check_cuda(batch)
        B, C, H, W = color0.shape
        if use_gt_pose:
            R_gt, t_gt = transform[:, :3, :3], transform[:, :3, 3]
            Rt = [R_gt, t_gt]
        else:
            with torch.no_grad():
                if options.obj_only:
                    output = tracker.forward(color0, color1, depth0, depth1, K,
                                             obj_mask0=mask0, obj_mask1=mask1,
                                             )
                else:
                    output = tracker.forward(color0, color1, depth0, depth1, K,
                                             index=idx,
                                             )
            Rt = output
        R, t = Rt
        outputs['R_est'][count_base:count_base + B] = R.cpu().numpy()
        outputs['t_est'][count_base:count_base + B] = t.cpu().numpy()
        outputs['seq_idx'][count_base:count_base + B] = names['seq_idx'].cpu().numpy()
        outputs['frame0'][count_base:count_base + B] = names['frame0'].cpu().numpy()
        outputs['frame1'][count_base:count_base + B] = names['frame1'].cpu().numpy()
        count_base += B
    return outputs

def test_object_warping(options):
    # loader = MovingObjects3D('', load_type='train', keyframes=[1])
    assert options.dataset == 'MovingObjects3D'
    keyframes = [int(x) for x in options.keyframes.split(',')]
    objects = ['boat', 'motorbike'] if options.object == '' else [options.object]
    eval_loaders = {}
    for test_object in objects:
        for kf in keyframes:
            np_loader = load_data(options.dataset, keyframes=[kf],
                                  load_type=options.eval_set,
                                  select_trajectory=test_object, options=options)
            eval_loaders['{:}_keyframe_{:}'.
                format(test_object, kf)] = data.DataLoader(np_loader,
                                                           batch_size=int(options.batch_per_gpu),
                                                           shuffle=False, num_workers = options.cpu_workers)
    use_gt_pose = options.gt_pose
    # method_list = {}
    if not use_gt_pose:
        tracker = select_method(options.method, options)
    else:
        tracker = None
        options.method = 'gt'
    method_list = {options.method: tracker}

    for k, loader in eval_loaders.items():
        for method_name in method_list:
        # method = method_list
            tracker = method_list[method_name]
            output_dir_method = osp.join(MOVING_OBJECTS_3D, 'visualization', method_name)
            output_dir = osp.join(output_dir_method, k)
            output_pkl = output_dir + '/pose.pkl'
            output_compose_dir = osp.join(output_dir, 'compose')
            output_input_dir = osp.join(output_dir, 'input')
            output_residual_dir = osp.join(output_dir, 'res')
            check_directory(output_pkl)
            check_directory(output_compose_dir + '/.png')
            check_directory(output_input_dir + '/.png')
            check_directory(output_residual_dir + '/.png')

            if options.recompute or not osp.isfile(output_pkl):
                info = compute_pose(loader, tracker, k, method_name, use_gt_pose)
                with open(output_pkl, 'wb') as output:
                    pickle.dump(info, output)
            else:
                with open(output_pkl, 'rb') as pkl_file:
                    info = pickle.load(pkl_file)
            # info = compute_pose(loader, tracker, k, method_name, use_gt_pose)

            # visualize residuals
            loader.dataset.fx_s = 1.0
            loader.dataset.fy_s = 1.0
            progress = tqdm(loader, ncols=100,
                            desc='compute residual for object {:} using {}'.format(k, method_name),
                            total=len(loader))
            count_base = 0
            for idx, batch in enumerate(progress):
                color0, color1, depth0, depth1, transform, K, mask0, mask1, names = check_cuda(batch)
                # color0, color1, depth0, depth1, transform, K, mask0, mask1, names = check_cuda(loader.dataset.get_original_size_batch(idx))
                B, C, H, W = color0.shape
                invD0 = 1.0 / depth0
                invD1 = 1.0 / depth1

                R = torch.stack(check_cuda(info['R_est'][count_base:count_base + B]), dim=0).float()
                t = torch.stack(check_cuda(info['t_est'][count_base:count_base + B]), dim=0).float()
                Rt = [R, t]
                # R_gt, t_gt = transform[:, :3, :3], transform[:, :3, 3]
                # Rt = [R_gt, t_gt]
                px, py = geometry.generate_xy_grid(B, H, W, K)
                u_warped, v_warped, inv_z_warped = geometry.batch_warp_inverse_depth(
                    px, py, invD0, Rt, K)
                x1_1to0 = geometry.warp_features(color1, u_warped, v_warped)
                crd = torch.cat((u_warped, v_warped), dim=1)
                occ = geometry.check_occ(inv_z_warped, invD1, crd, DIC_version=True)

                residuals = x1_1to0 - color0  # equation (12)
                # remove occlusion
                x1_1to0[occ.expand(B, C, H, W)] = 0

                if mask0 is not None:
                    bg_mask0 = ~mask0
                    res_occ = occ | (bg_mask0.view(B, 1, H, W))
                else:
                    res_occ = occ
                residuals[res_occ.expand(B, C, H, W)] = 0
                residuals = residuals.mean(dim=1, keepdim=True)

                # # for each image
                for idx in range(B):
                    feat_residual = display.create_mosaic([color0[idx:idx+1], color1[idx:idx+1], x1_1to0[idx:idx+1], residuals[idx:idx+1]],
                                                          cmap=['NORMAL', 'NORMAL', 'NORMAL', cv2.COLORMAP_JET],
                                                          order='CHW')
                    input0 = feat_residual[0:H, 0:W, :].copy()
                    input1 = feat_residual[0:H, W:, :].copy()
                    warped = feat_residual[H:, 0:W, :].copy()
                    res = feat_residual[H:, W:, :].copy()
                    obj_mask0 = mask0[idx:idx+1].squeeze().cpu().numpy().astype(np.uint8)*255
                    obj_mask1 = mask1[idx:idx+1].squeeze().cpu().numpy().astype(np.uint8)*255
                    contours0, hierarchy0 = cv2.findContours(obj_mask0, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    contours1, hierarchy1 = cv2.findContours(obj_mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                    input0 = cv2.drawContours(input0, contours0, -1, (0, 0, 255), 1)
                    input1 = cv2.drawContours(input1, contours0, -1, (0, 0, 255), 1)
                    input1 = cv2.drawContours(input1, contours1, -1, (0, 255, 0), 1)
                    overlay = cv2.drawContours(warped, contours0, -1, (0, 0, 255), 1)
                    overlay = cv2.drawContours(overlay, contours1, -1, (0, 255, 0), 1)

                    # visualization for debugging
                    if options.save_img:
                        idx_in_batch=count_base+idx
                        seq_idx = int(info['seq_idx'][idx_in_batch])
                        idx0 = int(info['frame0'][idx_in_batch])
                        idx1 = int(info['frame1'][idx_in_batch])
                        index = "_" + str.zfill(str(idx_in_batch), 5) + '.png'
                        image_name = osp.join(output_compose_dir, 'compose'+index)
                        cv2.imwrite(image_name, feat_residual)
                        overlay_name = osp.join(output_residual_dir, 'overlay'+index)
                        input_name = osp.join(output_input_dir, 'input0'+index)
                        cv2.imwrite(overlay_name, overlay)
                        cv2.imwrite(input_name, input0)
                        cv2.imwrite(input_name.replace('input0', "input1"), input1)

                        pair_dir = osp.join(output_dir, 'sequence',
                                            'seq' + str(seq_idx) + "_" + str(idx0) + "_" + str(idx1),
                                            )
                        check_directory(pair_dir + '/.png')
                        cv2.imwrite(overlay_name.replace('overlay', "residual"), res)
                        cv2.imwrite(pair_dir+"/overlay.png", overlay)
                        cv2.imwrite(pair_dir+"/input0.png", input0)
                        cv2.imwrite(pair_dir+"/input1.png", input1)
                        cv2.imwrite(pair_dir+"/residual.png", res)
                    else:
                        cv2.imshow("overlay", overlay)
                        cv2.imshow("input0", input0)
                        cv2.imshow("input1", input1)

                        window_name = "feature-metric residuals"
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.imshow(window_name, feat_residual)
                        # image_name = osp.join(output_dir, 'residual'+str(idx)+'.png')
                        # cv2.imwrite(image_name, feat_residual)
                        cv2.waitKey(0)
                count_base += B


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluate the network")
    config.add_basics_config(parser)
    config.add_tracking_config(parser)
    config.add_object_config(parser)
    options = parser.parse_args()

    print('---------------------------------------')
    test_object_warping(options)

