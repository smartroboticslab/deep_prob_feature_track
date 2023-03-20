""" 
An implementation of RGBD odometry using Open3D library for comparison in the paper
# SPDX-FileCopyrightText: 2021 Binbin Xu
# SPDX-License-Identifier: BSD-3-Clause
"""

import open3d as o3d
import numpy as np
import torch
import copy

class RGBDOdometry():

    def __init__(self, mode='RGBD'):

        self.odo_opt = None
        if mode == "RGBD":
            print("Using RGB-D Odometry")
            self.odo_opt = o3d.odometry.RGBDOdometryJacobianFromColorTerm()
        elif mode == "COLOR_ICP":
            print("Using Hybrid RGB-D Odometry")
            self.odo_opt = o3d.odometry.RGBDOdometryJacobianFromHybridTerm()
        else:
            raise NotImplementedError()

    def set_K(self, K, width, height):
        fx, fy, cx, cy = K
        K = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        return K

    def batch_track(self, batch_rgb0, batch_dpt0, batch_rgb1, batch_dpt1, batch_K,
                    batch_objmask0=None, batch_objmask1=None, vis_pcd=True):
        assert batch_rgb0.ndim == 4
        B = batch_rgb0.shape[0]
        batch_R = []
        batch_t = []
        if batch_objmask0 is not None:
            batch_dpt0 = batch_dpt0 * batch_objmask0
        if batch_objmask1 is not None:
            batch_dpt1 = batch_dpt1 * batch_objmask1
        for i in range(B):
            rgb0 = batch_rgb0[i].permute(1,2,0).cpu().numpy()
            dpt0 = batch_dpt0[i].permute(1,2,0).cpu().numpy()
            rgb1 = batch_rgb1[i].permute(1,2,0).cpu().numpy()
            dpt1 = batch_dpt1[i].permute(1,2,0).cpu().numpy()
            K = batch_K[i].cpu().numpy().tolist()
            pose10, _ = self.track(rgb0, dpt0, rgb1, dpt1, K)
            batch_R.append(pose10[0])
            batch_t.append(pose10[1])

        batch_R = torch.tensor(batch_R).type_as(batch_K)
        batch_t = torch.tensor(batch_t).type_as(batch_K)
        return batch_R, batch_t

    def draw_registration_result(self, source, target, transformation, name='Open3D'):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp], window_name=name)

    def track(self, rgb0, dpt0, rgb1, dpt1, K, vis_pcd=True, odo_init=None):
        H, W, _ = rgb0.shape
        intrinsic = self.set_K(K, H, W)
        rgbd_0 = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb0), o3d.geometry.Image(dpt0), depth_scale=1, depth_trunc=3.0)
        rgbd_1 = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb1), o3d.geometry.Image(dpt1), depth_scale=1, depth_trunc=3.0)
        if odo_init is None:
            odo_init = np.identity(4)
        if vis_pcd:
            pcd_0 = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_0, intrinsic)
            pcd_1 = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_1, intrinsic)

        # option = o3d.odometry.OdometryOption()
        option = o3d.odometry.OdometryOption(min_depth=0.01, max_depth_diff=1.0)
        # print(option)

        [is_success, T_10, info] = o3d.odometry.compute_rgbd_odometry(
            rgbd_0, rgbd_1, intrinsic,
            odo_init, self.odo_opt, option)

        trs = T_10[0:3, 3]
        if (trs>1).sum(): #is_success and vis_pcd:
            print(T_10)
            print(is_success)
            # pcd_0 = o3d.geometry.PointCloud.create_from_rgbd_image(
            #     rgbd_0, intrinsic)
            # pcd_0.transform(T_10)
            # o3d.visualization.draw_geometries([pcd_1, pcd_0])
            self.draw_registration_result(pcd_0, pcd_1, odo_init, 'init')
            self.draw_registration_result(pcd_0, pcd_1, T_10, 'aligned')

        trs = T_10[0:3, 3]
        rot = T_10[0:3, 0:3]
        pose10 = [rot, trs]

        return pose10, is_success