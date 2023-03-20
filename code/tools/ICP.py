""" 
An implementation of ICP odometry using Open3D library for comparison in the paper
# SPDX-FileCopyrightText: 2021 Binbin Xu
# SPDX-License-Identifier: BSD-3-Clause
"""

import open3d as o3d
import numpy as np
import torch
import copy
import cv2

class ICP_Odometry:

    def __init__(self, mode='Point2Plane'):
        self.mode = mode
        if mode == 'Point2Plane':
            print("Using Point-to-plane ICP")
        elif mode == 'Point2Point':
            print("Using Point-to-point ICP")
        elif mode == "ColorICP":
            print("using ColorICP")
        elif mode == 'Iter_Point2Plane':
            print("Using iterative Point-to-plane ICP")
        elif mode == "Iter_ColorICP":
            print("using iterative ColorICP")
        else:
            raise NotImplementedError()

    def set_K(self, K, width, height):
        fx, fy, cx, cy = K
        K = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        return K

    def batch_track(self, batch_rgb0, batch_dpt0, batch_rgb1, batch_dpt1, batch_K,
                    batch_objmask0=None, batch_objmask1=None, vis_pcd=False):
        assert batch_dpt0.ndim == 4
        B = batch_dpt0.shape[0]
        batch_R = []
        batch_t = []
        if batch_objmask0 is not None:
            batch_dpt0 = batch_dpt0 * batch_objmask0
        if batch_objmask1 is not None:
            batch_dpt1 = batch_dpt1 * batch_objmask1
        for i in range(B):
            rgb0 = batch_rgb0[i].permute(1, 2, 0).cpu().numpy()
            dpt0 = batch_dpt0[i].permute(1,2,0).cpu().numpy()
            rgb1 = batch_rgb1[i].permute(1, 2, 0).cpu().numpy()
            dpt1 = batch_dpt1[i].permute(1,2,0).cpu().numpy()
            K = batch_K[i].cpu().numpy().tolist()
            pose10 = self.track(rgb0, dpt0, rgb1, dpt1, K)
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
        H, W, _ = dpt0.shape
        intrinsic = self.set_K(K, H, W)
        # pcd_0 = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(dpt0),
        #                                                         intrinsic=intrinsic,
        #                                                         depth_scale=1.0,
        #                                                         depth_trunc=5.0)
        # pcd_1 = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(dpt1),
        #                                                         intrinsic=intrinsic,
        #                                                         depth_scale=1.0,
        #                                                         depth_trunc=5.0)
        rgbd_0 = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb0), o3d.geometry.Image(dpt0), depth_scale=1, depth_trunc=4.0)
        rgbd_1 = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb1), o3d.geometry.Image(dpt1), depth_scale=1, depth_trunc=4.0)
        pcd_0 = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_0, intrinsic)
        pcd_1 = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_1, intrinsic)

        if odo_init is None:
            odo_init = np.identity(4)

        # point-to-point ICP
        if self.mode == 'Point2Point':
            reg_p2p = o3d.registration.registration_icp(
                pcd_0, pcd_1, 0.02, odo_init,
                o3d.registration.TransformationEstimationPointToPoint())
            T_10 = reg_p2p.transformation

        # point-to-plane ICP
        elif self.mode == 'Point2Plane':
            # radius = 0.01
            # source_down = pcd_0.voxel_down_sample(radius)
            # target_down = pcd_1.voxel_down_sample(radius)
            #
            # # print("3-2. Estimate normal.")
            # source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
            # target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
            # reg_p2l = o3d.registration.registration_icp(source_down, target_down, 0.2, odo_init,
            #                                             o3d.registration.TransformationEstimationPointToPlane(),
            #                                             o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
            #                                                                                     relative_rmse=1e-6,
            #                                                                                     max_iteration=50)
            #
            #                                             )
            iter = 10
            pcd_0.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30))
            pcd_1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30))
            reg_p2l = o3d.registration.registration_icp(
                pcd_0, pcd_1, 0.4, odo_init,
                o3d.registration.TransformationEstimationPointToPlane(),
                o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                        relative_rmse=1e-6,
                                                        max_iteration=iter)
            )

            T_10 = reg_p2l.transformation

        elif self.mode == 'ColorICP':
            pcd_0.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30))
            pcd_1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30))
            reg_p2l = o3d.registration.registration_colored_icp(
                pcd_0, pcd_1, 0.02, odo_init,)
            T_10 = reg_p2l.transformation

        elif self.mode in ['Iter_Point2Plane', 'Iter_ColorICP']:
            voxel_radius = [0.04, 0.02, 0.01]
            max_iter = [50, 30, 14]
            T_10 = odo_init
            for scale in range(3):
                iter = max_iter[scale]
                radius = voxel_radius[scale]

                pcd0_down = pcd_0.voxel_down_sample(radius)
                pcd1_down = pcd_1.voxel_down_sample(radius)

                pcd0_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                pcd1_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

                # point-to-plane ICP
                if self.mode == 'Iter_Point2Plane':
                    result_icp = o3d.registration.registration_icp(
                        pcd0_down, pcd1_down, radius, T_10,
                        o3d.registration.TransformationEstimationPointToPlane(),
                        o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                relative_rmse=1e-6,
                                                                max_iteration=iter)
                    )
                elif self.mode == 'Iter_ColorICP':
                    # colored ICP
                    result_icp = o3d.registration.registration_colored_icp(
                        pcd0_down, pcd1_down, radius*2, T_10,
                        o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                relative_rmse=1e-6,
                                                                max_iteration=iter))
                else:
                    raise NotImplementedError
                T_10 = result_icp.transformation
        else:
            raise NotImplementedError()

        # T_10 = result_icp.transformation
        trs = T_10[0:3, 3]
        rot = T_10[0:3, 0:3]
        pose10 = [rot, trs]

        if (trs > 1).sum():
            print('pose', T_10)

            # cv2.imshow('rgb0', rgb0)
            # cv2.imshow('rgb1', rgb1)
            # cv2.waitKey(0)
            # self.draw_registration_result(pcd_0, pcd_1, odo_init, name='init')
            # self.draw_registration_result(pcd_0, pcd_1, T_10, name='aligned')

            T_10 = odo_init
            trs = T_10[0:3, 3]
            rot = T_10[0:3, 0:3]
            pose10 = [rot, trs]


        return pose10