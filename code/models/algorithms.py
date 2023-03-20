"""
The algorithm backbone, primarily the contributions proposed in our paper

# SPDX-FileCopyrightText: 2021 Binbin Xu
# SPDX-License-Identifier: BSD-3-Clause

@author: Zhaoyang Lv
@date: March, 2019
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as func

import models.geometry as geometry
from models.submodules import convLayer as conv
from models.submodules import convLayer1d as conv1d
from models.submodules import fcLayer, initialize_weights
from tools import display
import cv2
import numpy as np

class TrustRegionBase(nn.Module):
    """ 
    This is the the base function of the trust-region based inverse compositional algorithm. 
    """
    def __init__(self,
        max_iter    = 3,
        mEst_func   = None,
        solver_func = None,
        timers      = None):
        """
        :param max_iter, maximum number of iterations
        :param mEst_func, the M-estimator function / network 
        :param solver_func, the trust-region function / network
        :param timers, if yes, counting time for each step
        """
        super(TrustRegionBase, self).__init__()

        self.max_iterations = max_iter
        self.mEstimator     = mEst_func
        self.directSolver   = solver_func
        self.timers         = timers

    def forward(self, pose, x0, x1, invD0, invD1, K, wPrior=None, vis_res=False,
                obj_mask0=None, obj_mask1=None):
        """
        :param pose, the initial pose
            (extrinsic of the target frame w.r.t. the referenc frame)
        :param x0, the template features
        :param x1, the image features
        :param invD0, the template inverse depth
        :param invD1, the image inverse depth
        :param K, the intrinsic parameters, [fx, fy, cx, cy]
        :param wPrior (optional), provide an initial weight as input to the convolutional m-estimator
        """
        B, C, H, W = x0.shape
        px, py = geometry.generate_xy_grid(B,H,W,K)

        if self.timers: self.timers.tic('pre-compute Jacobians')
        J_F_p = self.precompute_Jacobian(invD0, x0, px, py, K)  # [B, HXW, 6]
        if self.timers: self.timers.toc('pre-compute Jacobians')

        if self.timers: self.timers.tic('compute warping residuals')
        residuals, occ = compute_warped_residual(pose, invD0, invD1, \
            x0, x1, px, py, K, obj_mask0=obj_mask0, obj_mask1=obj_mask1)  # [B, 1, H, W]
        if self.timers: self.timers.toc('compute warping residuals')

        if self.timers: self.timers.tic('robust estimator')
        weights = self.mEstimator(residuals, x0, x1, wPrior)  # [B, C, H, W]
        wJ = weights.view(B,-1,1) * J_F_p    # [B, HXW, 6]
        if self.timers: self.timers.toc('robust estimator')

        if self.timers: self.timers.tic('pre-compute JtWJ')
        JtWJ = torch.bmm(torch.transpose(J_F_p, 1, 2) , wJ)  # [B, 6, 6]
        if self.timers: self.timers.toc('pre-compute JtWJ')

        for idx in range(self.max_iterations):
            if self.timers: self.timers.tic('solve x=A^{-1}b')
            pose = self.directSolver(JtWJ,
                torch.transpose(J_F_p,1,2), weights, residuals,
                pose, invD0, invD1, x0, x1, K, obj_mask1=obj_mask1)
            if self.timers: self.timers.toc('solve x=A^{-1}b')
    
            if self.timers: self.timers.tic('compute warping residuals')
            residuals, occ = compute_warped_residual(pose, invD0, invD1, \
                x0, x1, px, py, K, obj_mask1=obj_mask1)
            if self.timers: self.timers.toc('compute warping residuals')

            if vis_res:
                with torch.no_grad():
                    u_warped, v_warped, inv_z_warped = geometry.batch_warp_inverse_depth(
                        px, py, invD0, pose, K)
                    x1_1to0 = geometry.warp_features(x1, u_warped, v_warped)
                    feat_residual = display.create_mosaic([x0, x1, x1_1to0, residuals],
                                                           cmap=['NORMAL', 'NORMAL', 'NORMAL', cv2.COLORMAP_JET],
                                                           order='CHW')
                    cv2.namedWindow("feature-metric residuals", cv2.WINDOW_NORMAL)
                    cv2.imshow("feature-metric residuals", feat_residual)
                    cv2.waitKey(10)
        return pose, weights

    def precompute_Jacobian(self, invD, x, px, py, K):
        """ Pre-compute the image Jacobian on the reference frame
        refer to equation (13) in the paper
        
        :param invD, template depth
        :param x, template feature
        :param px, normalized image coordinate in cols (x)
        :param py, normalized image coordinate in rows (y)
        :param K, the intrinsic parameters, [fx, fy, cx, cy]

        ------------
        :return precomputed image Jacobian on template
        """
        Jf_x, Jf_y = feature_gradient(x)   # [B, 1, H, W], [B, 1, H, W]
        Jx_p, Jy_p = compute_jacobian_warping(invD, K, px, py)  # [B, HXW, 6], [B, HXW, 6]
        J_F_p = compute_jacobian_dIdp(Jf_x, Jf_y, Jx_p, Jy_p)  # [B, HXW, 6]
        # Use the jacobian from the original paper, have a sign difference with the true jacobian
        # J_F_p = - J_F_p
        return J_F_p

    def forward_residuals(self, pose, x0, x1, invD0, invD1, K, wPrior=None,
                          vis_res=False, obj_mask0=None, obj_mask1=None):
        B, C, H, W = x0.shape
        px, py = geometry.generate_xy_grid(B, H, W, K)

        residuals, occ = compute_warped_residual(pose, invD0, invD1, \
                                                 x0, x1, px, py, K,
                                                 obj_mask0=obj_mask0,
                                                 obj_mask1=obj_mask1)  # [B, 1, H, W]

        # weighting via learned robust cost function
        weights = self.mEstimator(residuals, x0, x1, wPrior)  # [B, C, H, W]
        residuals = (weights * residuals)

        loss = compute_avg_loss([residuals, ], occ)

        return loss


class TrustRegionICP(nn.Module):
    def __init__(self,
                 max_iter=3,
                 mEst_func=None,
                 solver_func=None,
                 timers=None):
        """
        :param max_iter, maximum number of iterations
        :param mEst_func, the M-estimator function / network
        :param solver_func, the trust-region function / network
        :param timers, if yes, counting time for each step
        """
        super(TrustRegionICP, self).__init__()

        self.max_iterations = max_iter
        self.mEstimator = mEst_func
        self.directSolver = solver_func
        self.timers = timers

    def forward(self, pose10, depth0, depth1, K, wPrior=None, vis_res=False, stop_loss_inc=False, obj_mask1=None):
        """
        :param pose, the initial pose
            (extrinsic of the target frame w.r.t. the referenc frame)
        :param x0, the template features
        :param x1, the image features
        :param invD0, the template inverse depth
        :param invD1, the image inverse depth
        :param K, the intrinsic parameters, [fx, fy, cx, cy]
        :param wPrior (optional), provide an initial weight as input to the convolutional m-estimator
        """
        B, C, H, W = depth0.shape
        px, py = geometry.generate_xy_grid(B, H, W, K)
        vertex0 = geometry.compute_vertex(depth0, px, py)
        vertex1 = geometry.compute_vertex(depth1, px, py)
        normal0 = compute_normal(vertex0)
        normal1 = compute_normal(vertex1)
        # pose01 = [geometry.batch_inverse_Rt(pose10[0], pose10[1])]

        #  # visualize surface normal image.
        # img = (((normal1.permute(0,2,3,1)[0,:,:,:]+1.0)*128.0).cpu().numpy()).astype('uint8')
        # cv2.imshow('normal', img)
        # cv2.waitKey(0)

        lowes_err = 1e10
        for idx in range(self.max_iterations):
            # compute residuals
            if self.timers: self.timers.tic('compute warping residuals')
            residuals, J_F_p, occ = self.compute_residuals_jacobian(vertex0, vertex1, normal0, normal1, pose10, K)
            if self.timers: self.timers.toc('compute warping residuals')

            if stop_loss_inc:
                err = compute_avg_res(residuals, occ)
                print(err)
                if err < lowes_err:
                    lowes_err = err
                else:
                    weights = torch.ones(residuals.shape).type_as(residuals)
                    return pose10, weights

            if self.timers: self.timers.tic('pre-compute JtWJ')
            JtWJ = self.compute_jtj(J_F_p)  # [B, 6, 6]
            JtR = self.compute_jtr(J_F_p, residuals)
            if self.timers: self.timers.toc('pre-compute JtWJ')

            if self.timers: self.timers.tic('solve x=A^{-1}b')
            pose10 = self.GN_solver(JtWJ, JtR, pose10)
            if self.timers: self.timers.toc('solve x=A^{-1}b')

        weights = torch.ones(residuals.shape).type_as(residuals)
        # print('---')
        return pose10, weights

    def compute_residuals_jacobian(self, vertex0, vertex1, normal0, normal1, pose10, K):
        R, t = pose10
        B, C, H, W = vertex0.shape

        rot_vertex0_to1 = torch.bmm(R, vertex0.view(B, 3, H*W))
        vertex0_to1 = rot_vertex0_to1 + t.view(B, 3, 1).expand(B, 3, H*W)
        normal0_to1 = torch.bmm(R, normal0.view(B, 3, H * W))

        fx, fy, cx, cy = torch.split(K, 1, dim=1)
        x_, y_, s_ = torch.split(vertex0_to1, 1, dim=1)
        u_ = (x_ / s_).view(B, -1) * fx + cx
        v_ = (y_ / s_).view(B, -1) * fy + cy

        inviews = (u_ > 0) & (u_ < W-1) & (v_ > 0) & (v_ < H-1)

        # # interpolation-version
        r_vertex1 = geometry.warp_features(vertex1, u_, v_)
        r_normal1 = geometry.warp_features(normal1, u_, v_)

        # round-version
        # u_i = (u_ + 0.5).int().type_as(vertex0)
        # v_i = (v_ + 0.5).int().type_as(vertex0)
        # u_i = u_.floor()  #  int().todouble()
        # v_i = v_.floor()  # int().double()
        # r_vertex1 = geometry.warp_features(vertex1, u_i, v_i)
        # r_normal1 = geometry.warp_features(normal1, u_i, v_i)

        diff = vertex0_to1 - r_vertex1.view(B, 3, H * W)
        normal_diff = (normal0_to1 * r_normal1.view(B, 3, H * W)).sum(dim=1, keepdim=True)

        # occlusion
        occ = ~inviews.view(B,1,H,W) | (diff.view(B,3,H,W).norm(p=2, dim=1, keepdim=True) > 0.1) #| \
              # (normal_diff.view(B,1,H,W) < 0.8)   # since normal is estimated from the noise depth, might not be very useful

        # point-to-plane residuals
        res = (r_normal1.view(B, 3, H*W)) * diff
        res = res.sum(dim=1, keepdim=True).view(B,1,H,W)  # [B,1,H,W]
        # jpoint-to-plane  acobians
        J_trs = r_normal1.view(B,3,-1).permute(0,2,1).contiguous().view(-1,3)  # [B*H*W, 3]
        J_rot = -torch.bmm(J_trs.unsqueeze(dim=1),
                           geometry.batch_skew(vertex0_to1.permute(0,2,1).contiguous().view(-1,3))).squeeze()   # [B*H*W, 3]

        # compose jacobians
        J_F_p = torch.cat((J_rot, J_trs), dim=-1)  # follow the order of [rot, trs]  [B*H*W, 6]
        J_F_p = J_F_p.view(B, 1, -1, 6)  # [B, 1, HXW, 6]

        # # point-to-point residuals
        # res = diff.view(B,3,H,W)  # [B,3,HXW]
        # # point-to-point jacobians
        # J_trs = torch.eye(3).type_as(res).view(1,1,3,3).expand(B,H*W,3,3)
        # J_rot = -geometry.batch_skew(rot_vertex0_to1.permute(0,2,1).contiguous().view(-1,3))
        # J_rot = J_rot.view(B, H*W, 3, 3)
        # # compose jacobians
        # J_F_p = torch.cat((J_rot, J_trs), dim=-1)  # follow the order of [rot, trs]  [B, H*W, 3, 6]
        # J_F_p = J_F_p.permute(0, 2, 1, 3)  # [B, 3, HXW, 6]
        # occ = occ.expand(B,3,H,W)

        # covariance-normalized
        dpt0 = vertex0[:,2:3,:,:]
        sigma_icp = self.compute_sigma(dpt_l=dpt0, normal_r=r_normal1, rot=R)
        res = res / (sigma_icp + 1e-8)
        J_F_p = J_F_p / (sigma_icp.view(B,1,H*W,1) + 1e-8)

        # follow the conversion of inversing the jacobian
        J_F_p = - J_F_p

        res[occ] = 1e-6

        return res, J_F_p, occ

    def compute_sigma(self, dpt_l, normal_r, rot, dataset='TUM'):
        # obtain sigma
        if dataset == 'TUM':
            sigma_disp = 0.4  # 5.5
            sigma_xy = 5.5  # 5.5
            baseline = 1.0  #0.075
            focal = 525.0
        else:
            raise NotImplementedError()

        B, C, H, W = normal_r.shape

        # compute sigma on depth using stereo model
        sigma_depth = torch.empty((B, 3, H, W)).type_as(dpt_l)
        sigma_depth[:, 0:2, :, :] = dpt_l / focal * sigma_xy
        sigma_depth[:, 2:3, :, :] = dpt_l * dpt_l * sigma_disp / (focal * baseline)

        # compute sigma for icp loss  # fast computation of dot product
        # cov_icp = (sigma_depth ** 2).sum(dim=1, keepdim=True)

        # sigma_icp = normal_r * sigma_depth
        # cov_icp = (sigma_icp ** 2).sum(dim=1, keepdim=True)

        # sigma_icp = torch.bmm(rot, sigma_depth.view(B, 3, H * W))
        # cov_icp = (sigma_icp.view(B,3,H,W) ** 2).sum(dim=1, keepdim=True)

        J = torch.bmm(normal_r.view(B,3,H*W).transpose(1,2), rot)
        J = J.transpose(1,2).view(B,3,H,W)
        cov_icp = (J * sigma_depth * sigma_depth * J).sum(dim=1, keepdim=True)

        sigma_icp = torch.sqrt(cov_icp + 1e-8)
        return sigma_icp

    def compute_jtj(self, jac):
        # J in the dimension of (B, C, HW, y)
        B, C, HW, y = jac.shape
        jac_reshape2 = jac.permute(0, 2, 1, 3).contiguous()  # [B, HW, C, 6]
        jac_reshape2 = jac_reshape2.view(-1, C, y)  # [B*HW, C, 6]
        jtj2 = torch.bmm(torch.transpose(jac_reshape2, 1, 2), jac_reshape2)  # [B*HW, 6, 6]
        jtj2 = jtj2.view(B, HW, y, y)
        jtj2 = jtj2.sum(dim=1)
        return jtj2  # [B, 6, 6]

    def compute_jtr(self, jac, res):
        # J in the dimension of (B, C, HW, y)
        # res in the dimension of [B, C, H, W]
        B,C,H,W = res.shape
        res = res.view(B, C, H*W, 1).permute(0,2,1,3).contiguous()  # [B, HW, C, 1]
        res = res.view(-1,C,1)  # [B*HW, C, 1]
        jac_reshape = jac.permute(0, 2, 1, 3).contiguous()  # [B, HW, C, 6]
        jac_reshape = jac_reshape.view(-1, C, 6)  # [B*HW, C, 6]

        jtr = torch.bmm(torch.transpose(jac_reshape, 1, 2), res)  # [B*HW, 6, 1]
        jtr = jtr.view(B, H*W, 6, 1)
        jtr = jtr.sum(dim=1)
        return jtr  # [B, 6, 1]

    def GN_solver(self, JtJ, JtR, pose0, direction='inverse'):

        B = JtJ.shape[0]

        # Add a small diagonal damping. Without it, the training becomes quite unstable
        # Do not see a clear difference by removing the damping in inference though
        Hessian = lev_mar_H(JtJ)
        # Hessian = JtJ

        updated_pose = forward_update_pose(Hessian, JtR, pose0)

        return updated_pose


class Inverse_ICP(nn.Module):
    def __init__(self,
                 max_iter=3,
                 mEst_func=None,
                 solver_func=None,
                 timers=None):
        """
        :param max_iter, maximum number of iterations
        :param mEst_func, the M-estimator function / network
        :param solver_func, the trust-region function / network
        :param timers, if yes, counting time for each step
        """
        super(Inverse_ICP, self).__init__()

        self.max_iterations = max_iter
        self.mEstimator = mEst_func
        self.directSolver = solver_func
        self.timers = timers

    def forward(self, pose10, depth0, depth1, K, wPrior=None, vis_res=False, stop_loss_inc=False, obj_mask1=None):
        """
        :param pose, the initial pose
            (extrinsic of the target frame w.r.t. the referenc frame)
        :param x0, the template features
        :param x1, the image features
        :param invD0, the template inverse depth
        :param invD1, the image inverse depth
        :param K, the intrinsic parameters, [fx, fy, cx, cy]
        :param wPrior (optional), provide an initial weight as input to the convolutional m-estimator
        """
        B, C, H, W = depth0.shape
        px, py = geometry.generate_xy_grid(B, H, W, K)
        vertex0 = geometry.compute_vertex(depth0, px, py)
        vertex1 = geometry.compute_vertex(depth1, px, py)
        normal0 = compute_normal(vertex0)
        normal1 = compute_normal(vertex1)
        # pose01 = [geometry.batch_inverse_Rt(pose10[0], pose10[1])]

        #  # visualize surface normal image.
        # img = (((normal1.permute(0,2,3,1)[0,:,:,:]+1.0)*128.0).cpu().numpy()).astype('uint8')
        # cv2.imshow('normal', img)
        # cv2.waitKey(0)

        lowes_err = 1e10
        for idx in range(self.max_iterations):
            # compute residuals
            if self.timers: self.timers.tic('compute warping residuals')
            residuals, J_F_p, occ = self.compute_residuals_jacobian(vertex0, vertex1, normal0, normal1, pose10, K)
            if self.timers: self.timers.toc('compute warping residuals')

            if stop_loss_inc:
                err = compute_avg_res(residuals, occ)
                print(err)
                if err < lowes_err:
                    lowes_err = err
                else:
                    weights = torch.ones(residuals.shape).type_as(residuals)
                    return pose10, weights

            if self.timers: self.timers.tic('pre-compute JtWJ')
            JtWJ = self.compute_jtj(J_F_p)  # [B, 6, 6]
            JtR = self.compute_jtr(J_F_p, residuals)
            if self.timers: self.timers.toc('pre-compute JtWJ')

            if self.timers: self.timers.tic('solve x=A^{-1}b')
            pose10 = self.GN_solver(JtWJ, JtR, pose10)
            if self.timers: self.timers.toc('solve x=A^{-1}b')

        weights = torch.ones(residuals.shape).type_as(residuals)
        # print('---')
        return pose10, weights

    def forward_residuals(self, pose10, depth0, depth1, K, wPrior=None, vis_res=False, obj_mask1=None):
        """
        :param pose, the initial pose
            (extrinsic of the target frame w.r.t. the referenc frame)
        :param x0, the template features
        :param x1, the image features
        :param invD0, the template inverse depth
        :param invD1, the image inverse depth
        :param K, the intrinsic parameters, [fx, fy, cx, cy]
        :param wPrior (optional), provide an initial weight as input to the convolutional m-estimator
        """
        B, C, H, W = depth0.shape
        px, py = geometry.generate_xy_grid(B, H, W, K)
        vertex0 = geometry.compute_vertex(depth0, px, py)
        vertex1 = geometry.compute_vertex(depth1, px, py)
        normal0 = compute_normal(vertex0)
        normal1 = compute_normal(vertex1)
        # pose01 = [geometry.batch_inverse_Rt(pose10[0], pose10[1])]

        # compute residuals and average loss
        residuals, _, occ = self.compute_residuals_jacobian(vertex0, vertex1, normal0, normal1, pose10, K)
        loss = compute_avg_loss([residuals, ], occ)

        return loss

    def compute_residuals_jacobian(self, vertex0, vertex1, normal0, normal1, pose10, K,
                                   normalize_residual=True):
        R, t = pose10
        B, C, H, W = vertex0.shape

        rot_vertex0_to1 = torch.bmm(R, vertex0.view(B, 3, H*W))
        vertex0_to1 = rot_vertex0_to1 + t.view(B, 3, 1).expand(B, 3, H*W)
        normal0_to1 = torch.bmm(R, normal0.view(B, 3, H * W))

        fx, fy, cx, cy = torch.split(K, 1, dim=1)
        x_, y_, s_ = torch.split(vertex0_to1, 1, dim=1)
        u_ = (x_ / s_).view(B, -1) * fx + cx
        v_ = (y_ / s_).view(B, -1) * fy + cy

        inviews = (u_ > 0) & (u_ < W-1) & (v_ > 0) & (v_ < H-1)

        # # interpolation-version
        r_vertex1 = geometry.warp_features(vertex1, u_, v_)
        r_normal1 = geometry.warp_features(normal1, u_, v_)

        # round-version
        # u_i = (u_ + 0.5).int().type_as(vertex0)
        # v_i = (v_ + 0.5).int().type_as(vertex0)
        # u_i = u_.floor()  #  int().todouble()
        # v_i = v_.floor()  # int().double()
        # r_vertex1 = geometry.warp_features(vertex1, u_i, v_i)
        # r_normal1 = geometry.warp_features(normal1, u_i, v_i)

        diff = vertex0_to1 - r_vertex1.view(B, 3, H * W)
        normal_diff = (normal0_to1 * r_normal1.view(B, 3, H * W)).sum(dim=1, keepdim=True)

        # occlusion
        occ = ~inviews.view(B,1,H,W) | (diff.view(B,3,H,W).norm(p=2, dim=1, keepdim=True) > 0.1) #| \
              # (normal_diff.view(B,1,H,W) < 0.8)   # since normal is estimated from the noise depth, might not be very useful

        # point-to-plane residuals
        res = (r_normal1.view(B, 3, H*W)) * diff
        res = res.sum(dim=1, keepdim=True).view(B,1,H,W)  # [B,1,H,W]
        # inverse point-to-plane jacobians
        NtC10 = torch.bmm(r_normal1.view(B,3,-1).permute(0,2,1), R)  # [B, H*W, 3]
        J_rot = torch.bmm(NtC10.view(-1,3).unsqueeze(dim=1),  #[B*H*W,1,3]
                           geometry.batch_skew(vertex0.view(B,3,-1).permute(0, 2, 1).contiguous().view(-1, 3))).squeeze()  # [B*H*W, 3]
        J_trs = -NtC10.view(-1,3)  # [B*H*W, 3]

        # compose jacobians
        J_F_p = torch.cat((J_rot, J_trs), dim=-1)  # follow the order of [rot, trs]  [B*H*W, 6]
        J_F_p = J_F_p.view(B, 1, -1, 6)  # [B, 1, HXW, 6]

        # covariance-normalized
        if normalize_residual:
            dpt0 = vertex0[:,2:3,:,:]
            sigma_icp = self.compute_sigma(dpt_l=dpt0, normal_r=r_normal1, rot=R)
            w_res = res / (sigma_icp + 1e-8)
            J_F_p = J_F_p / (sigma_icp.view(B,1,H*W,1) + 1e-8)
        else:
            w_res = res
            J_F_p = J_F_p

        # follow the conversion of inversing the jacobian
        J_F_p = - J_F_p

        w_res[occ] = 1e-6

        return w_res, J_F_p, occ

    def compute_sigma(self, dpt_l, normal_r, rot, dataset='TUM'):
        # obtain sigma
        if dataset == 'TUM':
            sigma_disp = 0.4  # 5.5
            sigma_xy = 5.5  # 5.5
            baseline = 1.2  #0.075
            focal = 525.0
        else:
            raise NotImplementedError()

        B, C, H, W = normal_r.shape

        # compute sigma on depth using stereo model
        sigma_depth = torch.empty((B, 3, H, W)).type_as(dpt_l)
        sigma_depth[:, 0:2, :, :] = dpt_l / focal * sigma_xy
        sigma_depth[:, 2:3, :, :] = dpt_l * dpt_l * sigma_disp / (focal * baseline)

        J = torch.bmm(normal_r.view(B,3,H*W).transpose(1,2), rot)
        J = J.transpose(1,2).view(B,3,H,W)
        cov_icp = (J * sigma_depth * sigma_depth * J).sum(dim=1, keepdim=True)

        sigma_icp = torch.sqrt(cov_icp + 1e-8)
        return sigma_icp

    def compute_jtj(self, jac):
        # J in the dimension of (B, C, HW, y)
        B, C, HW, y = jac.shape
        jac_reshape2 = jac.permute(0, 2, 1, 3).contiguous()  # [B, HW, C, 6]
        jac_reshape2 = jac_reshape2.view(-1, C, y)  # [B*HW, C, 6]
        jtj2 = torch.bmm(torch.transpose(jac_reshape2, 1, 2), jac_reshape2)  # [B*HW, 6, 6]
        jtj2 = jtj2.view(B, HW, y, y)
        jtj2 = jtj2.sum(dim=1)
        return jtj2  # [B, 6, 6]

    def compute_jtr(self, jac, res):
        # J in the dimension of (B, C, HW, y)
        # res in the dimension of [B, C, H, W]
        B,C,H,W = res.shape
        res = res.view(B, C, H*W, 1).permute(0,2,1,3).contiguous()  # [B, HW, C, 1]
        res = res.view(-1,C,1)  # [B*HW, C, 1]
        jac_reshape = jac.permute(0, 2, 1, 3).contiguous()  # [B, HW, C, 6]
        jac_reshape = jac_reshape.view(-1, C, 6)  # [B*HW, C, 6]

        jtr = torch.bmm(torch.transpose(jac_reshape, 1, 2), res)  # [B*HW, 6, 1]
        jtr = jtr.view(B, H*W, 6, 1)
        jtr = jtr.sum(dim=1)
        return jtr  # [B, 6, 1]

    def GN_solver(self, JtJ, JtR, pose0, direction='inverse'):

        B = JtJ.shape[0]

        # Add a small diagonal damping. Without it, the training becomes quite unstable
        # Do not see a clear difference by removing the damping in inference though
        Hessian = lev_mar_H(JtJ)
        # Hessian = JtJ

        updated_pose = inverse_update_pose(Hessian, JtR, pose0)

        return updated_pose


class TrustRegionInverseWUncertainty(nn.Module):
    """
    Direct Dense tracking based on trust region and feature-metric uncertainty
    """

    def __init__(self,
                 max_iter=3,
                 mEst_func=None,
                 solver_func=None,
                 timers=None,
                 uncer_prop=False,
                 combine_icp=False,
                 scale_func=None,
                 remove_tru_sigma=False,
                 ):
        """
        :param max_iter, maximum number of iterations
        :param mEst_func, the M-estimator function / network
        :param solver_func, the trust-region function / network
        :param timers, if yes, counting time for each step
        """
        super(TrustRegionInverseWUncertainty, self).__init__()

        self.max_iterations = max_iter
        self.mEstimator = mEst_func
        self.directSolver = solver_func
        self.timers = timers
        self.uncer_prop = uncer_prop
        self.combine_icp = combine_icp
        self.scale_func = scale_func
        self.remove_tru_sigma = remove_tru_sigma  # remove truncated uncertainty

    def forward(self, pose10, x0, x1, invD0, invD1, K, sigma0, sigma1, wPrior=None, 
                depth0=None, depth1=None, vis_res=True, obj_mask0=None, obj_mask1=None):
        """
        :param pose10, the initial pose
            (extrinsic of the target frame w.r.t. the referenc frame)
        :param x0, the template features
        :param x1, the image features
        :param invD0, the template inverse depth
        :param invD1, the image inverse depth
        :param K, the intrinsic parameters, [fx, fy, cx, cy]
        :param wPrior (optional), provide an initial weight as input to the convolutional scaler
        """
        assert sigma0 is not None and sigma1 is not None

        B, C, H, W = x0.shape
        px, py = geometry.generate_xy_grid(B, H, W, K)

        if self.combine_icp:
            assert depth0 is not None and depth1 is not None
            if self.timers: self.timers.tic('compute vertex and normal')
            vertex0 = geometry.compute_vertex(depth0, px, py)
            vertex1 = geometry.compute_vertex(depth1, px, py)
            normal0 = compute_normal(vertex0)
            normal1 = compute_normal(vertex1)
            if self.timers: self.timers.toc('compute vertex and normal')

        if self.timers: self.timers.tic('compute pre-computable Jacobian components')
        grad_f0, grad_sigma0, Jx_p, Jy_p = self.precompute_jacobian_components(invD0, x0, sigma0, px, py, K)
        if self.timers: self.timers.toc('compute pre-computable Jacobian components')

        #
        # if self.timers: self.timers.tic('robust estimator')
        # weights = self.mEstimator(weighted_res, x0, x1, wPrior)  # [B, C, H, W]
        # if self.timers: self.timers.toc('robust estimator')
        w_icp = None
        for idx in range(self.max_iterations):
            if self.timers: self.timers.tic('compute warping residuals')
            weighted_res, res, \
            sigma, occ= compute_inverse_residuals(pose10, invD0, invD1,
                                                  x0, x1, sigma0, sigma1, px, py, K,
                                                  obj_mask0=obj_mask0, obj_mask1=obj_mask1,
                                                  remove_truncate_uncertainty=self.remove_tru_sigma)  #[B,C,H,W]
            if self.timers: self.timers.toc('compute warping residuals')

            # compute batch-wise average residual
            # print("feat:", compute_avg_res(weighted_res, occ))

            if self.timers: self.timers.tic('compose Jacobian components')
            J_F_p, _, _ = self.compose_inverse_jacobians(res, sigma, sigma0, grad_f0, grad_sigma0, Jx_p, Jy_p)  # [B,C,HXW,6]
            if self.timers: self.timers.toc('compose Jacobian components')

            if self.timers: self.timers.tic('compute JtWJ and JtR')
            # wJ = weights.view(B, -1, 1) * J_F_p  # [B,CXHXW,6]
            JtWJ = self.compute_jtj(J_F_p)  # [B, 6, 6]
            JtR = self.compute_jtr(J_F_p, weighted_res)  # [B, 6, 1]
            if self.timers: self.timers.toc('compute JtWJ and JtR')

            if self.combine_icp:
                if self.timers: self.timers.tic('compute ICP residuals and jacobians')
                icp_residuals, icp_J, icp_occ = self.compute_ICP_residuals_jacobian(vertex0, vertex1, normal0, normal1, pose10, K,
                                                                                    obj_mask0=obj_mask0, obj_mask1=obj_mask1)
                if self.timers: self.timers.toc('compute ICP residuals and jacobians')

                # use the scale computed at the first iteration
                # @TODO: test if we should also scale feature residuals
                if self.timers: self.timers.tic('compute scale function')
                if idx == 0 or w_icp is None:
                    w_icp = self.scale_func(icp_residuals, weighted_res, wPrior)  # [B,1,H,W]
                icp_residuals = w_icp * icp_residuals
                icp_J = w_icp.view(B,1,H*W,1) * icp_J
                if self.timers: self.timers.toc('compute scale function')
                # print("icp:", compute_avg_res(icp_residuals, icp_occ))

                if self.timers: self.timers.tic('compute ICP JtWJ and JtR')
                icp_JtWJ = self.compute_jtj(icp_J)  # [B, 6, 6]
                icp_JtR = self.compute_jtr(icp_J, icp_residuals)
                JtWJ = JtWJ + icp_JtWJ
                JtR = JtR + icp_JtR
                if self.timers: self.timers.toc('compute ICP JtWJ and JtR')

            if self.timers: self.timers.tic('solve x=A^{-1}b')
            pose10 = self.GN_solver(JtWJ, JtR, pose10)
            if self.timers: self.timers.toc('solve x=A^{-1}b')

            if vis_res:
                with torch.no_grad():
                    u_warped, v_warped, inv_z_warped = geometry.batch_warp_inverse_depth(
                        px, py, invD0, pose10, K)
                    x1_1to0 = geometry.warp_features(x1, u_warped, v_warped)
                    feat_0 = display.visualize_feature_channels(x0, order='CHW')
                    feat_1 = display.visualize_feature_channels(x1, order='CHW')
                    feat_1_0 = display.visualize_feature_channels(x1_1to0, order='CHW')
                    feat_res = display.visualize_feature_channels(weighted_res, order='CHW')

                    feat_residual = display.create_mosaic([feat_0, feat_1, feat_1_0, feat_res],
                                                           cmap=[cv2.COLORMAP_JET, cv2.COLORMAP_JET, cv2.COLORMAP_JET, cv2.COLORMAP_JET],
                                                           order='CHW')
                    cv2.namedWindow("feature-metric residuals", cv2.WINDOW_NORMAL)
                    cv2.imshow("feature-metric residuals", feat_residual)
                    cv2.waitKey(10)
        # print("--->")
        if self.combine_icp:
            weights = w_icp
        else:
            weights = torch.ones(weighted_res.shape).type_as(weighted_res)
        if self.uncer_prop:
            # compute sigma_ksi: 6X6
            inv_sigma_ksi = JtWJ
            #Hessian = lev_mar_H(JtWJ)
            #sigma_ksi = invH(Hessian)
            return pose10, weights, inv_sigma_ksi
        else:
            return pose10, weights

    def forward_residuals(self, pose10, x0, x1, invD0, invD1, K, sigma0, sigma1, wPrior=None,
                          depth0=None, depth1=None, vis_res=True, obj_mask0=None, obj_mask1=None):
        """
        compute the residual/loss in one forward pass, without need of computing jacoabian
        :param pose10, the initial pose
            (extrinsic of the target frame w.r.t. the referenc frame)
        :param x0, the template features
        :param x1, the image features
        :param invD0, the template inverse depth
        :param invD1, the image inverse depth
        :param K, the intrinsic parameters, [fx, fy, cx, cy]
        :param wPrior (optional), provide an initial weight as input to the convolutional scaler
        :return residuals under current init pose
        """
        assert sigma0 is not None and sigma1 is not None

        B, C, H, W = x0.shape
        px, py = geometry.generate_xy_grid(B, H, W, K)

        if self.combine_icp:
            assert depth0 is not None and depth1 is not None
            if self.timers: self.timers.tic('compute vertex and normal')
            vertex0 = geometry.compute_vertex(depth0, px, py)
            vertex1 = geometry.compute_vertex(depth1, px, py)
            normal0 = compute_normal(vertex0)
            normal1 = compute_normal(vertex1)
            if self.timers: self.timers.toc('compute vertex and normal')

        w_icp = None
        if self.timers: self.timers.tic('compute warping residuals')
        weighted_res, res, \
        sigma, occ = compute_inverse_residuals(pose10, invD0, invD1,
                                               x0, x1, sigma0, sigma1, px, py, K,
                                               obj_mask0=obj_mask0, obj_mask1=obj_mask1,
                                               remove_truncate_uncertainty=self.remove_tru_sigma)  # [B,C,H,W]
        if self.timers: self.timers.toc('compute warping residuals')

        # compute batch-wise average residual
        # print("avg feat res:", compute_avg_res(weighted_res, occ))
        # print("avg feat loss:", compute_avg_loss([weighted_res, ], occ))

        if self.combine_icp:
            if self.timers: self.timers.tic('compute ICP residuals and jacobians')
            icp_residuals, icp_J, icp_occ = self.compute_ICP_residuals_jacobian(vertex0, vertex1, normal0, normal1,
                                                                                pose10, K)
            if self.timers: self.timers.toc('compute ICP residuals and jacobians')

            # use the scale computed at the first iteration
            # @TODO: test if we should also scale feature residuals
            if self.timers: self.timers.tic('compute scale function')
            w_icp = self.scale_func(icp_residuals, weighted_res, wPrior)  # [B,1,H,W]
            icp_residuals = w_icp * icp_residuals
            if self.timers: self.timers.toc('compute scale function')
            # print("avg icp res:", compute_avg_res(icp_residuals, icp_occ))
            # print("avg icp loss:", compute_avg_loss([icp_residuals, ], icp_occ))
            combined_occ = occ | icp_occ
            # print("combined loss sum:", compute_avg_loss([weighted_res, icp_residuals], combined_occ))
            loss = compute_avg_loss([weighted_res, icp_residuals], combined_occ)
        else:
            loss = compute_avg_loss([weighted_res,], occ)

        return loss

    # def compute_inverse_jacobian(self, pose_10, invD0, invD1, x0, x1, sigma0, sigma1, px, py, K, obj_mask=None):
    #
    #     grad_f0, grad_sigma0, Jx_p, Jy_p = self.precompute_jacobian_components(invD0, x0, sigma0, px, py, K)
    #
    #     # pose is updated, and thus the crd needs to be re-computed for evaluating res and sigma
    #     u_warped, v_warped, inv_z_warped = geometry.batch_warp_inverse_depth(
    #         px, py, invD0, pose_10, K)
    #     warped_crd = torch.cat((u_warped, v_warped), dim=1)
    #     occ = geometry.check_occ(inv_z_warped, invD1, warped_crd)
    #
    #     B, C, H, W = x0.shape
    #     if obj_mask is not None:
    #         # determine whether the object is in-view
    #         occ = occ & (obj_mask.view(B, 1, H, W) < 1)
    #
    #     J_res_x, J_res_y = self.compute_j_e_u_inverse(warped_crd, occ, x0, x1, sigma0, sigma1)
    #     Jx_p, Jy_p = compute_jacobian_warping(invD0, K, px, py)  # [B, HXW, 6], [B, HXW, 6]
    #     J_res_p = compute_jacobian_dIdp(J_res_x, J_res_y, Jx_p, Jy_p)  # [B, HXW, 6]
    #
    #     J_res_rot, J_res_trs = J_res_p.view(B, H, W, 6).split(3, dim=-1)  # [B, H, W, 3]
    #     J_res_trs = J_res_trs.permute(0, 3, 1, 2)
    #     J_res_rot = J_res_rot.permute(0, 3, 1, 2)
    #     return J_res_p, J_res_trs, J_res_rot

    def compute_jtj(self, jac):
        # J in the dimension of (B, C, HW, y)
        B, C, HW, y = jac.shape
        jac_reshape2 = jac.permute(0, 2, 1, 3).contiguous()  # [B, HW, C, 6]
        jac_reshape2 = jac_reshape2.view(-1, C, y)  # [B*HW, C, 6]
        jtj2 = torch.bmm(torch.transpose(jac_reshape2, 1, 2), jac_reshape2)  # [B*HW, 6, 6]
        jtj2 = jtj2.view(B, HW, y, y)
        jtj2 = jtj2.sum(dim=1)
        return jtj2  # [B, 6, 6]

    def compute_jtr(self, jac, res):
        # J in the dimension of (B, C, HW, y)
        # res in the dimension of [B, C, H, W]
        B,C,H,W = res.shape
        res = res.view(B, C, H*W, 1).permute(0,2,1,3).contiguous()  # [B, HW, C, 1]
        res = res.view(-1,C,1)  # [B*HW, C, 1]
        jac_reshape = jac.permute(0, 2, 1, 3).contiguous()  # [B, HW, C, 6]
        jac_reshape = jac_reshape.view(-1, C, 6)  # [B*HW, C, 6]

        jtr = torch.bmm(torch.transpose(jac_reshape, 1, 2), res)  # [B*HW, 6, 1]
        jtr = jtr.view(B, H*W, 6, 1)
        jtr = jtr.sum(dim=1)
        return jtr  # [B, 6, 1]

    def GN_solver(self, JtJ, JtR, pose0, direction='inverse'):

        B = JtJ.shape[0]

        # Add a small diagonal damping. Without it, the training becomes quite unstable
        # Do not see a clear difference by removing the damping in inference though
        Hessian = lev_mar_H(JtJ)

        updated_pose = inverse_update_pose(Hessian, JtR, pose0)

        return updated_pose

    def precompute_jacobian_components(self, invD0, f0, sigma0, px, py, K, grad_interp=False, crd0=None):
        if not grad_interp:
            # inverse: no need for interpolation in gradients: (linearized at origin)
            f0_gradx, f0_grady = feature_gradient(f0)
            sigma0_gradx, sigma0_grady = feature_gradient(sigma0)
        else:
            # gradients of bilinear interpolation
            if crd0 is None:
                _, _, H, W = f0.shape
                crd0 = geometry.gen_coordinate_tensors(W, H).unsqueeze(dim=0).double()
            f0_gradx, f0_grady = geometry.grad_bilinear_interpolation(crd0, f0, replace_nan_as_eps=True)
            sigma0_gradx, sigma0_grady = geometry.grad_bilinear_interpolation(crd0, sigma0, replace_nan_as_eps=True)

        grad_f0 = torch.stack((f0_gradx, f0_grady), dim=2)
        grad_sigma0 = torch.stack((sigma0_gradx, sigma0_grady), dim=2)
        Jx_p, Jy_p = compute_jacobian_warping(invD0, K, px, py)  # [B, HXW, 6], [B, HXW, 6]

        return grad_f0, grad_sigma0, Jx_p, Jy_p

    def compose_inverse_jacobians(self, res, sigma, sigma0, grad_f0, grad_sigma0, Jx_p, Jy_p):
        B, C, H, W = sigma0.shape
        res = res.unsqueeze(dim=2)
        sigma = sigma.unsqueeze(dim=2)
        sigma0 = sigma0.unsqueeze(dim=2)
        J_res_crd = - grad_f0 / sigma - res * (sigma0 * grad_sigma0 / (sigma ** 3))
        J_res_x, J_res_y = J_res_crd.split(1, dim=2)
        J_res_x.squeeze_(dim=2)
        J_res_y.squeeze_(dim=2)


        J_res_p = compute_jacobian_dIdp(J_res_x, J_res_y, Jx_p, Jy_p)  # [B, CXHXW, 6]
        J_res_rot, J_res_trs = J_res_p.view(B, C, H, W, 6).split(3, dim=-1)  # [B, C, H, W, 3]
        J_res_trs = J_res_trs.permute(0, 1, 4, 2, 3)  # [B, C, 3, H, W]
        J_res_rot = J_res_rot.permute(0, 1, 4, 2, 3)  # [B, C, 3, H, W]
        # follow the conversion of inverse the jacobian
        J_res_p = - J_res_p
        # separate channel and batch and pixel number
        J_res_p = J_res_p.view(B, C, -1, 6)  # [B, C, HXW, 6]
        assert check_nan(J_res_p) == 0
        return J_res_p, J_res_trs, J_res_rot

    def compose_j_e_u(self, invD0, f0, f1, sigma0, sigma1, px, py, K, warped_crd, invalid_mask, eps=1e-6, grad_interp=False, crd0=None):

        grad_f0, grad_sigma0, Jx_p, Jy_p = self.precompute_jacobian_components(invD0, f0, sigma0, px, py, K, grad_interp=grad_interp, crd0=crd0)
        _, res, sigma, _, _, invalid_mask = compose_residuals(warped_crd, invalid_mask, f0, f1, sigma0, sigma1, remove_tru_sigma=self.remove_tru_sigma)

        res = res.unsqueeze(dim=2)
        sigma = sigma.unsqueeze(dim=2)
        sigma0 = sigma0.unsqueeze(dim=2)
        J_res_crd = - grad_f0 / sigma - res * (sigma0 * grad_sigma0 / (sigma ** 3))
        J_res_x, J_res_y = J_res_crd.split(1, dim=2)
        J_res_x.squeeze_(dim=2)
        J_res_y.squeeze_(dim=2)

        # handle nan invalidity
        removed_area = torch.ones_like(J_res_x) * eps
        J_res_x = torch.where(invalid_mask, removed_area, J_res_x)
        J_res_y = torch.where(invalid_mask, removed_area, J_res_y)
        if check_nan(J_res_x):
            import cv2
            nan_mask = torch.isnan(J_res_x).to(torch.uint8).squeeze().numpy()
            cv2.imshow("nan_mask", nan_mask)
            cv2.waitKey(0)
            check_nan(J_res_x)
        # assert check_nan(J_res_x) == 0
        # assert check_nan(J_res_y) == 0
        return J_res_x, J_res_y

    def compute_ICP_residuals_jacobian(self, vertex0, vertex1, normal0, normal1, pose10, K,
                                       obj_mask0=None, obj_mask1=None):
        R, t = pose10
        B, C, H, W = vertex0.shape

        rot_vertex0_to1 = torch.bmm(R, vertex0.view(B, 3, H*W))
        vertex0_to1 = rot_vertex0_to1 + t.view(B, 3, 1).expand(B, 3, H*W)
        # normal0_to1 = torch.bmm(R, normal0.view(B, 3, H * W))

        fx, fy, cx, cy = torch.split(K, 1, dim=1)
        x_, y_, s_ = torch.split(vertex0_to1, 1, dim=1)
        u_ = (x_ / s_).view(B, -1) * fx + cx
        v_ = (y_ / s_).view(B, -1) * fy + cy

        inviews = (u_ > 0) & (u_ < W-1) & (v_ > 0) & (v_ < H-1)

        # # interpolation-version
        r_vertex1 = geometry.warp_features(vertex1, u_, v_)
        r_normal1 = geometry.warp_features(normal1, u_, v_)

        diff = vertex0_to1 - r_vertex1.view(B, 3, H * W)
        # normal_diff = (normal0_to1 * r_normal1.view(B, 3, H * W)).sum(dim=1, keepdim=True)

        # occlusion
        occ = ~inviews.view(B,1,H,W) | (diff.view(B,3,H,W).norm(p=2, dim=1, keepdim=True) > 0.1) #| \
        if obj_mask0 is not None:
            bg_mask0 = ~obj_mask0
            occ = occ | (bg_mask0.view(B, 1, H, W))
        if obj_mask1 is not None:
            obj_mask1_r = geometry.warp_features(obj_mask1.float(), u_, v_) > 0
            bg_mask1 = ~obj_mask1_r
            occ = occ | (bg_mask1.view(B, 1, H, W))

        # point-to-plane residuals
        res = (r_normal1.view(B, 3, H*W)) * diff
        res = res.sum(dim=1, keepdim=True).view(B,1,H,W)  # [B,1,H,W]
        # inverse point-to-plane jacobians
        NtC10 = torch.bmm(r_normal1.view(B,3,-1).permute(0,2,1), R)  # [B, H*W, 3]
        J_rot = torch.bmm(NtC10.view(-1,3).unsqueeze(dim=1),  #[B*H*W,1,3]
                           geometry.batch_skew(vertex0.view(B,3,-1).permute(0, 2, 1).contiguous().view(-1, 3))).squeeze()  # [B*H*W, 3]
        J_trs = -NtC10.view(-1,3)  # [B*H*W, 3]

        # compose jacobians
        J_F_p = torch.cat((J_rot, J_trs), dim=-1)  # follow the order of [rot, trs]  [B*H*W, 6]
        J_F_p = J_F_p.view(B, 1, -1, 6)  # [B, 1, HXW, 6]

        # covariance-normalized
        dpt0 = vertex0[:,2:3,:,:]
        sigma_icp = self.compute_icp_sigma(dpt_l=dpt0, normal_r=r_normal1, rot=R)
        res = res / (sigma_icp + 1e-8)
        J_F_p = J_F_p / (sigma_icp.view(B,1,H*W,1) + 1e-8)

        # follow the conversion of inversing the jacobian
        J_F_p = - J_F_p

        res[occ] = 1e-6

        return res, J_F_p, occ

    def compute_icp_sigma(self, dpt_l, normal_r, rot, dataset='TUM'):
        # obtain sigma
        if dataset == 'TUM':
            sigma_disp = 0.4  # 5.5
            sigma_xy = 5.5  # 5.5
            baseline = 1.2  #0.075
            focal = 525.0
        else:
            raise NotImplementedError()

        B, C, H, W = normal_r.shape

        # compute sigma on depth using stereo model
        sigma_depth = torch.empty((B, 3, H, W)).type_as(dpt_l)
        sigma_depth[:, 0:2, :, :] = dpt_l / focal * sigma_xy
        sigma_depth[:, 2:3, :, :] = dpt_l * dpt_l * sigma_disp / (focal * baseline)

        J = torch.bmm(normal_r.view(B,3,H*W).transpose(1,2), rot)
        J = J.transpose(1,2).view(B,3,H,W)
        cov_icp = (J * sigma_depth * sigma_depth * J).sum(dim=1, keepdim=True)

        sigma_icp = torch.sqrt(cov_icp + 1e-8)
        return sigma_icp

class TrustRegionWUncertainty(nn.Module):
    """
    Direct Dense tracking based on trust region and feature-metric uncertainty
    """

    def __init__(self,
                 max_iter=3,
                 mEst_func=None,
                 solver_func=None,
                 timers=None):
        """
        :param max_iter, maximum number of iterations
        :param mEst_func, the M-estimator function / network
        :param solver_func, the trust-region function / network
        :param timers, if yes, counting time for each step
        """
        super(TrustRegionWUncertainty, self).__init__()

        self.max_iterations = max_iter
        self.mEstimator = mEst_func
        self.directSolver = solver_func
        self.timers = timers

    def forward(self, pose, x0, x1, dpt0, dpt1, K, sigma0, sigma1, wPrior=None, vis_res=True):
        """
        :param pose, the initial pose
            (extrinsic of the target frame w.r.t. the referenc frame)
        :param x0, the template features
        :param x1, the image features
        :param invD0, the template inverse depth
        :param invD1, the image inverse depth
        :param K, the intrinsic parameters, [fx, fy, cx, cy]
        :param wPrior (optional), provide an initial weight as input to the convolutional m-estimator
        """
        if sigma0 is None or sigma1 is None:
            assert sigma0 is not None and sigma1 is not None

        B, C, H, W = x0.shape
        
        for idx in range(self.max_iterations):
            # compute residuals
            if self.timers: self.timers.tic('compute warping residuals')
            rot, trs = pose
            normalized_residuals, residuals, uncertainty = self.compute_residuals(trs, rot, dpt0, dpt1, K, x0, x1, sigma0, sigma1)
            if self.timers: self.timers.toc('compute warping residuals')

            if vis_res:
                with torch.no_grad():
                    feat_residual = display.create_mosaic([normalized_residuals, residuals, uncertainty],
                                                          cmap=[cv2.COLORMAP_JET, cv2.COLORMAP_JET, cv2.COLORMAP_JET],
                                                          order='CHW')
                    cv2.namedWindow("feature-metric residuals", cv2.WINDOW_NORMAL)
                    cv2.imshow("feature-metric residuals", feat_residual)
                    cv2.waitKey(10)

            # compute jacobians
            if self.timers: self.timers.tic('compute jacobian')
            J_res_trs, J_res_rot = self.compute_Jacobian(trs, rot, dpt0, dpt1, K, x0, x1, sigma0, sigma1)
            if self.timers: self.timers.tic('compute jacobian')

            # compose jacobian
            J_res_p = torch.cat((J_res_rot, J_res_trs), dim=1)  # follow the order of [rot, trs]
            J_res_p = J_res_p.view(B, 6, -1).permute(0, 2, 1)  # [B, H*W, 6]
            # follow the conversion of inversing the jacobian
            J_res_p = - J_res_p

            if self.timers: self.timers.tic('robust estimator')
            weights = self.mEstimator(normalized_residuals, x0, x1, wPrior)
            wJ = weights.view(B, -1, 1) * J_res_p  # [B, H*W, 6]
            if self.timers: self.timers.toc('robust estimator')

            if self.timers: self.timers.tic('pre-compute JtWJ')
            JtWJ = torch.bmm(torch.transpose(J_res_p, 1, 2), wJ)
            if self.timers: self.timers.toc('pre-compute JtWJ')

            if self.timers: self.timers.tic('solve x=A^{-1}b')
            # did not really use dpt/invD for trust region optimisation
            pose = self.directSolver(JtWJ,
                                     torch.transpose(J_res_p, 1, 2), weights, normalized_residuals,
                                     pose, dpt0, dpt1, x0, x1, K)
            if self.timers: self.timers.toc('solve x=A^{-1}b')
        return pose, weights

    def compute_residuals(self, trs, rot, dpt0, dpt1, K, f0, f1, sigma0, sigma1):
        """
        compute uncertainty-normalized residuals
        :param trs: translation
        :type trs: Union[numpy.ndarray, torch.Tensor] #(B x 3 X 1)
        :param rot: rotation
        :type rot:  Union[numpy.ndarray, torch.Tensor] #(B x 3 X 3)
        :param dpt0: template depth
        :type dpt0: torch.Tensor, size [(B x 1 x H x W)]
        :param dpt1: live depth
        :type dpt1: torch.Tensor, size [(B x 1 x H x W)]
        :param K: intrinsics
        :type K: torch.Tensor, size [(B x 4)]
        :param f0: template feature image
        :type f0: torch.Tensor, size [(B x 1 x H x W)]
        :param f1: live feature image
        :type f1: torch.Tensor, size [(B x 1 x H x W)]
        :param sigma0: template feature uncertainty
        :type sigma0: torch.Tensor, size [(B x 1 x H x W)]
        :param sigma1: live feature uncertainty
        :type sigma1: torch.Tensor, size [(B x 1 x H x W)]
        :return: weighted_res: normlized residuals
        :rtype: weighted_res: torch.Tensor, size [(B x 1 x H x W)]
        :return: res: residuals
        :rtype: res: torch.Tensor, size [(B x 1 x H x W)]
        :return: sigma: combined uncertainty
        :rtype: sigma: torch.Tensor, size [(B x 1 x H x W)]
        """
        crd, dpt_r, depth_valid = geometry.warp_net(dpt0, trs, rot, K, doJac=False, debug=False)
        occ = geometry.check_occ(dpt_r, dpt1, crd, depth_valid=depth_valid)
        weighted_res, res, sigma, _, _, occ = compose_residuals(crd, occ, f0, f1, sigma0, sigma1)
        return weighted_res, res, sigma


    def compute_j_e_u(self, crd, invalid_mask, f0, f1, sigma0, sigma1, eps=1e-6, use_grad_interpolation=False):
        _, res, sigma, _, sigma_r, invalid_mask = compose_residuals(crd, invalid_mask, f0, f1, sigma0, sigma1)

        # # only use it for unit test, not training
        if use_grad_interpolation:
            # gradient of bilinear interpolation
            grad_f1 = geometry.grad_bilinear_interpolation(crd, f1, valid_mask=~invalid_mask, replace_nan_as_eps=True)
            grad_sigma1 = geometry.grad_bilinear_interpolation(crd, sigma1, valid_mask=~invalid_mask, replace_nan_as_eps=True)
            grad_f1_r = torch.stack(grad_f1, dim=2)  # [B, C, 2, H, W]
            grad_sigma1_r = torch.stack(grad_sigma1, dim=2)
        else:
            # bilinear interpolation of gradients: 1. compute gradients 2. do interpolation on that
            u, v = crd.split(1, dim=1)
            f1_gradx, f1_grady = feature_gradient(f1)
            sigma1_gradx, sigma1_grady = feature_gradient(sigma1)
            inp = [f1_gradx, f1_grady, sigma1_gradx, sigma1_grady]
            out = [geometry.warp_features(image, u, v) for image in inp]
            grad_f1_r = torch.stack((out[0], out[1]), dim=2)  # [B, C, 2, H, W]
            grad_sigma1_r = torch.stack((out[2], out[3]), dim=2)

        res = res.unsqueeze(dim=2)
        sigma = sigma.unsqueeze(dim=2)
        sigma_r = sigma_r.unsqueeze(dim=2)
        J_res_crd = grad_f1_r / sigma - res * (sigma_r * grad_sigma1_r / (sigma ** 3))
        J_res_x, J_res_y = J_res_crd.split(1, dim=2)
        J_res_x.squeeze_(dim=2)
        J_res_y.squeeze_(dim=2)

        # handle nan invalidity
        removed_area = torch.ones_like(J_res_x) * eps
        J_res_x = torch.where(invalid_mask, removed_area, J_res_x)
        J_res_y = torch.where(invalid_mask, removed_area, J_res_y)
        assert check_nan(J_res_x) == 0
        assert check_nan(J_res_y) == 0
        return J_res_x, J_res_y

    def compute_Jacobian(self, trs, rot, dpt0, dpt1, K, f0, f1, sigma0, sigma1,
                         use_grad_interpolation=False):
        """ Compute the image Jacobian on the reference frame
        need to find a way to improve the efficiency (like how to pre-compute it)

        :param trs: translation
        :type trs: Union[numpy.ndarray, torch.Tensor] #(B x 3 X 1)
        :param rot: rotation
        :type rot:  Union[numpy.ndarray, torch.Tensor] #(B x 3 X 3)
        :param dpt0: template depth
        :type dpt0: torch.Tensor, size [(B x 1 x H x W)]
        :param dpt1: live depth
        :type dpt1: torch.Tensor, size [(B x 1 x H x W)]
        :param f0: template feature image
        :type f0: torch.Tensor, size [(B x 1 x H x W)]
        :param f1: live feature image
        :type f1: torch.Tensor, size [(B x 1 x H x W)]
        :param sigma0: template feature uncertainty
        :type sigma0: torch.Tensor, size [(B x 1 x H x W)]
        :param sigma1: live feature uncertainty
        :type sigma1: torch.Tensor, size [(B x 1 x H x W)]
        :param use_grad_interpolation: use gradient of interpolation, otherwise use interpolation of gradients
        :type use_grad_interpolation: bool
        :return: J_res_trs: jacobian of residual w.r.t. translation
        :rtype: J_res_trs: torch.Tensor, size [(B x 3 x H x W)]
        :return: J_res_rot: jacobian of residual w.r.t. rotation
        :rtype: J_res_rot: torch.Tensor, size [(B x 3 x H x W)]
        ------------
        """
        crd, dpt_r, depth_valid, _, crd_J_trs, crd_J_rot, \
        _, _, _ = geometry.warp_net(dpt0, trs, rot, K, doJac=True, debug=True)
        occ = geometry.check_occ(dpt_r, dpt1, crd, depth_valid=depth_valid)
        J_res_x, J_res_y = self.compute_j_e_u(crd, occ, f0, f1, sigma0, sigma1,
                                              use_grad_interpolation=use_grad_interpolation)

        # Compose Jacobians
        B, C, H, W = J_res_x.shape
        # to ensure the correct dimension slice in the matmul function later
        J_res_crd = torch.stack((J_res_x, J_res_y), dim=2)  #[B, C, 2, H, W]
        J_res_crd = J_res_crd.contiguous().view(B, -1, H, W)  #[B, 1X2, H, W]

        J_res_trs = geometry.matmul(J_res_crd, crd_J_trs, [C, 3])
        J_res_rot = geometry.matmul(J_res_crd, crd_J_rot, [C, 3])
        # when I need to wrap J_res_trs, J_res_rot together
        # J_res_p = torch.cat((J_res_trs, J_res_rot), dim=1)
        # J_res_p = J_res_p.view(B, 6, -1).permute(0, 2, 1)  # [B, H*W, 6]
        return J_res_trs, J_res_rot


class ImagePyramids(nn.Module):
    """ Construct the pyramids in the image / depth space
    """
    def __init__(self, scales, pool='avg'):
        super(ImagePyramids, self).__init__()
        if pool == 'avg':
            self.multiscales = [nn.AvgPool2d(1<<i, 1<<i) for i in scales]
        elif pool == 'max':
            self.multiscales = [nn.MaxPool2d(1<<i, 1<<i) for i in scales]
        else:
            raise NotImplementedError()

    def forward(self, x):
        if x.dtype == torch.bool:
            x = x.to(torch.float32)
            x_out = [f(x).to(torch.bool) for f in self.multiscales]
        else:
            x_out = [f(x) for f in self.multiscales]
        return x_out

class FeaturePyramid(nn.Module):
    """ 
    The proposed feature-encoder (A).
    It also supports to extract features using one-view only.
    """
    # @todo: try different output feature channel number
    def __init__(self,
                 D,
                 w_uncertainty='None',
                 feature_channel=1,
                 feature_extract='conv',   # 1by1, conv, skip, average, prob_fuse
                 uncertainty_channel=1,
                 ):
        """
        :param D: input channel dimension
        :type D: int
        :param C: output feature channel dimension
        :type C: int
        :param w_uncertainty:
        :type w_uncertainty: str
        """
        super(FeaturePyramid, self).__init__()
        assert uncertainty_channel == feature_channel or uncertainty_channel == 1
        self.w_uncertainty = False if w_uncertainty == 'None' else True
        self.uncertainty_type = w_uncertainty
        self.output_C = feature_channel
        self.uncertainty_C = uncertainty_channel
        self.feature_extract = feature_extract
        self.f_channels = [32, 64, 96, 128]
        self.net0 = nn.Sequential(
            conv(True, D,  16, 3), 
            conv(True, 16, 32, 3, dilation=2),
            conv(True, 32, self.f_channels[0], 3, dilation=2))
        self.net1 = nn.Sequential(
            conv(True, 32, 32, 3),
            conv(True, 32, 64, 3, dilation=2),
            conv(True, 64, self.f_channels[1], 3, dilation=2))
        self.net2 = nn.Sequential(
            conv(True, 64, 64, 3),
            conv(True, 64, 96, 3, dilation=2),
            conv(True, 96, self.f_channels[2], 3, dilation=2))
        self.net3 = nn.Sequential(
            conv(True, 96, 96, 3),
            conv(True, 96, 128, 3, dilation=2),
            conv(True, 128, self.f_channels[3], 3, dilation=2))
        initialize_weights(self.net0)
        initialize_weights(self.net1)
        initialize_weights(self.net2)
        initialize_weights(self.net3)

        # @todo: check if l2 normalization is necessary
        if self.feature_extract != 'average' and self.feature_extract != 'skip':
            if feature_extract == 'conv':
                self.f_conv0 = conv(True, self.f_channels[0], self.output_C, kernel_size=1)
                self.f_conv1 = conv(True, self.f_channels[1], self.output_C, kernel_size=1)
                self.f_conv2 = conv(True, self.f_channels[2], self.output_C, kernel_size=1)
                self.f_conv3 = conv(True, self.f_channels[3], self.output_C, kernel_size=1)
            elif feature_extract == '1by1':
                self.f_conv0 = nn.Conv2d(self.f_channels[0], self.output_C, kernel_size=(1, 1))
                self.f_conv1 = nn.Conv2d(self.f_channels[1], self.output_C, kernel_size=(1, 1))
                self.f_conv2 = nn.Conv2d(self.f_channels[2], self.output_C, kernel_size=(1, 1))
                self.f_conv3 = nn.Conv2d(self.f_channels[3], self.output_C, kernel_size=(1, 1))
            elif self.feature_extract == 'prob_fuse':
                self.output_C = 8 * 2
                # self.f_conv0 = nn.Conv2d(self.f_channels[0], self.output_C, kernel_size=(1, 1))
                # self.f_conv1 = nn.Conv2d(self.f_channels[1], self.output_C, kernel_size=(1, 1))
                # self.f_conv2 = nn.Conv2d(self.f_channels[2], self.output_C, kernel_size=(1, 1))
                # self.f_conv3 = nn.Conv2d(self.f_channels[3], self.output_C, kernel_size=(1, 1))
                self.f_conv0 = conv(True, self.f_channels[0], self.output_C, kernel_size=1)
                self.f_conv1 = conv(True, self.f_channels[1], self.output_C, kernel_size=1)
                self.f_conv2 = conv(True, self.f_channels[2], self.output_C, kernel_size=1)
                self.f_conv3 = conv(True, self.f_channels[3], self.output_C, kernel_size=1)
            else:
                raise NotImplementedError("not supported feature extraction option")
            initialize_weights((self.f_conv0, self.f_conv1, self.f_conv2, self.f_conv3))

        if self.w_uncertainty and self.uncertainty_type != 'identity':
            # @todo: also try deconv and sigmoid, different kernel size
            if self.uncertainty_type == 'feature':
                self.sigma_conv0 = conv(True, self.f_channels[0], self.output_C, 1)
                self.sigma_conv1 = conv(True, self.f_channels[1], self.output_C, 1)
                self.sigma_conv2 = conv(True, self.f_channels[2], self.output_C, 1)
                self.sigma_conv3 = conv(True, self.f_channels[3], self.output_C, 1)
            elif self.uncertainty_type in ('gaussian', 'laplacian', 'old_gaussian', 'old_laplacian', 'sigmoid'):
                self.sigma_conv0 = nn.Sequential(
                    conv(True, self.f_channels[0], 16, kernel_size=1),
                    nn.Conv2d(16, self.uncertainty_C, kernel_size=1),
                )
                self.sigma_conv1 = nn.Sequential(
                    conv(True, self.f_channels[1], 16, kernel_size=1),
                    nn.Conv2d(16, self.uncertainty_C, kernel_size=1),
                )
                self.sigma_conv2 = nn.Sequential(
                    conv(True, self.f_channels[2], 16, kernel_size=1),
                    nn.Conv2d(16, self.uncertainty_C, kernel_size=1),
                )
                self.sigma_conv3 = nn.Sequential(
                    conv(True, self.f_channels[3], 16, kernel_size=1),
                    nn.Conv2d(16, self.uncertainty_C, kernel_size=1),
                )
            initialize_weights((self.sigma_conv0, self.sigma_conv1, self.sigma_conv2, self.sigma_conv3))

        self.downsample = torch.nn.AvgPool2d(kernel_size=2)

    def __Nto1(self, x):
        """ Take the average of multi-dimension feature into one dimensional,
            which boostrap the optimization speed
        """
        C = x.shape[1]
        return x.sum(dim=1, keepdim=True) / C

    def _prob_fuse(self, x, conv):
        # should we do conv once before softmax?
        x_ = conv(x)
        B, C, H, W = x_.shape
        f, p = x_.split(int(C/2), dim=1)
        p = func.sigmoid(p)  # better than softmax
        # p = func.softmax(p, dim=1)
        out = torch.sum(f * p, dim=1, keepdim=True)
        return out

    def forward(self, x):
        x0 = self.net0(x)
        x0s= self.downsample(x0)
        x1 = self.net1(x0s)
        x1s= self.downsample(x1)
        x2 = self.net2(x1s)
        x2s= self.downsample(x2)
        x3 = self.net3(x2s)
        x = [x0, x1, x2, x3]

        if self.feature_extract == 'skip':
            f = [x0, x1, x2, x3]
        elif self.feature_extract == 'average':
            x = (x0, x1, x2, x3)
            f = [self.__Nto1(a) for a in x]
        elif self.feature_extract == 'prob_fuse':
            f0 = self._prob_fuse(x0, self.f_conv0)
            f1 = self._prob_fuse(x1, self.f_conv1)
            f2 = self._prob_fuse(x2, self.f_conv2)
            f3 = self._prob_fuse(x3, self.f_conv3)
            f = [f0, f1, f2, f3]
        elif self.feature_extract in ('conv', '1by1'):
            f0 = self.f_conv0(x0)
            f1 = self.f_conv1(x1)
            f2 = self.f_conv2(x2)
            f3 = self.f_conv3(x3)
            f = [f0, f1, f2, f3]
        else:
            raise NotImplementedError()

        if self.w_uncertainty:
            if self.uncertainty_type == 'feature':
                sigma0 = self.sigma_conv0(x0)
                sigma1 = self.sigma_conv1(x1)
                sigma2 = self.sigma_conv2(x2)
                sigma3 = self.sigma_conv3(x3)
                sigma = [sigma0, sigma1, sigma2, sigma3]
            elif self.uncertainty_type == 'identity':
                # if do not predict uncertainty, use all identity as uncertainty
                # sigma0 = torch.ones(x0.shape[0], self.output_C, x0.shape[2], x0.shape[3]).to(device=f[0].device)
                # sigma1 = torch.ones(x1.shape[0], self.output_C, x1.shape[2], x1.shape[3]).to(device=f[1].device)
                # sigma2 = torch.ones(x2.shape[0], self.output_C, x2.shape[2], x2.shape[3]).to(device=f[2].device)
                # sigma3 = torch.ones(x3.shape[0], self.output_C, x3.shape[2], x3.shape[3]).to(device=f[3].device)
                # sigma = [sigma0, sigma1, sigma2, sigma3]
                sigma = [torch.ones_like(f_i) for f_i in f]
            elif self.uncertainty_type == 'gaussian':
                sigma0 = self.sigma_conv0(x0)
                sigma1 = self.sigma_conv1(x1)
                sigma2 = self.sigma_conv2(x2)
                sigma3 = self.sigma_conv3(x3)
                sigma = [sigma0, sigma1, sigma2, sigma3]
                sigma = [torch.exp(0.5 * torch.clamp(sigma_i, min=-6, max=6)) for sigma_i in sigma]
            elif self.uncertainty_type == 'laplacian':
                sigma0 = self.sigma_conv0(x0)
                sigma1 = self.sigma_conv1(x1)
                sigma2 = self.sigma_conv2(x2)
                sigma3 = self.sigma_conv3(x3)
                sigma = [sigma0, sigma1, sigma2, sigma3]
                sigma = [torch.exp(torch.clamp(sigma_i, min=-3, max=3)) for sigma_i in sigma]
            elif self.uncertainty_type == 'sigmoid':
                sigma0 = self.sigma_conv0(x0)
                sigma1 = self.sigma_conv1(x1)
                sigma2 = self.sigma_conv2(x2)
                sigma3 = self.sigma_conv3(x3)
                sigma = [sigma0, sigma1, sigma2, sigma3]
                sigma = [torch.sigmoid(sigma_i) for sigma_i in sigma]
            elif self.uncertainty_type == 'old_gaussian':
                sigma0 = self.sigma_conv0(x0)
                sigma1 = self.sigma_conv1(x1)
                sigma2 = self.sigma_conv2(x2)
                sigma3 = self.sigma_conv3(x3)
                sigma = [sigma0, sigma1, sigma2, sigma3]
                sigma = [torch.exp(0.5 * torch.clamp(sigma_i, min=1e-3, max=1e3)) for sigma_i in sigma] 
            elif self.uncertainty_type == 'old_laplacian':
                sigma0 = self.sigma_conv0(x0)
                sigma1 = self.sigma_conv1(x1)
                sigma2 = self.sigma_conv2(x2)
                sigma3 = self.sigma_conv3(x3)
                sigma = [sigma0, sigma1, sigma2, sigma3]
                sigma = [torch.exp(torch.clamp(sigma_i, min=1e-3, max=1e3)) for sigma_i in sigma] 
            else:
                raise NotImplementedError()

            # if uncertainty channel is 1: repeat to each channel
            if self.uncertainty_C == 1 and self.uncertainty_type != 'identity' and self.output_C != 1:
                sigma = [sigma_i.repeat((1, self.output_C, 1, 1)) for sigma_i in sigma]
        else:
            sigma = [None for f_i in f]
        return f, sigma, x

class DeepRobustEstimator(nn.Module):
    """ The M-estimator 

    When use estimator_type = 'MultiScale2w', it is the proposed convolutional M-estimator
    """

    def __init__(self, estimator_type):
        super(DeepRobustEstimator, self).__init__()

        if estimator_type == 'MultiScale2w':
            self.D = 4
        elif estimator_type == 'None':
            self.mEst_func = self.__constant_weight
            self.D = -1
        else:
            raise NotImplementedError()

        if self.D > 0:
            self.net = nn.Sequential(
                conv(True, self.D, 16, 3, dilation=1),
                conv(True, 16, 32, 3, dilation=2),
                conv(True, 32, 64, 3, dilation=4),
                conv(True, 64, 1,  3, dilation=1),
                nn.Sigmoid() )
            initialize_weights(self.net)
        else:
            self.net = None

    def forward(self, residual, x0, x1, ws=None):
        """
        :param residual, the residual map
        :param x0, the feature map of the template
        :param x1, the feature map of the image
        :param ws, the initial weighted residual
        """
        if self.D == 1: # use residual only
            context = residual.abs()
            w = self.net(context)
        elif self.D == 4:
            B, C, H, W = residual.shape
            wl = func.interpolate(ws, (H,W), mode='bilinear', align_corners=True)
            context = torch.cat((residual.abs(), x0, x1, wl), dim=1)
            w = self.net(context)
        elif self.D < 0:
            w = self.mEst_func(residual)

        return w

    def __weight_Huber(self, x, alpha = 0.02):
        """ weight function of Huber loss:
        refer to P. 24 w(x) at
        https://members.loria.fr/moberger/Enseignement/Master2/Documents/ZhangIVC-97-01.pdf

        Note this current implementation is not differentiable.
        """
        abs_x = torch.abs(x)
        linear_mask = abs_x > alpha
        w = torch.ones(x.shape).type_as(x)

        if linear_mask.sum().item() > 0: 
            w[linear_mask] = alpha / abs_x[linear_mask]
        return w

    def __constant_weight(self, x):
        """ mimic the standard least-square when weighting function is constant
        """
        return torch.ones(x.shape).type_as(x)


class ScaleNet(nn.Module):
    """ The scale-estimator

    When use estimator_type = 'MultiScale2w', it is the proposed convolutional M-estimator
    """

    def __init__(self, estimator_type, scale=None):
        super(ScaleNet, self).__init__()
        self.estimator_type = estimator_type 
        if estimator_type == 'oneResidual':
            self.D = 1
        elif estimator_type == 'twoResidual':
            self.D = 2
        elif estimator_type == 'MultiScale2w':
            self.D = 3
        elif estimator_type == "expMultiScale":
            self.D = 3
        elif estimator_type == 'None':
            self.mEst_func = self.__constant_weight
            self.D = -1
        else:
            raise NotImplementedError()

        if self.D > 0:
            self.net = nn.Sequential(
                conv(True, self.D, 16, 3, dilation=1),
                conv(True, 16, 32, 3, dilation=2),
                conv(True, 32, 64, 3, dilation=4),
                conv(True, 64, 16,  3, dilation=1),
                nn.Conv2d(16, 1, kernel_size=1, padding=0),
                )
            initialize_weights(self.net)
        else:
            self.net = None
        self.scale = 0.01 if scale is None else scale

    def forward(self, residual, another_residual=None, ws=None):
        """
        :param residual, the residual map
        :param x0, the feature map of the template
        :param x1, the feature map of the image
        :param ws, the initial weighted residual
        """
        if self.D == 1: # use one residual only
            rtr = self.compute_rtr(residual)
            w = self.net(rtr)
        elif self.D == 2: # use two residual together
            rtr1 = self.compute_rtr(residual)
            rtr2 = self.compute_rtr(another_residual)
            rtr = torch.cat((rtr1, rtr2), dim=1)
            w = self.net(rtr)
        elif self.D == 3:
            B, C, H, W = residual.shape
            wl = func.interpolate(ws, (H,W), mode='bilinear', align_corners=True)
            rtr1 = self.compute_rtr(residual)
            rtr2 = self.compute_rtr(another_residual)
            rtr = torch.cat((rtr1, rtr2, wl), dim=1)
            w = self.net(rtr)
            if self.estimator_type == "expMultiScale":
                w = torch.exp(torch.clamp(w, min=-6, max=6))
            else:
                w = func.sigmoid(w)
        elif self.D < 0:
            w = self.mEst_func(residual)
        else:
            raise NotImplementedError()
        w = w * self.scale
        return w

    def compute_rtr(self, res):
        # res in the dimension of [B, C, H, W]
        rtr = res * res
        rtr = rtr.sum(dim=1, keepdim=True)
        return rtr

    def __constant_weight(self, x):
        """ mimic the standard least-square when weighting function is constant
        """
        # the hyperparameter we found to balance the icp and feature to same scale factor
        return torch.ones(x.shape).type_as(x)


class DirectSolverNet(nn.Module):

    # the enum types for direct solver
    SOLVER_NO_DAMPING       = 0
    SOLVER_RESIDUAL_VOLUME  = 1

    def __init__(self, solver_type, samples=10, direction='inverse'):
        super(DirectSolverNet, self).__init__()
        self.direction = direction  # 'inverse' or 'forward'
        if solver_type == 'Direct-Nodamping':
            self.net = None
            self.type = self.SOLVER_NO_DAMPING
        elif solver_type == 'Direct-ResVol':
            # flattened JtJ and JtR (number of samples, currently fixed at 10)
            self.samples = samples
            self.net = deep_damping_regressor(D=6*6+6*samples)
            self.type = self.SOLVER_RESIDUAL_VOLUME
            initialize_weights(self.net)
        else: 
            raise NotImplementedError()

    def forward(self, JtJ, Jt, weights, R, pose0, invD0, invD1, x0, x1, K, obj_mask1=None):
        """
        :param JtJ, the approximated Hessian JtJ, [B, 6, 6]
        :param Jt, the trasposed Jacobian, [B, 6, CXHXW]
        :param weights, the weight matrix, [B, C, H, W]
        :param R, the residual, [B, C, H, W]
        :param pose0, the initial estimated pose
        :param invD0, the template inverse depth map
        :param invD1, the image inverse depth map
        :param x0, the template feature map, [B, C, H, W]
        :param x1, the image feature map, [B, C, H, W]
        :param K, the intrinsic parameters

        -----------
        :return updated pose
        """

        B = JtJ.shape[0]

        wR = (weights * R).view(B, -1, 1)   # [B, CXHXW, 1]
        JtR = torch.bmm(Jt, wR)  # [B, 6, 1]

        if self.type == self.SOLVER_NO_DAMPING:
            # Add a small diagonal damping. Without it, the training becomes quite unstable
            # Do not see a clear difference by removing the damping in inference though
            Hessian = lev_mar_H(JtJ)
        elif self.type == self.SOLVER_RESIDUAL_VOLUME:
            Hessian = self.__regularize_residual_volume(JtJ, Jt, JtR, weights,
                pose0, invD0, invD1, x0, x1, K, sample_range=self.samples, obj_mask1=obj_mask1)
        else:
            raise NotImplementedError()

        if self.direction == 'forward':
            updated_pose = forward_update_pose(Hessian, JtR, pose0)
        elif self.direction == 'inverse':
            updated_pose = inverse_update_pose(Hessian, JtR, pose0)
        else:
            raise NotImplementedError('pose updated should be either forward or inverse')
        return updated_pose

    def __regularize_residual_volume(self, JtJ, Jt, JtR, weights, pose,
        invD0, invD1, x0, x1, K, sample_range, obj_mask1=None):
        """ regularize the approximate with residual volume

        :param JtJ, the approximated Hessian JtJ
        :param Jt, the trasposed Jacobian
        :param JtR, the Right-hand size residual
        :param weights, the weight matrix
        :param pose, the initial estimated pose
        :param invD0, the template inverse depth map
        :param invD1, the image inverse depth map
        :param K, the intrinsic parameters
        :param x0, the template feature map
        :param x1, the image feature map
        :param sample_range, the numerb of samples

        ---------------
        :return the damped Hessian matrix
        """
        # the following current support only single scale
        JtR_volumes = []

        B, C, H, W = x0.shape
        px, py = geometry.generate_xy_grid(B, H, W, K)

        diag_mask = torch.eye(6).view(1,6,6).type_as(JtJ)
        diagJtJ = diag_mask * JtJ
        traceJtJ = torch.sum(diagJtJ, (2,1))
        epsilon = (traceJtJ * 1e-6).view(B,1,1) * diag_mask
        n = sample_range
        lambdas = torch.logspace(-5, 5, n).type_as(JtJ)

        for s in range(n):
            # the epsilon is to prevent the matrix to be too ill-conditioned
            D = lambdas[s] * diagJtJ + epsilon
            Hessian = JtJ + D
            pose_s = inverse_update_pose(Hessian, JtR, pose)

            res_s,_= compute_warped_residual(pose_s, invD0, invD1, x0, x1, px, py, K, obj_mask1=obj_mask1)
            JtR_s = torch.bmm(Jt, (weights * res_s).view(B,-1,1))
            JtR_volumes.append(JtR_s)

        JtR_flat = torch.cat(tuple(JtR_volumes), dim=2).view(B,-1)
        JtJ_flat = JtJ.view(B,-1)
        damp_est = self.net(torch.cat((JtR_flat, JtJ_flat), dim=1))
        R = diag_mask * damp_est.view(B,6,1) + epsilon # also lift-up

        return JtJ + R


class PoseNetFeat(nn.Module):
    def __init__(self):
        super(PoseNetFeat, self).__init__()
        # @todo: compare: 1.global feature only; 2. combined dense features
        self.final_C = 1024
        self.sep_cov = nn.Sequential(
            conv(True, 128, 128, kernel_size=3, stride=2),
            conv(True, 128, 128, kernel_size=3, stride=2),
        )
        self.conv1 = conv1d(True, 128, 256, 1)
        self.conv2 = conv1d(True, 256, 512, 1)
        self.conv3 = conv1d(True, 512, self.final_C, 1)

        initialize_weights((self.sep_cov, self.conv1, self.conv2, self.conv3))

    def forward(self, feat_map):
        B, C, H, W = feat_map.shape
        feat_map = self.sep_cov(feat_map)
        feat_map_1d = feat_map.view(B, C, -1)
        emb1 = self.conv1(feat_map_1d)
        emb2 = self.conv2(emb1)
        final_emb = self.conv3(emb2)
        N = final_emb.shape[-1]
        ap_x = func.avg_pool1d(final_emb, N)
        ap_x = ap_x.view(B, self.final_C, 1).repeat(1, 1, N)
        return emb1, emb2, ap_x  # 256 +  512 + 1024 = 1792


class PoseNet(nn.Module):
    def __init__(self, scale_motion=0.01, multi_hypo='None', res_input=False):
        super(PoseNet, self).__init__()
        # feature net
        # @todo: test if we should use point-cloud instead of inverse-depth

        self.feat_net = PoseNetFeat()
        self.input_C = 3584
        self.scale_motion = scale_motion
        self.net_r = nn.Sequential(
            conv1d(True, self.input_C, 640, 1),
            conv1d(True, 640, 256, 1),
            conv1d(True, 256, 128, 1),
            nn.Conv1d(128, 3, kernel_size=1, padding=0),
        )  # rotation
        self.net_t = nn.Sequential(
            conv1d(True, self.input_C, 640, 1),
            conv1d(True, 640, 256, 1),
            conv1d(True, 256, 128, 1),
            nn.Conv1d(128, 3, kernel_size=1, padding=0),
        )  # translation
        self.net_c = nn.Sequential(
            conv1d(True, self.input_C, 640, 1),
            conv1d(True, 640, 256, 1),
            conv1d(True, 256, 128, 1),
            nn.Conv1d(128, 1, kernel_size=1, padding=0),
        )  # confidence
        initialize_weights((self.net_r, self.net_t, self.net_c))
    
    def forward(self, x0, x1):
        # @todo: test if we should share 2d features or train a new network to extract features
        # x = torch.cat((x0, x1), dim=1)
        x0_emb1, x0_emb2, x0_ap_x = self.feat_net(x0)
        x1_emb1, x1_emb2, x1_ap_x = self.feat_net(x1)
        feat = torch.cat([x0_emb1, x1_emb1, x0_emb2, x1_emb2, x0_ap_x, x1_ap_x], dim=1)

        rot = self.net_r(feat)
        trs = self.net_t(feat)
        conf = self.net_c(feat)

        # weighted average - competition
        conf = func.softmax(conf, dim=-1)
        rot = self.scale_motion * torch.sum(rot * conf, dim=-1)
        trs = self.scale_motion * torch.sum(trs * conf, dim=-1)  # [B,3]
        rot = geometry.batch_euler2mat(rot[:, 0], rot[:, 1], rot[:, 2])  # [B,3,3]
        return rot, trs


class SFMPoseNet(nn.Module):
    def __init__(self, scale_motion=0.01, multi_hypo='None', res_input=False):
        super(SFMPoseNet, self).__init__()
        self.batchnorm = True
        self.res_input = res_input
        if self.res_input:
            conv_planes = [384, 256, 256]
        else:
            conv_planes = [256, 256, 256]
        self.scale_motion = scale_motion
        self.multi_hypo = multi_hypo
        self.hypo_num = 16 if self.multi_hypo != 'None' else 1
        # @todo: we can also try stride 2, as in sfm-learner
        self.net = nn.Sequential(
            conv(self.batchnorm, conv_planes[0], conv_planes[1], kernel_size=3, dilation=2),
            conv(self.batchnorm, conv_planes[1], conv_planes[2], kernel_size=3, dilation=2),
        )
        if self.multi_hypo == 'None':
            self.final_layer = nn.Conv2d(conv_planes[2], 6, kernel_size=1, padding=0)
        elif self.multi_hypo == 'average':
            self.final_layer = nn.Sequential(
                conv1d(True, conv_planes[2], 128, 1),
                nn.Conv1d(128, self.hypo_num * 6, kernel_size=1, padding=0),
            )
        elif self.multi_hypo == 'prob_fuse':
            self.final_layer = nn.Sequential(
                conv1d(True, conv_planes[2], 128, 1),
                nn.Conv1d(128, self.hypo_num * 7, kernel_size=1, padding=0),
            )
        initialize_weights(self.net, self.final_layer)

    def forward(self, x0, x1):
        if self.res_input:
            res = x0 - x1
            input = torch.cat((x0, x1, res), dim=1)
        else:
            input = torch.cat((x0, x1), dim=1)
        B = input.shape[0]
        pose = self.net(input)
        if self.multi_hypo == 'None':
            pose = self.final_layer(pose)
            pose = pose.mean(dim=3).mean(dim=2)
        elif self.multi_hypo == 'average':
            C = pose.shape[1]
            pose = pose.view(B, C, -1)
            pose = self.final_layer(pose)
            pose = pose.mean(dim=-1)
            pose = pose.view(B, 6, -1).mean(dim=-1)
        elif self.multi_hypo == 'prob_fuse':
            C = pose.shape[1]
            pose = pose.view(B, C, -1)
            pose = self.final_layer(pose)
            pose = pose.mean(dim=-1)
            pose = pose.view(B, -1, 7)
            poses, conf = pose.split((6, 1), dim=-1)
            conf = func.softmax(conf, dim=1)
            pose = torch.sum(poses * conf, dim=1)

        pose = self.scale_motion * pose.view(B, 6)
        # @TODO: compare euler - quat
        rot, trs = geometry.pose_vec2mat(pose, rotation_mode='euler')
        return rot, trs


def deep_damping_regressor(D):
    """ Output a damping vector at each dimension
    """
    net = nn.Sequential(
        fcLayer(in_planes=D,   out_planes=128, bias=True),
        fcLayer(in_planes=128, out_planes=256, bias=True),
        fcLayer(in_planes=256, out_planes=6, bias=True)
    ) # the last ReLU makes sure every predicted value is positive
    return net

def feature_gradient(img, normalize_gradient=True):
    """ Calculate the gradient on the feature space using Sobel operator
    :param the input image 
    -----------
    :return the gradient of the image in x, y direction
    """
    B, C, H, W = img.shape
    # to filter the image equally in each channel
    wx = torch.FloatTensor([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]).view(1,1,3,3).type_as(img)
    wy = torch.FloatTensor([[-1,-2,-1],[ 0, 0, 0],[ 1, 2, 1]]).view(1,1,3,3).type_as(img)

    img_reshaped = img.view(-1, 1, H, W)
    img_pad = func.pad(img_reshaped, (1,1,1,1), mode='replicate')
    img_dx = func.conv2d(img_pad, wx, stride=1, padding=0)
    img_dy = func.conv2d(img_pad, wy, stride=1, padding=0)

    if normalize_gradient:
        mag = torch.sqrt((img_dx ** 2) + (img_dy ** 2)+ 1e-8)
        img_dx = img_dx / mag 
        img_dy = img_dy / mag

    return img_dx.view(B,C,H,W), img_dy.view(B,C,H,W)

def compute_jacobian_dIdp(Jf_x, Jf_y, Jx_p, Jy_p):
    """ chained gradient of image w.r.t. the pose
    :param the Jacobian of the feature map in x direction  [B, C, H, W]
    :param the Jacobian of the feature map in y direction  [B, C, H, W]
    :param the Jacobian of the x map to manifold p  [B, HXW, 6]
    :param the Jacobian of the y map to manifold p  [B, HXW, 6]
    ------------
    :return the image jacobian in x, y, direction, Bx2x6 each
    """
    B, C, H, W = Jf_x.shape

    # precompute J_F_p, JtWJ
    Jf_p = Jf_x.view(B,C,-1,1) * Jx_p.view(B,1,-1,6) + \
        Jf_y.view(B,C,-1,1) * Jy_p.view(B,1,-1,6)
    
    return Jf_p.view(B,-1,6)

def compute_jacobian_warping(p_invdepth, K, px, py, pose=None):
    """ Compute the Jacobian matrix of the warped (x,y) w.r.t. the inverse depth
    (linearized at origin)
    :param p_invdepth the input inverse depth
    :param the intrinsic calibration
    :param the pixel x map
    :param the pixel y map
     ------------
    :return the warping jacobian in x, y direction
    """
    B, C, H, W = p_invdepth.size()
    assert(C == 1)

    if pose is not None:
        x_y_invz = torch.cat((px, py, p_invdepth), dim=1)
        R, t = pose
        warped = torch.bmm(R, x_y_invz.view(B, 3, H * W)) + \
                 t.view(B, 3, 1).expand(B, 3, H * W)
        px, py, p_invdepth = warped.split(1, dim=1)

    x = px.view(B, -1, 1)
    y = py.view(B, -1, 1)
    invd = p_invdepth.view(B, -1, 1)

    xy = x * y
    O = torch.zeros((B, H*W, 1)).type_as(p_invdepth)

    # This is cascaded Jacobian functions of the warping function
    # Refer to the supplementary materials for math documentation
    dx_dp = torch.cat((-xy,     1+x**2, -y, invd, O, -invd*x), dim=2)
    dy_dp = torch.cat((-1-y**2, xy,     x, O, invd, -invd*y), dim=2)

    fx, fy, cx, cy = torch.split(K, 1, dim=1)
    return dx_dp*fx.view(B,1,1), dy_dp*fy.view(B,1,1)

def compute_warped_residual(pose10, invD0, invD1, x0, x1, px, py, K,
                            obj_mask0=None, obj_mask1=None):
    """ Compute the residual error of warped target image w.r.t. the reference feature map.
    refer to equation (12) in the paper

    :param the forward warping pose from the reference camera to the target frame.
        Note that warping from the target frame to the reference frame is the inverse of this operation.
    :param the reference inverse depth
    :param the target inverse depth
    :param the reference feature image
    :param the target feature image
    :param the pixel x map
    :param the pixel y map
    :param the intrinsic calibration
    -----------
    :return the residual (of reference image), and occlusion information
    """
    u_warped, v_warped, inv_z_warped = geometry.batch_warp_inverse_depth(
        px, py, invD0, pose10, K)
    x1_1to0 = geometry.warp_features(x1, u_warped, v_warped)
    crd = torch.cat((u_warped, v_warped), dim=1)
    occ = geometry.check_occ(inv_z_warped, invD1, crd, DIC_version=True)

    residuals = x1_1to0 - x0 # equation (12)

    B, C, H, W = x0.shape
    # determine whether the object is in-view
    if obj_mask0 is not None:
        bg_mask0 = ~obj_mask0
        occ = occ | (bg_mask0.view(B, 1, H, W))
    if obj_mask1 is not None:
        # determine whether the object is in-view
        obj_mask1_r = geometry.warp_features(obj_mask1.float(), u_warped, v_warped)>0
        bg_mask1 = ~obj_mask1_r
        occ = occ | (bg_mask1.view(B, 1, H, W))

    residuals[occ.expand(B,C,H,W)] = 1e-3

    return residuals, occ


def compose_residuals(warped_crd, invalid_mask, f0, f1, sigma0, sigma1,
                      eps=1e-6, perturbed_crd0=None, remove_tru_sigma=False):
    u, v = warped_crd.split(1, dim=1)
    [f_r, sigma_r] = [geometry.warp_features(img, u, v) for img in [f1, sigma1]]

    # if crd0 is perturbed, mainly for unit test
    if perturbed_crd0 is not None:
        u0, v0 = perturbed_crd0.split(1, dim=1)
        [f0, sigma0] = [geometry.warp_features(img, u0, v0) for img in [f0, sigma0]]
    res = f_r - f0
    # sigma = torch.sqrt(sigma_r.pow(2) + sigma0.pow(2) + eps) + eps
    sigma = torch.sqrt(sigma_r.pow(2) + sigma0.pow(2))
    weighted_res = res / sigma

    # handle the truncated uncertainty areas
    if remove_tru_sigma:
        sigma_tru = (sigma_r == sigma_r.min()) | (sigma_r == sigma_r.max()) | \
                    (sigma0 == sigma0.min()) | (sigma0 == sigma0.max())
        sigma_tru = sigma_tru[:, 0:1, :, :]  # a dirty way to handle 1-dim uncertainty
        invalid_mask = invalid_mask | sigma_tru

    # handle invalidity
    removed_area = torch.ones_like(f_r[0]) * eps
    weighted_res = torch.where(invalid_mask, removed_area, weighted_res)
    # res = torch.where(invalid_mask, removed_area, res)
    # sigma = torch.where(invalid_mask, removed_area, sigma)

    # assert check_nan(sigma) == 0
    assert check_nan(weighted_res) == 0
    return weighted_res, res, sigma, f_r, sigma_r, invalid_mask


def compute_inverse_residuals(pose_10, invD0, invD1, x0, x1, sigma0, sigma1, px, py, K,
                              obj_mask0=None, obj_mask1=None, remove_truncate_uncertainty=False):
    u_warped, v_warped, inv_z_warped = geometry.batch_warp_inverse_depth(
        px, py, invD0, pose_10, K)
    warped_crd = torch.cat((u_warped, v_warped), dim=1)
    occ = geometry.check_occ(inv_z_warped, invD1, warped_crd)

    B, C, H, W = x0.shape
    if obj_mask0 is not None:
        bg_mask0 = ~obj_mask0
        occ = occ | (bg_mask0.view(B, 1, H, W))
    if obj_mask1 is not None:
        # determine whether the object is in-view
        u, v = warped_crd.split(1, dim=1)
        obj_mask1_r = geometry.warp_features(obj_mask1.float(), u, v)>0
        bg_mask1 = ~obj_mask1_r
        occ = occ | (bg_mask1.view(B, 1, H, W))

    (weighted_res, res, sigma,
     f_r, sigma_r, occ) = compose_residuals(warped_crd, occ, x0,
                                            x1, sigma0, sigma1,
                                            eps=1e-6,
                                            remove_tru_sigma=remove_truncate_uncertainty)
    return weighted_res, res, sigma, occ

def least_square_solve(H, Rhs):
    """
    x =  - H ^{-1} * Rhs
    importantly: use pytorch inverse to have a differential inverse operation
    :param H: Hessian
    :type H: [B, 6, 6]
    :param  Rhs: Right-hand side vector
    :type Rhs: [B, 6, 1]
    :return: solved ksi
    :rtype:  [B, 6, 1]
    """
    inv_H = invH(H)  # [B, 6, 6] square matrix
    xi = torch.bmm(inv_H, Rhs)
    # because the jacobian is also including the minus signal, it should be (J^T * J) J^T * r
    # xi = - xi
    return xi


def inverse_update_pose(H, Rhs, pose):
    """ Ues left-multiplication for the pose update 
    in the inverse compositional form
    refer to equation (10) in the paper 
    here ksi (se3) is formulated in the order of [rot, trs]
    :param the (approximated) Hessian matrix
    :param Right-hand side vector
    :param the initial pose (forward transform inverse of xi)
    ---------
    :return the forward updated pose (inverse of xi)
    """

    xi = least_square_solve(H, Rhs)
    # simplifed inverse compositional for SE3: (delta_ksi)^-1
    d_R = geometry.batch_twist2Mat(-xi[:, :3].view(-1,3))
    d_t = -torch.bmm(d_R, xi[:, 3:])

    R, t = pose
    pose = geometry.batch_Rt_compose(R, t, d_R, d_t) 
    return pose


def forward_update_pose(H, Rhs, pose):
    """ Ues left-multiplication for the pose update
    in the forward compositional form
    ksi_k o (delta_ksi)
    :param the (approximated) Hessian matrix
    :param Right-hand side vector
    :param the initial pose (forward transform inverse of xi)
    ---------
    :return the forward updated pose (inverse of xi)
    """
    xi = least_square_solve(H, Rhs)
    # forward compotional for SE3: delta_ksi
    d_R = geometry.batch_twist2Mat(xi[:, :3].view(-1, 3))
    d_t = xi[:, 3:]

    R, t = pose
    pose = geometry.batch_Rt_compose(R, t, d_R, d_t)
    return pose


def invH(H):
    """ Generate (H+damp)^{-1}, with predicted damping values
    :param approximate Hessian matrix JtWJ
    -----------
    :return the inverse of Hessian
    """
    # GPU is much slower for matrix inverse when the size is small (compare to CPU)
    # works (50x faster) than inversing the dense matrix in GPU
    if H.is_cuda:
        # invH = bpinv((H).cpu()).cuda()
        # invH = torch.inverse(H)
        invH = torch.inverse(H.cpu()).cuda()
    else:
        invH = torch.inverse(H)
    return invH


def lev_mar_H(JtWJ):
    # Add a small diagonal damping. Without it, the training becomes quite unstable
    # Do not see a clear difference by removing the damping in inference though
    B, _, _ = JtWJ.shape
    diag_mask = torch.eye(6).view(1, 6, 6).type_as(JtWJ)
    diagJtJ = diag_mask * JtWJ
    traceJtJ = torch.sum(diagJtJ, (2, 1))
    epsilon = (traceJtJ * 1e-6).view(B, 1, 1) * diag_mask
    Hessian = JtWJ + epsilon
    return Hessian


def check_nan(T):
    return torch.isnan(T).sum().item() + torch.isinf(T).sum().item()


def compute_avg_res(x, invalid_area):
    B, C, H, W = x.shape
    removed_area = torch.zeros_like(x)
    valid_res = torch.where(invalid_area, removed_area, x)
    valid_num = B * 1 * H * W - invalid_area.sum().item()  # since x might be C-dim while pixel is 1-dim
    avg_res = valid_res.norm(2).item()/ valid_num
    return avg_res


def compute_avg_loss(x_list: list, invalid_area):
    # the batch-wise difference with the compute_avg_res function is that no (**0.5) computation involved

    assert isinstance(x_list, list)
    valid_res_list = []
    for x in x_list:
        removed_area = torch.zeros_like(x)
        valid_res = torch.where(invalid_area, removed_area, x)
        valid_res_list.append(valid_res)
    B, C, H, W = invalid_area.shape
    valid_pixel_num = H*W - invalid_area.sum(dim=[2,3]).squeeze()

    loss_t = torch.zeros_like(invalid_area).float()
    for valid_res in valid_res_list:
        loss_t += (valid_res ** 2).sum(dim=1, keepdim=True)
    loss_sum = loss_t.sum(dim=[2,3], keepdim=True).squeeze()
    avg_loss = loss_sum / valid_pixel_num
    # print("avg", loss_sum**0.5/valid_pixel_num)
    return avg_loss


def warp_images(invD0, pose10, img1, invD1, K):
    B, C, H, W = img1.shape
    px, py = geometry.generate_xy_grid(B, H, W, K)
    u_warped, v_warped, inv_z_warped = geometry.batch_warp_inverse_depth(
        px, py, invD0, pose10, K)
    [img1_r, invD1_r] = [geometry.warp_features(img, u_warped, v_warped) for img in [img1, invD1]]
    return img1_r, invD1_r

def compute_normal(vertex_map):
    """ Calculate the normal map from a depth map
    :param the input depth image
    -----------
    :return the normal map
    """
    B, C, H, W = vertex_map.shape
    img_dx, img_dy = feature_gradient(vertex_map, normalize_gradient=False)

    # img_dz = torch.ones_like(img_pad)
    # normal = torch.cat([-img_dx, -img_dy, img_dz])
    normal = torch.cross(img_dx.view(B,3,-1).permute(0,2,1).contiguous().view(-1,3),
                         img_dy.view(B,3,-1).permute(0,2,1).contiguous().view(-1,3))
    normal = normal.view(B,H*W,3).permute(0,2,1)  # [B,3,H*W]

    mag = torch.norm(normal, p=2, dim=1, keepdim=True)
    normal = normal / (mag + 1e-8)

    # filter out invalid pixels
    depth = vertex_map[:,2:3,:,:]
    invalid_mask = (depth == depth.min()) | (depth == depth.max())
    zero_normal = torch.zeros_like(normal)
    normal = torch.where(invalid_mask.view(B,1,-1).expand(B,3,H*W), zero_normal, normal)

    return normal.view(B,C,H,W)