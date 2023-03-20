"""
Some training criterions tested in the experiments

# SPDX-FileCopyrightText: 2021 Binbin Xu
# SPDX-License-Identifier: BSD-3-Clause

@author: Zhaoyang Lv
@date: March, 2019
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as func
import models.geometry as geo
from models.algorithms import invH, lev_mar_H


def EPE3D_loss(input_flow, target_flow, invalid=None):
    """
    :param the estimated optical / scene flow
    :param the ground truth / target optical / scene flow
    :param the invalid mask, the mask has value 1 for all areas that are invalid
    """
    epe_map = torch.norm(target_flow-input_flow,p=2,dim=1)
    B = epe_map.shape[0]

    invalid_flow = (target_flow != target_flow) # check Nan same as torch.isnan

    mask = (invalid_flow[:,0,:,:] | invalid_flow[:,1,:,:] | invalid_flow[:,2,:,:])
    if invalid is not None:
        mask = mask | (invalid.view(mask.shape) > 0)

    epes = []
    for idx in range(B):
        epe_sample = epe_map[idx][~mask[idx].data]
        if len(epe_sample) == 0:
            epes.append(torch.zeros(()).type_as(input_flow))
        else:
            epes.append(epe_sample.mean())

    return torch.stack(epes)


def RPE(R, t):
    """ Calcualte the relative pose error 
    (a batch version of the RPE error defined in TUM RGBD SLAM TUM dataset)
    :param relative rotation
    :param relative translation
    """
    angle_error = geo.batch_mat2angle(R)
    trans_error = torch.norm(t, p=2, dim=1) 
    return angle_error, trans_error


def compute_RPE_uncertainty(R_est, t_est, R_gt, t_gt, inv_var):
    loss = 0
    if R_est.dim() > 3: # training time [batch, num_poses, rot_row, rot_col]

        for idx in range(R_est.shape[1]):
            dR = geo.batch_mat2twist(R_gt.detach().contiguous()) - geo.batch_mat2twist(R_est[:,idx])
            dt = t_gt.detach() - t_est[:,idx]
            dKsi = torch.cat([dR, dt], dim=-1).unsqueeze(dim=-1)
            Hessian = lev_mar_H(inv_var[:,idx])
            sigma_ksi = invH(Hessian)
            det_var = torch.det(sigma_ksi)
            # clamp
            det_var = det_var.clamp(min=1e-9)
            weighted_error = torch.bmm(dKsi.transpose(1, 2), torch.bmm(inv_var[:,idx], dKsi)).squeeze()
            regularization = torch.log(1e-6 + det_var)
            loss += (weighted_error + regularization).sum()
    return loss


def compute_RPE_loss(R_est, t_est, R_gt, t_gt):
    """
    :param estimated rotation matrix Bx3x3
    :param estimated translation vector Bx3
    :param ground truth rotation matrix Bx3x3
    :param ground truth translation vector Bx3
    """
    B=R_est.shape[0]
    if R_est.dim() > 3:  # training time [batch, num_poses, rot_row, rot_col]
        angle_error = 0
        trans_error = 0
        for idx in range(R_est.shape[1]):
            dR, dt = geo.batch_Rt_between(R_est[:, idx], t_est[:, idx], R_gt, t_gt)
            angle_error_i, trans_error_i = RPE(dR, dt)
            angle_error += angle_error_i.norm(p=2).sum()
            trans_error += trans_error_i.norm(p=2).sum()
    else:
        dR, dt = geo.batch_Rt_between(R_est, t_est, R_gt, t_gt)
        angle_error, trans_error = RPE(dR, dt)
    return angle_error, trans_error


def compute_RT_EPE_loss(R_est, t_est, R_gt, t_gt, depth0, K, invalid=None):
    """ Compute the epe point error of rotation & translation
    :param estimated rotation matrix Bx3x3
    :param estimated translation vector Bx3
    :param ground truth rotation matrix Bx3x3
    :param ground truth translation vector Bx3
    :param reference depth image,
    :param camera intrinsic
    """

    loss = 0
    if R_est.dim() > 3:  # training time [batch, num_poses, rot_row, rot_col]
        rH, rW = 60, 80  # we train the algorithm using a downsized input, (since the size of the input is not super important at training time)

        B, C, H, W = depth0.shape
        rdepth = func.interpolate(depth0, size=(rH, rW), mode='bilinear')
        rinvalid = func.interpolate(invalid.float(), size=(rH, rW), mode='bilinear')
        rK = K.clone()
        rK[:, 0] *= float(rW) / W
        rK[:, 1] *= float(rH) / H
        rK[:, 2] *= float(rW) / W
        rK[:, 3] *= float(rH) / H
        xyz = geo.batch_inverse_project(rdepth, rK)
        flow_gt = geo.batch_transform_xyz(xyz, R_gt, t_gt, get_Jacobian=False)

        for idx in range(R_est.shape[1]):
            flow_est = geo.batch_transform_xyz(xyz, R_est[:, idx], t_est[:, idx], get_Jacobian=False)
            loss += EPE3D_loss(flow_est, flow_gt.detach(), invalid=rinvalid)  # * (1<<idx) scaling does not help that much
    else:
        xyz = geo.batch_inverse_project(depth0, K)
        flow_gt = geo.batch_transform_xyz(xyz, R_gt, t_gt, get_Jacobian=False)

        flow_est = geo.batch_transform_xyz(xyz, R_est, t_est, get_Jacobian=False)
        loss = EPE3D_loss(flow_est, flow_gt, invalid=invalid)

    return loss


def UEPE3D_loss(input_flow, target_flow, variance, uncer_loss_type, invalid=None):
    """
    :param the estimated optical / scene flow
    :param the ground truth / target optical / scene flow
    :param the invalid mask, the mask has value 1 for all areas that are invalid
    """
    if uncer_loss_type is None:
        epe_map = torch.norm(target_flow-input_flow,p=2,dim=1)
    else:
        assert type(variance) != None
        deltaP = target_flow - input_flow
        o_epe_map = torch.norm(deltaP, p=2, dim=1)
        B,C,H,W = deltaP.shape
        # dimension permutation (B,3,H,W) -> (B*H*w,3,1)
        deltaP = deltaP.view(B,3,-1).permute(0,2,1).contiguous().view(-1,3).unsqueeze(dim=-1)
        dim_ind = True
        if uncer_loss_type == 'gaussian':
            # if assume each dimension independent
            if dim_ind:
                diag_mask = torch.eye(3).view(1, 3, 3).type_as(variance)
                variance = torch.clamp(variance, min=1e-3)
                variance = diag_mask * variance

            # inv_var = invH(variance)
            inv_var = torch.inverse(variance)
            weighted_error = torch.bmm(deltaP.transpose(1, 2), torch.bmm(inv_var, deltaP)).squeeze()
            if dim_ind:
                regularization = variance.diagonal(dim1=1, dim2=2).log().sum(dim=1)
            else:
                det_var = torch.det(variance)
                # make it numerical stable
                det_var = torch.clamp(det_var, min=1e-3)
                regularization = torch.log(det_var)
            epe_map = weighted_error + regularization
        elif uncer_loss_type == 'laplacian':
            raise NotImplementedError("the bessel function needs to be implemented")
            sigma = torch.sqrt(variance + 1e-7)
            weighted_error = torch.abs(deltaP) / (sigma + 1e-7)
            regularization = torch.log(torch.det(sigma))
            epe_map = weighted_error + regularization
        else:
            raise NotImplementedError()
        epe_map = epe_map.view(B,H,W)
    B = epe_map.shape[0]

    invalid_flow = (target_flow != target_flow) # check Nan same as torch.isnan

    mask = (invalid_flow[:,0,:,:] | invalid_flow[:,1,:,:] | invalid_flow[:,2,:,:])
    if invalid is not None:
        mask = mask | (invalid.view(mask.shape) > 0)

    epes = []
    o_epes = []
    for idx in range(B):
        epe_sample = epe_map[idx][~mask[idx].data]
        if len(epe_sample) == 0:
            epes.append(torch.zeros(()).type_as(input_flow))
        else:
            epes.append(epe_sample.mean())

        if uncer_loss_type is not None:
            o_epe_sample = o_epe_map[idx][~mask[idx].data]
            if len(o_epe_sample) == 0:
                o_epes.append(torch.zeros(()).type_as(input_flow))
            else:
                o_epes.append(o_epe_sample.mean())
    if uncer_loss_type is not None:
        return torch.stack(epes), torch.stack(o_epes)
    else:
        return torch.stack(epes)


def compute_RT_EPE_uncertainty_loss(R_est, t_est, R_gt, t_gt, depth0, K, sigma_ksi, uncertainty_type, invalid=None):
    """ Compute the epe point error of rotation & translation
    :param estimated rotation matrix BxNx3x3
    :param estimated translation vector BxNx3
    :param ground truth rotation matrix Bx3x3
    :param ground truth translation vector Bx3
    :param reference depth image, 
    :param camera intrinsic 
    """
    
    loss = 0
    epe = 0
    assert R_est.dim() > 3 # training time [batch, num_poses, rot_row, rot_col]
    rH, rW = 60, 80 # we train the algorithm using a downsized input, (since the size of the input is not super important at training time)

    B,C,H,W = depth0.shape
    rdepth = func.interpolate(depth0, size=(rH, rW), mode='bilinear')
    rinvalid = func.interpolate(invalid.float(), size=(rH,rW), mode='bilinear')
    rK = K.clone()
    rK[:,0] *= float(rW) / W
    rK[:,1] *= float(rH) / H
    rK[:,2] *= float(rW) / W
    rK[:,3] *= float(rH) / H
    xyz = geo.batch_inverse_project(rdepth, rK)
    flow_gt = geo.batch_transform_xyz(xyz, R_gt, t_gt, get_Jacobian=False)

    for idx in range(R_est.shape[1]):
        flow_est, J_flow = geo.batch_transform_xyz(xyz, R_est[:,idx], t_est[:,idx], get_Jacobian=True)
        # uncertainty propagation: J*sigma*J^T
        sigma_ksai_level = sigma_ksi[:,idx:idx+1,:,:].repeat(1,rH*rW,1,1).view(-1,6,6)
        J_sigmaKsi = torch.bmm(J_flow, sigma_ksai_level)
        sigma_deltaP = torch.bmm(J_sigmaKsi, torch.transpose(J_flow, 1, 2))
        loss_l, epe_l = UEPE3D_loss(flow_est, flow_gt.detach(), variance=sigma_deltaP, uncer_loss_type=uncertainty_type, invalid=rinvalid,)
        loss += loss_l
        epe += epe_l

    return loss, epe

