"""
A collection of geometric transformation operations

# SPDX-FileCopyrightText: 2021 Binbin Xu
# SPDX-License-Identifier: BSD-3-Clause

@author: Zhaoyang Lv 
@Date: March, 2019
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch import sin, cos, atan2, acos

_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def meshgrid(H, W, B=None, is_cuda=False):
    """ torch version of numpy meshgrid function

    :input
    :param height
    :param width
    :param batch size
    :param initialize a cuda tensor if true
    -------
    :return 
    :param meshgrid in column
    :param meshgrid in row
    """
    u = torch.arange(0, W)
    v = torch.arange(0, H)

    if is_cuda:
        u, v = u.cuda(), v.cuda()

    u = u.repeat(H, 1).view(1,H,W)
    v = v.repeat(W, 1).t_().view(1,H,W)

    if B is not None:
        u, v = u.repeat(B,1,1,1), v.repeat(B,1,1,1)
    return u, v

def generate_xy_grid(B, H, W, K):
    """ Generate a batch of image grid from image space to world space 
        px = (u - cx) / fx
        py = (y - cy) / fy

        function tested in 'test_geometry.py'

    :input
    :param batch size
    :param height
    :param width
    :param camera intrinsic array [fx,fy,cx,cy] 
    ---------
    :return 
    :param 
    :param 
    """
    fx, fy, cx, cy = K.split(1,dim=1)
    uv_grid = meshgrid(H, W, B)
    u_grid, v_grid = [uv.type_as(cx) for uv in uv_grid]
    px = ((u_grid.view(B,-1) - cx) / fx).view(B,1,H,W)
    py = ((v_grid.view(B,-1) - cy) / fy).view(B,1,H,W)
    return px, py

def batch_inverse_Rt(R, t):
    """ The inverse of the R, t: [R' | -R't] 

        function tested in 'test_geometry.py'

    :input 
    :param rotation Bx3x3
    :param translation Bx3
    ----------
    :return 
    :param rotation inverse Bx3x3
    :param translation inverse Bx3
    """
    R_t = R.transpose(1,2)
    t_inv = -torch.bmm(R_t, t.contiguous().view(-1, 3, 1))

    return R_t, t_inv.view(-1,3)

def batch_Rt_compose(d_R, d_t, R0, t0):
    """ Compose operator of R, t: [d_R*R | d_R*t + d_t] 
        We use left-mulitplication rule here. 

        function tested in 'test_geometry.py'
    
    :input
    :param rotation incremental Bx3x3
    :param translation incremental Bx3
    :param initial rotation Bx3x3
    :param initial translation Bx3
    ----------
    :return 
    :param composed rotation Bx3x3
    :param composed translation Bx3
    """
    R1 = d_R.bmm(R0)
    t1 = d_R.bmm(t0.view(-1,3,1)) + d_t.view(-1,3,1)
    return R1, t1.view(-1,3)

def batch_Rt_between(R0, t0, R1, t1): 
    """ Between operator of R, t, transform of T_0=[R0, t0] to T_1=[R1, t1]
        which is T_1 \compose T^{-1}_0 

        function tested in 'test_geometry.py'
    
    :input 
    :param rotation of source Bx3x3
    :param translation of source Bx3
    :param rotation of target Bx3x3
    :param translation of target Bx3
    ----------
    :return 
    :param incremental rotation Bx3x3
    :param incremnetal translation Bx3
    """
    R0t = R0.transpose(1,2)
    dR = R1.bmm(R0t)
    dt = t1.view(-1,3) - dR.bmm(t0.view(-1,3,1)).view(-1,3)
    return dR, dt

def batch_skew(w):
    """ Generate a batch of skew-symmetric matrices. 

        function tested in 'test_geometry.py'

    :input
    :param skew symmetric matrix entry Bx3
    ---------
    :return 
    :param the skew-symmetric matrix Bx3x3
    """
    B, D = w.size()
    assert(D == 3)
    o = torch.zeros(B).type_as(w)
    w0, w1, w2 = w[:, 0], w[:, 1], w[:, 2]
    return torch.stack((o, -w2, w1, w2, o, -w0, -w1, w0, o), 1).view(B, 3, 3)

def batch_twist2Mat(twist):
    """ The exponential map from so3 to SO3

        Calculate the rotation matrix using Rodrigues' Rotation Formula
        http://electroncastle.com/wp/?p=39 
        or Ethan Eade's lie group note:
        http://ethaneade.com/lie.pdf equation (13)-(15) 

        functioned tested with cv2.Rodrigues implementation in 'test_geometry.py'

    :input
    :param twist/axis angle Bx3 \in \so3 space 
    ----------
    :return 
    :param Rotation matrix Bx3x3 \in \SO3 space
    """
    B = twist.size()[0]
    theta = twist.norm(p=2, dim=1).view(B, 1)
    w_so3 = twist / theta.expand(B, 3)
    W = batch_skew(w_so3)
    return torch.eye(3).repeat(B,1,1).type_as(W) \
        + W*sin(theta.view(B,1,1)) \
        + W.bmm(W)*(1-cos(theta).view(B,1,1))

def batch_mat2angle(R):
    """ Calcuate the axis angles (twist) from a batch of rotation matrices

        Ethan Eade's lie group note:
        http://ethaneade.com/lie.pdf equation (17)

        function tested in 'test_geometry.py'

    :input
    :param Rotation matrix Bx3x3 \in \SO3 space
    --------
    :return 
    :param the axis angle B
    """
    R1 = [torch.trace(R[i]) for i in range(R.size()[0])]
    R_trace = torch.stack(R1)
    # clamp if the angle is too large (break small angle assumption)
    # @todo: not sure whether it is absoluately necessary in training.
    eps = 1e-7
    angle = acos( ((R_trace - 1)/2).clamp(-1+eps,1-eps))
    return angle

def batch_mat2twist(R):
    """ The log map from SO3 to so3

        Calculate the twist vector from Rotation matrix 

        Ethan Eade's lie group note:
        http://ethaneade.com/lie.pdf equation (18)

        function tested in 'test_geometry.py'

        @note: it currently does not consider extreme small values. 
        If you use it as training loss, you may run into problems

    :input
    :param Rotation matrix Bx3x3 \in \SO3 space 
    --------
    :param the twist vector Bx3 \in \so3 space
    """
    B = R.size()[0]
    eps = 1e-8
    R1 = [torch.trace(R[i]) for i in range(R.size()[0])]
    tr = torch.stack(R1)

    r11,r12,r13,r21,r22,r23,r31,r32,r33 = torch.split(R.view(B,-1),1,dim=1)
    res = torch.cat([r32-r23, r13-r31, r21-r12],dim=1)
    cos_theta = (tr - 1) * 0.5

    so3 = []
    for i in range(B):
        cos_theta_i = cos_theta[i]
        res_i = res[i]
        if cos_theta_i.abs().lt(1. - eps):
            theta_i = acos((cos_theta_i))
            magnitude = (0.5 * theta_i / sin(theta_i))
        else:
            magnitude = 0.5
        so3.append(magnitude * res_i)
    so3 = torch.stack(so3)

    return so3


def batch_quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of rx, ry, rz, tx, ty, tz -- [B, 6]
    Returns:
        rotation [B, 3, 3], translation [B, 3, 1]
    """
    trs = vec[:, 3:]  # [B, 3]
    rot_compact = vec[:,3:]
    if rotation_mode == 'euler':
        rot = batch_euler2mat(rot_compact[:, 0], rot_compact[:, 1], rot_compact[:, 2])  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot = batch_quat2mat(rot_compact)  # [B, 3, 3]
    return rot, trs


def batch_warp_inverse_depth(p_x, p_y, p_invD0, pose_10, K):
    """ Compute the warping grid w.r.t. the SE3 transform given the inverse depth

    :input
    :param p_x the x coordinate map
    :param p_y the y coordinate map
    :param p_invD0 the inverse depth in frame 0
    :param pose_10 the 3D transform in SE3
    :param K the intrinsics
    --------
    :return 
    :param projected u coordinate in image space Bx1xHxW
    :param projected v coordinate in image space Bx1xHxW
    :param projected inverse depth Bx1XHxW
    """
    [R, t] = pose_10
    B, _, H, W = p_x.shape

    I = torch.ones((B,1,H,W)).type_as(p_invD0)
    x_y_1 = torch.cat((p_x, p_y, I), dim=1)

    warped = torch.bmm(R, x_y_1.view(B,3,H*W)) + \
             t.view(B,3,1).expand(B,3,H*W) * p_invD0.view(B, 1, H * W).expand(B, 3, H * W)

    x_, y_, s_ = torch.split(warped, 1, dim=1)
    fx, fy, cx, cy = torch.split(K, 1, dim=1)

    u_ = (x_ / s_).view(B,-1) * fx + cx
    v_ = (y_ / s_).view(B,-1) * fy + cy

    inv_z_ = p_invD0 / s_.view(B, 1, H, W)

    return u_.view(B,1,H,W), v_.view(B,1,H,W), inv_z_

def batch_warp_affine(pu, pv, affine):
    # A = affine[:,:,:2]
    # t = affine[:,:, 2]
    B,_,H,W = pu.shape
    ones = torch.ones(pu.shape).type_as(pu)
    uv = torch.cat((pu, pv, ones), dim=1)
    uv = torch.bmm(affine, uv.view(B,3,-1)) #+ t.view(B,2,1)
    return uv[:,0].view(B,1,H,W), uv[:,1].view(B,1,H,W)

def check_occ(inv_z_buffer, inv_z_ref, crd, thres=1e-1, depth_valid=None, DIC_version=False):
    """ z-buffering check of occlusion 
    :param inverse depth of target frame
    :param inverse depth of reference frame
    """
    B, _, H, W = inv_z_buffer.shape
    u, v = crd.split(1, dim=1)
    inv_z_warped = warp_features(inv_z_ref, u, v)

    # this is much better than ((inv_z_buffer - inv_z_warped).abs() < thres), espcially for wide baseline
    inlier = (inv_z_buffer > inv_z_warped - thres)
    inviews = inlier & (u > 0) & (u < W) & \
              (v > 0) & (v < H)
    if depth_valid is not None:
        inviews = inviews & depth_valid

    return inviews.logical_not()


def warp_features(F, u, v):
    """
    Warp the feature map (F) w.r.t. the grid (u, v)
    """
    B, C, H, W = F.shape

    u_norm = u / ((W-1)/2) - 1
    v_norm = v / ((H-1)/2) - 1
    uv_grid = torch.cat((u_norm.view(B,H,W,1), v_norm.view(B,H,W,1)), dim=3)
    F_warped = nn.functional.grid_sample(F, uv_grid,
                                         align_corners=True,
                                         mode='bilinear', padding_mode='border')
    return F_warped


def render_features(F, crd, valid_mask=None, use_pytorch_func=True):
    """
    Warp the feature map (F) w.r.t. the grid (u, v)
    """
    return bilinear_interpolation_pytorch(crd, F, valid_mask=valid_mask,
                                          torch_sample=use_pytorch_func)


def batch_transform_xyz(xyz_tensor, R, t, get_Jacobian=True):
    '''
    transform the point cloud w.r.t. the transformation matrix
    :param xyz_tensor: B * 3 * H * W
    :param R: rotation matrix B * 3 * 3
    :param t: translation vector B * 3
    '''
    B, C, H, W = xyz_tensor.size()
    t_tensor = t.contiguous().view(B,3,1).repeat(1,1,H*W)
    p_tensor = xyz_tensor.contiguous().view(B, C, H*W)
    # the transformation process is simply:
    # p' = t + R*p
    xyz_t_tensor = torch.baddbmm(t_tensor, R, p_tensor)

    if get_Jacobian:
        # return both the transformed tensor and its Jacobian matrix
        # J_r = R.bmm(batch_skew_symmetric_matrix(-1*p_tensor.permute(0,2,1)))
        rotated_tensor = torch.bmm(R, p_tensor).permute(0,2,1).contiguous().view(-1,3)
        J_r = batch_skew(rotated_tensor)  # [B*H*W, 3, 3]
        J_t = -1 * torch.eye(3).view(1,3,3).expand(B*H*W,3,3).to(device=J_r.device)
        J = torch.cat((J_r, J_t), dim=-1)  # [B*H*W, 3, 6]
        return xyz_t_tensor.view(B, C, H, W), J
    else:
        return xyz_t_tensor.view(B, C, H, W)

def flow_from_rigid_transform(depth, extrinsic, intrinsic):
    """
    Get the optical flow induced by rigid transform [R,t] and depth
    """
    [R, t] = extrinsic
    [fx, fy, cx, cy] = intrinsic

def batch_project(xyz_tensor, K):
    """ Project a point cloud into pixels (u,v) given intrinsic K
    [u';v';w] = [K][x;y;z]
    u = u' / w; v = v' / w

    :param the xyz points 
    :param calibration is a torch array composed of [fx, fy, cx, cy]
    -------
    :return u, v grid tensor in image coordinate
    (tested through inverse project)
    """
    B, _, H, W = xyz_tensor.size()
    batch_K = K.expand(H, W, B, 4).permute(2,3,0,1)

    x, y, z = torch.split(xyz_tensor, 1, dim=1)
    fx, fy, cx, cy = torch.split(batch_K, 1, dim=1)

    u = fx*x / z + cx
    v = fy*y / z + cy
    return torch.cat((u,v), dim=1)

def batch_inverse_project(depth, K):
    """ Inverse project pixels (u,v) to a point cloud given intrinsic 
    :param depth dim B*H*W
    :param calibration is torch array composed of [fx, fy, cx, cy]
    :param color (optional) dim B*3*H*W
    -------
    :return xyz tensor (batch of point cloud)
    (tested through projection)
    """
    if depth.dim() == 3:
        B, H, W = depth.size()
    else: 
        B, _, H, W = depth.size()

    x, y = generate_xy_grid(B,H,W,K)
    z = depth.view(B,1,H,W)
    return torch.cat((x*z, y*z, z), dim=1)

def batch_euler2mat(ai, aj, ak, axes='sxyz'):
    """ A torch implementation euler2mat from transform3d:
    https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/euler.py
    :param ai : First rotation angle (according to `axes`).
    :param aj : Second rotation angle (according to `axes`).
    :param ak : Third rotation angle (according to `axes`).
    :param axes : Axis specification; one of 24 axis sequences as string or encoded tuple - e.g. ``sxyz`` (the default).
    -------
    :return rotation matrix, array-like shape (B, 3, 3)

    Tested w.r.t. transforms3d.euler module
    """
    B = ai.size()[0]

    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]
    order = [i, j, k]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = sin(ai), sin(aj), sin(ak)
    ci, cj, ck = cos(ai), cos(aj), cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    # M = torch.zeros(B, 3, 3).cuda()
    if repetition:
        c_i = [cj, sj*si, sj*ci]
        c_j = [sj*sk, -cj*ss+cc, -cj*cs-sc]
        c_k = [-sj*ck, cj*sc+cs, cj*cc-ss]
    else:
        c_i = [cj*ck, sj*sc-cs, sj*cc+ss]
        c_j = [cj*sk, sj*ss+cc, sj*cs-sc]
        c_k = [-sj, cj*si, cj*ci]

    def permute(X): # sort X w.r.t. the axis indices
        return [ x for (y, x) in sorted(zip(order, X)) ]

    c_i = permute(c_i)
    c_j = permute(c_j)
    c_k = permute(c_k)

    r =[torch.stack(c_i, 1),
        torch.stack(c_j, 1),
        torch.stack(c_k, 1)]
    r = permute(r)

    return torch.stack(r, 1)

def batch_mat2euler(M, axes='sxyz'): 
    """ A torch implementation euler2mat from transform3d:
    https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/euler.py
    :param array-like shape (3, 3) or (4, 4). Rotation matrix or affine.
    :param  Axis specification; one of 24 axis sequences as string or encoded tuple - e.g. ``sxyz`` (the default).
    --------
    :returns 
    :param ai : First rotation angle (according to `axes`).
    :param aj : Second rotation angle (according to `axes`).
    :param ak : Third rotation angle (according to `axes`).
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if repetition:
        sy = torch.sqrt(M[:, i, j]**2 + M[:, i, k]**2)
        # A lazy way to cope with batch data. Can be more efficient
        mask = ~(sy > 1e-8) 
        ax = atan2( M[:, i, j],  M[:, i, k])
        ay = atan2( sy,          M[:, i, i])
        az = atan2( M[:, j, i], -M[:, k, i])
        if mask.sum() > 0:
            ax[mask] = atan2(-M[:, j, k][mask], M[:, j, j][mask])
            ay[mask] = atan2( sy[mask],         M[:, i, i][mask])
            az[mask] = 0.0
    else:
        cy = torch.sqrt(M[:, i, i]**2 + M[:, j, i]**2)
        mask = ~(cy > 1e-8)
        ax = atan2( M[:, k, j],  M[:, k, k])
        ay = atan2(-M[:, k, i],  cy)
        az = atan2( M[:, j, i],  M[:, i, i])
        if mask.sum() > 0:
            ax[mask] = atan2(-M[:, j, k][mask],  M[:, j, j][mask])
            ay[mask] = atan2(-M[:, k, i][mask],  cy[mask])
            az[mask] = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def to_nan_mask(mask):
    """
    convert bool-indicated mask to non-indicated mask
    :param mask: normal mask, in bool
    :type mask: bool[torch.Tensor]
    :return: mask where invalid pixel (False, 0) indicated as nan, otherwise (True/1) as 0
    :rtype: List[torch.Tensor]
    """
    non_mask = [0.0 * 1.0 / m.float() for m in mask]
    return non_mask[0]


def bilinear_interpolation_pytorch(crd, image, valid_mask=None,
                                   torch_sample=True, debug=False):
    """
    pytorch bilinear interpolation/sampling (dense)
    :param crd: sub-pixel coordinate
    :type crd: torch.Tensor: (B x 2 x H x W)
    :param image: image to be sampled
    :type image: torch.Tensor: (B x N x H x W)
    :param torch_sample: if True, use pytorch grid_sample function to interpolate
    :type torch_sample: bool
    :param debug:
    :type debug: bool
    :return: sampled/interpolated image
    :rtype: Union[None, torch.Tensor]
    """
    crd = crd.to(image.device)  # match cpu/cuda for crd and image
    B, N, H, W = image.shape

    # clip and use mask to indicate which pixel is out of the region
    if valid_mask is None:
        valid_mask = torch.ones_like(crd[:, 0:1, :, :], dtype=torch.bool)
    nan_valid = to_nan_mask((crd[:, 0:1, :, :] > 0) & (crd[:, 1:2, :, :] > 0) &
                            (crd[:, 0:1, :, :] < (W - 1)) & (crd[:, 1:2, :, :] < (H - 1)) &
                            torch.isfinite(crd[:, 0:1, :, :]) &
                            torch.isfinite(crd[:, 1:2, :, :]) &
                            (valid_mask))
    nan_valid = nan_valid.to(image.dtype)

    # use grid_sample
    # scale grid to [-1,1]
    if torch_sample:
        sample_crd = crd.clone()
        sample_crd[:, 0, :, :] = 2.0 * sample_crd[:, 0, :, :] / max(W - 1, 1) - 1.0
        sample_crd[:, 1, :, :] = 2.0 * sample_crd[:, 1, :, :] / max(H - 1, 1) - 1.0
        sample_crd = sample_crd.permute(0, 2, 3, 1)
        interp_I = torch.nn.functional.grid_sample(image, sample_crd,
                                                   align_corners=True,
                                                   mode='bilinear',
                                                   padding_mode='border')

    else:
        crd00 = torch.floor(crd)  # (B x 2 x H x W)
        crd11 = torch.ceil(crd)  # (B x 2 x H x W)
        crd00_u = torch.clamp(crd00[:, 0, :, :], 0, W - 1)
        crd00_v = torch.clamp(crd00[:, 1, :, :], 0, H - 1)
        crd11_u = torch.clamp(crd11[:, 0, :, :], 0, W - 1)
        crd11_v = torch.clamp(crd11[:, 1, :, :], 0, H - 1)

        crd00 = torch.stack((crd00_u,crd00_v), dim=1)
        crd11 = torch.stack((crd11_u, crd11_v), dim=1)
        crd01 = torch.stack((crd00_u, crd11_v), dim=1)  # (B x 2 x H x W)
        crd10 = torch.stack((crd11_u, crd00_v), dim=1)  # (B x 2 x H x W)
        crd_subpix = crd - crd00  # (B x 2 x H x W)

        interp_I_local_list = []
        for b in range(B):
            image_local = image[b, :, :, :]  # (N x H x W)
            crd_subpix_local = crd_subpix[b, :, :, :]  # (2 X H X W)
            crd00_local = crd00[b, :, :, :].long()  # (2 X H X W)
            crd01_local = crd01[b, :, :, :].long()  # (2 X H X W)
            crd10_local = crd10[b, :, :, :].long()  # (2 X H X W)
            crd11_local = crd11[b, :, :, :].long()  # (2 X H X W)

            I00 = image_local[:, crd00_local[1, :, :], crd00_local[0, :, :]]  # (N x H x W)
            I01 = image_local[:, crd01_local[1, :, :], crd01_local[0, :, :]]  # (N x H x W)
            I10 = image_local[:, crd10_local[1, :, :], crd10_local[0, :, :]]  # (N x H x W)
            I11 = image_local[:, crd11_local[1, :, :], crd11_local[0, :, :]]  # (N x H x W)

            I0_local = lerp_tensor(I00, I01, crd_subpix_local[1:2, :, :])  # (N x H x W)
            I1_local = lerp_tensor(I10, I11, crd_subpix_local[1:2, :, :])  # (N x H x W)
            interp_I_local = lerp_tensor(I0_local, I1_local, crd_subpix_local[0:1, :, :])  # (N x H x W)
            interp_I_local_list.append(interp_I_local)

        interp_I = torch.stack(interp_I_local_list, dim=0)

    interp_I = nan_valid + interp_I
    invalid_mask = torch.isnan(nan_valid).repeat(B, 1, 1, 1)
    return interp_I, invalid_mask


def grad_bilinear_interpolation(crd, image, valid_mask=None, replace_nan_as_eps=False,
                                debug=False):
    """
    pytorch bilinear interpolation/sampling (dense)
    :param crd: sub-pixel coordinate
    :type crd: torch.Tensor: (B x 2 x H x W)
    :param image: image to be sampled
    :type image: torch.Tensor: (B x N x H x W)
    :param torch_sample: if True, use pytorch grid_sample function to interpolate
    :type torch_sample: bool
    :param debug:
    :type debug: bool
    :return: sampled/interpolated image
    :rtype: Union[None, torch.Tensor]
    """
    crd = crd.to(image.device)  # match cpu/cuda for crd and image
    B, N, H, W = image.shape

    # clip and use mask to indicate which pixel is out of the region
    if valid_mask is None:
        valid_mask = torch.ones_like(crd[:, 0:1, :, :]).to(torch.bool)
    nan_valid = to_nan_mask((crd[:, 0:1, :, :] > 0) & (crd[:, 1:2, :, :] > 0) &
                            (crd[:, 0:1, :, :] < (W - 1)) & (crd[:, 1:2, :, :] < (H - 1)) &
                            torch.isfinite(crd[:, 0:1, :, :]) &
                            torch.isfinite(crd[:, 1:2, :, :]) &
                            (valid_mask))
    nan_valid = nan_valid.to(image.dtype)

    crd00 = torch.floor(crd)  # (B x 2 x H x W)
    crd11 = torch.floor(crd) + 1.0 # torch.ceil(crd) is wrong for int crd  # (B x 2 x H x W)
    crd00_u = torch.clamp(crd00[:, 0, :, :], 0, W - 1)
    crd00_v = torch.clamp(crd00[:, 1, :, :], 0, H - 1)
    crd11_u = torch.clamp(crd11[:, 0, :, :], 0, W - 1)
    crd11_v = torch.clamp(crd11[:, 1, :, :], 0, H - 1)

    crd00 = torch.stack((crd00_u, crd00_v), dim=1)
    crd11 = torch.stack((crd11_u, crd11_v), dim=1)
    crd01 = torch.stack((crd00_u, crd11_v), dim=1)  # (B x 2 x H x W)
    crd10 = torch.stack((crd11_u, crd00_v), dim=1)  # (B x 2 x H x W)
    crd_subpix = crd - crd00  # (B x 2 x H x W)

    inp_I_gx_local_list = []
    inp_I_gy_local_list = []
    for b in range(B):
        image_local = image[b, :, :, :]  # (N x H x W)
        crd_subpix_local = crd_subpix[b, :, :, :]  # (2 X H X W)
        crd00_local = crd00[b, :, :, :].long()  # (2 X H X W)
        crd01_local = crd01[b, :, :, :].long()  # (2 X H X W)
        crd10_local = crd10[b, :, :, :].long()  # (2 X H X W)
        crd11_local = crd11[b, :, :, :].long()  # (2 X H X W)

        I00 = image_local[:, crd00_local[1, :, :], crd00_local[0, :, :]]  # (N x H x W)
        I01 = image_local[:, crd01_local[1, :, :], crd01_local[0, :, :]]  # (N x H x W)
        I10 = image_local[:, crd10_local[1, :, :], crd10_local[0, :, :]]  # (N x H x W)
        I11 = image_local[:, crd11_local[1, :, :], crd11_local[0, :, :]]  # (N x H x W)

        Ix0_local = lerp_tensor(I00, I01, crd_subpix_local[1:2, :, :])  # (N x H x W)
        Ix1_local = lerp_tensor(I10, I11, crd_subpix_local[1:2, :, :])  # (N x H x W)
        inp_I_gx = -Ix0_local + Ix1_local
        inp_I_gx_local_list.append(inp_I_gx)

        Iy0_local = lerp_tensor(I00, I10, crd_subpix_local[0:1, :, :])  # (N x H x W)
        Iy1_local = lerp_tensor(I01, I11, crd_subpix_local[0:1, :, :])  # (N x H x W)
        inp_I_gy = -Iy0_local + Iy1_local
        inp_I_gy_local_list.append(inp_I_gy)

    interp_I_gx = torch.stack(inp_I_gx_local_list, dim=0)
    interp_I_gy = torch.stack(inp_I_gy_local_list, dim=0)

    interp_I_gx = nan_valid + interp_I_gx
    interp_I_gy = nan_valid + interp_I_gy

    if replace_nan_as_eps:
        invalid_mask = torch.isnan(nan_valid).repeat(B, 1, 1, 1)
        zeros = torch.ones_like(interp_I_gx) * 1e-6
        interp_I_gx = torch.where(invalid_mask, zeros, interp_I_gx)
        interp_I_gy = torch.where(invalid_mask, zeros, interp_I_gy)
    return interp_I_gx, interp_I_gy



def matmul(A, B, size):  # (B x LM x H x W), (B x MN x H x W), (L, N)
    """
    matrix multilication for stretched vectors (pytorch is row major)
    :param A:
    :type A: torch.Tensor  (B x LM x H x W)
    :param B:
    :type B: torch.Tensor   (B x MN x H x W)
    :param size:
    :type size: List[int]  (L, N)
    :return:
    :rtype: torch.Tensor  #  (B x LN x H x W)
    """
    L, N = size
    M = A.shape[1] // L
    assert (M == B.shape[1] // N)
    mult_list = []
    for i in range(L):
        for j in range(N):
            entry = torch.sum(A[:, (i * M):((i + 1) * M), :, :] *
                              B[:, j::N, :, :], dim=1)  # (B x H x W)
            mult_list.append(entry)  # LN x (B x H x W)
    return torch.stack(mult_list, dim=1)  # (B x LN x H x W)


def lerp_tensor(x, y, a):
    # same as torch.lerp (at least io), but enable weight and its derivative
    return x + a * (y - x)


def strip_scalar_tensors(tensor_list: list):
    return [x.item() if torch.is_tensor(x) else x for x in tensor_list]


def gen_coordinate_tensors(width, height, debug=False):
    """
    generate grid-type UV coordinate
    :param width:
    :type width: int
    :param height:
    :type height: int
    :param debug:
    :type debug: bool
    :return:
    :rtype: torch.Tensor
    """
    u = torch.tensor(np.kron(np.arange(width, dtype=np.float32),
                             np.ones([height, 1], dtype=np.float32)))
    v = torch.tensor(np.kron(np.arange(height, dtype=np.float32),
                             np.ones([width, 1], dtype=np.float32)).transpose())
    crd = torch.stack([u, v], dim=0)
    return crd


def back_project(depth0, intrinsics, doJac=False, debug=False):
    """[back-projection depth points into camera coordinate]

    Return:
    points in the world coordinate
    corresponding Jacobian
    :param depth0: [depth image]
    :type depth0: torch.Tensor, size [(B x 1 x H x W)]
    :param intrinsics: [intrinsic matrix]
    :type intrinsics: Dict[str, float]
    :param doJac:
    :type doJac: bool Defaults to False. [if doing Jacobian]
    :param debug:
    :type debug: bool Defaults to False. [if debug]
    :return:
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """

    fx, fy, ux, uy = read_intrinsics(intrinsics)

    # back-projection
    B, _, image_height, image_width = depth0.shape
    crd = gen_coordinate_tensors(image_width, image_height).to(dtype=depth0.dtype, device=depth0.device)  # (2, H, H)
    u = crd[0].repeat(B, 1, 1, 1)
    v = crd[1].repeat(B, 1, 1, 1)
    x = ((u.view(B,-1) - ux) / fx).view(B,1,image_height,image_width)
    y = ((v.view(B,-1) - uy) / fy).view(B,1,image_height,image_width)
    z = torch.ones_like(x)  # (B X 1 X H x W)
    pnt_n = torch.cat([x, y, z], dim=1).to(depth0.device)  # (B x 3 x H x W)

    pnt = pnt_n * depth0  # (B x 3 x H x W)
    pnt_J_dpt = pnt_n  # (B x 3 x H x W)

    if not doJac:
        # (B x 3 x H x W)
        return pnt
    else:
        # (B x 3 x H x W), (1 x 3 x H x W)
        return pnt, pnt_J_dpt


def project(pnt, intrinsics, dpt_thr=1e-6, doJac=False, debug=False):
    """
    project 3d points from world to image plan
    :param pnt: world points
    :type pnt: torch.Tensor of size # (B x 3 x H x W)
    :param intrinsics: [intrinsic matrix]
    :type intrinsics: Dict[str, float]
    :param dpt_thr: clip depth to lowest range, to avoid dividing zero
    :type dpt_thr: Union[bool, float]
    :param doJac:
    :type doJac: bool
    :param debug:
    :type debug: bool
    :return: crd: projected coordinate
             dpt: rendered depth, without being clipped
             depth_valid: mask of valid depth, larger than depth_threshold
             crd_J_pnt: Jacobian of projected points wrt input points
             [fx/Z , 0, -fx * X/(Z * Z)
             0, fy/Z, -fy * Y/(Z * Z)]
             dpt_J_pnt: Jacobian of rendered depth wrt wrt world points: [0, 0, 1]
    :rtype:  torch.Tensor (B x 3 x H x W),  (B x 9 x 1 x 1), (1 x 9 x 1 x 1), (B x 9 x H x W)
    :rtype: crd: torch.Tensor, float  # (B x 2 x H x W)
            dpt: torch.Tensor, float  # (B x 1 x H x W)
            depth_valid: torch.Tensor # (B x 1 x H x W)
            crd_J_pnt: torch.Tensor # (B x 6 x H x W)
            dpt_J_pnt: torch.Tensor  # (1 x 3 x 1 x 1)
    """
    fx, fy, ux, uy = read_intrinsics(intrinsics)
    B, _, H, W = pnt.shape
    dpt = pnt[:, 2:3, :, :]  # (B x 1 x H x W)

    # # filter too small (zero) depth (B x 1 x H x W)
    dpt_thr_tensor = torch.ones_like(dpt) * dpt_thr
    dpt_clipped = torch.where((dpt >= 0) & (dpt < dpt_thr), dpt_thr_tensor, dpt)  # positive small depth
    dpt_clipped = torch.where((dpt_clipped < 0) & (dpt_clipped > -dpt_thr), -dpt_thr_tensor, dpt_clipped)  # negative

    depth_valid = torch.where(dpt.abs() > dpt_thr, torch.ones_like(dpt), torch.zeros_like(dpt)).to(torch.bool)

    pnt_n = pnt[:, 0:2, :, :] / dpt_clipped  # (B x 2 x H x W)
    u = pnt_n[:, 0, :, :].view(B, -1) * fx + ux
    v = pnt_n[:, 1, :, :].view(B, -1) * fy + uy
    crd = torch.cat([u.view(B, 1, H, W),
                       v.view(B, 1, H, W)], dim=1)  # (B x 2 x H x W)
    if not doJac:
        return crd, dpt, depth_valid
    else:
        fx_z = (fx / dpt_clipped.view(B, -1)).view(B, 1, H, W)  # fx/Z (B x 1 x H x W)
        fy_z = (fy / dpt_clipped.view(B, -1)).view(B, 1, H, W)  # fy/Z (B x 1 x H x W)
        mxfx_z2 = -fx_z / dpt_clipped * pnt[:, 0:1, :, :]  # -fx * X/(Z * Z) (B x 1 x H x W)
        myfy_z2 = -fy_z / dpt_clipped * pnt[:, 1:2, :, :]  # -fy * Y/(Z * Z) (B x 1 x H x W)
        zero = -torch.zeros_like(dpt_clipped)  # (B x 1 x H x W)
        du = torch.cat([fx_z, zero, mxfx_z2], dim=1)  # (B x 3 x H x W)
        dv = torch.cat([zero, fy_z, myfy_z2], dim=1)  # (B x 3 x H x W)
        crd_J_pnt = torch.cat([du, dv], dim=1)  # (B x 6 x H x W)
        dpt_J_pnt = torch.from_numpy(np.array([[[[0.0]], [[0.0]], [[1.0]]]],
                                              )).to(dpt.device, pnt.dtype)  # (1 x 3 x 1 x 1)

        # (B x 2 x H x W), (B x 1 x H x W), (B x 6 x H x W), (1 x 3 x 1 x 1)
        return crd, dpt, depth_valid, crd_J_pnt, dpt_J_pnt


def read_intrinsics(intrinsics):
    if type(intrinsics) is dict:
        fx, fy, ux, uy = strip_scalar_tensors([intrinsics[k] for k in ['fx', 'fy', 'ux', 'uy']])
    elif torch.is_tensor(intrinsics):
        fx, fy, ux, uy = intrinsics.split(1, dim=1)
    else:
        raise NotImplementedError()

    return fx, fy, ux, uy


def get_dim_size(T):
    if type(T).__module__ == 'numpy':
        return T.ndim
    elif torch.is_tensor(T):
        return T.dim()
    else:
        raise NotImplementedError("not support current vec version")


def skew_matrix(vec, debug=False):
    """
    skew matrix, the calculation
    [0, -a2, a1,
     a2, 0, -a0,
    -a1, a0, 0]
    :param vec: input
    :type vec: numpy.ndarray  # B x 3
    :param debug:
    :type debug: bool
    :return:
    :rtype: numpy.ndarray
    """
    if isinstance(vec, list):
        vec = np.asarray(vec, dtype=np.float32)

    dim_size = get_dim_size(vec)

    if dim_size == 1:  # size of 3
        if type(vec).__module__ == 'numpy':
            mat = np.zeros([3, 3])
            mat[0, 1] = -vec[2]
            mat[0, 2] = vec[1]
            mat[1, 0] = vec[2]
            mat[1, 2] = -vec[0]
            mat[2, 0] = -vec[1]
            mat[2, 1] = vec[0]
        elif torch.is_tensor(vec):
            mat = torch.zeros(3, 3)
            mat[0, 1] = -vec[2]
            mat[0, 2] = vec[1]
            mat[1, 0] = vec[2]
            mat[1, 2] = -vec[0]
            mat[2, 0] = -vec[1]
            mat[2, 1] = vec[0]
        else:
            raise NotImplementedError("not support current vec type")
    elif dim_size == 2:
        if type(vec).__module__ == 'numpy':
            v0 = vec[:, 0]  # B
            v1 = vec[:, 1]  # B
            v2 = vec[:, 2]  # B
            zero = np.zeros_like(v0)  # B

            col0 = np.stack([zero, v2, -v1], -1)  # B x 3
            col1 = np.stack([-v2, zero, v0], -1)  # B x 3
            col2 = np.stack([v1, -v0, zero], -1)  # B x 3

            mat = np.stack([col0, col1, col2], -1)  # B x 3 x 3
        elif torch.is_tensor(vec):
            v0 = vec[:, 0]  # B
            v1 = vec[:, 1]  # B
            v2 = vec[:, 2]  # B
            zero = torch.zeros_like(v0)  # B

            col0 = torch.stack([zero, v2, -v1], -1)  # B x 3
            col1 = torch.stack([-v2, zero, v0], -1)  # B x 3
            col2 = torch.stack([v1, -v0, zero], -1)  # B x 3

            mat = torch.stack([col0, col1, col2], -1)  # B x 3 x 3
        else:
            raise NotImplementedError("not support current vec type")
    else:
        raise NotImplementedError("only support vec dim size of 2 or 3, currently=>", dim_size)

    return mat


def transform_points(pnt0, trs, rot, invert=False, doJac=False, debug=False):
    """
    NOTE: this is implemented in SO3 perturbation
    transformation: qBA * ArAP + BrBA => BrBP
    if invert: qAB^-1 * (ArAP + ArAB^-1) => BrBP
    :param pnt0: 3D points in the camera coordinate ArAP
    :type pnt0: torch.Tensor # (B x 3 x H x W)
    :param trs: translation BrBA or ArAB if invert
    :type trs: Union[numpy.ndarray, torch.Tensor] #(B x 3)
    :param rot: rotation qBA or qAB if invert
    :type rot: Union[numpy.ndarray, torch.Tensor] #(B x 3 x 3)
    :param invert: if the given trs and rot are inverted pose
    :type invert: bool
    :param doJac: do Jacobian
    :type doJac: bool
    :param debug:
    :type debug: bool
    :return: pnt1: transformed points BrBP
             pnt1_J_pnt0: Jacobian of BrBP wrt ArAP: qBA | qAB^-1 if inverted
             pnt1_J_trs: Jacobian of transformed points wrt translation: I | -I
             pnt1_J_rot: Jacobian of transformed points wrt rotation: -BrAP_skewed | q_BA * A_r_BP_skewed?
    :rtype:  torch.Tensor (B x 3 x H x W),  (B x 9 x 1 x 1), (1 x 9 x 1 x 1), (B x 9 x H x W)
    """
    B, _, H, W = pnt0.shape
    ArAP = torch.reshape(pnt0, [B, 3, H * W])  # (B x 3 x HW)
    if trs.shape[-1] != 1:
        trs = trs.unsqueeze(dim=-1)
    if not invert:
        BrBA = trs  # (B x 3)
        qBA = rot  # (B x 3 x 3)
        BrAP = torch.matmul(qBA, ArAP)  # (B x 3 x HW)
        BrBP = BrAP + BrBA  # (B x 3 x HW)
        pnt1 = torch.reshape(BrBP, [B, 3, H, W])  # (B x 3 x H x W)
        if doJac:
            pnt1_J_pnt0 = torch.unsqueeze(torch.unsqueeze(torch.reshape(
                qBA, [B, 9]), -1), -1)  # (B x 9 x 1 x 1)
            pnt1_J_trs = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.reshape(
                torch.eye(3), [9]), -1), -1), 0).to(pnt0.dtype)  # (1 x 9 x 1 x 1)
            BrAP_f = torch.reshape(BrAP.permute(0, 2, 1), [B * H * W, 3])  # BHW x 3 flat
            BrAP_s = skew_matrix(BrAP_f, debug)  # BHW x 3 x 3 skew matrix
            pnt1_J_rot = torch.reshape(-BrAP_s, [B, H, W, 9]).permute(0, 3, 1, 2)  # (B x 9 x H x W)
    else:
        ArAB = trs  # (B x 3)
        qAB = rot  # (B x 3 x 3)
        qBA = qAB.permute(0, 2, 1)  # for rotation, inverse = transpose  #(B x 3 x 3)
        ArBP = ArAP - ArAB  # (B x 3 x HW)
        BrBP = torch.matmul(qBA, ArBP)  # (B x 3 x HW)
        pnt1 = torch.reshape(BrBP, [B, 3, H, W])  # (B x 3 x H x W)
        if doJac:
            pnt1_J_pnt0 = torch.unsqueeze(torch.unsqueeze(torch.reshape(
                qBA, [B, 9]), -1), -1)  # (B x 9 x 1 x 1)
            pnt1_J_trs = -pnt1_J_pnt0  # (B x 9 x 1 x 1)
            BrBP_f = torch.reshape(BrBP.permute(0, 2, 1), [B * H * W, 3])
            BrBP_s = skew_matrix(BrBP_f, debug)  # BHW x 3 x 3
            pnt1_J_qBA = torch.reshape(-BrBP_s, [B, H, W, 9]).permute(0, 3, 1, 2)  # (B x 9 x H x W)
            pnt1_J_rot = matmul(pnt1_J_qBA,
                                torch.reshape(-qBA, [B, 9, 1, 1]), [3, 3])  # (B x 9 x H x W)

        # here ,our custom matmul did exactly a matrix multiplication..
        # the reason we use is the since we have flattened the jacobian along the second dimension, to do a row x colum in common matrix multiplication, the first input variable takes continuous N elements
        # and the second input variable takes every N elements to form the colum.

            layers.layer_logging('transform', [pnt0, trs, rot],
                                 [pnt1, pnt1_J_pnt0, pnt1_J_trs, pnt1_J_rot], printInfo=debug)

    # (B x 3 x H x W),  (B x 9 x 1 x 1), (1 x 9 x 1 x 1), (B x 9 x H x W)
    if doJac:
        pnt1_J_trs = pnt1_J_trs.to(pnt0.device)
        return pnt1, pnt1_J_pnt0, pnt1_J_trs, pnt1_J_rot
    else:
        return pnt1


def TRC(inp, C, B):
    out = torch.cat(torch.reshape(inp.permute(0, 2, 3, 1), (B, -1, C)), 1)
    return out



def warp_net(dpt0, trs_10, rot_10, intrinsics, doJac=True, invert=False, debug=False):
    """

    :param dpt0: [depth image]
    :type dpt0: torch.Tensor, size [(B x 1 x H x W)]
    :param trs_10:
    :type trs_10: Union[numpy.ndarray, torch.Tensor] #(B x 3)
    :param rot_10:
    :type rot_10: Union[numpy.ndarray, torch.Tensor] #(B x 3 X 3)
    :param intrinsics:
    :type intrinsics: Dict[str, float]
    :param doJac:
    :type doJac: bool
    :param invert:
    :type invert: bool
    :param debug:
    :type debug: bool
    :return:
    crd, (B x 2 x H x W)
    dpt1,  (B x 1 x H x W)
    depth_valid,  (B x 6 x H x W )
    crd_J_dpt0, (B x 2 x H x W)
    crd_J_trs, (B x 6 x H x W)
    crd_J_rot, (B x 6 x H x W)
    dpt1_J_dpt0, (B x 1 x H x W)
    dpt1_J_trs, (B x 3 x H x W)
    dpt1_J_rot (B x 3 x H x W)
    """
    B, _, H, W = dpt0.shape
    rot_10 = rot_10.to(dpt0.device)
    trs_10 = trs_10.to(dpt0.device)

    if doJac:
        # (B x 3 x H x W), (1 x 3 x H x W)
        pnt0, pnt0_J_dpt0 = back_project(dpt0, intrinsics, doJac=doJac, debug=debug)

        # (B x 3 x H x W),  (B x 9 x 1 x 1), (1 x 9 x 1 x 1), (B x 9 x H x W)
        pnt1, pnt1_J_pnt0, pnt1_J_trs, pnt1_J_rot = transform_points(pnt0, trs_10, rot_10, doJac=doJac, invert=invert,
                                                                     debug=debug)

        # (B x 2 x H x W), (B x 1 x H x W), (B x 6 x H x W ), (1 x 3 x 1 x 1)
        crd, dpt1, depth_valid, crd_J_pnt1, dpt1_J_pnt1 = project(pnt1, intrinsics, doJac=doJac, debug=debug)

        # (B x 2 x H x W)
        pnt1_J_dpt0 = matmul(pnt1_J_pnt0, pnt0_J_dpt0, [3, 1])
        crd_J_dpt0 = matmul(crd_J_pnt1, pnt1_J_dpt0, [2, 1])

        # (B x 6 x H x W)
        crd_J_trs = matmul(crd_J_pnt1, pnt1_J_trs, [2, 3])

        # (B x 6 x H x W)
        crd_J_rot = matmul(crd_J_pnt1, pnt1_J_rot, [2, 3])

        # (B x 1 x H x W)
        dpt1_J_dpt0 = matmul(dpt1_J_pnt1, pnt1_J_dpt0, [1, 1])

        # (B x 3 x H x W)
        dpt1_J_trs = matmul(dpt1_J_pnt1, pnt1_J_trs, [1, 3])

        # (B x 3 x H x W)
        dpt1_J_rot = matmul(dpt1_J_pnt1, pnt1_J_rot, [1, 3])

        return crd, dpt1, depth_valid, crd_J_dpt0, crd_J_trs, crd_J_rot, dpt1_J_dpt0, dpt1_J_trs, dpt1_J_rot

    else:
        # only warping, no Jacobian calculation
        # (B x 3 x H x W)
        pnt0 = back_project(dpt0, intrinsics, doJac=doJac, debug=debug)

        # (B x 3 x H x W)
        pnt1 = transform_points(pnt0, trs_10, rot_10, doJac=doJac, invert=invert, debug=debug)

        # (B x 2 x H x W), (B x 1 x H x W), (B x 6 x H x W ), (1 x 3 x 1 x 1)
        crd, dpt1, depth_valid = project(pnt1, intrinsics, doJac=doJac, debug=debug)

        return crd, dpt1, depth_valid


def compute_vertex(depth_map, px, py):
    B, _, H, W = px.shape
    I = torch.ones((B,1,H,W)).type_as(depth_map)
    x_y_1 = torch.cat((px, py, I), dim=1)

    vertex = x_y_1 * depth_map

    return vertex


def batch_create_transform(trs, rot, debug=False):  # B x 3   B x 3 x 3
    if type(trs).__module__ == 'numpy':
        top = np.hstack((rot, trs.reshape(3, 1)))
        bot = np.hstack((np.zeros([1, 3]), np.ones([1, 1])))
        out = np.vstack((top, bot))
    else:
        B = trs.shape[0]   # B x 3
        top = torch.cat((rot, torch.unsqueeze(trs, 2)), dim=2)  # B x 3 x 4
        bot = torch.cat((torch.zeros(B, 1, 3), torch.ones(B, 1, 1)), 2).to(top.device)  # B x 1 x 4
        out = torch.cat((top, bot), dim=1)
    return out