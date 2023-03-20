""" 
Unit test for the jacobians used in the tracker
# SPDX-FileCopyrightText: 2021 Binbin Xu
# SPDX-License-Identifier: BSD-3-Clause
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as func
from scipy.stats import ortho_group
import models.geometry as geometry
from models.submodules import convLayer as conv
from models.submodules import fcLayer, initialize_weights
from models.algorithms import TrustRegionWUncertainty, DeepRobustEstimator, DirectSolverNet, compute_inverse_residuals, compose_residuals, compute_jacobian_warping, compute_normal
from models.algorithms import TrustRegionInverseWUncertainty as U_IC
from models.algorithms import TrustRegionICP as ICP


def test_normal_forward_jacobian():
    import math
    import numpy as np
    from skimage import io
    import cv2
    from PIL import Image

    import math

    from data.TUM_RGBD import tq2mat
    from models.algorithms import feature_gradient
    from models.geometry import grad_bilinear_interpolation

    # from dense_feature.tool import rotation

    """
    ground truth data used for unit test:
    1.351500  0.586800  1.582300  0.836900  0.288700  -0.181000  -0.428300 /media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/rgb/1305031454.027662.png /media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/depth/1305031454.040976.png
    1.343600  0.577000  1.583100  0.830700  0.302300  -0.186300  -0.428800 /media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/rgb/1305031454.059654.png /media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/depth/1305031454.072690.png
    """
    B = 1
    eps = 1e-6
    VIS = False

    # captured data:
    img0_path = '/media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/rgb/1305031454.027662.png'
    dpt0_path = '/media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/depth/1305031454.040976.png'
    img1_path = '/media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/rgb/1305031454.059654.png'
    dpt1_path = '/media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/depth/1305031454.072690.png'
    timestamp0_pose = (1.351500, 0.586800, 1.582300, 0.836900, 0.288700, -0.181000, -0.428300)
    timestamp1_pose = (1.343600, 0.577000, 1.583100, 0.830700, 0.302300, -0.186300, -0.428800)
    DEPTH_SCALE = 1.0 / 5000
    img0 = io.imread(img0_path, as_gray=True)
    img1 = io.imread(img1_path, as_gray=True)
    dpt0 = io.imread(dpt0_path) / 5e3
    dpt1 = io.imread(dpt1_path) / 5e3
    # truncate the depth to accurate range
    invalid_depth_mask = (dpt0 < 0.5) | (dpt0 > 5.0) | (dpt1 < 0.5) | (dpt1 > 5.0)
    cv2.imshow("invalid_depth_mask", invalid_depth_mask * 255.0)
    # cv2.waitKey(0)

    T_W0 = tq2mat(timestamp0_pose)
    T_W1 = tq2mat(timestamp1_pose)
    T_10 = np.dot(np.linalg.inv(T_W1), T_W0)  # .astype(np.float32)
    trs = T_10[0:3, 3]  + 0.1
    rot = T_10[0:3, 0:3]
    # rot = ortho_group.rvs(3)
    # trs = np.random.random((3))
    K = torch.tensor([525.0, 525.0, 319.5, 239.5]).unsqueeze(dim=0).repeat(B, 1).double()
    intrinsics = {'fx': 525.0, 'fy': 525.0, 'ux': 319.5, 'uy': 239.5}
    H, W = img0.shape
    C = 1

    # convert to torch tensor
    img0 = torch.from_numpy(img0).unsqueeze(dim=0).unsqueeze(dim=0)
    img1 = torch.from_numpy(img1).unsqueeze(dim=0).unsqueeze(dim=0)
    dpt0 = torch.from_numpy(dpt0).unsqueeze(dim=0).unsqueeze(dim=0)
    dpt1 = torch.from_numpy(dpt1).unsqueeze(dim=0).unsqueeze(dim=0)
    invD0 = 1.0 / dpt0
    invD1 = 1.0 / dpt1
    trs = torch.from_numpy(trs).unsqueeze(dim=0)
    rot = torch.from_numpy(rot).unsqueeze(dim=0)
    pose = [rot, trs]

    # compute gradients
    img1_gx, img1_gy = feature_gradient(img1)
    # if VIS:
    # grad = img1_gx.squeeze().numpy()
    # cv2.imshow("gradient", grad)
    # cv2.waitKey(0)

    crd, dpt3, depth_valid, crd_J_dpt, crd_J_trs, crd_J_rot, \
    dpt3_J_dpt, dpt3_J_trs, dpt3_J_rot = geometry.warp_net(dpt0, trs, rot, intrinsics, doJac=True, debug=True)

    px, py = geometry.generate_xy_grid(B, H, W, K)
    # u_warped, v_warped, inv_z_warped = geometry.batch_warp_inverse_depth(
    #     px, py, invD0, pose, K)

    img_r, invalid_mask = geometry.render_features(img1, crd, valid_mask=depth_valid)

    # if VIS:
    warped_img = img_r.squeeze().numpy()
    cv2.imshow("rendering", warped_img)
    cv2.waitKey(0)

    res = img_r - img0
    # if VIS:
    warped_residual = res.squeeze().numpy()
    cv2.imshow("rendering", warped_residual)
    cv2.waitKey(0)

    # gradient interpolation
    # inp = [img1, img1_gx, img1_gy]
    # out = [geometry.warp_features(image, crd, valid_mask=depth_valid, use_pytorch_func=False) for image in inp]
    # imgr, imgr_gx, imgr_gy = out

    # instead, use gradient of bilinear interpolation, much better than interpolation on gradients
    imgr_gx, imgr_gy = grad_bilinear_interpolation(crd, img1, valid_mask=depth_valid)

    def debug_jacobian_interpolation(i_x, i_y):
        print(num_imgr_gx[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(imgr_gx[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(abs_gx_crd_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(rel_gx_crd_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(imgr_gy[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(num_imgr_gy[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(abs_gy_crd_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(rel_gy_crd_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(outliers_x[i_x:i_x + 3, i_y:i_y + 3])
        print(outliers_y[i_x:i_x + 3, i_y:i_y + 3])


    # verify J_I_u: numerical verification
    crd_x_p = crd.clone()
    crd_y_p = crd.clone()
    crd_x_p[:, 0, :, :] += eps
    crd_y_p[:, 1, :, :] += eps
    [I_r, I_x, I_y] = [geometry.render_features(img1, pos, valid_mask=depth_valid, use_pytorch_func=True) for
                       pos in [crd, crd_x_p, crd_y_p]]

    num_imgr_gx = (I_x[0] - I_r[0]) / eps
    num_imgr_gy = (I_y[0] - I_r[0]) / eps
    abs_gx_crd_diff = (num_imgr_gx - imgr_gx).abs()
    abs_gy_crd_diff = (num_imgr_gy - imgr_gy).abs()
    rel_gx_crd_diff = abs_gx_crd_diff / num_imgr_gx.abs()
    rel_gy_crd_diff = abs_gy_crd_diff / num_imgr_gy.abs()
    outliers_x = (rel_gx_crd_diff.abs() > 0.1) & (abs_gx_crd_diff.abs() > 0.1)
    outliers_y = (rel_gy_crd_diff.abs() > 0.1) & (abs_gy_crd_diff.abs() > 0.1)
    outliers_x = outliers_x.squeeze().numpy() * 255.0
    outliers_y = outliers_y.squeeze().numpy() * 255.0
    outliers_x[invalid_depth_mask] = 0
    outliers_y[invalid_depth_mask] = 0
    debug_jacobian_interpolation(197, 462)
    debug_jacobian_interpolation(446, 262)
    cv2.imshow("gx_outliers_x", outliers_x)
    cv2.imshow("gy_outliers_y", outliers_y)
    cv2.waitKey(0)

    # compose jacobians
    err_J_crd = torch.stack((imgr_gx, imgr_gy), dim=1)  # [B, 2, C, H, W]
    # to ensure the correct dimension slice in the matmul function later
    err_J_crd = err_J_crd.permute(0, 2, 1, 3, 4)  # [B, C, 2, H, W]
    err_J_crd = err_J_crd.contiguous().view(1, -1, H, W)  # [B, C*2, H, W]

    h_h = round(H / 2)
    h_w = round(W / 2)
    px, py = geometry.generate_xy_grid(B, H, W, K)
    invD0 = torch.clamp(1.0 / dpt0, 0, 10)
    for i in range(3):
        print("\n=>translation perturbation: ", i)
        trs_p = trs.clone()
        trs_p[0, i] += eps
        crd_p, dpt3_p, depth_valid = geometry.warp_net(dpt0, trs_p, rot, intrinsics, doJac=False, debug=True)
        u_perturbed, v_perturbed, inv_z_perturbed = geometry.batch_warp_inverse_depth(
            px, py, invD0, [rot, trs_p], K)
        crd_p = torch.cat((u_perturbed, v_perturbed), dim=1)
        # dpt3_p = 1.0/inv_z_perturbed

        # gradient calculation
        print("=>Jacoabian on Warping:")
        num_J_warping = (crd_p - crd) / eps
        print("Numerical in the middle", num_J_warping[:, :, h_h: h_h + 2, h_w: h_w + 2])
        print("Analytic in the middle", crd_J_trs[0, :, h_h: h_h + 2, h_w: h_w + 2][[i, i + 3]])

        def debug_warping(i_x, i_y):
            print(num_J_warping[:, 0, i_x:i_x + 3, i_y:i_y + 3])
            print(crd_J_trs[:, i, i_x:i_x + 3, i_y:i_y + 3])
            print(abs_warped_crd_diff[:, 0, i_x:i_x + 3, i_y:i_y + 3])
            print(rel_warped_crd_diff[:, 0, i_x:i_x + 3, i_y:i_y + 3])
            print(outliers_x[i_x:i_x + 3, i_y:i_y + 3])
            print(num_J_warping[:, 1, i_x:i_x + 3, i_y:i_y + 3])
            print(crd_J_trs[:, i+3, i_x:i_x + 3, i_y:i_y + 3])
            print(abs_warped_crd_diff[:, 1, i_x:i_x + 3, i_y:i_y + 3])
            print(rel_warped_crd_diff[:, 1, i_x:i_x + 3, i_y:i_y + 3])
            print(outliers_y[i_x:i_x + 3, i_y:i_y + 3])

        # check difference in the warping
        abs_warped_crd_diff = (num_J_warping[:, :, :, :] - crd_J_trs[:, i::3, :, :]).abs()
        rel_warped_crd_diff = abs_warped_crd_diff / num_J_warping.abs()
        outliers_x = (abs_warped_crd_diff[:, 0, :, :].abs() > 0.1) & (rel_warped_crd_diff[:, 0, :, :].abs() > eps) & (
                    num_J_warping[:, 0, :, :].abs() > eps) & (crd_J_trs[:, i, :, :].abs() > eps)
        outliers_y = (abs_warped_crd_diff[:, 1, :, :].abs() > 0.1) & (rel_warped_crd_diff[:, 1, :, :].abs() > eps) & (
                    num_J_warping[:, 1, :, :].abs() > eps) & (crd_J_trs[:, i+3, :, :].abs() > eps)

        outliers_x = outliers_x.squeeze().numpy() * 255.0
        outliers_y = outliers_y.squeeze().numpy() * 255.0
        outliers_x[invalid_depth_mask] = 0
        outliers_y[invalid_depth_mask] = 0
        cv2.imshow("warping_outliers_x", outliers_x)
        cv2.imshow("warping_outliers_y", outliers_y)

        # numerical jacoabian computation
        print('=>Whole jacobian')
        feat_p = geometry.render_features(img1, crd_p, valid_mask=depth_valid)
        res_p = feat_p[0] - img0
        J_res_trs_num = (res_p - res) / eps

        # analytic jacobian computation
        J_res_trs = geometry.matmul(err_J_crd, crd_J_trs, [C, 3])
        # remove nan
        J_res_trs_num[J_res_trs_num != J_res_trs_num] = 0
        J_res_trs[J_res_trs != J_res_trs] = 0
        avg_j_num = J_res_trs_num.mean()
        avg_j_ana = J_res_trs[:, i, :, :].mean()
        print("avg num jac:", avg_j_num)
        print("avg ana jac:", avg_j_ana)
        print("numerical jacobian:", J_res_trs_num[:, :, h_h: h_h + 2, h_w: h_w + 2])
        print("analytic jacobian:", J_res_trs[:, i, h_h: h_h + 2, h_w: h_w + 2])
        num_jacobian = J_res_trs_num.squeeze().numpy()
        ana_jacobian = J_res_trs.squeeze().numpy()

        # check large difference pixels
        abs_diff = (J_res_trs_num - J_res_trs[:, i, :, :]).abs()
        rel_diff = abs_diff / J_res_trs[:, i, :, :].abs()
        # outliers = (abs_diff > 0.1)
        outliers = (rel_diff > 0.1) & (abs_diff > 0.1)
        outliers = outliers.squeeze().numpy() * 255.0


        def debug_numerical_jacobian(i_x, i_y):
            print(J_res_trs_num[:, :, i_x:i_x + 3, i_y:i_y + 3])
            print(J_res_trs[:, i, i_x:i_x + 3, i_y:i_y + 3])
            print(abs_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
            print(rel_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
            print(outliers[i_x:i_x + 3, i_y:i_y + 3])


        debug_numerical_jacobian(200, 440)
        outliers[invalid_depth_mask] = 0
        cv2.imshow("outliers", outliers)
        cv2.waitKey(0)
        print("-------------------")

    for i in range(3):
        print("\n=>rotation perturbation: ", i)
        rot_p = rot.clone()
        pert_vec = torch.zeros(1, 3)
        pert_vec[:, i] = eps
        rot_perturb = geometry.batch_twist2Mat(pert_vec).double()
        rot_p = torch.bmm(rot_perturb, rot_p)

        crd_p, dpt3_p, _ = geometry.warp_net(dpt0, trs, rot_p, intrinsics, doJac=False, debug=True)
        print("=>Jacobian on Warping:")
        num_J_warping = (crd_p - crd) / eps
        print("Numerical in the middle", num_J_warping[:, :, h_h: h_h + 2, h_w: h_w + 2])
        print("Analytic in the middle", crd_J_rot[0, :, h_h: h_h + 2, h_w: h_w + 2][i::3])

        # check difference in the warping
        abs_warped_crd_diff = (num_J_warping[:, :, :, :] - crd_J_rot[:, i::3, :, :]).abs()
        rel_warped_crd_diff = abs_warped_crd_diff / num_J_warping.abs()
        outliers_x = (rel_warped_crd_diff[:, 0, :, :].abs() > 0.2) & (abs_warped_crd_diff[:, 0, :, :].abs() > 0.2)
        outliers_y = (rel_warped_crd_diff[:, 1, :, :].abs() > 0.2) & (abs_warped_crd_diff[:, 1, :, :].abs() > 0.2)
        outliers_x = outliers_x.squeeze().numpy() * 255.0
        outliers_y = outliers_y.squeeze().numpy() * 255.0
        outliers_x[invalid_depth_mask] = 0
        outliers_y[invalid_depth_mask] = 0
        num_jacobian = num_J_warping.squeeze().numpy()
        ana_jacobian = crd_J_rot.squeeze().numpy()
        cv2.imshow("warping_outliers_x", outliers_x)
        cv2.imshow("warping_outliers_y", outliers_y)

        i_x = 233
        i_y = 526


        def debug_crd_warping(i_x, i_y):
            print(num_J_warping[:, :, i_x:i_x + 3, i_y:i_y + 3])
            print(crd_J_rot[:, i::3, i_x:i_x + 3, i_y:i_y + 3])
            print(abs_warped_crd_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
            print(rel_warped_crd_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
            print(outliers_x[i_x:i_x + 3, i_y:i_y + 3])
            print(outliers_y[i_x:i_x + 3, i_y:i_y + 3])


        debug_crd_warping(i_x, i_y)

        # numerical jacoabian computation
        print('=>Whole jacobian')
        feat_p = geometry.render_features(img1, crd_p, valid_mask=depth_valid)
        res_p = feat_p[0] - img0
        J_res_rot_num = (res_p - res) / eps

        # analytic jacobian computation
        J_res_rot = geometry.matmul(err_J_crd, crd_J_rot, [C, 3])
        # remove nan
        J_res_rot_num[J_res_rot_num != J_res_rot_num] = 0
        J_res_rot[J_res_rot != J_res_rot] = 0
        avg_j_num = J_res_rot_num.mean()
        avg_j_ana = J_res_rot[:, i, :, :].mean()
        print("avg num jac:", avg_j_num)
        print("avg ana jac:", avg_j_ana)
        print("numerical jacobian in the middle:", J_res_rot_num[:, :, h_h: h_h + 2, h_w: h_w + 2])
        print("analytic jacobian in the middle:", J_res_rot[:, i, h_h: h_h + 2, h_w: h_w + 2])
        # num_jacobian = J_res_rot_num.squeeze().numpy()
        # ana_jacobian = J_res_rot.squeeze().numpy()

        # check large difference pixels
        # outliers = (abs(J_res_rot_num - J_res_rot[:, i, :, :]) > 0.1)
        abs_diff = (J_res_rot_num - J_res_rot[:, i, :, :]).abs()
        rel_diff = abs_diff / J_res_rot[:, i, :, :].abs()
        diff = rel_diff.squeeze().numpy()
        # outliers = (abs_diff > 0.1)
        outliers = (rel_diff > 0.1) & (abs_diff > 0.1)
        outliers = outliers.squeeze().numpy() * 255.0
        outliers[invalid_depth_mask] = 0
        cv2.imshow("outliers", outliers)
        cv2.waitKey(0)
        print("-------------------")


def test_forward_jacobian():
    import math
    import numpy as np
    from skimage import io
    import cv2
    from PIL import Image

    import math

    from data.TUM_RGBD import tq2mat
    from models.algorithms import feature_gradient
    from models.geometry import grad_bilinear_interpolation

    # from dense_feature.tool import rotation

    """
    ground truth data used for unit test:
    1.351500  0.586800  1.582300  0.836900  0.288700  -0.181000  -0.428300 /media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/rgb/1305031454.027662.png /media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/depth/1305031454.040976.png
    1.343600  0.577000  1.583100  0.830700  0.302300  -0.186300  -0.428800 /media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/rgb/1305031454.059654.png /media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/depth/1305031454.072690.png
    """
    B = 1
    eps = 1e-6
    VIS = False

    # captured data:
    img0_path = '/media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/rgb/1305031454.027662.png'
    dpt0_path = '/media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/depth/1305031454.040976.png'
    img1_path = '/media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/rgb/1305031454.059654.png'
    dpt1_path = '/media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/depth/1305031454.072690.png'
    timestamp0_pose = (1.351500, 0.586800, 1.582300, 0.836900, 0.288700, -0.181000, -0.428300)
    timestamp1_pose = (1.343600, 0.577000, 1.583100, 0.830700, 0.302300, -0.186300, -0.428800)
    DEPTH_SCALE = 1.0 / 5000
    img0 = io.imread(img0_path, as_gray=True)
    img1 = io.imread(img1_path, as_gray=True)
    dpt0 = io.imread(dpt0_path) / 5e3
    dpt1 = io.imread(dpt1_path) / 5e3
    # truncate the depth to accurate range
    invalid_depth_mask = (dpt0 < 0.5) | (dpt0 > 5.0) | (dpt1 < 0.5) | (dpt1 > 5.0)
    cv2.imshow("invalid_depth_mask", invalid_depth_mask * 255.0)
    # cv2.waitKey(0)

    T_W0 = tq2mat(timestamp0_pose)
    T_W1 = tq2mat(timestamp1_pose)
    T_10 = np.dot(np.linalg.inv(T_W1), T_W0)  # .astype(np.float32)
    trs = T_10[0:3, 3] + 0.1
    rot = T_10[0:3, 0:3]
    # rot = ortho_group.rvs(3)
    # trs = np.random.random((3))
    K = torch.tensor([525.0, 525.0, 319.5, 239.5]).unsqueeze(dim=0).repeat(B, 1).double()
    intrinsics = {'fx': 525.0, 'fy': 525.0, 'ux': 319.5, 'uy': 239.5}
    H, W = img0.shape
    feature_C = 8
    sigma_C = 8

    # convert to torch tensor
    img0 = torch.from_numpy(img0).unsqueeze(dim=0).unsqueeze(dim=0).repeat([1, feature_C, 1, 1])
    img1 = torch.from_numpy(img1).unsqueeze(dim=0).unsqueeze(dim=0).repeat([1, feature_C, 1, 1])
    dpt0 = torch.from_numpy(dpt0).unsqueeze(dim=0).unsqueeze(dim=0)
    dpt1 = torch.from_numpy(dpt1).unsqueeze(dim=0).unsqueeze(dim=0)
    invD0 = 1.0 / dpt0
    invD1 = 1.0 / dpt1
    trs = torch.from_numpy(trs).unsqueeze(dim=0)
    rot = torch.from_numpy(rot).unsqueeze(dim=0)
    pose = [rot, trs]

    # compute gradients
    img1_gx, img1_gy = feature_gradient(img1)
    # if VIS:
    # grad = img1_gx.squeeze().numpy()
    # cv2.imshow("gradient", grad)
    # cv2.waitKey(0)

    crd, dpt3, depth_valid, crd_J_dpt, crd_J_trs, crd_J_rot, \
    dpt3_J_dpt, dpt3_J_trs, dpt3_J_rot = geometry.warp_net(dpt0, trs, rot, intrinsics, doJac=True, debug=True)

    # numerical test of uncertainty-normalized feature=metric residual
    mEstimator = DeepRobustEstimator('None')
    solver_func0 = DirectSolverNet('Direct-Nodamping')
    tracker = TrustRegionWUncertainty(max_iter=mEstimator, solver_func=solver_func0, timers=True)

    # test J_e_u
    print("===========> J_e_u")

    def debug_jacobian_e_u(i_x, i_y):
        print(num_J_res_x[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(J_res_x[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(abs_J_res_x_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(rel_J_res_x_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(J_res_y[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(num_J_res_y[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(abs_J_res_y_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(rel_J_res_y_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(outliers_res_x[i_x:i_x + 3, i_y:i_y + 3])
        print(outliers_res_y[i_x:i_x + 3, i_y:i_y + 3])

    # use
    # confidence in feature-metric is 1, for debug
    # sigma0 = torch.ones(B, sigma_C, H, W, dtype=torch.double)
    # sigma1 = torch.ones(B, sigma_C, H, W, dtype=torch.double)
    # sigma0 = img0.clone()
    # sigma1 = img1.clone()
    sigma0 = torch.randn(B, sigma_C, H, W, dtype=torch.double)
    sigma1 = torch.randn(B, sigma_C, H, W, dtype=torch.double)
    crd_x_p = crd.clone()
    crd_y_p = crd.clone()
    crd_x_p[:, 0, :, :] += eps
    crd_y_p[:, 1, :, :] += eps
    [res, res_x, res_y] = [compose_residuals(pos, depth_valid, img0, img1, sigma0, sigma1) for
                           pos in [crd, crd_x_p, crd_y_p]]

    num_J_res_x = (res_x[0] - res[0]) / eps
    num_J_res_y = (res_y[0] - res[0]) / eps

    J_res_x, J_res_y = tracker.compute_j_e_u(crd, depth_valid, img0, img1, sigma0, sigma1, use_grad_interpolation=True)
    abs_J_res_x_diff = (num_J_res_x - J_res_x).abs()
    abs_J_res_y_diff = (num_J_res_y - J_res_y).abs()
    rel_J_res_x_diff = abs_J_res_x_diff / num_J_res_x.abs()
    rel_J_res_y_diff = abs_J_res_y_diff / num_J_res_y.abs()
    outliers_res_x = (rel_J_res_x_diff.abs() > 0.1) & (abs_J_res_x_diff.abs() > eps)
    outliers_res_y = (rel_J_res_y_diff.abs() > 0.1) & (abs_J_res_y_diff.abs() > eps)
    outliers_res_x = outliers_res_x[0, 0, :, :].numpy() * 255.0
    outliers_res_y = outliers_res_y[0, 0, :, :].numpy() * 255.0
    outliers_res_x[invalid_depth_mask] = 0
    outliers_res_y[invalid_depth_mask] = 0
    debug_jacobian_e_u(120, 500)
    cv2.imshow("outliers_res_x", outliers_res_x)
    cv2.imshow("outliers_res_y", outliers_res_y)
    cv2.waitKey(0)

    # test whole jacobian
    res, _, _ = tracker.compute_residuals(trs, rot, dpt0, dpt1, intrinsics,
                                          img0, img1, sigma0, sigma1)
    J_res_trs, J_res_rot = tracker.compute_Jacobian(trs, rot, dpt0, dpt1, intrinsics,
                                                    img0, img1, sigma0, sigma1,
                                                    use_grad_interpolation=True)

    for i in range(3):
        print("\n=>translation perturbation: ", i)
        trs_p = trs.clone()
        trs_p[0, i] += eps
        # numerical jacoabian computation
        print('=>Whole jacobian')
        res_p, _, _ = tracker.compute_residuals(trs_p, rot, dpt0, dpt1, intrinsics,
                                                img0, img1, sigma0, sigma1)
        J_res_trs_num = (res_p - res) / eps
        J_res_trs_ana = J_res_trs[:, i:i+1, :, :]

        # remove nan
        J_res_trs_num[J_res_trs_num != J_res_trs_num] = 0
        J_res_trs_ana[J_res_trs_ana != J_res_trs_ana] = 0
        avg_j_num = J_res_trs_num.mean()
        avg_j_ana = J_res_trs_ana.mean()
        print("avg num jac:", avg_j_num)
        print("avg ana jac:", avg_j_ana)
        num_jacobian = J_res_trs_num.squeeze().numpy()
        ana_jacobian = J_res_trs_ana.squeeze().numpy()

        # check large difference pixels
        abs_diff = (J_res_trs_num - J_res_trs_ana).abs()
        rel_diff = abs_diff / J_res_trs_ana.abs()
        # outliers = (abs_diff > 0.1)
        outliers = (rel_diff > 0.1) & (abs_diff > eps) & (J_res_trs_num > eps) & (J_res_trs_ana > eps)
        outliers = outliers[0, 0, :, :].numpy() * 255.0

        def debug_numerical_jacobian_j_res_trs(i_x, i_y):
            print(J_res_trs_num[:, :, i_x:i_x + 3, i_y:i_y + 3])
            print(J_res_trs_ana[:, :, i_x:i_x + 3, i_y:i_y + 3])
            print(abs_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
            print(rel_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
            print(outliers[i_x:i_x + 3, i_y:i_y + 3])

        debug_numerical_jacobian_j_res_trs(200, 440)
        outliers[invalid_depth_mask] = 0
        cv2.imshow("outliers", outliers)
        cv2.waitKey(0)
        print("-------------------")

    for i in range(3):
        print("\n=>rotation perturbation: ", i)
        rot_p = rot.clone()
        pert_vec = torch.zeros(1, 3)
        pert_vec[:, i] = eps
        rot_perturb = geometry.batch_twist2Mat(pert_vec).double()
        rot_p = torch.bmm(rot_perturb, rot_p)

        # numerical jacoabian computation
        print('=>Whole jacobian')
        res_p, _, _ = tracker.compute_residuals(trs, rot_p, dpt0, dpt1, intrinsics,
                                                img0, img1, sigma0, sigma1)
        J_res_rot_num = (res_p - res) / eps
        J_res_rot_ana = J_res_rot[:, i:i + 1, :, :]

        # remove nan
        J_res_rot_num[J_res_rot_num != J_res_rot_num] = 0
        J_res_rot_ana[J_res_rot_ana != J_res_rot_ana] = 0
        avg_j_num = J_res_rot_num.mean()
        avg_j_ana = J_res_rot_ana.mean()
        print("avg num jac:", avg_j_num)
        print("avg ana jac:", avg_j_ana)
        num_jacobian = J_res_rot_num.squeeze().numpy()
        ana_jacobian = J_res_rot_ana.squeeze().numpy()

        def debug_numerical_jacobian_j_res_rot(i_x, i_y):
            print(J_res_rot_num[:, :, i_x:i_x + 3, i_y:i_y + 3])
            print(J_res_rot_ana[:, :, i_x:i_x + 3, i_y:i_y + 3])
            print(abs_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
            print(rel_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
            print(outliers[i_x:i_x + 3, i_y:i_y + 3])

        # check large difference pixels
        abs_diff = (J_res_rot_num - J_res_rot_ana).abs()
        rel_diff = abs_diff / J_res_rot_ana.abs()
        outliers = (rel_diff > 0.1) & (abs_diff > eps) & (J_res_rot_num > eps) & (J_res_rot_ana > eps)
        outliers = outliers[0, 0, :, :].numpy() * 255.0
        outliers[invalid_depth_mask] = 0
        debug_numerical_jacobian_j_res_rot(239, 319)
        cv2.imshow("outliers", outliers)
        cv2.waitKey(0)
        print("-------------------")


def test_inverse_jacobian():
    import math
    import numpy as np
    from skimage import io
    import cv2
    from PIL import Image

    import math

    from data.TUM_RGBD import tq2mat
    from models.algorithms import feature_gradient
    from models.geometry import grad_bilinear_interpolation

    # from dense_feature.tool import rotation

    """
    ground truth data used for unit test:
    1.351500  0.586800  1.582300  0.836900  0.288700  -0.181000  -0.428300 /media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/rgb/1305031454.027662.png /media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/depth/1305031454.040976.png
    1.343600  0.577000  1.583100  0.830700  0.302300  -0.186300  -0.428800 /media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/rgb/1305031454.059654.png /media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/depth/1305031454.072690.png
    """
    B = 1
    eps = 1e-6

    # captured data:
    img0_path = '/media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/rgb/1305031454.027662.png'
    dpt0_path = '/media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/depth/1305031454.040976.png'
    img1_path = '/media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/rgb/1305031454.059654.png'
    dpt1_path = '/media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/depth/1305031454.072690.png'
    timestamp0_pose = (1.351500, 0.586800, 1.582300, 0.836900, 0.288700, -0.181000, -0.428300)
    timestamp1_pose = (1.343600, 0.577000, 1.583100, 0.830700, 0.302300, -0.186300, -0.428800)
    img0 = io.imread(img0_path, as_gray=True)
    img1 = io.imread(img1_path, as_gray=True)
    dpt0 = io.imread(dpt0_path) / 5e3
    dpt1 = io.imread(dpt1_path) / 5e3

    # truncate the depth to accurate range
    invalid_depth_mask = (dpt0 < 0.31) | (dpt0 > 10.0) | (dpt1 < 0.3) | (dpt1 > 10.0)
    cv2.imshow("invalid_depth_mask", invalid_depth_mask * 255.0)
    # cv2.waitKey(0)

    T_W0 = tq2mat(timestamp0_pose)
    T_W1 = tq2mat(timestamp1_pose)
    T_10 = np.dot(np.linalg.inv(T_W1), T_W0)  # .astype(np.float32)
    trs = T_10[0:3, 3]
    rot = T_10[0:3, 0:3]
    rot = ortho_group.rvs(3)
    trs = np.random.random((3))
    K = torch.tensor([525.0, 525.0, 319.5, 239.5]).unsqueeze(dim=0).repeat(B, 1).double()
    intrinsics = {'fx': 525.0, 'fy': 525.0, 'ux': 319.5, 'uy': 239.5}
    H, W = img0.shape
    f_C = 8
    s_C = 8

    # convert to torch tensor
    img0 = torch.from_numpy(img0).unsqueeze(dim=0).unsqueeze(dim=0).repeat([1, f_C, 1, 1])
    img1 = torch.from_numpy(img1).unsqueeze(dim=0).unsqueeze(dim=0).repeat([1, f_C, 1, 1])
    dpt0 = torch.from_numpy(dpt0).unsqueeze(dim=0).unsqueeze(dim=0)
    dpt1 = torch.from_numpy(dpt1).unsqueeze(dim=0).unsqueeze(dim=0)
    trs = torch.from_numpy(trs).unsqueeze(dim=0)
    rot = torch.from_numpy(rot).unsqueeze(dim=0)
    pose10 = [rot, trs]

    # pre-processing the inverse depth, all the invalid inputs depth are set to 0
    invD0 = torch.clamp(1.0 / dpt0, 0, 10)
    invD1 = torch.clamp(1.0 / dpt1, 0, 10)
    invD0[invD0 == invD0.min()] = 0
    invD1[invD1 == invD1.min()] = 0
    invD0[invD0 == invD0.max()] = 0
    invD1[invD1 == invD1.max()] = 0

    # use
    # confidence in feature-metric is 1, for debug
    # sigma0 = torch.ones(B, s_C, H, W, dtype=torch.double)
    # sigma1 = torch.ones(B, s_C, H, W, dtype=torch.double)
    # sigma0 = img0.clone()
    # sigma1 = img1.clone()
    sigma0 = torch.randn(B, s_C, H, W, dtype=torch.double)
    sigma1 = torch.randn(B, s_C, H, W, dtype=torch.double)

    # compute gradients
    px, py = geometry.generate_xy_grid(B, H, W, K)
    u_warped, v_warped, inv_z_warped = geometry.batch_warp_inverse_depth(
        px, py, invD0, pose10, K)
    warped_crd = torch.cat((u_warped, v_warped), dim=1)
    # occ = geometry.check_occ(inv_z_warped, invD1, warped_crd)
    occ = torch.zeros_like(invD1, dtype=torch.bool)
    occ_mask = occ.squeeze().numpy()*255.0
    cv2.imshow("occ", occ_mask)

    # numerical test of uncertainty-normalized feature=metric residual
    mEstimator = DeepRobustEstimator('None')
    solver_func0 = DirectSolverNet('Direct-Nodamping')
    tracker = U_IC(max_iter=mEstimator, solver_func=solver_func0, timers=True)
    weighted_res, _, _, _, _ = compose_residuals(warped_crd, occ, img0, img1, sigma0, sigma1)

    def debug_jacobian_e_u(i_x, i_y):
        print(J_e_x[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(num_J_res_x[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(abs_J_res_x_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(rel_J_res_x_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(outliers_res_x[i_x:i_x + 3, i_y:i_y + 3])

        print(J_e_y[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(num_J_res_y[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(abs_J_res_y_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(rel_J_res_y_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(outliers_res_y[i_x:i_x + 3, i_y:i_y + 3])

    def debug_warping(i_x, i_y):
        print(num_J_warping[:, 0, i_x:i_x + 3, i_y:i_y + 3])
        print(ana_J_warping[:, 0, i_x:i_x + 3, i_y:i_y + 3])
        print(abs_warped_crd_diff[:, 0, i_x:i_x + 3, i_y:i_y + 3])
        print(rel_warped_crd_diff[:, 0, i_x:i_x + 3, i_y:i_y + 3])
        print(outliers_x[i_x:i_x + 3, i_y:i_y + 3])
        print(num_J_warping[:, 1, i_x:i_x + 3, i_y:i_y + 3])
        print(ana_J_warping[:, 1, i_x:i_x + 3, i_y:i_y + 3])
        print(abs_warped_crd_diff[:, 1, i_x:i_x + 3, i_y:i_y + 3])
        print(rel_warped_crd_diff[:, 1, i_x:i_x + 3, i_y:i_y + 3])
        print(outliers_y[i_x:i_x + 3, i_y:i_y + 3])


    def debug_numerical_jacobian_j_res_trs(i_x, i_y):
        print(invalid_depth_mask[i_x:i_x + 3, i_y:i_y + 3])
        print(res_p[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(res0[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(J_res_trs_num[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(J_res_trs_ana[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(abs_diff_trs[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(rel_diff_trs[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(outliers_trs[i_x:i_x + 3, i_y:i_y + 3])

    def debug_numerical_jacobian_j_res_rot(i_x, i_y):
        print(J_res_rot_num[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(J_res_rot_ana[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(abs_diff_rot[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(rel_diff_rot[:, :, i_x:i_x + 3, i_y:i_y + 3])
        print(outliers_rot[i_x:i_x + 3, i_y:i_y + 3])


    # test J_e_u
    print("===========> testing J_e_u")
    crd0 = geometry.gen_coordinate_tensors(W, H).unsqueeze(dim=0).double()
    crd_x_p = crd0.clone()
    crd_y_p = crd0.clone()
    crd_x_p[:, 0, :, :] += eps
    crd_y_p[:, 1, :, :] += eps
    [res_x, res_y] = [compose_residuals(warped_crd, occ, img0, img1, sigma0, sigma1, perturbed_crd0=pos) for
                           pos in [crd_x_p, crd_y_p]]

    # numerical
    num_J_res_x = (res_x[0] - weighted_res) / eps
    num_J_res_y = (res_y[0] - weighted_res) / eps
    # analytical
    J_e_x, J_e_y = tracker.compose_j_e_u(invD0, img0, img1, sigma0, sigma1, px, py, K, warped_crd, occ, grad_interp=True)

    avg_j_res_x_num = num_J_res_x.mean()
    avg_j_res_x = J_e_x.mean()
    print("avg num jac_res_x:", avg_j_res_x_num)
    print("avg ana jac_res_x:", avg_j_res_x)
    avg_j_res_y_num = num_J_res_y.mean()
    avg_j_res_y = J_e_y.mean()
    print("avg num jac_res_y:", avg_j_res_y_num)
    print("avg ana jac_res_y:", avg_j_res_y)

    abs_J_res_x_diff = (num_J_res_x - J_e_x).abs()
    abs_J_res_y_diff = (num_J_res_y - J_e_y).abs()
    rel_J_res_x_diff = abs_J_res_x_diff / num_J_res_x.abs()
    rel_J_res_y_diff = abs_J_res_y_diff / num_J_res_y.abs()
    outliers_res_x = (rel_J_res_x_diff.abs() > 0.1) & (abs_J_res_x_diff.abs() > eps) & (num_J_res_x.abs() > eps) & (J_e_x.abs() > eps)
    outliers_res_y = (rel_J_res_y_diff.abs() > 0.1) & (abs_J_res_y_diff.abs() > eps) & (num_J_res_y.abs() > eps) & (J_e_y.abs() > eps)
    outliers_res_x = outliers_res_x[0, 0, :, :].numpy() * 255.0
    outliers_res_y = outliers_res_y[0, 0, :, :].numpy() * 255.0
    outliers_res_x[invalid_depth_mask] = 0
    outliers_res_y[invalid_depth_mask] = 0
    debug_jacobian_e_u(120, 500)
    cv2.imshow("outliers_res_x", outliers_res_x)
    cv2.imshow("outliers_res_y", outliers_res_y)
    cv2.waitKey(0)

    # test warping as well
    rdm_rot_0 = np.identity(3) + eps  # np.identity(3) + eps # ortho_group.rvs(3)
    rdm_trs_0 = (np.zeros((3)) + eps) # np.random.random((3))
    rdm_trs_0 = torch.from_numpy(rdm_trs_0).unsqueeze(dim=0)
    rdm_rot_0 = torch.from_numpy(rdm_rot_0).unsqueeze(dim=0)
    rdm_pose_0 = [rdm_rot_0, rdm_trs_0]
    crd, dpt1, depth_valid = geometry.warp_net(dpt0, rdm_trs_0, rdm_rot_0, intrinsics, doJac=False, debug=False)
    depth_valid = depth_valid.squeeze().numpy()
    invalid_depth_mask = invalid_depth_mask | ~depth_valid
    u0, v0, _ = geometry.batch_warp_inverse_depth(
        px, py, invD0, rdm_pose_0, K)
    crd0 = torch.cat((u0, v0), dim=1)
    res0, res_, sigma_, _, _ = compose_residuals(warped_crd, occ, img0, img1, sigma0, sigma1, perturbed_crd0=crd0)
    grad_f0, grad_sigma0, Jx_p, Jy_p = tracker.precompute_jacobian_components(invD0, img0, sigma0, px, py, K)
    J_F_p, _, _ = tracker.compose_inverse_jacobians(res_, sigma_, sigma0, grad_f0, grad_sigma0, Jx_p, Jy_p)

    # crd = geometry.gen_coordinate_tensors(W, H).unsqueeze(dim=0).double()
    # u, v = crd.split(1, dim=1)
    J_res_x, J_res_y = tracker.compose_j_e_u(invD0, img0, img1, sigma0, sigma1, px, py, K, warped_crd, occ, grad_interp=True, crd0=crd0)
    Jx_p, Jy_p = compute_jacobian_warping(invD0, K, px, py, rdm_pose_0)
    crd_J_p = torch.cat((Jx_p, Jy_p), dim=1)
    crd_J_p = crd_J_p.view(-1, H, W, 6)
    crd_J_p = crd_J_p.permute(0, 3, 1, 2)
    crd_J_rot, crd_J_trs = crd_J_p.split(3, dim=1)

    J_res_crd = torch.stack((J_res_x, J_res_y), dim=2) #[B, C, 2, H, W]
    J_res_crd = J_res_crd.contiguous().view(B, -1, H, W)  # [B, 1X2, H, W]
    J_res_trs = geometry.matmul(J_res_crd, crd_J_trs.contiguous().view(1, -1, H, W), [f_C, 3])
    J_res_rot = geometry.matmul(J_res_crd, crd_J_rot.contiguous().view(1, -1, H, W), [f_C, 3])


    # # test whole jacobian, using a more random matrix instead of identity matrix
    # j_res_trs = geometry.matmul(J_res_crd, crd_J_trs.contiguous().view(1, -1, H, W), [f_C, 3])
    # res, _, _, occ = compute_inverse_residuals(pose10, invD0, invD1, img0, img1, sigma0, sigma1,
    #                                               px, py, K)
    # J_res_p, J_res_trs, J_res_rot = tracker.compute_inverse_jacobian(pose10, invD0, invD1, img0, img1, sigma0, sigma1,
    #                                                         px, py, K)
    # diff = (j_res_trs - J_res_trs).sum().item()


    # use trs, rot for now
    for i in range(3):
        print("\n=>translation perturbation: ", i)
        trs_p = rdm_trs_0.clone()
        trs_p[0, i] += eps
        pert_pose = [rdm_rot_0, trs_p]

        u_perturbed, v_perturbed, inv_z_perturbed = geometry.batch_warp_inverse_depth(
            px, py, invD0, pert_pose, K)
        perturbed_crd = torch.cat((u_perturbed, v_perturbed), dim=1)
        # numerical jacobian computation
        print('=>warping jacobian')
        J_x_p_num = (u_perturbed - u0) / eps
        J_y_p_num = (v_perturbed - v0) / eps
        num_J_warping = torch.cat((J_x_p_num, J_y_p_num), dim=1)
    
        # check difference in the warping
        ana_J_warping = crd_J_trs[:, i, :, :].unsqueeze(dim=0)
        abs_warped_crd_diff = (num_J_warping - ana_J_warping).abs()
        rel_warped_crd_diff = abs_warped_crd_diff / num_J_warping.abs()
        outliers_x = (abs_warped_crd_diff[:, 0, :, :].abs() > 0.1) & (rel_warped_crd_diff[:, 0, :, :].abs() > eps) & (num_J_warping[:, 0, :, :].abs() > eps) & (ana_J_warping[:, 0, :, :].abs() > eps)
        outliers_y = (abs_warped_crd_diff[:, 1, :, :].abs() > 0.1) & (rel_warped_crd_diff[:, 1, :, :].abs() > eps) & (num_J_warping[:, 1, :, :].abs() > eps )& (ana_J_warping[:, 0, :, :].abs() > eps)
        outliers_x = outliers_x[0, :, :].numpy() * 255.0
        outliers_y = outliers_y[0, :, :].numpy() * 255.0
        outliers_x[invalid_depth_mask] = 0
        outliers_y[invalid_depth_mask] = 0
        cv2.imshow("warping_outliers_x", outliers_x)
        cv2.imshow("warping_outliers_y", outliers_y)
        debug_warping(239, 319)

        print('=>Whole jacobian')
        res_p, _, _, _, _ = compose_residuals(warped_crd, occ, img0, img1, sigma0, sigma1, perturbed_crd0=perturbed_crd)
        J_res_trs_num = (res_p - res0) / eps
        J_res_trs_ana = J_res_trs[:, i:i + 1, :, :]

        # remove nan
        J_res_trs_num[J_res_trs_num != J_res_trs_num] = 0
        J_res_trs_ana[J_res_trs_ana != J_res_trs_ana] = 0
        avg_j_num = J_res_trs_num.mean()
        avg_j_ana = J_res_trs_ana.mean()
        print("avg num jac:", avg_j_num)
        print("avg ana jac:", avg_j_ana)
        num_jacobian = J_res_trs_num.squeeze().numpy()
        ana_jacobian = J_res_trs_ana.squeeze().numpy()

        # check large difference pixels
        abs_diff_trs = (J_res_trs_num - J_res_trs_ana).abs()
        rel_diff_trs = abs_diff_trs / J_res_trs_ana.abs()
        # outliers = (abs_diff > 0.1)
        outliers_trs = (rel_diff_trs > 0.1) & (abs_diff_trs > eps) & (J_res_trs_num.abs() > eps) & (J_res_trs_ana.abs() > eps)
        outliers_trs = outliers_trs[0, 0, :, :].numpy() * 255.0
        outliers_trs[invalid_depth_mask] = 0
        debug_numerical_jacobian_j_res_trs(239, 319)
        cv2.imshow("outliers", outliers_trs)
        cv2.waitKey(0)
        print("-------------------")

    for i in range(3):
        print("\n=>rotation perturbation: ", i)
        rdm_rot_0_p = rdm_rot_0.clone()
        pert_vec = torch.zeros(1, 3)
        pert_vec[:, i] = eps
        rot_perturb = geometry.batch_twist2Mat(pert_vec).double()
        rdm_rot_0_p = torch.bmm(rot_perturb, rdm_rot_0_p)
        pert_pose = [rdm_rot_0_p, rdm_trs_0]

        u_perturbed, v_perturbed, inv_z_perturbed = geometry.batch_warp_inverse_depth(
            px, py, invD0, pert_pose, K)
        perturbed_crd = torch.cat((u_perturbed, v_perturbed), dim=1)

        print('=>warping jacobian')
        J_x_p_num = (u_perturbed - u0) / eps
        J_y_p_num = (v_perturbed - v0) / eps
        num_J_warping = torch.cat((J_x_p_num, J_y_p_num), dim=1)

        # check difference in the warping
        ana_J_warping = crd_J_rot[:, i, :, :].unsqueeze(dim=0)
        abs_warped_crd_diff = (num_J_warping - ana_J_warping).abs()
        rel_warped_crd_diff = abs_warped_crd_diff / num_J_warping.abs()
        outliers_x = (abs_warped_crd_diff[:, 0, :, :].abs() > 0.1) & (rel_warped_crd_diff[:, 0, :, :].abs() > eps) & (
                num_J_warping[:, 0, :, :].abs() > eps) & (ana_J_warping[:, 0, :, :].abs() > eps)
        outliers_y = (abs_warped_crd_diff[:, 1, :, :].abs() > 0.1) & (rel_warped_crd_diff[:, 1, :, :].abs() > eps) & (
                num_J_warping[:, 1, :, :].abs() > eps) & (ana_J_warping[:, 0, :, :].abs() > eps)
        outliers_x = outliers_x[0, :, :].numpy() * 255.0
        outliers_y = outliers_y[0, :, :].numpy() * 255.0
        outliers_x[invalid_depth_mask] = 0
        outliers_y[invalid_depth_mask] = 0
        debug_warping(239, 319)
        cv2.imshow("warping_outliers_x", outliers_x)
        cv2.imshow("warping_outliers_y", outliers_y)



        # numerical jacobian computation
        print('=>Whole jacobian')
        res_p, _, _, _, _ = compose_residuals(warped_crd, occ, img0, img1, sigma0, sigma1,
                                              perturbed_crd0=perturbed_crd)
        J_res_rot_num = (res_p - res0) / eps
        J_res_rot_ana = J_res_rot[:, i:i + 1, :, :]

        # remove nan
        J_res_rot_num[J_res_rot_num != J_res_rot_num] = 0
        J_res_rot_ana[J_res_rot_ana != J_res_rot_ana] = 0
        avg_j_num = J_res_rot_num.mean()
        avg_j_ana = J_res_rot_ana.mean()
        print("avg num jac:", avg_j_num)
        print("avg ana jac:", avg_j_ana)
        num_jacobian = J_res_rot_num.squeeze().numpy()
        ana_jacobian = J_res_rot_ana.squeeze().numpy()

        # check large difference pixels
        abs_diff_rot = (J_res_rot_num - J_res_rot_ana).abs()
        rel_diff_rot = abs_diff_rot / J_res_rot_ana.abs()
        outliers_rot = (rel_diff_rot > 0.1) & (abs_diff_rot > eps) & (J_res_rot_num.abs() > eps) & (J_res_rot_ana.abs() > eps)
        outliers_rot = outliers_rot[0, 0, :, :].numpy() * 255.0
        debug_numerical_jacobian_j_res_rot(239, 319)
        outliers_rot[invalid_depth_mask] = 0
        cv2.imshow("outliers", outliers_rot)
        cv2.waitKey(0)
        print("-------------------")

def test_multi_channel_jacobian():
    B = 10
    C = 32
    HW = 120
    y = 6

    jac = torch.rand((B, C, HW, y)).double()*10

    jac_reshape1 = jac.view(B, -1, y)
    jtj1 = torch.bmm(torch.transpose(jac_reshape1, 1, 2), jac_reshape1)

    jac_reshape2 = jac.permute(0, 2, 1, 3).contiguous() # [B, HW, C, 6]
    jac_reshape2 = jac_reshape2.view(-1, C, y)  # [B*HW, C, 6]
    jtj2 = torch.bmm(torch.transpose(jac_reshape2, 1, 2), jac_reshape2)  # [B*HW, 6, 6]
    jtj2 = jtj2.view(B, HW, y, y)
    jtj2 = jtj2.sum(dim=1)
    print(torch.abs(jtj1-jtj2))
    print(torch.eq(jtj1, jtj2))



def test_ICP_jacobian():
    import math
    import numpy as np
    from skimage import io
    import cv2
    from PIL import Image

    import math

    from data.TUM_RGBD import tq2mat
    from models.algorithms import feature_gradient
    from models.geometry import grad_bilinear_interpolation

    # from dense_feature.tool import rotation

    """
    ground truth data used for unit test:
    1.351500  0.586800  1.582300  0.836900  0.288700  -0.181000  -0.428300 /media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/rgb/1305031454.027662.png /media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/depth/1305031454.040976.png
    1.343600  0.577000  1.583100  0.830700  0.302300  -0.186300  -0.428800 /media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/rgb/1305031454.059654.png /media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/depth/1305031454.072690.png
    """
    B = 1
    eps = 1e-6
    VIS = False

    # captured data:
    dpt0_path = '/media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/depth/1305031454.040976.png'
    dpt1_path = '/media/binbin/data/dataset/tum-rgbd/rgbd_dataset_freiburg1_desk/depth/1305031454.072690.png'
    timestamp0_pose = (1.351500, 0.586800, 1.582300, 0.836900, 0.288700, -0.181000, -0.428300)
    timestamp1_pose = (1.343600, 0.577000, 1.583100, 0.830700, 0.302300, -0.186300, -0.428800)
    dpt0 = io.imread(dpt0_path) / 5e3
    dpt1 = io.imread(dpt1_path) / 5e3
    # truncate the depth to accurate range
    # invalid_depth_mask = (dpt0 < 0.5) | (dpt0 > 5.0) | (dpt1 < 0.5) | (dpt1 > 5.0)
    # cv2.imshow("invalid_depth_mask", invalid_depth_mask * 255.0)
    # cv2.waitKey(0)

    T_W0 = tq2mat(timestamp0_pose)
    T_W1 = tq2mat(timestamp1_pose)
    T_10 = np.dot(np.linalg.inv(T_W1), T_W0)  # .astype(np.float32)
    trs = T_10[0:3, 3] # + 0.1
    rot = T_10[0:3, 0:3]
    # rot = ortho_group.rvs(3)
    # trs = np.random.random((3))
    K = torch.tensor([525.0, 525.0, 319.5, 239.5]).unsqueeze(dim=0).repeat(B, 1).double()
    intrinsics = {'fx': 525.0, 'fy': 525.0, 'ux': 319.5, 'uy': 239.5}
    H, W = dpt0.shape
    # C = 1

    # convert to torch tensor
    depth0 = torch.from_numpy(dpt0).unsqueeze(dim=0).unsqueeze(dim=0).double()
    depth1 = torch.from_numpy(dpt1).unsqueeze(dim=0).unsqueeze(dim=0).double()
    trs = torch.from_numpy(trs).unsqueeze(dim=0).double()
    rot = torch.from_numpy(rot).unsqueeze(dim=0).double()
    pose10 = [rot, trs]

    px, py = geometry.generate_xy_grid(B, H, W, K)
    vertex0 = geometry.compute_vertex(depth0, px, py)
    vertex1 = geometry.compute_vertex(depth1, px, py)
    normal0 = compute_normal(vertex0)
    normal1 = compute_normal(vertex1)

    mEstimator = DeepRobustEstimator('None')
    solver_func0 = DirectSolverNet('Direct-Nodamping')
    tracker = ICP(max_iter=mEstimator, solver_func=solver_func0, timers=True)
    residuals, J_F_p, occ = tracker.compute_residuals_jacobian(vertex0, vertex1, normal0, normal1, pose10, K)
    # match inverse
    J_F_p = -J_F_p
    C = J_F_p.shape[1]

    occ_mask = occ[0].permute(1,2,0).to(dtype=torch.uint8).cpu().numpy() * 255
    cv2.imshow("occ", occ_mask)
    # cv2.waitKey(0)

    for i in range(3):
        print("\n=>translation perturbation: ", i)
        trs_p = trs.clone()
        trs_p[0, i] += eps
        pose10_p = [rot, trs_p]
        residuals_p, _, _ = tracker.compute_residuals_jacobian(vertex0, vertex1, normal0, normal1, pose10_p, K)

        # numerical jacoabian computation
        print('=>Whole jacobian')
        J_res_trs_num = (residuals_p - residuals) / eps
        residuals_np = residuals.cpu().numpy()
        residuals_p_np = residuals_p.cpu().numpy()
        J_res_trs_num_np = J_res_trs_num.cpu().numpy()
        J_res_trs_np = J_res_trs.cpu().numpy()

        # analytic jacobian computation
        J_res_trs = J_F_p.view(B,C,H,W,6)[:,:,:,:,i+3]
        # remove nan
        J_res_trs_num[J_res_trs_num != J_res_trs_num] = 0
        J_res_trs[J_res_trs != J_res_trs] = 0
        avg_j_num = J_res_trs_num.mean()
        avg_j_ana = J_res_trs.mean()
        print("avg num jac:", avg_j_num)
        print("avg ana jac:", avg_j_ana)
        num_jacobian = J_res_trs_num.squeeze().numpy()
        ana_jacobian = J_res_trs.squeeze().numpy()


        print(torch.eq(J_res_trs_num, J_res_trs).sum())
        # check large difference pixels
        abs_diff = (J_res_trs_num - J_res_trs).abs()
        rel_diff = abs_diff / J_res_trs.abs()
        rel_diff_np = rel_diff.numpy()
        outliers = (rel_diff > 0.1) & (abs_diff > eps) & (J_res_trs_num.abs() > eps) & (J_res_trs.abs() > eps)
        outliers = outliers[0].permute(1,2,0).to(dtype=torch.uint8).cpu().numpy() * 255
        #
        #
        # def debug_numerical_jacobian(i_x, i_y):
        #     print(J_res_trs_num[:, :, i_x:i_x + 3, i_y:i_y + 3])
        #     print(J_res_trs[:, i, i_x:i_x + 3, i_y:i_y + 3])
        #     print(abs_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
        #     print(rel_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
        #     print(outliers[i_x:i_x + 3, i_y:i_y + 3])
        #
        #
        # debug_numerical_jacobian(200, 440)
        # outliers[occ_mask] = 0
        cv2.imshow("outliers", outliers)
        cv2.waitKey(0)
        print("-------------------")

    for i in range(3):
        print("\n=>rotation perturbation: ", i)
        rot_p = rot.clone()
        pert_vec = torch.zeros(1, 3)
        pert_vec[:, i] = eps
        rot_perturb = geometry.batch_twist2Mat(pert_vec).double()
        rot_p = torch.bmm(rot_perturb, rot_p)

        pose10_p = [rot_p, trs]
        residuals_p, _, _ = tracker.compute_residuals_jacobian(vertex0, vertex1, normal0, normal1, pose10_p, K)

        # numerical jacoabian computation
        print('=>Whole jacobian')
        J_res_rot_num = (residuals_p - residuals) / eps
        residuals_np = residuals.cpu().numpy()
        residuals_p_np = residuals_p.cpu().numpy()
        J_res_rot_num_np = J_res_rot_num.cpu().numpy()

        # analytic jacobian computation
        J_res_rot = J_F_p.view(B, C, H, W, 6)[:, :, :, :, i]
        # remove nan
        J_res_rot_num[J_res_rot_num != J_res_rot_num] = 0
        J_res_rot[J_res_rot != J_res_rot] = 0
        avg_j_num = J_res_rot_num.mean()
        avg_j_ana = J_res_rot.mean()
        print("avg num jac:", avg_j_num)
        print("avg ana jac:", avg_j_ana)
        num_jacobian = J_res_rot_num.squeeze().numpy()
        ana_jacobian = J_res_rot.squeeze().numpy()

        print(torch.eq(J_res_rot_num, J_res_rot).sum())
        # check large difference pixels
        abs_diff = (J_res_rot_num - J_res_rot).abs()
        rel_diff = abs_diff / J_res_rot.abs()
        rel_diff_np = rel_diff.numpy()
        outliers = (rel_diff > 0.1) & (abs_diff > eps) & (J_res_rot_num.abs() > eps) & (J_res_rot.abs() > eps)
        outliers = outliers[0].permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy() * 255
        #
        #
        # def debug_numerical_jacobian(i_x, i_y):
        #     print(J_res_trs_num[:, :, i_x:i_x + 3, i_y:i_y + 3])
        #     print(J_res_trs[:, i, i_x:i_x + 3, i_y:i_y + 3])
        #     print(abs_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
        #     print(rel_diff[:, :, i_x:i_x + 3, i_y:i_y + 3])
        #     print(outliers[i_x:i_x + 3, i_y:i_y + 3])
        #
        #
        # debug_numerical_jacobian(200, 440)
        # outliers[occ_mask] = 0
        cv2.imshow("outliers", outliers)
        cv2.waitKey(0)
        print("-------------------")


if __name__ == "__main__":
    # test_normal_forward_jacobian()
    # test_forward_jacobian()
    # test_inverse_jacobian()
    # test_multi_channel_jacobian()
    test_ICP_jacobian()