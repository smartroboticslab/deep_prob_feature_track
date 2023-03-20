"""
Main function to call Deep Probabilistic Feature-metric Tracking.
Support both ego-motion and object-motion tracking
# SPDX-FileCopyrightText: 2021 Binbin Xu
# SPDX-License-Identifier: BSD-3-Clause

"""
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
import cv2

from models.submodules import convLayer as conv
from models.submodules import color_normalize

from models.algorithms import TrustRegionWUncertainty as TrustRegionU
from models.algorithms import TrustRegionBase as TrustRegion
from models.algorithms import TrustRegionInverseWUncertainty as TR_U_IC
from models.algorithms import Inverse_ICP as TR_ICP
from models.algorithms import ImagePyramids, DirectSolverNet, FeaturePyramid, DeepRobustEstimator, SFMPoseNet, PoseNet
from models.algorithms import ScaleNet
from tools import display
from models.algorithms import warp_images
from models.geometry import batch_Rt_compose
from Logger import check_directory


class LeastSquareTracking(nn.Module):

    # all enum types
    NONE                = -1
    RGB                 = 0

    CONV_RGBD           = 1
    CONV_RGBD2          = 2

    def __init__(self,
                 encoder_name,
                 uncertainty_type,
                 max_iter_per_pyr,
                 mEst_type,
                 solver_type,
                 tr_samples = 10,
                 direction='inverse',
                 options=None,
                 vis_res=False,
                 add_init_noise=False,
                 no_weight_sharing=False,
                 init_pose_type=None,
                 feature_channel=None,
                 uncertainty_channel=None,
                 remove_tru_sigma=None,
                 feature_extract=None,
                 combine_ICP=None,
                 scaler=None,
                 scale_scaler=None,
                 timers = None):
        """
        :param the backbone network used for regression.
        :param the maximum number of iterations at each pyramid levels
        :param the type of weighting functions.
        :param the type of solver. 
        :param number of samples in trust-region solver
        :param True if we do not want to share weight at different pyramid levels
        :param (optional) time to benchmark time consumed at each step
        """
        super(LeastSquareTracking, self).__init__()
        self.scales = [0,1,2,3]
        self.construct_image_pyramids = ImagePyramids(self.scales, pool='avg')
        self.construct_depth_pyramids = ImagePyramids(self.scales, pool='max')

        self.timers = timers
        self.direction = direction
        self.mEst_type = mEst_type
        self.vis_res = vis_res
        self.add_init_noise = add_init_noise

        """ =============================================================== """
        """              use option to transfer parameters                  """
        """ =============================================================== """
        # used in the forward function
        self.vis_feat_uncer = options.vis_feat
        self.uncer_prop=options.train_uncer_prop
        self.combine_icp = options.combine_ICP if combine_ICP is None else combine_ICP
        # feature & uncertainty pyramid
        feature_extract = options.feature_extract if feature_extract is None else feature_extract
        self.uncertainty_type = uncertainty_type
        feature_channel = options.feature_channel if feature_channel is None else feature_channel
        uncertainty_channel = options.uncertainty_channel if uncertainty_channel is None else uncertainty_channel
        # scaling function
        scale_type = options.scaler if scaler is None else scaler
        # tracking solving function
        remove_tru_sigma = options.remove_tru_sigma if remove_tru_sigma is None else remove_tru_sigma
        train_uncer_prop = options.train_uncer_prop
        # pose initialization function
        if init_pose_type is None: init_pose_type = options.init_pose
        self.predict_init_pose = False if init_pose_type == 'identity' else True
        self.train_init_pose = options.train_init_pose
        init_pose_scale = options.scale_init_pose
        init_pose_multi_hypo = options.multi_hypo
        res_input_for_init_pose = options.res_input
        self.checkpoint = options.checkpoint

        """ =============================================================== """
        """             Initialize the Deep Feature Extractor               """
        """ =============================================================== """

        if encoder_name == 'RGB':
            print('The network will use raw image as measurements.')
            self.encoder = None
            self.encoder_type = self.RGB
            context_dim = 1
        elif encoder_name == 'ConvRGBD':
            print('Use a network with RGB-D information \
            to extract the features')
            context_dim = 4
            self.encoder = FeaturePyramid(D=context_dim,
                                          w_uncertainty=uncertainty_type,
                                          feature_channel=feature_channel,
                                          feature_extract=feature_extract,
                                          uncertainty_channel=uncertainty_channel,
                                          )
            self.encoder_type = self.CONV_RGBD
        elif encoder_name == 'ConvRGBD2':
            print('Use two stream network with two frame input')
            context_dim = 8
            self.encoder = FeaturePyramid(D=context_dim,
                                          w_uncertainty=uncertainty_type,
                                          feature_channel=feature_channel,
                                          feature_extract=feature_extract,
                                          uncertainty_channel=uncertainty_channel,
                                          )
            self.encoder_type = self.CONV_RGBD2
        elif encoder_name == 'ICP':
            print('ICP not using images, just for visualization')
            self.encoder = None
            self.encoder_type = self.RGB
            context_dim = 1
        else:
            raise NotImplementedError()

        """ =============================================================== """
        """             Initialize the Scale Estimator                     """
        """ =============================================================== """
        if no_weight_sharing:
            self.scaler_func0 = ScaleNet(scale_type, scale=scale_scaler)
            self.scaler_func1 = ScaleNet(scale_type, scale=scale_scaler)
            self.scaler_func2 = ScaleNet(scale_type, scale=scale_scaler)
            self.scaler_func3 = ScaleNet(scale_type, scale=scale_scaler)
            scaler_funcs = [self.scaler_func0, self.scaler_func1, self.scaler_func2,
            self.scaler_func3]
        else:
            self.scaler_func = ScaleNet(scale_type, scale=scale_scaler)
            scaler_funcs = [self.scaler_func, self.scaler_func, self.scaler_func,
            self.scaler_func]

        """ =============================================================== """
        """             Initialize the Robust Estimator                     """
        """ =============================================================== """

        if no_weight_sharing:
            self.mEst_func0 = DeepRobustEstimator(mEst_type)
            self.mEst_func1 = DeepRobustEstimator(mEst_type)
            self.mEst_func2 = DeepRobustEstimator(mEst_type)
            self.mEst_func3 = DeepRobustEstimator(mEst_type)
            mEst_funcs = [self.mEst_func0, self.mEst_func1, self.mEst_func2,
            self.mEst_func3]
        else:
            self.mEst_func = DeepRobustEstimator(mEst_type)
            mEst_funcs = [self.mEst_func, self.mEst_func, self.mEst_func,
            self.mEst_func]

        """ =============================================================== """
        """             Initialize the Trust-Region Damping                 """
        """ =============================================================== """

        if no_weight_sharing:
            # for residual volume, the input K is not assigned correctly
            self.solver_func0 = DirectSolverNet(solver_type, samples=tr_samples, direction=direction)
            self.solver_func1 = DirectSolverNet(solver_type, samples=tr_samples, direction=direction)
            self.solver_func2 = DirectSolverNet(solver_type, samples=tr_samples, direction=direction)
            self.solver_func3 = DirectSolverNet(solver_type, samples=tr_samples, direction=direction)
            solver_funcs = [self.solver_func0, self.solver_func1,
            self.solver_func2, self.solver_func3]
        else:
            self.solver_func = DirectSolverNet(solver_type, samples=tr_samples, direction=direction)
            solver_funcs = [self.solver_func, self.solver_func,
                self.solver_func, self.solver_func]

        """ =============================================================== """
        """             Initialize the Trust-Region Method                  """
        """ =============================================================== """

        self.track_type = None  # currently support: IC, U_IC, U_FC
        if uncertainty_type == 'ICP':
            print("Inverse ICP tracking method")
            self.track_type = 'ICP'
            self.tr_update0 = TR_ICP(max_iter_per_pyr,
                                     mEst_func=mEst_funcs[0],
                                     solver_func=solver_funcs[0],
                                     timers=timers)
            self.tr_update1 = TR_ICP(max_iter_per_pyr,
                                     mEst_func=mEst_funcs[1],
                                     solver_func=solver_funcs[1],
                                     timers=timers)
            self.tr_update2 = TR_ICP(max_iter_per_pyr,
                                     mEst_func=mEst_funcs[2],
                                     solver_func=solver_funcs[2],
                                     timers=timers)
            self.tr_update3 = TR_ICP(max_iter_per_pyr,
                                     mEst_func=mEst_funcs[3],
                                     solver_func=solver_funcs[3],
                                     timers=timers)
        elif uncertainty_type == 'None' and direction == 'inverse':
            print("Deep inverse compositional algorithm, without uncertainty")
            self.track_type = 'IC'
            self.tr_update0 = TrustRegion(max_iter_per_pyr,
                mEst_func   = mEst_funcs[0],
                solver_func = solver_funcs[0],
                timers      = timers)
            self.tr_update1 = TrustRegion(max_iter_per_pyr,
                mEst_func   = mEst_funcs[1],
                solver_func = solver_funcs[1],
                timers      = timers)
            self.tr_update2 = TrustRegion(max_iter_per_pyr,
                mEst_func   = mEst_funcs[2],
                solver_func = solver_funcs[2],
                timers      = timers)
            self.tr_update3 = TrustRegion(max_iter_per_pyr,
                mEst_func   = mEst_funcs[3],
                solver_func = solver_funcs[3],
                timers      = timers)
        elif uncertainty_type != 'None' and direction == 'forward':
            print("=> Deep forward compositional algorithm, with uncertainty of", uncertainty_type)
            self.track_type = 'U_FC'
            self.tr_update0 = TrustRegionU(max_iter_per_pyr,
                                           mEst_func=mEst_funcs[0],
                                           solver_func=solver_funcs[0],
                                           timers=timers)
            self.tr_update1 = TrustRegionU(max_iter_per_pyr,
                                           mEst_func=mEst_funcs[1],
                                           solver_func=solver_funcs[1],
                                           timers=timers)
            self.tr_update2 = TrustRegionU(max_iter_per_pyr,
                                           mEst_func=mEst_funcs[2],
                                           solver_func=solver_funcs[2],
                                           timers=timers)
            self.tr_update3 = TrustRegionU(max_iter_per_pyr,
                                           mEst_func=mEst_funcs[3],
                                           solver_func=solver_funcs[3],
                                           timers=timers)
        elif uncertainty_type != 'None' and direction == 'inverse':
            print("=> Deep Inverse Compositional algorithm, with uncertainty of", uncertainty_type)
            self.track_type = 'U_IC'
            self.tr_update0 = TR_U_IC(max_iter_per_pyr,
                                      mEst_func=mEst_funcs[0],
                                      solver_func=solver_funcs[0],
                                      timers=timers,
                                      combine_icp=self.combine_icp,
                                      scale_func=scaler_funcs[0],
                                      uncer_prop=train_uncer_prop,
                                      remove_tru_sigma=remove_tru_sigma,
                                      )
            self.tr_update1 = TR_U_IC(max_iter_per_pyr,
                                      mEst_func=mEst_funcs[1],
                                      solver_func=solver_funcs[1],
                                      timers=timers,
                                      combine_icp=self.combine_icp,
                                      scale_func=scaler_funcs[1],
                                      uncer_prop=train_uncer_prop,
                                      remove_tru_sigma=remove_tru_sigma,
                                      )
            self.tr_update2 = TR_U_IC(max_iter_per_pyr,
                                      mEst_func=mEst_funcs[2],
                                      solver_func=solver_funcs[2],
                                      timers=timers,
                                      combine_icp=self.combine_icp,
                                      scale_func=scaler_funcs[2],
                                      uncer_prop=train_uncer_prop,
                                      remove_tru_sigma=remove_tru_sigma,
                                      )
            self.tr_update3 = TR_U_IC(max_iter_per_pyr,
                                      mEst_func=mEst_funcs[3],
                                      solver_func=solver_funcs[3],
                                      timers=timers,
                                      combine_icp=self.combine_icp,
                                      scale_func=scaler_funcs[3],
                                      uncer_prop=train_uncer_prop,
                                      remove_tru_sigma=remove_tru_sigma,
                                      )
        else:
            raise NotImplementedError("Not supported tracking method, check again about the configs of uncertainty and direction.")

        """ =============================================================== """
        """             Initialize Pose Predictor Network                 """
        """ =============================================================== """
        print("=> Init pose is estimated by", init_pose_type)
        if self.predict_init_pose:
            if self.train_init_pose:
                print("=> Add predicted pose to the output pose (joint train the init pose)")
            print("=> # init pose hypothesis:", init_pose_multi_hypo)
            if init_pose_type == 'sfm_net':
                self.pose_predictor = SFMPoseNet(scale_motion=init_pose_scale,
                                                 multi_hypo=init_pose_multi_hypo,
                                                 res_input=res_input_for_init_pose)
            elif init_pose_type == 'dense_net':
                self.pose_predictor = PoseNet(scale_motion=init_pose_scale,
                                              multi_hypo=init_pose_multi_hypo,
                                              res_input=res_input_for_init_pose)
            else:
                raise NotImplementedError('unsupported pose predictor network')

    def forward(self, img0, img1, depth0, depth1, K, init_only=False,
                logger=None, iteration=0, vis=False, obj_mask0=None, obj_mask1=None, index=None):
        """
        :input
        :param the reference image
        :param the target image
        :param the inverse depth of the reference image
        :param the inverse depth of the target image
        :param the pin-hole camera instrinsic (in vector) [fx, fy, cx, cy] 
        :param the initial pose [Rotation, translation]
        --------
        :return 
        :param estimated transform 
        """
        preprocessed_data = self._preprocess(img0, img1, depth0, depth1,
                                             poseI=None, obj_mask0=obj_mask0, obj_mask1=obj_mask1)
        (I0, I1, x0, x1, sigma0, sigma1, dpt0_pyr, dpt1_pyr, inv_d0, inv_d1,
         obj_mask0_pyr, obj_mask1_pyr, poseI) = preprocessed_data

        poses_to_train = [[],[]] # '[rotation, translation]'
        sigma_ksi = []

        # initial pose prediction
        if self.predict_init_pose and self.train_init_pose:
            [R0, t0] = poseI
            poses_to_train[0].append(R0)
            poses_to_train[1].append(t0)
            if self.uncer_prop:
                B = inv_d0[0].shape[0]
                sigma_ksi.append(torch.eye(6).view(1, 6, 6).type_as(R0).repeat(B,1,1))

        # the prior of the mask
        prior_W = torch.ones(inv_d0[3].shape).type_as(inv_d0[3]) * 0.001

        if self.timers: self.timers.tic('trust-region update')
        # trust region update on level 3
        K3 = K >> 3
        if self.track_type == 'U_FC':
            output3 = self.tr_update3(poseI, x0[3], x1[3], dpt0_pyr[3], dpt1_pyr[3], K3,
                                      sigma0=sigma0[3], sigma1=sigma1[3], wPrior=prior_W, vis_res=self.vis_res,
                                      obj_mask1=obj_mask1_pyr[3])
        elif self.track_type == 'U_IC':
            output3 = self.tr_update3(poseI, x0[3], x1[3], inv_d0[3], inv_d1[3], K3,
                                      sigma0=sigma0[3], sigma1=sigma1[3], wPrior=prior_W, vis_res=self.vis_res,
                                      depth0=dpt0_pyr[3], depth1=dpt1_pyr[3],
                                      obj_mask0=obj_mask0_pyr[3], obj_mask1=obj_mask1_pyr[3])
        elif self.track_type == 'IC':
            output3 = self.tr_update3(poseI, x0[3], x1[3], inv_d0[3], inv_d1[3], K3, prior_W, vis_res=self.vis_res,
                                      obj_mask0=obj_mask0_pyr[3], obj_mask1=obj_mask1_pyr[3])
        elif self.track_type == 'ICP':
            output3 = self.tr_update3(poseI, dpt0_pyr[3], dpt1_pyr[3], K3, prior_W, vis_res=self.vis_res,
                                      obj_mask1=obj_mask1_pyr[3])
        else:
            raise NotImplementedError()
        pose3, mEst_W3 = output3[0], output3[1]
        poses_to_train[0].append(pose3[0])
        poses_to_train[1].append(pose3[1])
        if self.uncer_prop:
            sigma_ksi.append(output3[2])
        # trust region update on level 2
        K2 = K >> 2
        if self.track_type == 'U_FC':
            output2 = self.tr_update2(pose3, x0[2], x1[2], dpt0_pyr[2], dpt1_pyr[2], K2,
                                      sigma0=sigma0[2], sigma1=sigma1[2], wPrior=mEst_W3, vis_res=self.vis_res,
                                      obj_mask1=obj_mask1_pyr[2])
        elif self.track_type == 'U_IC':
            output2 = self.tr_update2(pose3, x0[2], x1[2], inv_d0[2], inv_d1[2], K2,
                                      sigma0=sigma0[2], sigma1=sigma1[2], wPrior=mEst_W3, vis_res=self.vis_res,
                                      depth0=dpt0_pyr[2], depth1=dpt1_pyr[2],
                                      obj_mask0=obj_mask0_pyr[2], obj_mask1=obj_mask1_pyr[2])
        elif self.track_type == 'IC':
            output2 = self.tr_update2(pose3, x0[2], x1[2], inv_d0[2], inv_d1[2], K2, wPrior=mEst_W3, vis_res=self.vis_res,
                                      obj_mask0=obj_mask0_pyr[2], obj_mask1=obj_mask1_pyr[2])
        elif self.track_type == 'ICP':
            output2 = self.tr_update2(pose3, dpt0_pyr[2], dpt1_pyr[2], K2, wPrior=mEst_W3, vis_res=self.vis_res,
                                      obj_mask1=obj_mask1_pyr[2])
        else:
            raise NotImplementedError()
        pose2, mEst_W2 = output2[0], output2[1]
        poses_to_train[0].append(pose2[0])
        poses_to_train[1].append(pose2[1])
        if self.uncer_prop:
            sigma_ksi.append(output2[2])
        # trust region update on level 1
        K1 = K >> 1
        if self.track_type == 'U_FC':
            output1 = self.tr_update1(pose2, x0[1], x1[1], dpt0_pyr[1], dpt1_pyr[1], K1,
                                      sigma0=sigma0[1], sigma1=sigma1[1], wPrior=mEst_W2, vis_res=self.vis_res,
                                      obj_mask1=obj_mask1_pyr[1])
        elif self.track_type == 'U_IC':
            output1 = self.tr_update1(pose2, x0[1], x1[1], inv_d0[1], inv_d1[1], K1,
                                      sigma0=sigma0[1], sigma1=sigma1[1], wPrior=mEst_W2, vis_res=self.vis_res,
                                      depth0=dpt0_pyr[1], depth1=dpt1_pyr[1],
                                      obj_mask0=obj_mask0_pyr[1], obj_mask1=obj_mask1_pyr[1])
        elif self.track_type == 'IC':
            output1 = self.tr_update1(pose2, x0[1], x1[1], inv_d0[1], inv_d1[1], K1,
                                      wPrior=mEst_W2, vis_res=self.vis_res,
                                      obj_mask0=obj_mask0_pyr[1], obj_mask1=obj_mask1_pyr[1])
        elif self.track_type == 'ICP':
            output1 = self.tr_update1(pose2, dpt0_pyr[1], dpt1_pyr[1], K1, wPrior=mEst_W2, vis_res=self.vis_res,
                                      obj_mask1=obj_mask1_pyr[1])
        else:
            raise NotImplementedError()
        pose1, mEst_W1 = output1[0], output1[1]
        poses_to_train[0].append(pose1[0])
        poses_to_train[1].append(pose1[1])
        if self.uncer_prop:
            sigma_ksi.append(output1[2])
        # trust-region update on the raw scale
        if self.track_type == 'U_FC':
            output0 = self.tr_update0(pose1, x0[0], x1[0], dpt0_pyr[0], dpt1_pyr[0], K,
                                      sigma0=sigma0[0], sigma1=sigma1[0], wPrior=mEst_W1, vis_res=self.vis_res,
                                      obj_mask1=obj_mask1_pyr[0])
        elif self.track_type == 'U_IC':
            output0 = self.tr_update0(pose1, x0[0], x1[0], inv_d0[0], inv_d1[0], K,
                                      sigma0=sigma0[0], sigma1=sigma1[0], wPrior=mEst_W1, vis_res=self.vis_res,
                                      depth0=dpt0_pyr[0], depth1=dpt1_pyr[0],
                                      obj_mask0=obj_mask0_pyr[0], obj_mask1=obj_mask1_pyr[0])
        elif self.track_type == 'IC':
            output0 = self.tr_update0(pose1, x0[0], x1[0], inv_d0[0], inv_d1[0], K,
                                      wPrior=mEst_W1, vis_res=self.vis_res,
                                      obj_mask0=obj_mask0_pyr[0], obj_mask1=obj_mask1_pyr[0])
        elif self.track_type == 'ICP':
            output0 = self.tr_update0(pose1, dpt0_pyr[0], dpt1_pyr[0], K, wPrior=mEst_W1, vis_res=self.vis_res,
                                      obj_mask1=obj_mask1_pyr[0])
        else:
            raise NotImplementedError()
        pose0 = output0[0]
        poses_to_train[0].append(pose0[0])
        poses_to_train[1].append(pose0[1])
        if self.uncer_prop:
            sigma_ksi.append(output0[2])
        if self.timers: self.timers.toc('trust-region update')

        with torch.no_grad():
            if logger is not None or vis or self.vis_feat_uncer:
                feat_0 = display.visualize_feature_channels(x0[0], rgb=I0, order='CHW')
                feat_1 = display.visualize_feature_channels(x1[0], rgb=I1, order='CHW')
                if self.uncertainty_type != 'None':
                    uncertainty_0 = display.visualize_feature_channels(sigma0[0], rgb=I0, order='CHW')
                    uncertainty_1 = display.visualize_feature_channels(sigma1[0], rgb=I1, order='CHW')
                if self.mEst_type != 'None':
                    robust_w = display.visualize_feature_channels(output0[1], rgb=I0, order='CHW')
                if self.combine_icp:
                    scale_w = display.visualize_feature_channels(output0[1], rgb=I0, order='CHW')
            if vis or self.vis_feat_uncer:
                cv2.namedWindow('feature: 0', cv2.WINDOW_NORMAL)
                cv2.imshow('feature: 0', feat_0)
                cv2.namedWindow('feature: 1', cv2.WINDOW_NORMAL)
                cv2.imshow('feature: 1', feat_1)
                if self.uncertainty_type != 'None':
                    cv2.namedWindow('uncertainty: 0', cv2.WINDOW_NORMAL)
                    cv2.imshow('uncertainty: 0', uncertainty_0)
                    cv2.namedWindow('uncertainty: 1', cv2.WINDOW_NORMAL)
                    cv2.imshow('uncertainty: 1', uncertainty_1)
                if self.mEst_type != 'None':
                    cv2.namedWindow('weights from robust network', cv2.WINDOW_NORMAL)
                    cv2.imshow('weights from robust network', robust_w)
                if self.combine_icp:
                    cv2.namedWindow('icp scales from scaler network', cv2.WINDOW_NORMAL)
                    cv2.imshow('icp scales from scaler network', scale_w)

                if not self.training and index is not None:
                    f_C=-1  # which channel to output, -1 show all channels
                    level=-1  # which level to save, -1 show all levels

                    if level<0:
                        levels = range(len(x0))
                    else:
                        levels = [level, ]

                    for i in levels:
                        img_index_png = str(index).zfill(5) + '_l_' + str(i) + '_c_' + str(f_C) + '.png'
                        if f_C < 0:
                            feat_0_save = display.visualize_feature_channels(x0[i], order='CHW', add_ftr_avg=False)
                            feat_1_save = display.visualize_feature_channels(x1[i], order='CHW', add_ftr_avg=False)
                        else:
                            feat_0_save = display.visualize_feature_channels(x0[i][0:1,f_C:f_C+1,], order='CHW',
                                                                             add_ftr_avg=False)
                            feat_1_save = display.visualize_feature_channels(x1[i][0:1,f_C:f_C+1,], order='CHW',
                                                                             add_ftr_avg=False)

                        output_folder = self.checkpoint.replace('.pth.tar', '')
                        feat0_folder = osp.join(output_folder, 'feat0')
                        feat1_folder = osp.join(output_folder, 'feat1')
                        feat0_img = osp.join(feat0_folder, img_index_png)
                        feat1_img = osp.join(feat1_folder, img_index_png)
                        check_directory(feat0_img)
                        check_directory(feat1_img)
                        cv2.imwrite(feat0_img, feat_0_save)
                        cv2.imwrite(feat1_img, feat_1_save)
                        if self.uncertainty_type != 'None':
                            if f_C < 0:
                                uncertainty_0_save = display.visualize_feature_channels(sigma0[i], order='CHW',
                                                                                        add_ftr_avg=False)
                                uncertainty_1_save = display.visualize_feature_channels(sigma1[i], order='CHW',
                                                                                        add_ftr_avg=False)
                            else:
                                uncertainty_0_save = display.visualize_feature_channels(sigma0[i][0:1,f_C:f_C+1,],
                                                                                        order='CHW', add_ftr_avg=False)
                                uncertainty_1_save = display.visualize_feature_channels(sigma1[i][0:1,f_C:f_C+1,],
                                                                                        order='CHW', add_ftr_avg=False)

                            unc0_folder = osp.join(output_folder, 'uncertainty0')
                            unc1_folder = osp.join(output_folder, 'uncertainty1')
                            unc0_img = osp.join(unc0_folder, img_index_png)
                            unc1_img = osp.join(unc1_folder, img_index_png)
                            check_directory(unc0_img)
                            check_directory(unc1_img)
                            cv2.imwrite(unc0_img, uncertainty_0_save)
                            cv2.imwrite(unc1_img, uncertainty_1_save)

                    if self.mEst_type != 'None':
                        robust_w = display.visualize_feature_channels(output0[1], add_ftr_avg=False, order='CHW')
                        mest_folder = osp.join(output_folder, 'mestimator')
                        mest_img = osp.join(mest_folder, img_index_png)
                        check_directory(mest_img)
                        cv2.imwrite(mest_img, robust_w)
                cv2.waitKey(1)

            if logger is not None:
                if self.training:
                    logger.add_images_to_tensorboard([feat_0, feat_1],
                                                     'trained_feature_maps',
                                                     iteration)
                    if self.uncertainty_type != 'None':
                        logger.add_images_to_tensorboard([uncertainty_0, uncertainty_1],
                                                         'trained_uncertainty_maps',
                                                         iteration)
                    if self.mEst_type != 'None':
                        logger.add_images_to_tensorboard([robust_w, ],
                                                         'trained_robust_cost_function_weights',
                                                         iteration)
                    if self.combine_icp:
                        logger.add_images_to_tensorboard([scale_w, ],
                                                         'trained_icp_scales',
                                                         iteration)
                else:
                    logger.add_images_to_tensorboard([feat_0, feat_1],
                                                     'eval_feature_maps',
                                                     iteration)
                    if self.uncertainty_type != 'None':
                        logger.add_images_to_tensorboard([uncertainty_0, uncertainty_1],
                                                         'eval_uncertainty_maps',
                                                         iteration)
                    if self.mEst_type != 'None':
                        logger.add_images_to_tensorboard([robust_w, ],
                                                         'eval_robust_cost_function_weights',
                                                         iteration)
                    if self.combine_icp:
                        logger.add_images_to_tensorboard([scale_w, ],
                                                         'eval_icp_scales',
                                                         iteration)

        if self.training:
            pyr_R = torch.stack(tuple(poses_to_train[0]), dim=1)
            pyr_t = torch.stack(tuple(poses_to_train[1]), dim=1) 
            if self.uncer_prop:
                sigma_ksi = torch.stack(sigma_ksi, dim=1)               
                return pyr_R, pyr_t, sigma_ksi
            else:
                return pyr_R, pyr_t
        else:
            return pose0

    def __encode_features(self, img0, invD0, img1, invD1):
        """ get the encoded features
        """
        if self.encoder_type == self.RGB:
            # In the RGB case, we will only use the intensity image
            I = self.__color3to1(img0)
            x = self.construct_image_pyramids(I)
            sigma = [torch.ones_like(a) for a in x]
            origin_x = x
        elif self.encoder_type == self.CONV_RGBD:
            m = torch.cat((img0, invD0), dim=1)
            x, sigma, origin_x = self.encoder.forward(m)
        elif self.encoder_type in [self.CONV_RGBD2]:
            m = torch.cat((img0, invD0, img1, invD1), dim=1)
            x, sigma, origin_x = self.encoder.forward(m)
        else:
            raise NotImplementedError()

        return x, sigma, origin_x

    def __color3to1(self, img):
        """ Return a gray-scale image
        """
        B, _, H, W = img.shape
        return (img[:,0] * 0.299 + img[:, 1] * 0.587 + img[:, 2] * 0.114).view(B,1,H,W)

    def compute_residual(self, img0, img1, depth0, depth1, K, poseI=None, level_i=0,
                         logger=None, vis=False, obj_mask0=None, obj_mask1=None):
        """
        Forward computing of the residuals: default on the level_i (final level)
        :input
        :param the reference image
        :param the target image
        :param the inverse depth of the reference image
        :param the inverse depth of the target image
        :param the pin-hole camera instrinsic (in vector) [fx, fy, cx, cy]
        :param the initial pose [Rotation, translation]
        --------
        :return
        :param estimated transform
        """
        preprocessed_data = self._preprocess(img0, img1, depth0, depth1, poseI=poseI, obj_mask0=obj_mask0, obj_mask1=obj_mask1)
        (I0, I1, x0, x1, sigma0, sigma1, dpt0_pyr, dpt1_pyr, inv_d0, inv_d1,
         obj_mask0_pyr, obj_mask1_pyr, poseI) = preprocessed_data

        i = level_i
        Ki = K >> i
        # the prior of the mask
        # @TODO: update scaling with the final scaling method
        prior_W = torch.ones(inv_d0[i].shape).type_as(inv_d0[i]) * 0.01
        if self.timers: self.timers.tic('trust-region update')
        # residual evaluation on the level_i of the feature pyramid
        if self.track_type == 'U_FC':
            avg_res = self.tr_update0.forward_residuals(poseI, x0[i], x1[i], dpt0_pyr[i], dpt1_pyr[i], Ki,
                                      sigma0=sigma0[i], sigma1=sigma1[i], wPrior=prior_W, vis_res=self.vis_res,
                                      obj_mask1=obj_mask1_pyr[i])
        elif self.track_type == 'U_IC':
            avg_res = self.tr_update0.forward_residuals(poseI, x0[i], x1[i], inv_d0[i], inv_d1[i], Ki,
                                      sigma0=sigma0[i], sigma1=sigma1[i], wPrior=prior_W, vis_res=self.vis_res,
                                      depth0=dpt0_pyr[i], depth1=dpt1_pyr[i],
                                      obj_mask0=obj_mask0_pyr[i], obj_mask1=obj_mask1_pyr[i])
        elif self.track_type == 'IC':
            prior_W = torch.ones(inv_d0[i].shape).type_as(inv_d0[i])
            avg_res = self.tr_update0.forward_residuals(poseI, x0[i], x1[i], inv_d0[i], inv_d1[i], Ki,
                                      wPrior=prior_W, vis_res=self.vis_res,
                                      obj_mask0=obj_mask0_pyr[i], obj_mask1=obj_mask1_pyr[i])
        elif self.track_type == 'ICP':
            avg_res = self.tr_update0.forward_residuals(poseI, dpt0_pyr[i], dpt1_pyr[i], Ki, wPrior=prior_W, vis_res=self.vis_res,
                                                        obj_mask1=obj_mask1_pyr[i])
        else:
            raise NotImplementedError()

        return poseI, avg_res

    def _preprocess(self, img0, img1, depth0, depth1, poseI=None, obj_mask0=None, obj_mask1=None):
        if self.timers: self.timers.tic('extract features')
        # pre-processing all the data, all the invalid inputs depth are set to 0
        invD0 = torch.clamp(1.0 / depth0, 0, 10)
        invD1 = torch.clamp(1.0 / depth1, 0, 10)
        invD0[invD0 == invD0.min()] = 0
        invD1[invD1 == invD1.min()] = 0
        invD0[invD0 == invD0.max()] = 0
        invD1[invD1 == invD1.max()] = 0

        I0 = color_normalize(img0)
        I1 = color_normalize(img1)

        x0, sigma0, orig_x0 = self.__encode_features(I0, invD0, I1, invD1)
        x1, sigma1, orig_x1 = self.__encode_features(I1, invD1, I0, invD0)
        inv_d0 = self.construct_depth_pyramids(invD0)
        inv_d1 = self.construct_depth_pyramids(invD1)

        if self.track_type in ['U_FC', 'ICP'] or self.combine_icp:
            # don't use truncated depth -> affect icp tracking
            dpt0_pyr = self.construct_depth_pyramids(depth0)
            dpt1_pyr = self.construct_depth_pyramids(depth1)
        else:
            dpt0_pyr = [None] * len(self.scales)
            dpt1_pyr = [None] * len(self.scales)
        if obj_mask0 is not None:
            obj_mask0_pyr = self.construct_image_pyramids(obj_mask0)
        else:
            obj_mask0_pyr = [None] * len(self.scales)
        if obj_mask1 is not None:
            obj_mask1_pyr = self.construct_image_pyramids(obj_mask1)
        else:
            obj_mask1_pyr = [None] * len(self.scales)
        if self.timers: self.timers.toc('extract features')

        # init pose
        if poseI is None:
            if self.predict_init_pose:
                R0, t0 = self.pose_predictor(orig_x0[3], orig_x1[3])  # use full resolution
            else:
                B = invD0.shape[0]
                R0 = torch.eye(3,dtype=torch.float).expand(B,3,3).type_as(I0)
                t0 = torch.zeros(B,3,1,dtype=torch.float).type_as(I0)
            poseI = [R0, t0]

        return (I0, I1, x0, x1, sigma0, sigma1, dpt0_pyr, dpt1_pyr, inv_d0, inv_d1,
                obj_mask0_pyr, obj_mask1_pyr, poseI)