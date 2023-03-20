""" 
A wrapper to select different methods for comparison
# SPDX-FileCopyrightText: 2021 Binbin Xu
# SPDX-License-Identifier: BSD-3-Clause
"""

import torch
from models.LeastSquareTracking import LeastSquareTracking

# load comparison methods
def select_method(method_name, options):
    assert method_name in ['DeepIC', 'RGB', 'ICP', 'RGB+ICP', 'feature', 'feature_icp']
    if method_name == 'DeepIC':
        print('==>Load DeepIC method')
        if options.dataset == 'MovingObjects3D':
            deepIC_checkpoint = '/media/binbin/code/SLAM/DeeperInverseCompositionalAlgorithm/logs/MovingObjects3D/124/cvpr124_ConvRGBD2_MultiScale2w_Direct-ResVol_MovingObjects3D_obj_False_uCh_1_None_rmT_False_fCh_1_average_iP_identity_mH_None_wICP_False_s_None_lr_0.0005_batch_64/checkpoint_epoch29.pth.tar'
        else:
            deepIC_checkpoint = '/media/binbin/code/SLAM/DeeperInverseCompositionalAlgorithm/code/trained_models/TUM_RGBD_ABC_final.pth.tar'
        deepIC = LeastSquareTracking(
            encoder_name='ConvRGBD2',
            direction='inverse',
            max_iter_per_pyr=3,
            mEst_type='MultiScale2w',
            # options=options,
            solver_type='Direct-ResVol',
            feature_channel=1,
            feature_extract='average',
            uncertainty_type='None',
            combine_ICP=False,
            scaler='None',
            init_pose_type='identity',
            options=options,
        )

        if torch.cuda.is_available(): deepIC.cuda()

        # check whether it is a single checkpoint or a directory
        deepIC.load_state_dict(torch.load(deepIC_checkpoint)['state_dict'])
        deepIC.eval()
        return deepIC

    if method_name == 'RGB':
        print('==>Load RGB method')
        rgb_tracker = LeastSquareTracking(
            encoder_name='RGB',
            combine_ICP=False,
            feature_channel=1,
            uncertainty_channel=1,
            # feature_extract='conv',
            uncertainty_type='None',
            scaler='None',
            direction='inverse',
            max_iter_per_pyr=options.max_iter_per_pyr,
            mEst_type='None',
            solver_type='Direct-Nodamping',
            init_pose_type='identity',
            options=options,
        )
        if torch.cuda.is_available(): rgb_tracker.cuda()
        rgb_tracker.eval()
        return rgb_tracker

    if method_name == 'ICP':
        print('==>Load ICP method')
        icp_tracker = LeastSquareTracking(
            encoder_name='ICP',
            combine_ICP=False,
            # feature_channel=1,
            # uncertainty_channel=1,
            # feature_extract='conv',
            uncertainty_type='ICP',
            scaler='None',
            direction='inverse',
            max_iter_per_pyr=options.max_iter_per_pyr,
            mEst_type='None',
            solver_type='Direct-Nodamping',
            init_pose_type='identity',
            options=options,
        )
        if torch.cuda.is_available(): icp_tracker.cuda()
        icp_tracker.eval()
        return icp_tracker

    if method_name == 'RGB+ICP':
        print('==>Load RGB+ICP method')
        rgbd_tracker = LeastSquareTracking(
            encoder_name='RGB',
            combine_ICP=True,
            # feature_channel=1,
            uncertainty_channel=1,
            # feature_extract='conv',
            uncertainty_type='identity',
            scaler='None',
            direction='inverse',
            max_iter_per_pyr=options.max_iter_per_pyr,
            mEst_type='None',
            solver_type='Direct-Nodamping',
            init_pose_type='identity',
            remove_tru_sigma=False,
            scale_scaler=0.2,
            options=options,
        )
        if torch.cuda.is_available(): rgbd_tracker.cuda()
        rgbd_tracker.eval()
        return rgbd_tracker

    if method_name == 'feature':
        # train_utils.load_checkpoint_test(options)
        #
        # net = ICtracking.LeastSquareTracking(
        #     encoder_name=options.encoder_name,
        #     uncertainty_type=options.uncertainty,
        #     direction=options.direction,
        #     max_iter_per_pyr=options.max_iter_per_pyr,
        #     mEst_type=options.mestimator,
        #     options=options,
        #     solver_type=options.solver,
        #     no_weight_sharing=options.no_weight_sharing)
        #
        # if torch.cuda.is_available(): net.cuda()
        # net.eval()
        #
        # # check whether it is a single checkpoint or a directory
        # net.load_state_dict(torch.load(options.checkpoint)['state_dict'])
        # method_list['trained_method'] = net
        # train_utils.load_checkpoint_test(options)
        print('==>Load our feature-metric method')
        net = LeastSquareTracking(
            encoder_name=options.encoder_name,
            uncertainty_type=options.uncertainty,
            direction=options.direction,
            max_iter_per_pyr=options.max_iter_per_pyr,
            mEst_type=options.mestimator,
            options=options,
            solver_type=options.solver,
            combine_ICP=False,
            no_weight_sharing=options.no_weight_sharing)

        if torch.cuda.is_available(): net.cuda()

        # check whether it is a single checkpoint or a directory
        if options.checkpoint == '':
            if options.dataset in ['TUM_RGBD', 'VaryLighting']:
                checkpoint = '/media/binbin/code/SLAM/DeeperInverseCompositionalAlgorithm/logs/TUM_RGBD/check_change/check4_ConvRGBD2_None_Direct-Nodamping_dataset_TUM_RGBD_obj_False_laplacian_uncerCh_1_featCh_8_conv_initPose_sfm_net_multiHypo_prob_fuse_uncer_prop_False_lr_0.0005_batch_64/checkpoint_epoch29.pth.tar'
            else:
                raise NotImplementedError()
            net.load_state_dict(torch.load(checkpoint)['state_dict'])
        else:
            net.load_state_dict(torch.load(options.checkpoint)['state_dict'])
        net.eval()
        return net

    if method_name == 'feature_icp':
        print('==>Load our feature-metric+ICP method')
        feature_icp = LeastSquareTracking(
            encoder_name=options.encoder_name,
            uncertainty_type=options.uncertainty,
            direction=options.direction,
            max_iter_per_pyr=options.max_iter_per_pyr,
            mEst_type=options.mestimator,
            options=options,
            solver_type=options.solver,
            combine_ICP=True,
            scale_scaler=options.scale_icp,
            no_weight_sharing=options.no_weight_sharing)

        if torch.cuda.is_available(): feature_icp.cuda()

        # check whether it is a single checkpoint or a directory
        if options.checkpoint == '':
            if options.dataset in ['TUM_RGBD', 'VaryLighting']:
                checkpoint = '/media/binbin/code/SLAM/DeeperInverseCompositionalAlgorithm/logs/TUM_RGBD/finetune/finetune_vl_icp_ConvRGBD2_None_Direct-Nodamping_TUM_RGBD_obj_False_uCh_1_laplacian_rmT_True_fCh_8_conv_iP_sfm_net_mH_prob_fuse_wICP_True_s_None_lr_0.0005_batch_64/checkpoint_epoch40.pth.tar'
            else:
                raise NotImplementedError()
            feature_icp.load_state_dict(torch.load(checkpoint)['state_dict'])
        else:
            feature_icp.load_state_dict(torch.load(options.checkpoint)['state_dict'])
        feature_icp.eval()
        return feature_icp
    else:
        raise NotImplementedError("unsupported test tracker: check argument of --tracker again")