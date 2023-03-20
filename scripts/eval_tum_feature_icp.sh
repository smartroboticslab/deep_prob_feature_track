#!/bin/bash

python code/evaluate.py \
--encoder_name ConvRGBD2 \
--mestimator None \
--solver Direct-Nodamping \
--dataset TUM_RGBD \
--keyframes 1,2,4,8 \
--cpu_workers 12 \
--batch_per_gpu 96 \
--feature_channel 8 \
--uncertainty_channel 1 \
--feature_extract conv \
--uncertainty laplacian \
--remove_tru_sigma \
--init_pose sfm_net \
--train_init_pose \
--multi_hypo prob_fuse \
--checkpoint combine_icp_checkpoint_epoch40.pth.tar \
--scaler None \
--combine_ICP