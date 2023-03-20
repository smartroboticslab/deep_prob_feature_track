#!/bin/bash

python code/train.py \
--dataset TUM_RGBD \
--encoder_name ConvRGBD2 \
--mestimator None \
--solver Direct-Nodamping \
--keyframes 1,2,4,8 \
--cpu_workers 0 \
--batch_per_gpu 64 \
--feature_channel 8 \
--feature_extract conv \
--uncertainty laplacian \
--remove_tru_sigma \
--uncertainty_channel 1 \
--init_pose sfm_net \
--train_init_pose \
--multi_hypo prob_fuse \
--combine_ICP \
--scaler None \
--resume_training \
--checkpoint $PROJECT_DIR$/../logs/TUM_RGBD/checkpoint_epoch29.pth.tar \
--prefix combine_ICP
