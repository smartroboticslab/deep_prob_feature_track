#!/bin/bash

python code/train.py \
--encoder_name \
ConvRGBD2 \
--mestimator \
None \
--solver \
Direct-Nodamping \
--dataset \
TUM_RGBD \
--batch_per_gpu \
32 \
--feature_channel \
8 \
--uncertainty_channel \
1 \
--feature_extract \
conv \
--uncertainty \
laplacian \
--remove_tru_sigma \
--init_pose \
sfm_net \
--train_init_pose \
--multi_hypo \
prob_fuse 