#!/bin/bash

python code/experiments/kf_vo.py \
--dataset VaryLighting \
--encoder_name ConvRGBD2 \
--mestimator None \
--solver Direct-Nodamping \
--cpu_workers 12 \
--batch_per_gpu 96 \
--feature_channel 8 \
--feature_extract conv \
--uncertainty laplacian \
--uncertainty_channel 1 \
--direction inverse \
--init_pose sfm_net \
--train_init_pose \
--multi_hypo prob_fuse \
--remove_tru_sigma \
--checkpoint $PROJECT_DIR/checkpoint/checkpoint_epoch29.pth.tar \
--vo feature

# keyframe tracking visualization
# --vo_type keyframe --two_view