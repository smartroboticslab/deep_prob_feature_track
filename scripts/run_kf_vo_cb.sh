#! /bin/bash

python code/convergence_basin.py \
--dataset TUM_RGBD \
--encoder_name ConvRGBD2 \
--mestimator None \
--solver Direct-Nodamping \
--cpu_workers 12 \
--batch_per_gpu 64 \
--feature_channel 8 \
--feature_extract conv \
--uncertainty laplacian \
--uncertainty_channel 1 \
--direction inverse \
--init_pose sfm_net \
--train_init_pose \
--multi_hypo prob_fuse \
--remove_tru_sigma \
--checkpoint $PROJECT_DIR$/checkpoint_epoch29.pth.tar \
--eval_set validation \
--trajectory structure_texture \
--keyframes 8 \
--cb_dimension 2D \
--save_img \
--reset_cb

# --combine_ICP \
# --checkpoint $PROJECT_DIR$/../logs/combine_ICP/checkpoint_epoch40.pth.tar \
