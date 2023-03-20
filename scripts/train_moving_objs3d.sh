#! /bin/bash

python train.py --dataset MovingObjs3D \
--keyframes 1,2,4,8 \
--cpu_workers 0 \
--batch_per_gpu 24 \
--encoder_name ConvRGBD2 \ 
--mestimator None \ 
--solver Direct-Nodamping \
--feature_channel 8 \ 
--feature_extract conv \
--uncertainty laplacian \
--uncertainty_channel 1 \
--init_pose sfm_net \
--remove_tru_sigma \
--train_init_pose \
--multi_hypo prob_fuse \
--prefix object \