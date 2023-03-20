""" 
Argparse configuration for training and testing in the paper
# SPDX-FileCopyrightText: 2021 Binbin Xu
# SPDX-License-Identifier: BSD-3-Clause
"""

def add_network_config(parser):
    parser.add_argument('--feature_channel', default=1, type=int,
        help='Specify the feature channel used for tracking. The default is 1.\n')
    parser.add_argument('--uncertainty_channel', default=1, type=int,
        help='Specify the uncertainty channel used for tracking when using uncerti. The default is 1.\n')
    parser.add_argument('--feature_extract', default='average', type=str,
                        choices=('1by1', 'conv', 'skip', 'average', 'prob_fuse'),
                        help='Specify the method to extract feature from the pyramid. The default is using 1by1.\n')
    parser.add_argument('--uncertainty',
                        default='None', type=str,
                        choices=('None', 'identity', 'sigmoid', 'feature', 'gaussian', 'laplacian', 'old_gaussian', 'old_laplacian'),
                        help='Choose a uncertainty function for feature-metric tracking. DIC is the original CVPR work. '
                             'None using identity matrix. and track is used in GN tracking\n')
    parser.add_argument('--combine_ICP',action='store_true',
                        help='Combine the training with ICP.\n')
    parser.add_argument('--add_init_pose_noise', action='store_true',
                        help='Add noise in the init pose (translation) only in training.\n')
    parser.add_argument('--init_pose', default='identity', type=str,
                        choices=('identity', 'sfm_net', 'dense_net'),
                        help='Use predicted pose as initial pose.\n')
    # @TODO: try different number 0.01 in sfm-learner, 0.1 in deeptam
    parser.add_argument('--scale_init_pose', default='0.01', type=float,
                        help='Scaling the initial predicted pose.\n')
    parser.add_argument('--train_init_pose', action='store_true',
                        help='Jointly train pose predictor by regressing predicted pose to the ground truth.\n')
    parser.add_argument('--multi_hypo',  default='None', type=str,
                        choices=('None', 'average', 'prob_fuse', 'res_prob_fuse'),
                        help='Use multi hypothesis for init pose guess.\n')
    parser.add_argument('--res_input', action='store_true',
                        help='Also input residual for posenet.\n')
    # virtual does not work well
    # parser.add_argument('--virtual_camera', action='store_true',
    #                     help='Use rendered virtual frame and virtual camera instead of img1 \n')
    parser.add_argument('--vis_feat', default=False, action='store_true',
                        help='visualize the feature maps in the training')
    parser.add_argument('--scannet_subset_train', default='0.25', type=float,
                        help='Subset ratio in scannet for training.\n')
    parser.add_argument('--scannet_subset_val', default='0.005', type=float,
                        help='Subset ratio in scannet for validation.\n')
    parser.add_argument('--train_uncer_prop', action='store_true',
                        help='Use uncertainty propagation in the training loss\n')
    parser.add_argument('--obj_only', action='store_true',
                        help='Use uncertainty propagation in the training loss\n')
    parser.add_argument('--loss', default='EPE3D', type=str,
                        choices=('EPE3D', 'RPE', 'UEPE3D', 'URPE'),
                        help='Training loss.\n')
    parser.add_argument('--remove_tru_sigma', action='store_true',
                        help='Remove truncated uncertainty areas in the tracking for training/testing\n')
    parser.add_argument('--scaler', default='None', type=str,
                        choices=('None', 'oneResidual', 'twoResidual', 'MultiScale2w', 'expMultiScale'),
                        help='Choose a scale function for combing ICP and feature methods. \n')
    parser.add_argument('--scale_icp', default='0.01', type=float,
                        help='Scaling the ICP w.r.t feature/RGB.\n')
    parser.add_argument('--add_vl_dataset', action='store_true',
                        help='Add varying lighting dataset to the TUM dataset for training/validation \n')

def add_tracking_config(parser):
    add_network_config(parser)
    parser.add_argument('--network',
        default='DeepIC', type=str,
        choices=('DeepIC', 'GaussNewton'),
        help='Choose a network to run. \n \
        The DeepIC is the proposed Deeper Inverse Compositional method. \n\
        The GuassNewton is the baseline for Inverse Compositional method which does not include \
        any learnable parameters\n')
    parser.add_argument('--mestimator',
        default='MultiScale2w', type=str,
        choices=('None', 'MultiScale2w'),
        help='Choose a weighting function for the Trust Region method.\n\
            The MultiScale2w is the proposed (B) convolutional M-estimator. \n')
    parser.add_argument('--solver',
        default='Direct-ResVol', type=str,
        choices=('Direct-Nodamping', 'Direct-ResVol'),
        help='Choose the solver function for the Trust Region method. \n\
            Direct-Nodamping is the Gauss-Newton algorithm, which does not use damping. \n\
            Direct-ResVol is the proposed (C) Trust-Region Network. \n\
            (default: Direct-ResVol) ')
    parser.add_argument('--direction',
        default='inverse', type=str,
        choices=('inverse', 'forward'),
        help='Choose the direction to update pose: inverse, or forward \n')
    parser.add_argument('--encoder_name',
        default='ConvRGBD2',
        choices=('ConvRGBD2', 'RGB', 'ConvRGBD'),
        help='The encoder architectures. \
            ConvRGBD2 takes the two-view features as input. \n\
            RGB is using the raw RGB images as input (converted to intensity afterwards).\n\
            (default: ConvRGBD2)')
    parser.add_argument('--max_iter_per_pyr',
        default=3, type=int,
        help='The maximum number of iterations at each pyramids.\n')
    parser.add_argument('--no_weight_sharing',
        action='store_true',
        help='If this flag is on, we disable sharing the weights across different backbone network when extracing \
         features. In default, we share the weights for all network in each pyramid level.\n')
    parser.add_argument('--tr_samples', default=10, type=int,
        help='Set the number of trust-region samples. (default: 10)\n')

def add_basics_config(parser):
    """ the basic setting
    (supposed to be shared through train and inference)
    """
    parser.add_argument('--cpu_workers', type=int, default=12,
        help="Number of cpu threads for data loader.\n")
    parser.add_argument('--dataset', type=str,
        choices=('TUM_RGBD', 'ScanNet', 'MovingObjects3D', 'VaryLighting'),
        help='Choose a dataset to train/val/evaluate.\n')
    parser.add_argument('--image_resize', type=float, default=None,
                        help='downsize ratio for input images')
    parser.add_argument('--time', dest='time', action='store_true',
        help='Count the execution time of each step.\n' )

def add_test_basics_config(parser):
    parser.add_argument('--tracker', default='learning_based', type=str,
                        choices=('learning_based', 'ICP', 'ColorICP', 'RGBD'))
    parser.add_argument('--batch_per_gpu', default=8, type=int,
        help='Specify the batch size during test. The default is 8.\n')
    parser.add_argument('--checkpoint', default='', type=str,
        help='Choose a checkpoint model to test.\n')
    parser.add_argument('--keyframes',
        default='1,2,4,8', type=str,
        help='Choose the number of keyframes to train the algorithm.\n')
    parser.add_argument('--verbose', action='store_true',
        help='Print/save all the intermediate representations')
    parser.add_argument('--eval_set', default='test',
        choices=('test', 'validation'))
    parser.add_argument('--trajectory', type=str, 
        default = '',
        help = 'Specify a trajectory to run.\n')

def add_train_basics_config(parser):
    """ add the basics about the training """
    parser.add_argument('--checkpoint', default='', type=str,
        help='Choose a pretrained checkpoint model to start with. \n')
    parser.add_argument('--batch_per_gpu', default=64, type=int,
        help='Specify the batch size during training.\n')
    parser.add_argument('--epochs',
        default=30, type=int,
        help='The total number of total epochs to run. Default is 30.\n' )
    parser.add_argument('--resume_training',
        dest='resume_training', action='store_true',
        help='Resume the training using the loaded checkpoint. If not, restart the training. \n\
            You will need to use the --checkpoint config to load the pretrained checkpoint' )
    parser.add_argument('--pretrained_model', default='', type=str,
        help='Initialize the model weights with pretrained model.\n')
    parser.add_argument('--no_val',
        default=False,
        action='store_true',
        help='Use no validatation set for training.\n')
    parser.add_argument('--keyframes',
        default='1,2,4,8', type=str,
        help='Choose the number of keyframes to train the algorithm')
    parser.add_argument('--verbose', action='store_true',
        help='Print/save all the intermediate representations.\n')

def add_train_log_config(parser):
    """ checkpoint and log options """
    parser.add_argument('--checkpoint_folder', default='', type=str,
        help='The folder name (postfix) to save the checkpoint.')
    parser.add_argument('--snapshot', default=1, type=int,
        help='Number of interations to save a snapshot')
    parser.add_argument('--save_checkpoint_freq',
        default=1, type=int,
        help='save the checkpoint for every N epochs')
    parser.add_argument('--prefix', default='', type=str,
        help='the prefix string added to the log files')
    parser.add_argument('-p', '--print_freq',
        default=10, type=int,
        help='print frequency (default: 10)')


def add_train_optim_config(parser):
    """ add training optimization options """
    parser.add_argument('--opt',
        type=str, default='adam', choices=('sgd','adam'),
        help='choice of optimizer (default: adam) \n')
    parser.add_argument('--lr',
        default=0.0005, type=float,
        help='initial learning rate. \n')
    parser.add_argument('--lr_decay_ratio',
        default=0.5, type=float,
        help='lr decay ratio (default:0.5)')
    parser.add_argument('--lr_decay_epochs',
        default=[5, 10, 20], type=int, nargs='+',
        help='lr decay epochs')
    parser.add_argument('--lr_min', default=1e-6, type=float,
        help='minimum learning rate')
    parser.add_argument('--lr_restart', default=10, type=int,
        help='restart learning after N epochs')

def add_train_loss_config(parser):
    """ add training configuration for the loss function """
    parser.add_argument('--regression_loss_type',
        default='SmoothL1', type=str, choices=('L1', 'SmoothL1'),
        help='Loss function for flow regression (default: SmoothL1 loss)')

def add_vo_config(parser):
    """ add testing configuration for kf-vo demo """
    parser.add_argument('--vo',  default='feature_icp', type=str,
                        choices=('DeepIC', 'RGB', 'ICP', 'RGB+ICP', 'feature', 'feature_icp'),
                        help='Select which tracking method to use for visual odometry.\n')
    parser.add_argument('--vo_type', default='incremental', type=str,
                        choices=('incremental', 'keyframe'),
                        help='Select which reference frame to use for tracking.\n')
    parser.add_argument('--two_view', action='store_true',
        help='Only visualization two views.\n' )
    parser.add_argument('--gt_tracker', action='store_true',
        help='Use ground truth pose for point cloud visualization')
    parser.add_argument('--save_img', action='store_true',
        help='Save visualizations.\n' )

def add_cb_config(parser):
    """ add visualization configurations for convergence basin """
    parser.add_argument('--cb_dimension',  default='2D', type=str,
                        choices=('1D', '2D', '6D'),
                        help='Select which dimension to visualize for convergence basin.\n')
    parser.add_argument('--save_img', action='store_true',
        help='Save visualizations.\n' )
    parser.add_argument('--reset_cb', action='store_true',
        help='Save visualizations.\n' )
    parser.add_argument('--pert_samples', default=31, type=int,
        help='perturbation samples in each pose dimension')

def add_object_config(parser):
    parser.add_argument('--method',  default='feature_icp', type=str,
                        choices=('DeepIC', 'RGB', 'ICP', 'RGB+ICP', 'feature', 'feature_icp'),
                        help='Select which tracking method to use for visual odometry.\n')
    parser.add_argument('--batch_per_gpu', default=64, type=int,
        help='Specify the batch size during test. The default is 8.\n')
    parser.add_argument('--checkpoint', default='', type=str,
        help='Choose a checkpoint model to test.\n')
    parser.add_argument('--keyframes',
        default='1,2,4', type=str,
        help='Choose the number of keyframes to train the algorithm.\n')
    parser.add_argument('--eval_set', default='test',
        choices=('test', 'validation'))
    parser.add_argument('--object', type=str,
        default = '',
        help = 'Specify a trajectory to run.\n')
    parser.add_argument('--save_img', action='store_true',
        help='Save visualizations.\n' )
    parser.add_argument('--gt_pose', action='store_true',
        help='Save visualizations.\n' )
    parser.add_argument('--recompute', action='store_true',
        help='Save visualizations.\n' )