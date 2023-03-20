""" The dataloaders for training and evaluation
# SPDX-FileCopyrightText: 2021 Binbin Xu
# SPDX-License-Identifier: BSD-3-Clause

@author: Zhaoyang Lv
@date: March 2019
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torchvision.transforms as transforms
import numpy as np
import os
import socket
import yaml
try:
    # use faster C loader if available
    from yaml import CLoader
except ImportError:
    from yaml import Loader as CLoader


def get_datasets_path(which_dataset):
    # utils_path = os.getcwd().split('/')
    # print(utils_path)
    # source_folder = '/'.join(utils_path[:-2])
    # print(source_folder)
    # return source_folder
    curr_path = os.path.realpath(__file__)
    env_file_path = os.path.realpath(os.path.join(curr_path, '../../../setup/datasets.yaml'))
    hostname = str(socket.gethostname())
    env_config = yaml.load(open(env_file_path), Loader=CLoader)
    return env_config[which_dataset][hostname]['dataset_root']

TUM_DATASET_DIR = get_datasets_path('TUM_RGBD')
MOVING_OBJECTS_3D = get_datasets_path('MOVING_OBJECTS_3D')
ScanNet_DATASET_DIR = get_datasets_path('SCANNET')
VL_DATASET_DIR = get_datasets_path('VaryLighting')

def load_data(dataset_name, keyframes = None, load_type = 'train',
    select_trajectory = '', load_numpy = False, image_resize=0.25, truncate_depth=True,
    options=None, pair='incremental'):
    """ Use two frame camera pose data loader
    """
    if select_trajectory == '':
        select_trajectory = None

    if not load_numpy:
        if load_type == 'train': 
            data_transform = image_transforms(['color_augment', 'numpy2torch'])
        else:
            data_transform = image_transforms(['numpy2torch'])
    else:
        data_transform = image_transforms([])

    if dataset_name == 'TUM_RGBD':
        from data.TUM_RGBD import TUM
        np_loader = TUM(TUM_DATASET_DIR, load_type, keyframes,
                        data_transform, select_trajectory,
                        image_resize=image_resize,
                        truncate_depth=truncate_depth,
                        add_vl_dataset=options.add_vl_dataset,
                        )
    elif dataset_name == 'ScanNet':
        from data.ScanNet import ScanNet
        np_loader = ScanNet(ScanNet_DATASET_DIR, load_type, keyframes,
                            data_transform, select_trajectory,
                            image_resize=image_resize,
                            truncate_depth=truncate_depth,
                            subset_train=options.scannet_subset_train,
                            subset_val=options.scannet_subset_val,
                            )
    elif dataset_name == 'MovingObjects3D': 
        from data.MovingObj3D import MovingObjects3D
        np_loader = MovingObjects3D(MOVING_OBJECTS_3D, load_type,
                                    keyframes, data_transform,
                                    category=select_trajectory,
                                    image_resize=image_resize,
                                    )
    # elif dataset_name == 'BundleFusion':
    #     from data.BundleFusion import BundleFusion
    #     np_loader = BundleFusion(load_type, keyframes, data_transform)
    # elif dataset_name == 'Refresh':
    #     from data.REFRESH import REFRESH
    #     np_loader = REFRESH(load_type, keyframes)
    elif dataset_name == 'VaryLighting':
        from data.VaryLighting import VaryLighting
        np_loader = VaryLighting(VL_DATASET_DIR, load_type, keyframes,
                                 data_transform, select_trajectory,
                                 pair=pair,
                                 image_resize=image_resize,
                                 truncate_depth=truncate_depth,
                                 )
    else:
        raise NotImplementedError()

    return np_loader

def image_transforms(options):

    transform_list = []

    if 'color_augment' in options: 
        augment_parameters = [0.9, 1.1, 0.9, 1.1, 0.9, 1.1]
        transform_list.append(AugmentImages(augment_parameters))

    if 'numpy2torch' in options:
        transform_list.append(ToTensor())

    # if 'color_normalize' in options: # we do it on the fly
    #     transform_list.append(ColorNormalize())

    return transforms.Compose(transform_list)

class ColorNormalize(object):

    def __init__(self):
        rgb_mean = (0.4914, 0.4822, 0.4465)
        rgb_std = (0.2023, 0.1994, 0.2010)
        self.transform = transforms.Normalize(mean=rgb_mean, std=rgb_std)

    def __call__(self, sample):
        return [self.transform(x) for x in sample]

class ToTensor(object):
    def __init__(self):
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        return [self.transform(x) for x in sample] 

class AugmentImages(object):
    def __init__(self, augment_parameters):
        self.gamma_low  = augment_parameters[0]         # 0.9
        self.gamma_high = augment_parameters[1]         # 1.1
        self.brightness_low  = augment_parameters[2]    # 0.9
        self.brightness_high = augment_parameters[3]    # 1,1
        self.color_low  = augment_parameters[4]         # 0.9
        self.color_high = augment_parameters[5]         # 1.1

        self.thresh = 0.5

    def __call__(self, sample):
        p = np.random.uniform(0, 1, 1)
        if p > self.thresh:
            random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
            random_brightness = np.random.uniform(self.brightness_low, self.brightness_high)
            random_colors = np.random.uniform(self.color_low, self.color_high, 3)
            for x in sample:
                x = x ** random_gamma             # randomly shift gamma
                x = x * random_brightness         # randomly shift brightness
                for i in range(3):                # randomly shift color
                    x[:, :, i] *= random_colors[i]
                    x[:, :, i] *= random_colors[i]
                x = np.clip(x, a_min=0, a_max=1)  # saturate
            return sample
        else:        
            return sample
