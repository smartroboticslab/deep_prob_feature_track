""" The dataloader for custom dataset
# SPDX-FileCopyrightText: 2021 Binbin Xu
# SPDX-License-Identifier: BSD-3-Clause
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys, os, random

import os.path as osp
import torch.utils.data as data

from imageio import imread
from cv2 import resize, INTER_NEAREST

import os.path
import glob

# import third party
import numpy as np
import random


def get_depth_from_corresponding_rgb(rgb_file_abs):
    rgb_dir, rgb_file_rel = os.path.split(rgb_file_abs)
    depth_dir = rgb_dir.replace("rgb", "depth")
    depth_file = os.path.join(depth_dir, rgb_file_rel)
    return depth_file


class VaryLighting(data.Dataset):

    """
    Dataset class for our varying lighting dataset
    """
    base = 'data'
    IMAGE_HEIGHT = 480
    IMAGE_WIDTH = 640
    DEPTH_SCALE = 1.0/1000.0
    K = [525.0, 525.0, 319.5, 239.5]

    def __init__(self, root='', category='keyframe',
                 keyframes=[1,], data_transform=None, select_traj=None,
                 image_resize=0.25, truncate_depth=True, pair='incremental',
                 ):

        super(VaryLighting, self).__init__()
        assert pair in ['incremental', 'keyframe']
        self.pair = pair

        self.image_seq = []  # list (seq) of list (frame) of string (rgb image path)
        self.timestamp = []  # empty
        self.depth_seq = []  # list (seq) of list (frame) of string (depth image path)
        self.invalid_seq = []  # empty
        self.cam_pose_seq = []  # list (seq) of list (frame) of 4 X 4 ndarray
        self.calib = []  # list (seq) of list (intrinsics: fx, fy, cx, cy)
        self.seq_names = []  # list (seq) of string (seq name)

        self.ids = 0
        self.seq_acc_ids = [0]
        self.keyframes = keyframes

        self.transforms = data_transform

        if category == 'test':
            # self.set_test_mode()
            self.__load_test(root, select_traj)
        elif category in ['train', 'validation']:  # train and validation
            raise NotImplementedError()
        elif category == 'kf':
            self.__load_kf(root, select_traj)
        else:
            raise NotImplementedError

        # downscale the input image to a quarter
        self.fx_s = image_resize
        self.fy_s = image_resize
        self.truncate_depth = truncate_depth

        print('Vary Lighting dataloader for {:} using keyframe {:}: \
                      {:} valid frames'.format(category, keyframes, self.ids))


    def __load_test(self, root, select_traj=None):
        """ Note:
        The test trajectory is loaded slightly different from the train/validation trajectory.
        We only select keyframes from the entire trajectory, rather than use every individual frame.
        For a given trajectory of length N, using key-frame 2, the train/validation set will use
        [[1, 3], [2, 4], [3, 5],...[N-1, N]],
        while test set will use pair
        [[1, 3], [3, 5], [5, 7],...[N-1, N]]
        This difference result in a change in the trajectory length when using different keyframes.

        The benefit of sampling keyframes of the test set is that the output is a more reasonable trajectory;
        And in training/validation, we fully leverage every pair of image.
        """

        assert(len(self.keyframes) == 1)
        kf = self.keyframes[0]
        self.keyframes = [1]
        track_scene = osp.join(root, "*/")
        scene_lists = glob.glob(track_scene, recursive=True)

        self._num_scenes = len(scene_lists)
        if self._num_scenes is None:
            raise ValueError("No sub-folder data in the training or validation dataset")
        for scene in scene_lists:
            scene_name = osp.basename(osp.dirname(scene))
            if select_traj is not None:
                if scene_name != select_traj: continue

            rgb_images_regex = os.path.join(scene, "rgb/*.png")
            all_rgb_images_in_scene = sorted(glob.glob(rgb_images_regex))
            total_num = len(all_rgb_images_in_scene)

            self.calib.append(self.K)


            images = [all_rgb_images_in_scene[idx] for idx in range(0, total_num, kf)]
            # fake timestamps
            timestamp = [os.path.splitext(os.path.basename(image))[0] for image in images]
            depths = [get_depth_from_corresponding_rgb(rgb_file) for rgb_file in images]
            extrin = [None] * len(images)  # [tq2mat(frames[idx][0]) for idx in range(0, total_num, kf)]
            self.image_seq.append(images)
            self.timestamp.append(timestamp)
            self.depth_seq.append(depths)
            self.cam_pose_seq.append(extrin)
            self.seq_names.append(scene)
            self.ids += max(0, len(images)-1)
            self.seq_acc_ids.append(self.ids)

    def __load_rgb_tensor(self, path):
        """ Load the rgb image
        """
        image = imread(path)[:, :, :3]
        image = image.astype(np.float32) / 255.0
        image = resize(image, None, fx=self.fx_s, fy=self.fy_s)
        return image

    def __load_depth_tensor(self, path):
        """ Load the depth:
            The depth images are scaled by a factor of 5000, i.e., a pixel
            value of 5000 in the depth image corresponds to a distance of
            1 meter from the camera, 10000 to 2 meter distance, etc.
            A pixel value of 0 means missing value/no data.
        """
        depth = imread(path).astype(np.float32) / 5e3
        depth = resize(depth, None, fx=self.fx_s, fy=self.fy_s, interpolation=INTER_NEAREST)
        if self.truncate_depth:
            depth = np.clip(depth, a_min=0.5, a_max=5.0) # the accurate range of kinect depth
        return depth[np.newaxis, :]

    def __getitem__(self, index):
        # pair in the way like [[1, 3], [3, 5], [5, 7],...[N-1, N]]
        seq_idx = max(np.searchsorted(self.seq_acc_ids, index + 1) - 1, 0)
        frame_idx = index - self.seq_acc_ids[seq_idx]

        this_idx = frame_idx
        next_idx = frame_idx + random.choice(self.keyframes)

        color0 = self.__load_rgb_tensor(self.image_seq[seq_idx][this_idx])
        color1 = self.__load_rgb_tensor(self.image_seq[seq_idx][next_idx])

        depth0 = self.__load_depth_tensor(self.depth_seq[seq_idx][this_idx])
        depth1 = self.__load_depth_tensor(self.depth_seq[seq_idx][next_idx])

        if self.transforms:
            color0, color1 = self.transforms([color0, color1])

            # normalize the coordinate
        calib = np.asarray(self.calib[seq_idx], dtype=np.float32)
        calib[0] *= self.fx_s
        calib[1] *= self.fy_s
        calib[2] *= self.fx_s
        calib[3] *= self.fy_s

        # cam_pose0 = self.cam_pose_seq[seq_idx][this_idx]
        # cam_pose1 = self.cam_pose_seq[seq_idx][next_idx]
        # transform = np.dot(np.linalg.inv(cam_pose1), cam_pose0).astype(np.float32)
        transform = None

        name = {'seq': self.seq_names[seq_idx],
                'frame0': this_idx,
                'frame1': next_idx}

        # camera_info = dict()
        camera_info = {"height": color0.shape[0],
                       "width": color0.shape[1],
                       "fx": calib[0],
                       "fy": calib[1],
                       "ux": calib[2],
                       "uy": calib[3]}
        return color0, color1, depth0, depth1, transform, calib, name, camera_info


    def get_keypair(self, index, kf_idx=0):
        # pair in the way like [[1, 3], [1, 5], [1, 7],...[1, N]]
        seq_idx = max(np.searchsorted(self.seq_acc_ids, index + 1) - 1, 0)
        frame_idx = index - self.seq_acc_ids[seq_idx]

        this_idx = kf_idx
        next_idx = frame_idx

        color0 = self.__load_rgb_tensor(self.image_seq[seq_idx][this_idx])
        color1 = self.__load_rgb_tensor(self.image_seq[seq_idx][next_idx])

        depth0 = self.__load_depth_tensor(self.depth_seq[seq_idx][this_idx])
        depth1 = self.__load_depth_tensor(self.depth_seq[seq_idx][next_idx])

        if self.transforms:
            color0, color1 = self.transforms([color0, color1])

            # normalize the coordinate
        calib = np.asarray(self.calib[seq_idx], dtype=np.float32)
        calib[0] *= self.fx_s
        calib[1] *= self.fy_s
        calib[2] *= self.fx_s
        calib[3] *= self.fy_s

        # cam_pose0 = self.cam_pose_seq[seq_idx][this_idx]
        # cam_pose1 = self.cam_pose_seq[seq_idx][next_idx]
        # transform = np.dot(np.linalg.inv(cam_pose1), cam_pose0).astype(np.float32)
        transform = None

        name = {'seq': self.seq_names[seq_idx],
                'frame0': this_idx,
                'frame1': next_idx}

        # camera_info = dict()
        camera_info = {"height": color0.shape[0],
                       "width": color0.shape[1],
                       "fx": calib[0],
                       "fy": calib[1],
                       "ux": calib[2],
                       "uy": calib[3]}
        return color0, color1, depth0, depth1, transform, calib, name, camera_info

    def __len__(self):
        return self.ids