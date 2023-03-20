""" The dataloader for ScanNet dataset
# SPDX-FileCopyrightText: 2021 Binbin Xu
# SPDX-License-Identifier: BSD-3-Clause
"""

import os, random
import os.path as osp

import numpy as np
from torch.utils.data import Dataset
from imageio import imread
from tqdm import tqdm
import pickle
from cv2 import resize, INTER_NEAREST


class ScanNet(Dataset):

    def __init__(self, root=None, category='train',
                 keyframes=[1], data_transform=None, select_traj=None,
                 image_resize=0.25, truncate_depth=True,
                 subset_train=0.95, subset_val=0.05):
        assert root is not None
        super(ScanNet, self).__init__()

        self.image_seq = []  # list (seq) of list (frame) of string (rgb image path)
        self.timestamp = []  # empty
        self.depth_seq = []  # list (seq) of list (frame) of string (depth image path)
        # self.invalid_seq = []  # empty
        self.cam_pose_seq = []  # list (seq) of list (frame) of 4 X 4 ndarray
        self.calib = []  # list (seq) of list (intrinsics: fx, fy, cx, cy)
        self.seq_names = []  # list (seq) of string (seq name)

        self.subset_train = subset_train   # only use subset for training
        self.subset_val = subset_val  # only use subset for validation
        assert self.subset_train + self.subset_val <= 1
        self.ids = 0
        self.seq_acc_ids = [0]
        self.keyframes = keyframes
        self.cam =  {
            'distCoeffs': None,
            'fx': 577.871,
            'fy': 577.871,
            'ux': 319.5,
            'uy': 239.5,
            'size': (640, 480),
        }
        self.depth_conversion = 1.0/5e3

        self.transforms = data_transform

        if category == 'test':
            self.__load_test(osp.join(root, 'val'), select_traj)
        else:  # train and validation
            self.__load_train_val(osp.join(root, 'train'), category)

        # downscale the input image to a quarter
        self.fx_s = image_resize
        self.fy_s = image_resize
        self.truncate_depth = truncate_depth

        print('ScanNet dataloader for {:} using keyframe {:}: \
            {:} valid frames'.format(category, keyframes, self.ids))

    def __read_scans(self, data_dir):
        # glob for sequences
        sequences = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print('Found {} sequences in directory: {}'.format(len(sequences), data_dir))

        scans = []
        with tqdm(total=len(sequences)) as t:
            for seq in sequences:
                seq_dir = os.path.join(data_dir, seq)
                # synchronized trajectory file
                sync_traj_file = osp.join(seq_dir, 'sync_trajectory.pkl')

                if not osp.isfile(sync_traj_file):
                    print("The synchronized trajectory file {:} has not been generated.".format(seq))
                    print("Generate it now...")

                    # get sequence length from the _info.txt file
                    nframes = int(open(os.path.join(seq_dir, '_info.txt')).readlines()[-1].split()[-1])

                    views = list()
                    for i in range(nframes):
                        frame = os.path.join(seq_dir, 'frame-{:06d}'.format(i))
                        pose_file = os.path.join(seq_dir, frame + '.pose.txt')
                        pose = np.loadtxt(open(pose_file, 'r'))

                        # do not use any frame with inf pose
                        if np.isinf(np.sum(pose)):
                            print(frame)
                            continue
                        views.append({'img': frame + '.color.jpg',
                                      'dpt': frame + '.merged_depth.png',
                                      'frame_id': i,
                                      'pose': pose})

                    # export trajectory file
                    with open(sync_traj_file, 'wb') as output:
                        pickle.dump(views, output)

                else:
                    with open(sync_traj_file, 'rb') as p:
                        views = pickle.load(p)

                scans.append(views)
                t.set_postfix({'seq': seq})
                t.update()
        return scans

    def __load_train_val(self, root, category):
        scans = self.__read_scans(root)

        for scene in scans:
            total_num = len(scene)
            # the ratio to split the train & validation set
            if category == 'train':
                start_idx, end_idx = 0, int(self.subset_train * total_num)
            else:
                start_idx, end_idx = int((1-self.subset_val) * total_num), total_num

            images = [scene[idx]['img'] for idx in range(start_idx, end_idx)]
            depths = [scene[idx]['dpt'] for idx in range(start_idx, end_idx)]
            extrin = [scene[idx]['pose'] for idx in range(start_idx, end_idx)]
            # fake timestamp with frame id
            frame_id = [scene[idx]['frame_id'] for idx in range(start_idx, end_idx)]
            seq_name = osp.basename(osp.dirname(images[0]))
            calib = [self.cam['fx'], self.cam['fy'], self.cam['ux'], self.cam['uy']]

            self.calib.append(calib)
            self.image_seq.append(images)
            self.depth_seq.append(depths)
            self.timestamp.append(frame_id)
            self.cam_pose_seq.append(extrin)
            self.seq_names.append(seq_name)
            self.ids += max(0, len(images) - max(self.keyframes))
            self.seq_acc_ids.append(self.ids)

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

        assert (len(self.keyframes) == 1)
        scans = self.__read_scans(root)
        kf = self.keyframes[0]
        self.keyframes = [1]

        for scene in scans:
            seq_name = osp.basename(osp.dirname(scene[0]['img']))
            if select_traj is not None:
                if seq_name != select_traj: continue

            calib = [self.cam['fx'], self.cam['fy'], self.cam['ux'], self.cam['uy']]
            self.calib.append(calib)

            total_num = len(scene)
            images = [scene[idx]['img'] for idx in range(0, total_num, kf)]
            depths = [scene[idx]['dpt'] for idx in range(0, total_num, kf)]
            extrin = [scene[idx]['pose'] for idx in range(0, total_num, kf)]

            # fake timestamp with frame id
            timestamp = [scene[idx]['frame_id'] for idx in range(0, total_num, kf)]
            self.image_seq.append(images)
            self.timestamp.append(timestamp)
            self.depth_seq.append(depths)
            self.cam_pose_seq.append(extrin)
            self.seq_names.append(seq_name)
            self.ids += max(0, len(images) - 1)
            self.seq_acc_ids.append(self.ids)

        if len(self.image_seq) == 0:
            raise Exception("The specified trajectory is not in the test set nor supported.")

    def __getitem__(self, index):
        seq_idx = max(np.searchsorted(self.seq_acc_ids, index + 1) - 1, 0)
        frame_idx = index - self.seq_acc_ids[seq_idx]

        this_idx = frame_idx
        next_idx = frame_idx + random.choice(self.keyframes)

        # if the next random keyframe is too far
        if self.timestamp[seq_idx][next_idx] - self.timestamp[seq_idx][this_idx] > max(self.keyframes):
            search_keyframes = self.keyframes[::-1] + [-kf for kf in self.keyframes]
            inf_pose_issue = True
            print("search:", self.timestamp[seq_idx][this_idx])
            for keyframe in search_keyframes:
                next_idx = frame_idx + keyframe
                if abs(self.timestamp[seq_idx][next_idx] - self.timestamp[seq_idx][this_idx]) <= max(self.keyframes):
                    inf_pose_issue = False
                    break
            if inf_pose_issue:
                next_idx = frame_idx + 1
                print("#invalid frame:", self.image_seq[seq_idx][this_idx])
                # raise ValueError
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

        cam_pose0 = self.cam_pose_seq[seq_idx][this_idx]
        cam_pose1 = self.cam_pose_seq[seq_idx][next_idx]
        transform = np.dot(np.linalg.inv(cam_pose1), cam_pose0).astype(np.float32)

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

    def __load_rgb_tensor(self, path):
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
        depth = imread(path).astype(np.float32) * self.depth_conversion
        depth = resize(depth, None, fx=self.fx_s, fy=self.fy_s, interpolation=INTER_NEAREST)
        if self.truncate_depth:
            depth = np.clip(depth, a_min=0.5, a_max=5.0) # the accurate range of kinect depth
        return depth[np.newaxis, :]