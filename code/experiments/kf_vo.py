""" Script to run keyframe visual odometry on a sequence of images 
using the proposed probablilistic feature-metric trackingmethod.
# SPDX-FileCopyrightText: 2021 Binbin Xu
# SPDX-License-Identifier: BSD-3-Clause
"""

#!/usr/bin/env python

# import standard library
import os
import sys
import argparse
import os.path as osp
# import third party
import cv2
from evaluate import create_eval_loaders

import numpy as np
# opengl/trimesh visualization
import pyglet
import trimesh
import trimesh.viewer as tv
import trimesh.transformations as tf
import torch
from imageio import imread

import config
from models.geometry import batch_create_transform
from experiments.select_method import select_method
from train_utils import check_cuda
from data.dataloader import load_data
from Logger import check_directory


def init_scene(scene):
    scene.geometry = {}
    scene.graph.clear()
    scene.init = True

    # clear poses
    scene.gt_poses = []
    scene.est_poses = []
    scene.timestamps = []

    return scene


def camera_transform(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ trimesh.transformations.rotation_matrix(
        np.deg2rad(-180), [1, 0, 0]
    )


def pointcloud_from_depth(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_type: str = 'z',
) -> np.ndarray:
    assert depth_type in ['z', 'euclidean'], 'Unexpected depth_type'
    assert depth.dtype.kind == 'f', 'depth must be float and have meter values'

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = ~np.isnan(depth)
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, z * (c - cx) / fx, np.nan)
    y = np.where(valid, z * (r - cy) / fy, np.nan)
    pc = np.dstack((x, y, z))

    if depth_type == 'euclidean':
        norm = np.linalg.norm(pc, axis=2)
        pc = pc * (z / norm)[:, :, None]
    return pc


def callback(scene):
    if not scene.is_play:
        return

    dataset = scene.dataloader
    options = scene.options
    if scene.index >= len(dataset):
        return

    if scene.vo_type == 'incremental':
        batch = dataset[scene.index - 1]
    else:
        batch = dataset.get_keypair(scene.index)
    color0, color1, depth0, depth1, GT_Rt, intrins, name = check_cuda(
        batch[:7])

    scene_id = name['seq']

    # Reset scene for new scene.
    if scene_id != scene.video_id:
        scene = init_scene(scene)
        scene.init_idx = scene.index
        scene.video_id = scene_id
    else:
        scene.init = False

    GT_WC = dataset.cam_pose_seq[0][scene.index]
    depth_file = dataset.depth_seq[0][scene.index]
    if not options.save_img:
        # half resolution
        rgb = color1.permute((1, 2, 0)).cpu().numpy()
        depth = imread(depth_file).astype(np.float32) / 5e3
        depth = cv2.resize(depth, None, fx=dataset.fx_s,
                           fy=dataset.fy_s, interpolation=cv2.INTER_NEAREST)
        K = {"fx": intrins[0].item(), "fy": intrins[1].item(),
             "ux": intrins[2].item(), "uy": intrins[3].item()}
    else:
        # original resolution for demo
        rgb = imread(dataset.image_seq[0][scene.index])
        depth = imread(depth_file).astype(np.float32) / 5e3
        calib = np.asarray(dataset.calib[0], dtype=np.float32)
        K = {"fx": calib[0], "fy": calib[1],
             "ux": calib[2], "uy": calib[3]}

        # save input rgb and depth images
        img_index_png = str(scene.index).zfill(5)+'.png'
        if options.dataset == 'VaryLighting':
            output_folder = osp.join(dataset.seq_names[0], 'kf_vo', options.vo)
        else:
            output_folder = os.path.join(
                '/home/binbin/Pictures', 'kf_vo', options.vo)

        rgb_img = osp.join(output_folder, 'rgb', img_index_png)
        depth_img = osp.join(output_folder, 'depth', img_index_png)
        check_directory(rgb_img)
        check_directory(depth_img)
        cv2.imwrite(rgb_img, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(depth_img, depth)

    if scene.init:
        if GT_WC is not None:
            T_WC = GT_WC
        else:
            T_WC = np.array([
                [0.0, -1.0, 0.0, -0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            # T_WC = np.eye(4)
        scene.T_WC = T_WC
        if scene.vo_type == 'keyframe':
            scene.T_WK = T_WC
    else:
        with torch.no_grad():
            color0 = color0.unsqueeze(dim=0)
            color1 = color1.unsqueeze(dim=0)
            depth0 = depth0.unsqueeze(dim=0)
            depth1 = depth1.unsqueeze(dim=0)
            intrins = intrins.unsqueeze(dim=0)
            if options.save_img:
                output = scene.network.forward(
                    color0, color1, depth0, depth1, intrins, index=scene.index)
            else:
                output = scene.network.forward(
                    color0, color1, depth0, depth1, intrins)
        R, t = output
        if scene.is_gt_tracking:
            T_WC = GT_WC
            scene.T_WC = T_WC
        else:
            if scene.vo_type == 'incremental':
                T_CR = batch_create_transform(t, R)
                # T_CR = GT_Rt
                T_CR = T_CR.squeeze(dim=0).cpu().numpy()
                T_WC = np.dot(scene.T_WC, np.linalg.inv(
                    T_CR)).astype(np.float32)
            elif scene.vo_type == 'keyframe':
                T_CK = batch_create_transform(t, R)
                # T_CK = GT_Rt
                T_CK = T_CK.squeeze(dim=0).cpu().numpy()
                T_WC = np.dot(scene.T_WK, np.linalg.inv(
                    T_CK)).astype(np.float32)

                # print large drift in keyframe tracking,
                # just for noticing a possible tracking failure, not usd later
                T_CC = np.dot(np.linalg.inv(T_WC),
                              scene.T_WC).astype(np.float32)
                trs_drift = np.copy(T_CC[0:3, 3:4]).transpose()
                if np.linalg.norm(trs_drift) > 0.02:
                    print(depth_file)
            else:
                raise NotImplementedError()
            scene.T_WC = T_WC

    pcd = pointcloud_from_depth(
        depth, fx=K['fx'], fy=K['fy'], cx=K['ux'], cy=K['uy']
    )
    nonnan = ~np.isnan(depth)
    geom = trimesh.PointCloud(vertices=pcd[nonnan], colors=rgb[nonnan])
    # XYZ->RGB, Z is blue
    if options.dataset == 'VaryLighting':
        axis = trimesh.creation.axis(0.005, origin_color=(0, 0, 0))
    elif options.dataset in ['TUM_RGBD', 'ScanNet']:
        axis = trimesh.creation.axis(0.01, origin_color=(0, 0, 0))
    else:
        raise NotImplementedError()

    # two view: keyframe - live frames visualization
    if scene.vo_type == 'keyframe' and scene.two_view:
        if scene.init:
            scene.add_geometry(geom, transform=scene.T_WK, geom_name='key')
            scene.add_geometry(geom, transform=T_WC, geom_name='live')
            scene.add_geometry(axis, transform=T_WC, geom_name='camera_view')
        else:
            # after the first view, delete the old live view and add new live view
            scene.delete_geometry('live')
            scene.delete_geometry('camera_view')
            scene.add_geometry(geom, transform=T_WC, geom_name='live')
            scene.add_geometry(axis, transform=T_WC, geom_name='camera_view')

    else:
        scene.add_geometry(geom, transform=T_WC)

        # draw camera trajectory
        trs = np.copy(T_WC[0:3, 3:4]).transpose()
        cam = trimesh.PointCloud(vertices=trs, colors=[255, 0, 0])
        scene.add_geometry(cam)

        scene.add_geometry(axis, transform=T_WC)

        if scene.last_pose is not None:
            poses_seg = np.stack((scene.last_pose, trs), axis=1)
            cam_seg = trimesh.load_path(poses_seg)
            scene.add_geometry(cam_seg)
        scene.last_pose = trs

    # A kind of current camera view, but a bit far away to see whole scene.
    scene.camera.resolution = (rgb.shape[1], rgb.shape[0])
    scene.camera.focal = (K['fx'], K['fy'])
    if dataset.realscene:
        if options.save_img:
            if scene.vo_type == 'keyframe':
                # T_see = np.array([
                #     [1.000, 0.000, 0.000, 0.2],
                #     [0.000, 0.866, 0.500, -0.7],
                #     [0.000, -0.500, 0.866, -0.7],
                #     [0.000, 0.000, 0.000, 1.0],
                # ])
                T_see = np.array([
                    [1.000, 0.000, 0.000, 0.2],
                    [0.000, 0.866, 0.500, -0.7],
                    [0.000, -0.500, 0.866, -0.8],
                    [0.000, 0.000, 0.000, 1.0],
                ])

                # T_see = np.array([
                #     [1.000, 0.000, 0.000, 0.2],
                #     [0.000, 0.985, 0.174, -0.3],
                #     [0.000, -0.174, 0.985, -0.6],
                #     [0.000, 0.000, 0.000, 1.0],
                # ])
                scene.camera_transform = camera_transform(
                    np.matmul(scene.T_WK, T_see)
                )
            else:
                # if scene.index < 140:
                #     T_see = np.array([
                #         [1.000, 0.000, 0.000, 0.2],
                #         [0.000, 0.866, 0.500, -2.0],
                #         [0.000, -0.500, 0.866, -2.0],
                #         [0.000, 0.000, 0.000, 1.0],
                #     ])
                #     scene.camera_transform = camera_transform(
                #         np.matmul(scene.T_WC, T_see)
                #     )
                pass
        else:
            # adjust which transformation use to set the see pose
            if scene.vo_type == 'keyframe':
                T_see = np.array([
                    [1.000, 0.000, 0.000, 0.2],
                    [0.000, 0.866, 0.500, -0.7],
                    [0.000, -0.500, 0.866, -0.8],
                    [0.000, 0.000, 0.000, 1.0],
                ])
                # T_see = np.array([
                #     [1.000, 0.000, 0.000, 0.2],
                #     [0.000, 0.985, 0.174, -0.3],
                #     [0.000, -0.174, 0.985, -0.6],
                #     [0.000, 0.000, 0.000, 1.0],
                # ])
                # T_see = np.array([
                #     [1.000, 0.000, 0.000, 0.2],
                #     [0.000, 0.985, 0.174, -0.3],
                #     [0.000, -0.174, 0.985, -0.6],
                #     [0.000, 0.000, 0.000, 1.0],
                # ])
                # T_see = np.array([
                #     [1.000, 0.000, 0.000, 0.2],
                #     [0.000, 0.866, 0.500, -0.8],
                #     [0.000, -0.500, 0.866, -0.8],
                #     [0.000, 0.000, 0.000, 1.0],
                # ])

                scene.camera_transform = camera_transform(
                    np.matmul(scene.T_WK, T_see)
                )
    else:
        scene.camera.transform = T_WC @ tf.translation_matrix([0, 0, 2.5])

    # if scene.index == scene.init_idx + 1:
    #     input()
    print(scene.index)
    scene.index += 1  # scene.track_config['frame_step']
    # print("<=================================")
    if options.save_img:
        return


def main(options):

    if options.dataset == 'TUM_RGBD':
        sequence_dir = 'rgbd_dataset_freiburg1_desk'
        np_loader = load_data('TUM_RGBD', keyframes=[1, ], load_type='test',
                              select_trajectory=sequence_dir,
                              truncate_depth=True,
                              options=options,
                              load_numpy=False)
    elif options.dataset == 'VaryLighting':
        np_loader = load_data('VaryLighting', keyframes=[1, ], load_type='test',
                              select_trajectory='scene17_demo',  # 'l_scene3',
                              truncate_depth=True,
                              load_numpy=False,
                              pair=options.vo_type,
                              )
    elif options.dataset == 'ScanNet':
        np_loader = load_data('ScanNet', keyframes=[1, ], load_type='test',
                              select_trajectory='scene0593_00',
                              truncate_depth=True,
                              load_numpy=False,
                              options=options,
                              )

    scene = trimesh.Scene()
    scene.dataloader = np_loader
    scene.dataloader.realscene = True
    # total_batch_size = options.batch_per_gpu * torch.cuda.device_count()

    # keyframes = [int(x) for x in options.keyframes.split(',')]
    # if options.dataset in ['BundleFusion', 'TUM_RGBD']:
    #     obj_has_mask = False
    # else:
    #     obj_has_mask = True

    # eval_loaders = create_eval_loaders(options, options.eval_set,
    #                                    [1,], total_batch_size, options.trajectory)

    tracker = select_method(options.vo, options)
    scene.network = tracker

    scene.index = 0  # config['start_frame']  # starting frame e.g. 60
    scene.video_id = None
    scene.last_pose = None
    scene.is_gt_tracking = options.gt_tracker
    scene.init = False    # True only for the first frame
    scene.is_play = True  # immediately start playing when called
    scene.vo_type = options.vo_type
    scene.two_view = options.two_view
    scene.options = options

    callback(scene)
    window = trimesh.viewer.SceneViewer(
        scene, callback=callback, start_loop=False, resolution=(1080, 720)
    )

    @window.event
    def on_key_press(symbol, modifiers):
        if modifiers == 0:
            if symbol == pyglet.window.key.P:
                scene.is_play = not scene.is_play

    print('Press P key to pause/resume.')

    if not options.save_img:
        # scene.show()
        pyglet.app.run()
    else:
        # import pyrender
        # scene_pyrender = pyrender.Scene.from_trimesh(scene)
        # renderer = pyrender.OffscreenRenderer(viewport_height=480, viewport_width=640, point_size=1)
        # rgb, depth = renderer.render(scene_pyrender)

        if options.dataset == 'VaryLighting':
            output_dir = osp.join(np_loader.seq_names[0], 'kf_vo', options.vo)
        else:
            output_dir = os.path.join(
                '/home/binbin/Pictures', 'kf_vo', options.vo)
        check_directory(output_dir + '/*.png')
        for frame_id in range(len(scene.dataloader)):
            # scene.save_image()
            callback(scene)
            file_name = os.path.join(output_dir, 'render', str(
                scene.index-1).zfill(5) + '.png')
            check_directory(file_name)
            with open(file_name, "wb") as f:
                f.write(scene.save_image())
                f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the network")
    config.add_basics_config(parser)
    config.add_test_basics_config(parser)
    config.add_tracking_config(parser)
    config.add_vo_config(parser)

    options = parser.parse_args()
    # to save visualization: --save_img and --vis_feat
    print('---------------------------------------')
    main(options)
