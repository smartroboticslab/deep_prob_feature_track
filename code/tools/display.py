""" 
visualisation tool for debugging and demo
# SPDX-FileCopyrightText: 2021 Binbin Xu
# SPDX-License-Identifier: BSD-3-Clause
"""

import cv2
# cv2.setNumThreads(0)

import math
import sys
import numpy as np
import torch

def convert_flow_for_display(flow):
    """
    Converts a 2D image (e.g. flow) to bgr

    :param flow:
    :type flow: optical flow of size [2, H, W]
    :return:
    :rtype:
    """

    ang = np.arctan2(flow[1, :, :], flow[0, :, :])
    ang[ang < 0] += 2 * np.pi
    ang /= 2 * np.pi
    mag = np.sqrt(flow[0, :, :] ** 2. + flow[1, :, :] ** 2.)
    mag = np.clip(mag / (np.percentile(mag, 99) + 1e-6), 0., 1.)
    hfill_hsv = np.stack([ang * 180, mag * 255, np.ones_like(ang) * 255], 2).astype(np.uint8)
    flow_rgb = cv2.cvtColor(hfill_hsv, cv2.COLOR_HSV2RGB) / 255
    return np.transpose(flow_rgb, [2, 0, 1])


def single_image_tensor_mat(T):  # [1, C, H, W]
    img_mat = T.squeeze(0).permute(1, 2, 0).numpy()
    show = cv2.cvtColor(img_mat, cv2.COLOR_BGR2RGB)
    return show


def image_to_display(image, cmap=cv2.COLORMAP_JET, order='CHW', normalize=False):
    """
    accepts a [1xHxW] or [DxHxW] float image with values in range [0,1]
    => change it range of (0, 255) for visualization
    :param image:
    :type image:
    :param cmap: cv2.COLORMAP_BONE or cv2.COLORMAP_JET, or NORMAL(no-processing)
    :type cmap:
    :param order:
    :type order:
    :param normalize: if true, noramalize to 0~1, otherwise clip to 0-1
    :type normalize:
    :return: a visiable BGR image in range(0,255), in fault a colored heat map (in JET color map)
    :rtype: opencv mat [H, W, C]
    """
    if order is 'HWC' and len(image.shape) > 2:
        image = np.rollaxis(image, axis=2)
        # image = np.moveaxis(image, 2, 0)
    image = np.squeeze(image)  # 2d or 3d

    if len(image.shape) == 3 and image.shape[0] == 2:
        image = convert_flow_for_display(image)

    if normalize:
        # handle nan pixels
        min_intensity = np.nanmin(image)
        max_intensity = np.nanmax(image)
        image = (image - min_intensity) / (max_intensity - min_intensity)
        image = np.uint8(image * 255)
    else:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)
    # image = (image * 1).astype(np.uint8)

    if image.ndim == 3:
        if image.shape[0] == 3:
            image = np.transpose(image, [1, 2, 0])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.ndim == 2:
        if cmap == "NORMAL":
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.applyColorMap(image, cmap)
    return image


# image_array needs to be 2d
def create_mosaic(image_array, cmap=None, points=None, order='CHW', normalize=False):
    """
    Stitch array of images into a big concancated images

    :param image_array: subimages to be displayed
    :type image_array: Two-dimensional lists (arrays) , if in a stretch 1D lisr, will be stretched back
                        each element is an image of  [DxHxW]
    :param cmap: list of color map => common: cv2.COLORMAP_BONE or cv2.COLORMAP_JET or 'NORMAL'
    :type cmap:
    :param order:
    :type order:
    :param normalize: normalize to 0-1 range
    :type normalize: bool
    :return: image to be showed
    :rtype: numpy array
    """
    batch_version = (len(image_array[0].shape) == 4)

    if not isinstance(image_array[0], list):  # if image_array is a stretch 1D list
        image_size = math.ceil(math.sqrt(len(image_array)))  # stretch back to 2D list [N by N]
        image_array = [image_array[i:min(i + image_size, len(image_array))]
                       for i in range(0, len(image_array), image_size)]

    max_cols = max([len(row) for row in image_array])   # because not every row (1st array) has the same size
    rows = []

    if cmap is None:
        cmap = [cv2.COLORMAP_JET]
    elif not isinstance(cmap, list):
        cmap = [cmap]

    if not isinstance(normalize, list):
        normalize = [normalize]

    if points is not None:
        if not isinstance(points, list):
            points = [points]

    i = 0
    for image_row in image_array:
        if len(image_row) == 0:
            continue
        image_row_processed = []
        for image in image_row:
            if torch.is_tensor(image):
                if batch_version:
                    image = image[0:1, :, :, :]
                if len(image.shape) == 4:  #[B. C, H, W]
                    image = image.squeeze(0)
                    if order == 'CHW':
                        image = image.permute(1, 2, 0)  # [H, W, C]
                    if image.shape[2] not in(0, 3):  # sum all channel features
                        image = image.sum(dim=2)
                image = image.cpu().numpy()
            image_colorized = image_to_display(image, cmap[i % len(cmap)],
                                               order,
                                               normalize[i % len(normalize)])
            if points is not None:
                image_colorized = visualize_matches_on_image(image_colorized, points[i % len(points)])
            image_row_processed.append(image_colorized)
            i += 1
        nimages = len(image_row_processed)
        if nimages < max_cols:  # padding zero(black) images in the empty areas
            image_row_processed += [np.zeros_like(image_row_processed[-1])] * (max_cols - nimages)
        rows.append(np.concatenate(image_row_processed, axis=1))  #horizontally
    return np.concatenate(rows, axis=0)  # vertically


def visualize_image_features(img_list: list, feat_list: list, feat_cmap=cv2.COLORMAP_WINTER):
    feat_sum_list = [feat[0, :, :, :].sum(dim=0) for feat in feat_list]
    if feat_cmap is not None and feat_cmap is not list:
        feat_cmap = [feat_cmap] * len(feat_sum_list)
    img_cmap = ['NORMAL'] * len(img_list)
    cmap = img_cmap + feat_cmap
    img_show = create_mosaic(img_list + feat_sum_list, cmap=cmap)

    return img_show


def visualize_matches_on_image(image, matches):
    """
    :param image: []
    :type image: Union[torch.Tensor, numpy.ndarray]
    :param matches:
    :type matches: torch.Tensor
    :return:
    :rtype: None
    """
    num_matches = matches.shape[1]
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    if torch.is_tensor(matches):
        matches = matches.detach().cpu().numpy()
    # just for visualization, round it:
    matches = matches.astype(int)
    output = image.copy()
    red = (0, 0, 255)
    alpha = 0.6
    radius = int(image.shape[1] / 64)  # should be 10 when the image width is 640
    for i in range(num_matches):
        image = cv2.circle(image, (matches[0, i], matches[1, i]), radius, red, -1)
    #blend
    output = cv2.addWeighted(image, alpha, output, 1 - alpha, 0, )
    return output


def visualize_feature_channels(feat_map, rgb=None, points=None, order='CHW', add_ftr_avg=True):
    """
    :param points: points to draw on images
    :type points: torch.Tensor [2, #samples]
    :param feat_map:
    :type feat_map: torch.Tensor [B, H, W, C]
    :param rgb:
    :type rgb: numpy.ndarray  [H, W, C]  or [B, C, H, W]
    :param order: 'HWC' or 'CHW'
    :type order: str
    :return:
    :rtype: numpy.ndarray
    """
    assert len(feat_map.shape) == 4, "feature-map should be a 4-dim tensor"
    assert order in ['HWC', 'CHW']

    batch_version = (feat_map.shape[0] != 1)
    feat_map = feat_map.detach()
    if points is not None: points = points.detach()
    if not batch_version:
        feat_map = feat_map.squeeze(dim=0)
        if points is not None: points = points.squeeze()
    else:
        # if in batch, only visualize the 1st feature map
        feat_map = feat_map[0, :, :, :]
        if points is not None: points = points[0, :, :]

    if order == 'CHW':
        feat_map = feat_map.permute(1, 2, 0)  # convert to [H, W, C]
    D = feat_map.shape[2]
    feat_map_sum = feat_map.sum(dim=2)

    if rgb is not None:
        if torch.is_tensor(rgb) and len(rgb.shape) == 4:
            rgb = rgb.detach()
            if not batch_version:
                rgb = rgb.squeeze()
            else:
                # if in batch, only visualize the 1st feature map
                rgb = rgb[0, :, :, :]
            rgb = rgb.permute(1, 2, 0)  # convert to [H, W, C]
        if add_ftr_avg:
            feat_map_channel_list = [rgb, feat_map_sum]
        else:
            feat_map_channel_list = [rgb]
    else:
        if add_ftr_avg:
            feat_map_channel_list = [feat_map_sum]
        else:
            feat_map_channel_list = []

    for d in range(D):
        feat_map_channel = feat_map[:, :, d]
        feat_map_channel_list.append(feat_map_channel)

    cmap = [cv2.COLORMAP_JET] * (D + 1)
    if rgb is not None:
        cmap = ['NORMAL'] + cmap
    feature_channels = create_mosaic(feat_map_channel_list, cmap=cmap, points=points, order='HWC', normalize=True)
    return feature_channels


def normalize_descriptor_channel_wise(res):
    """
    Normalizes the descriptor into RGB color space for each channel
    :param res: numpy.array [H,W,D]
        Output of the network, per-pixel dense descriptor
    :param stats: dict, with fields ['min', 'max', 'mean'], which are used to normalize descriptor
    :return: numpy.array
        normalized descriptor [H,W,D]
    """

    # get #channel
    D = np.shape(res)[-1]
    normed_res = np.zeros_like(res)
    eps = 1e-10

    for d in range(D):
        res_min = np.min(res[:, :, d])
        res_max = np.max(res[:, :, d])
        scale_factor = res_max - res_min + eps
        normed_res[:, :, d] = (res[:, :, d] - res_min) / scale_factor

    return normed_res


def colorize(hue, lightness, normalize_hue=True, lightness_range=1.0):
    """
    Project images onto input images
    hue is normalized channel&image-wise

    :param hue: Features to be visualized
    :type hue: size of [#batch, #channel,  H, W], ith its range is supposed to be [-1.0, 1.0]
    :param lightness: input image (grey)
    :type lightness:  size [#batch, 1,  H, W], its value range is supposed to be [0, 1.0]
    :param normalize_hue: normalize hue to [0, 1]
    :type normalize_hue:
    :param lightness_range:
    :type lightness_range:
    :return: hue overlapped on the lightness
    :rtype: size of [#batch, #channel, 3, H, W]
    """
    # process the input value to be visualisation range
    lightness /= lightness_range

    out = np.zeros(list(hue.shape) + [3])  # now size become [#batch, #channel,  H, W, 3]
    if normalize_hue:
        image_num = np.shape(hue)[0]
        normed_hue = np.zeros_like(hue)
        # for i in xrange(image_num):
        #     hue_per_image = hue[i, :, :, :]
        #     hue_per_image = np.transpose(hue_per_image, [1, 2, 0])
        #     normalized_hue_image = normalize_descriptor_channel_wise(hue_per_image)
        #     normalized_hue_image = np.transpose(normalized_hue_image, [2, 0, 1])
        #     normalize_hue[i, :, :, :] = normalized_hue_image
        channel_num = np.shape(hue)[1]
        eps = 1e-10

        for i in range(image_num):
            for c in range(channel_num):
                hue_min = np.min(hue[i, c, :, :])
                hue_max = np.max(hue[i, c, :, :])
                scale_factor = hue_max - hue_min + eps
                normed_hue[i, c, :, :] = (hue[i, c, :, :] - hue_min) / scale_factor
    else:
        normed_hue = np.clip(hue, 0, 1.0) * 0.5 + 0.5
    out[:, :, :, :, 0] = normed_hue * 120.0 / 255.0
    out[:, :, :, :, 1] = (lightness - 0.5) * 0.5 + 0.5
    # out[:, :, :, :, 2] = np.ones(hue.shape) * (np.abs(np.clip(hue, -1.0, 1.0) * 1.0) + 0.0)
    out[:, :, :, :, 2] = np.ones(hue.shape) # * (normed_hue)
    out = np.reshape(out, [-1, hue.shape[3], 3])  # [#batch * #channel * H, W, 3]
    out = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_HLS2RGB).astype(np.float32) / 255
    out = np.reshape(out, list(hue.shape) + [3])  # [#batch, #channel,  H, W, 3]
    out = np.transpose(out, [0, 1, 4, 2, 3])  # [#batch, #channel, 3, H, W]. this is to meet the create_mosaic function
    return out


def visualise_frames(mat, name, max_img_visual=None, max_channel_visual=None, step_image=1, step_channel=1,
                     mosaic_save=None):
    """
    visualize batches of images in n-dimsional
    :param mat: images to be showed
    :type mat: numpy array of [#batch, #channel, 3, H, W]
    :param name: opencv window name
    :type name: string
    :param max_img_visual: image number to be showed
    :type max_img_visual: int
    :param max_channel_visual:channel number to be shoed
    :type max_channel_visual: int
    :param step_image: the step to skip image number
    :type step_image: int
    :param step_channel: the step to skip channel number
    :type step_channel: int
    :param mosaic_save: if not none, the directory to save the mosaic image
    :type mosaic_save: string
    :return: mosaic image -> deprecated currently
    :rtype:
    """
    array = list()
    max_img = mat.shape[0] if max_img_visual is None else min(max_img_visual, mat.shape[0])
    max_channel = mat.shape[1] if max_channel_visual is None else min(max_channel_visual, mat.shape[1])
    for i in range(0, max_img, step_image):
        sub = []
        for j in range(0, max_channel, step_channel):
            sub.append(mat[i, j])
        array.append(sub)
    mosaic = create_mosaic(array)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, mosaic)
    if mosaic is not None:
        cv2.imwrite(mosaic_save, mosaic)
    return mosaic



class Toolbar:
    def reset(self, width, tot, title=None):
        self.width = max(int(min(width, tot)), 1)
        self.tot = int(max(tot, 1))
        self.current = 0
        if title is not None:
            print(title)
        sys.stdout.write("[%s]" % (" " * self.width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.width + 1))

    def incr(self):
        n = self.current + 1
        if math.floor(n * self.width / self.tot) > math.floor(self.current * self.width / self.tot):
            sys.stdout.write("-")
            sys.stdout.flush()
        self.current = n
        if self.current == self.tot:
            sys.stdout.write("\n")
