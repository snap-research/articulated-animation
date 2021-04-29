"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import numpy as np
import torch
import torch.nn.functional as F
import imageio

import os
from skimage.draw import circle

import matplotlib.pyplot as plt
import collections


class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=8, log_file_name='log.txt',
                 train_mode='reconstruction'):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.train_mode = train_mode
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None
        self.models = {}

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp['driving'], inp['source'], out)
        path = os.path.join(self.visualizations_dir,
                            "%s-" % str(self.epoch).zfill(self.zfill_num) + self.train_mode + '.png')
        imageio.imsave(path, image)

    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch_' + self.train_mode] = self.epoch
        basename = '%s-cpk-' % str(self.epoch).zfill(self.zfill_num) + self.train_mode + '.pth'
        cpk_path = os.path.join(self.cpk_dir, basename)
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, generator=None, region_predictor=None, bg_predictor=None, avd_network=None,
                 optimizer_reconstruction=None, optimizer_avd=None):
        checkpoint = torch.load(checkpoint_path)
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if region_predictor is not None:
            region_predictor.load_state_dict(checkpoint['region_predictor'])
        if bg_predictor is not None:
            bg_predictor.load_state_dict(checkpoint['bg_predictor'])
        if avd_network is not None:
            if 'avd_network' in checkpoint:
                avd_network.load_state_dict(checkpoint['avd_network'])

        if optimizer_reconstruction is not None:
            optimizer_reconstruction.load_state_dict(checkpoint['optimizer_reconstruction'])
            return checkpoint['epoch_reconstruction']

        if optimizer_avd is not None:
            if 'optimizer_avd' in checkpoint:
                optimizer_avd.load_state_dict(checkpoint['optimizer_avd'])
                return checkpoint['epoch_avd']
            return 0

        return 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, inp, out):
        self.epoch = epoch
        self.models.update(models)
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)
        self.visualize_rec(inp, out)


def draw_colored_heatmap(heatmap, colormap, bg_color):
    parts = []
    weights = []
    bg_color = np.array(bg_color).reshape((1, 1, 1, 3))
    num_regions = heatmap.shape[-1]
    for i in range(num_regions):
        color = np.array(colormap(i / num_regions))[:3]
        color = color.reshape((1, 1, 1, 3))
        part = heatmap[:, :, :, i:(i + 1)]
        part = part / np.max(part, axis=(1, 2), keepdims=True)
        weights.append(part)

        color_part = part * color
        parts.append(color_part)

    weight = sum(weights)
    bg_weight = 1 - np.minimum(1, weight)
    weight = np.maximum(1, weight)
    result = sum(parts) / weight + bg_weight * bg_color
    return result


class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow', region_bg_color=(0, 0, 0)):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)
        self.region_bg_color = np.array(region_bg_color)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_regions = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = circle(kp[1], kp[0], self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_regions))[:3]
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, driving, source, out):
        images = []

        # Source image with region centers
        source = source.data.cpu()
        source_region_params = out['source_region_params']['shift'].data.cpu().numpy()
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source, source_region_params))

        # Equivariance visualization
        if 'transformed_frame' in out:
            transformed = out['transformed_frame'].data.cpu().numpy()
            transformed = np.transpose(transformed, [0, 2, 3, 1])
            transformed_kp = out['transformed_region_params']['shift'].data.cpu().numpy()
            images.append((transformed, transformed_kp))

        # Driving image with  region centers
        driving_region_params = out['driving_region_params']['shift'].data.cpu().numpy()
        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, driving_region_params))

        # Deformed image
        if 'deformed' in out:
            deformed = out['deformed'].data.cpu().numpy()
            deformed = np.transpose(deformed, [0, 2, 3, 1])
            images.append(deformed)

        # Result
        prediction = out['prediction'].data.cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        images.append(prediction)

        # Occlusion map
        if 'occlusion_map' in out:
            occlusion_map = out['occlusion_map'].data.cpu().repeat(1, 3, 1, 1)
            occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
            images.append(occlusion_map)

        # Heatmaps visualizations
        if 'heatmap' in out['driving_region_params']:
            driving_heatmap = F.interpolate(out['driving_region_params']['heatmap'], size=source.shape[1:3])
            driving_heatmap = np.transpose(driving_heatmap.data.cpu().numpy(), [0, 2, 3, 1])
            images.append(draw_colored_heatmap(driving_heatmap, self.colormap, self.region_bg_color))

        if 'heatmap' in out['source_region_params']:
            source_heatmap = F.interpolate(out['source_region_params']['heatmap'], size=source.shape[1:3])
            source_heatmap = np.transpose(source_heatmap.data.cpu().numpy(), [0, 2, 3, 1])
            images.append(draw_colored_heatmap(source_heatmap, self.colormap, self.region_bg_color))

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
