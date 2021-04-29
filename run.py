"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import matplotlib

matplotlib.use('Agg')

import os
import sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import FramesDataset

from modules.generator import Generator
from modules.bg_motion_predictor import BGMotionPredictor
from modules.region_predictor import RegionPredictor
from modules.avd_network import AVDNetwork

import torch

from train import train
from reconstruction import reconstruction
from animate import animate
from train_avd import train_avd

if __name__ == "__main__":

    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "train_avd", "reconstruction", "animate"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

    generator = Generator(num_regions=config['model_params']['num_regions'],
                          num_channels=config['model_params']['num_channels'],
                          revert_axis_swap=config['model_params']['revert_axis_swap'],
                          **config['model_params']['generator_params'])

    if torch.cuda.is_available():
        generator.to(opt.device_ids[0])
    if opt.verbose:
        print(generator)

    region_predictor = RegionPredictor(num_regions=config['model_params']['num_regions'],
                                       num_channels=config['model_params']['num_channels'],
                                       estimate_affine=config['model_params']['estimate_affine'],
                                       **config['model_params']['region_predictor_params'])

    if torch.cuda.is_available():
        region_predictor.to(opt.device_ids[0])

    if opt.verbose:
        print(region_predictor)

    bg_predictor = BGMotionPredictor(num_channels=config['model_params']['num_channels'],
                                     **config['model_params']['bg_predictor_params'])
    if torch.cuda.is_available():
        bg_predictor.to(opt.device_ids[0])
    if opt.verbose:
        print(bg_predictor)

    avd_network = AVDNetwork(num_regions=config['model_params']['num_regions'],
                             **config['model_params']['avd_network_params'])
    if torch.cuda.is_available():
        avd_network.to(opt.device_ids[0])
    if opt.verbose:
        print(avd_network)

    dataset = FramesDataset(is_train=(opt.mode.startswith('train')), **config['dataset_params'])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if opt.mode == 'train':
        print("Training...")
        train(config, generator, region_predictor, bg_predictor, opt.checkpoint, log_dir, dataset, opt.device_ids)
    elif opt.mode == 'train_avd':
        print("Training Animation via Disentaglement...")
        train_avd(config, generator, region_predictor, bg_predictor, avd_network, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
        reconstruction(config, generator, region_predictor, bg_predictor, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'animate':
        print("Animate...")
        animate(config, generator, region_predictor, avd_network, opt.checkpoint, log_dir, dataset)
