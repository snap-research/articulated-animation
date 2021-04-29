"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

from tqdm import trange
import torch
from torch.utils.data import DataLoader
from logger import Logger
from torch.optim.lr_scheduler import MultiStepLR
from frames_dataset import DatasetRepeater


def random_scale(region_params, scale):
    theta = torch.rand(region_params['shift'].shape[0], 2) * (2 * scale) + (1 - scale)
    theta = torch.diag_embed(theta).unsqueeze(1).type(region_params['shift'].type())
    new_region_params = {'shift': torch.matmul(theta, region_params['shift'].unsqueeze(-1)).squeeze(-1),
                         'affine': torch.matmul(theta, region_params['affine'])}
    return new_region_params


def train_avd(config, generator, region_predictor, bg_predictor,
              avd_network, checkpoint, log_dir, dataset):
    train_params = config['train_avd_params']

    optimizer = torch.optim.Adam(avd_network.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator=generator, region_predictor=region_predictor,
                                      bg_predictor=bg_predictor, avd_network=avd_network,
                                      optimizer_avd=optimizer)
    else:
        raise AttributeError("Checkpoint should be specified for mode='train_avd'.")

    scheduler = MultiStepLR(optimizer, train_params['epoch_milestones'], gamma=0.1)

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True,
                            num_workers=train_params['dataloader_workers'], drop_last=True)
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'],
                checkpoint_freq=train_params['checkpoint_freq'], train_mode='avd') as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            avd_network.train()
            for x in dataloader:
                with torch.no_grad():
                    regions_params_id = region_predictor(x['source'].cuda())
                    regions_params_pose_gt = region_predictor(x['driving'].cuda())
                    regions_params_pose = random_scale(regions_params_pose_gt, scale=train_params['random_scale'])
                rec = avd_network(regions_params_id, regions_params_pose)

                reconstruction_shift = train_params['lambda_shift'] * \
                                       torch.abs(regions_params_pose_gt['shift'] - rec['shift']).mean()
                reconstruction_affine = train_params['lambda_affine'] * \
                                        torch.abs(regions_params_pose_gt['affine'] - rec['affine']).mean()

                loss_dict = {'rec_shift': reconstruction_shift, 'rec_affine': reconstruction_affine}
                loss = reconstruction_shift + reconstruction_affine

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in loss_dict.items()}
                logger.log_iter(losses=losses)

            # Visualization
            avd_network.eval()
            with torch.no_grad():
                source = x['source'][:6].cuda()
                driving = torch.cat([x['driving'][[0, 1]].cuda(), source[[2, 3, 2, 1]]], dim=0)
                source_region_params = region_predictor(source)
                driving_region_params = region_predictor(driving)

                out = avd_network(source_region_params, driving_region_params)
                out['covar'] = torch.matmul(out['affine'], out['affine'].permute(0, 1, 3, 2))
                driving_region_params = out
                generated = generator(source, source_region_params=source_region_params,
                                      driving_region_params=driving_region_params)

                generated['driving_region_params'] = driving_region_params
                generated['source_region_params'] = source_region_params

            scheduler.step(epoch)
            logger.log_epoch(epoch,
                             {'generator': generator,
                              'bg_predictor': bg_predictor,
                              'region_predictor': region_predictor,
                              'optimizer_reconstruction': optimizer,
                              'avd_network': avd_network,
                              'optimizer_avd': optimizer},
                             inp={'source': source, 'driving': driving},
                             out=generated)
