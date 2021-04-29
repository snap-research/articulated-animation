"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""
import torch
from torch import nn


class AVDNetwork(nn.Module):
    """
    Animation via Disentanglement network
    """

    def __init__(self, num_regions, id_bottle_size=64, pose_bottle_size=64, revert_axis_swap=True):
        super(AVDNetwork, self).__init__()
        input_size = (2 + 4) * num_regions
        self.num_regions = num_regions
        self.revert_axis_swap = revert_axis_swap

        self.id_encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, id_bottle_size)
        )

        self.pose_encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, pose_bottle_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(pose_bottle_size + id_bottle_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_size)
        )

    @staticmethod
    def region_params_to_emb(x):
        mean = x['shift']
        jac = x['affine']
        emb = torch.cat([mean, jac.view(jac.shape[0], jac.shape[1], -1)], dim=-1)
        emb = emb.view(emb.shape[0], -1)
        return emb

    def emb_to_region_params(self, emb):
        emb = emb.view(emb.shape[0], self.num_regions, 6)
        mean = emb[:, :, :2]
        jac = emb[:, :, 2:].view(emb.shape[0], emb.shape[1], 2, 2)
        return {'shift': mean, 'affine': jac}

    def forward(self, x_id, x_pose, alpha=0.2):
        if self.revert_axis_swap:
            affine = torch.matmul(x_id['affine'], torch.inverse(x_pose['affine']))
            sign = torch.sign(affine[:, :, 0:1, 0:1])
            x_id = {'affine': x_id['affine'] * sign, 'shift': x_id['shift']}

        pose_emb = self.pose_encoder(self.region_params_to_emb(x_pose))
        id_emb = self.id_encoder(self.region_params_to_emb(x_id))

        rec = self.decoder(torch.cat([pose_emb, id_emb], dim=1))

        rec = self.emb_to_region_params(rec)
        rec['covar'] = torch.matmul(rec['affine'], rec['affine'].permute(0, 1, 3, 2))
        return rec
