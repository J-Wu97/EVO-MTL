import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from core.models.common_layers import get_nddr
from core.utils import AttrDict
from core.tasks import get_tasks
from core.utils.losses import poly, entropy_loss, l1_loss

from core.models.vgg16_lfov_bn_16_stages import DeepLabLargeFOVBN16
from core.config import cfg
from core.data import loader


class GeneralizedMTLNASNet(nn.Module):
    def __init__(self, cfg, net1, net2,
                 net1_connectivity_matrix,
                 net2_connectivity_matrix
                 ):
        """
        :param net1: task one network
        :param net2: task two network
        :param net1_connectivity_matrix: Adjacency list for the Single sided NDDR connections
        :param net2_connectivity_matrix: Adjacency list for the Single sided NDDR connections
        """
        super(GeneralizedMTLNASNet, self).__init__()
        self.cfg = cfg
        self.net1 = net1
        self.net2 = net2
        assert len(net1.stages) == len(net2.stages)
        print("Model has %d stages" % len(net1.stages))
        self.task1, self.task2 = get_tasks(cfg)
        self.num_stages = len(net1.stages)
        self.net1_connectivity_matrix = net1_connectivity_matrix
        self.net2_connectivity_matrix = net2_connectivity_matrix
        net1_in_degrees = net1_connectivity_matrix.sum(axis=1)
        net2_in_degrees = net2_connectivity_matrix.sum(axis=1)
        net1_fusion_ops = []  # used for incoming feature fusion
        net2_fusion_ops = []  # used for incoming feature fusion

        for stage_id in range(self.num_stages):
            n_channel = net1.stages[stage_id].out_channels
            net1_op = get_nddr(cfg,
                               (net1_in_degrees[stage_id] + 1) * n_channel,  # +1 for original upstream input
                               n_channel)
            net2_op = get_nddr(cfg,
                               (net2_in_degrees[stage_id] + 1) * n_channel,  # +1 for original upstream input
                               n_channel)
            net1_fusion_ops.append(net1_op)
            net2_fusion_ops.append(net2_op)

        net1_fusion_ops = nn.ModuleList(net1_fusion_ops)
        net2_fusion_ops = nn.ModuleList(net2_fusion_ops)

        self.net1_alphas = nn.Parameter(torch.zeros(net1_connectivity_matrix.shape))
        self.net2_alphas = nn.Parameter(torch.zeros(net2_connectivity_matrix.shape))

        self.paths = nn.ModuleDict({
            'net1_paths': net1_fusion_ops,
            'net2_paths': net2_fusion_ops,
        })
        self._net_parameters = dict()
        for k, v in self.named_parameters():
            self._net_parameters[k] = v
        self.supernet = False
        if cfg.MODEL.SUPERNET:
            print("Running Supernet Baseline")
            self.supernet = True
            
    def net_parameters(self):
        return self._net_parameters.values()
    
    def named_net_parameters(self):
        return self._net_parameters.items()
    
    def loss(self, image, labels):
        label_1, label_2 = labels
        result = self.forward(image)
        result.loss1 = self.task1.loss(result.out1, label_1)
        result.loss2 = self.task2.loss(result.out2, label_2)
        result.loss = result.loss1 + self.cfg.TRAIN.TASK2_FACTOR * result.loss2
        return result

    def forward(self, x):
        N, C, H, W = x.size()
        y = x.clone()
        # x = self.net1.base(x)
        # y = self.net2.base(y)
        xs, ys = [], []
        for stage_id in range(self.num_stages):
            x = self.net1.stages[stage_id](x)
            y = self.net2.stages[stage_id](y)
            if isinstance(x, list):
                xs.append(x[0])
                ys.append(y[0])
            else:
                xs.append(x)
                ys.append(y)

            net1_path_ids = np.nonzero(self.net1_connectivity_matrix[stage_id])[0]
            net2_path_ids = np.nonzero(self.net2_connectivity_matrix[stage_id])[0]

            if isinstance(x, list):
                net1_fusion_input = [x[0]]
                net2_fusion_input = [y[0]]
            else:
                net1_fusion_input = [x]
                net2_fusion_input = [y]

            # net1_fusion_input = [x]
            # net2_fusion_input = [y]

            for idx, input_id in enumerate(net1_path_ids):
                net1_fusion_input.append(ys[input_id])
            for idx, input_id in enumerate(net2_path_ids):
                net2_fusion_input.append(xs[input_id])

            if isinstance(x, list):
                x[0] = self.paths['net1_paths'][stage_id](net1_fusion_input)
                y[0] = self.paths['net2_paths'][stage_id](net2_fusion_input)
            else:
                x = self.paths['net1_paths'][stage_id](net1_fusion_input)
                y = self.paths['net2_paths'][stage_id](net2_fusion_input)

        x = self.net1.head(x)
        y = self.net2.head(y)
        return AttrDict({'out1': x, 'out2': y})