import numpy as np
import random
from core.models.supernet import EvoNet
from core.models.vgg16_lfov_bn_16_stages import DeepLabLargeFOVBN16
def depth_limited_connectivity_matrix(stage_config, limit=3):

    network_depth = np.sum(stage_config)
    stage_depths = np.cumsum([0] + stage_config)
    matrix = np.zeros((network_depth, network_depth)).astype('int')
    for i in range(network_depth):
        j_limit = stage_depths[np.argmax(stage_depths > i) - 1]
        for j in range(network_depth):
            if j <= i and i - j < limit and j >= j_limit:
                matrix[i, j] = 1.
    return matrix

def vgg_connectivity():
    matrix=depth_limited_connectivity_matrix([2, 2, 3, 3, 3])
    size = matrix.shape[0]
    for i in range(size):
        for j in range(size):
            if matrix[i, j] == 1:
                if np.random.rand() > 0.5:
                    matrix[i, j]= 0
    print(matrix)
    return matrix


def get_model(cfg, indi):
    if cfg.TASK == 'pixel':
        if cfg.MODEL.BACKBONE == 'VGG16_13_Stage':
            net1 = DeepLabLargeFOVBN16(3, cfg.MODEL.NET1_CLASSES, weights=cfg.TRAIN.WEIGHT_1)
            net2 = DeepLabLargeFOVBN16(3, cfg.MODEL.NET2_CLASSES, weights=cfg.TRAIN.WEIGHT_2)
        else:
            raise NotImplementedError

  

        model = EvoNet(cfg, net1, net2,
                                     net1_connectivity_matrix=indi.indi_net1,
                                     net2_connectivity_matrix=indi.indi_net2)
    else:
        raise NotImplementedError
    return model
