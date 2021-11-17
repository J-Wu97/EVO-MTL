# from core.models.connect_matrix import *
from core.models.individual import Individual
from core.models.vgg16_lfov_bn_16_stages import DeepLabLargeFOVBN16
from core.models.supernet import EvoNet


def get_model(cfg, indi, ask1, task2):
    if cfg.TASK == 'pixel':
        if cfg.MODEL.BACKBONE == 'VGG16_13_Stage':
            net1 = DeepLabLargeFOVBN16(3, cfg.MODEL.NET1_CLASSES, weights=cfg.TRAIN.WEIGHT_1)
            net2 = DeepLabLargeFOVBN16(3, cfg.MODEL.NET2_CLASSES, weights=cfg.TRAIN.WEIGHT_2)
        else:
            raise NotImplementedError

    if cfg.ARCH.SEARCHSPACE == 'EvoNet':
        if cfg.MODEL.BACKBONE == 'VGG16_13_Stage':
            connectivity = indi.indi
        else:
            raise NotImplementedError

        model = EvoNet(cfg, net1, net2,
                                     net1_connectivity_matrix=connectivity(),
                                     net2_connectivity_matrix=connectivity())
    else:
        raise NotImplementedError
    return model
