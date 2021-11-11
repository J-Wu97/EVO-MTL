import os
import argparse
import random
import numpy as np
import random

import torch
import torch.nn as nn
import torch.distributed as dist

# ad-hoc way to deal with python 3.7.4
import os, sys
lib_path = os.path.abspath(os.path.join('.'))
print(lib_path)
sys.path.append(lib_path)



import os
import argparse
import random
import numpy as np
import random
import os, sys
lib_path = os.path.abspath(os.path.join('.'))
print(lib_path)
sys.path.append(lib_path)
lib_path2 = r'/home/jiewu/GA_MTL/MTL/core'
print(lib_path2)
sys.path.append(lib_path2)
lib_path3 = r'/home/jiewu/GA_MTL/MTL/core/data'
print(lib_path3)
sys.path.append(lib_path3)
lib_path4 = r'/home/jiewu/GA_MTL/MTL/core/models'
print(lib_path4)
sys.path.append(lib_path4)

import torch
import torch.nn as nn
import torch.distributed as dist

# ad-hoc way to deal with python 3.7.4
import os, sys
lib_path = os.path.abspath(os.path.join('.'))
print(lib_path)
sys.path.append(lib_path)

from config import cfg
from tasks import get_tasks
from data import get_dataset
from models.connect_matrix import *

from eval import evaluate
import datetime

import torch
import torchvision
import numpy as np
from loader import MultiTaskDataset
from models import vgg16_lfov_bn_16_stages
from models.connect_matrix import *
import tasks
from config import cfg
import torch.optim as optim
from eval import evaluate
import os



def main():

    # load the data
    # load the data
    data_dir = '//home//jiewu//MTLNAS//datasets//nyu_v2'
    image_mean = '//nyu_v2_mean.npy'
    data_list_1 = '//list//training_seg.txt'
    data_list_2 = '//list//training_normal_mask.txt'
    test_data_list_1 = '//list//testing_seg.txt'
    test_data_list_2 = '//list//testing_normal_mask.txt'

    test_data = MultiTaskDataset(data_dir, image_mean, test_data_list_1, test_data_list_2, None, None,
                                 False, False, False, 255)
    print('The test dataset length: {}'.format(len(test_data)))

    test_sampler = None
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, sampler=test_sampler)

    task1, task2 = tasks.get_tasks(cfg)
    # model = get_model(cfg, task1, task2)



    ckpt_path = os.path.join(cfg.SAVE_DIR, 'vgg_nyuv2_default', 'ckpt-%s.pth' % str(cfg.TEST.CKPT_ID).zfill(5))
    print("Evaluating Checkpoint at %s" % ckpt_path)
    # ckpt = torch.load(ckpt_path)
    # compatibility with ddp saved checkpoint when evaluating without ddp
    # pretrain_dict = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
    # model_dict = model.state_dict()
    # model_dict.update(pretrain_dict)
    # model.load_state_dict(ckpt, strict=False)
    model = torch.load(ckpt_path)
    if cfg.CUDA:
        model = model.cuda()

    model.eval()

    task1_metric, task2_metric = evaluate(test_loader, model, task1, task2, False)

    for k, v in task1_metric.items():
        print('{}: {:.9f}'.format(k, v))
    for k, v in task2_metric.items():
        print('{}: {:.9f}'.format(k, v))


if __name__ == '__main__':
    main()
