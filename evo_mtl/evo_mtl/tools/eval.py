import os
import argparse
import random
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F

# ad-hoc way to deal with python 3.7.4
import os, sys

import torch.nn as nn
from core.config import cfg

from core.models.connect_matrix import get_model

from core import tasks
from core.data.loader import MultiTaskDataset
import torch.optim as optim
import datetime
from tools import utils
from tools.evaluate import evaluate




class Evaluate:
    def __init__(self, pops, distributed=False, local_rank=None):
        self.pops = pops
        self.distributed = distributed
        self.local_rank = local_rank

    def parse_individual(self, indi):

        distributed = False
        gpus = [0,1]
        seg_task, normal_task = tasks.get_tasks(cfg=cfg)
        # 准备模型
        model = get_model(cfg, indi)
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
        # 准备数据和加载参数
        data_dir = '//home//jiewu//MTLNAS//datasets//nyu_v2'
        image_mean = '//nyu_v2_mean.npy'
        data_list_1 = '//list//training_seg.txt'
        data_list_2 = '//list//training_normal_mask.txt'
        test_data_list_1 = '//list//testing_seg.txt'
        test_data_list_2 = '//list//testing_normal_mask.txt'

        output_size = (321, 321)
        train_color_jitter = False
        train_random_scale = True
        train_random_mirror = True
        train_random_crop = True
        ignore_label = 255

        full_data = MultiTaskDataset(data_dir, image_mean, data_list_1, data_list_2, output_size, train_color_jitter,
                                     train_random_scale, train_random_mirror, train_random_crop, ignore_label)
        # print('The whole dataset length: {}'.format(len(full_data)))
        num_train = len(full_data)
        indices = list(range(num_train))
        split = int(np.floor(cfg.ARCH.TRAIN_SPLIT * num_train))

        test_data = MultiTaskDataset(data_dir, image_mean, test_data_list_1, test_data_list_2, None, None,
                                     False, False, False, ignore_label)

        if cfg.TRAIN.EVAL_CKPT:

            if distributed:
                test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
            else:
                test_sampler = None

            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, sampler=test_sampler)

        # if distributed:
        #     # Important: Double check if BN is working as expected
        #
        #     model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        #     model = MyDataParallel(
        #         model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        #     )

        # print('The test dataset length: {}'.format(len(test_data)))
        num_data = len(full_data)
        # 模型在训练过程中需要优化和更新的参数
        nddr_params = []
        fc8_weights = []
        fc8_bias = []
        base_params = []

        for k, v in model.module.named_net_parameters():
            if 'path' in k:
                nddr_params.append(v)
            elif model.module.net1.fc_id in k:
                if 'weight' in k:
                    fc8_weights.append(v)
                else:
                    assert 'bias' in k
                    fc8_bias.append(v)
            else:
                base_params.append(v)
        parameter_dict = [
            {'params': base_params},
            {'params': fc8_weights, 'lr': cfg.TRAIN.LR * cfg.TRAIN.FC8_WEIGHT_FACTOR},
            {'params': fc8_bias, 'lr': cfg.TRAIN.LR * cfg.TRAIN.FC8_BIAS_FACTOR},
            {'params': nddr_params, 'lr': cfg.TRAIN.LR * cfg.TRAIN.NDDR_FACTOR}
        ]

        # 训练
        optimizer = optim.SGD(parameter_dict, lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.SCHEDULE == 'Poly':
            if cfg.TRAIN.WARMUP > 0.:
                scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                        lambda step: min(1., float(step) / cfg.TRAIN.WARMUP) * (
                                                                1 - float(step) / cfg.TRAIN.STEPS) ** cfg.TRAIN.POWER,
                                                        last_epoch=-1)
            else:
                scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                        lambda step: (1 - float(
                                                            step) / cfg.TRAIN.STEPS) ** cfg.TRAIN.POWER,
                                                        last_epoch=-1)

        model.train()
        epoch_num = 20
        batch_size = 4
        # data_loader = torch.utils.data.DataLoader(
        #     full_data, batch_size=batch_size, shuffle=True,
        #     pin_memory=True)
        # test_loader = torch.utils.data.DataLoader(
        #     test_data, batch_size=batch_size, shuffle=False
        # )
        epoc = 1
        while epoc < epoch_num:

            train_data = torch.utils.data.Subset(full_data, indices[:split])
            val_data = torch.utils.data.Subset(full_data, indices[split:num_train])

            if distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                train_sampler = None
                val_sampler = None

            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=batch_size,
                pin_memory=True, sampler=train_sampler)

            val_loader = torch.utils.data.DataLoader(
                val_data, batch_size=batch_size,
                pin_memory=True, sampler=val_sampler)

            val_iter = iter(val_loader)

            if distributed:
                train_sampler.set_epoch(epoc)  # steps is used to seed RNG
                val_sampler.set_epoch(epoc)
            start = datetime.datetime.now()
            print("epoc:{},start_time{}".format(epoc, start))
            for step, (image, label_1, label_2) in enumerate(train_loader):
                if torch.cuda.is_available():
                    image, label_1, label_2 = image.cuda(), label_1.cuda(), label_2.cuda()
                optimizer.zero_grad()
                result = model.module.loss(image, (label_1, label_2))
                out1, out2 = result.out1, result.out2

                loss1 = result.loss1
                loss2 = result.loss2

                loss = result.loss
                loss = loss.cuda()
                loss.backward()
                optimizer.step()
                scheduler.step()
                # Print out the loss periodically.
                if step % 265 == 0:
                    print('Train Step: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss1: {:.6f}\tLoss2: {:.6f}'.format(
                        epoc, step * len(image), len(train_loader.dataset),
                              100. * step / len(train_loader), loss.data.item(),
                        loss1.data.item(), loss2.data.item()))

            if epoc % 10 == 0:
                if cfg.TRAIN.EVAL_CKPT:

                    model.eval()
                    path = os.getcwd() + '/out10.dat'

                    torch.cuda.empty_cache()
                    task1_metric, task2_metric = evaluate(val_loader, model, seg_task, normal_task)
                    indi.MIou = task1_metric['Mean IoU']
                    indi.PAcc = task1_metric['Pixel Acc']
                    indi.Mean = task2_metric['Mean']
                    indi.Median = task2_metric['Median']
                    indi.RMSE = task2_metric['RMSE']
                    indi.within_11 = task2_metric['11.25']
                    indi.within_22 = task2_metric['22.5']
                    indi.within_30 = task2_metric['30']
                    indi.within_45 = task2_metric['45']
                    for k, v in task1_metric.items():
                        string = '{}: {:.4f}'.format(k, v)
                        print(string)
                        with open(path, 'a') as myfile:
                            myfile.write(string)
                            myfile.write('\n')
                    for k, v in task2_metric.items():
                        string = '{}: {:.4f}'.format(k, v)
                        print(string)
                        with open(path, 'a') as myfile:
                            myfile.write(string)
                            myfile.write('\n')
                    model.train()
                    torch.cuda.empty_cache()

                # torch.save(model, os.path.join(cfg.SAVE_DIR, 'vgg_nyuv2_default',
                #                                'ckpt-%s.pth' % str(epoc).zfill(5)))
            if epoc >= epoch_num:
                break
            epoc = epoc + 1
            end = datetime.datetime.now()
            print("epoc:{},end_time{}".format(epoc, end))

    def parse_population(self, gen_no, evaluated_num):
        save_dir = os.getcwd() + '/save_data/gen_{:03d}'.format(gen_no)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        for i in range(evaluated_num, self.pops.get_pop_size()):
            indi = self.pops.get_individual_at(i)
            print('this is the {} generation the {} individual\n {}\n {}'.format(gen_no, i, indi.indi_net1,
                                                                                 indi.indi_net2))
            self.parse_individual(indi)
            list_save_path = os.getcwd() + '/save_data/gen_{:03d}/pop.txt'.format(gen_no)
            utils.save_append_individual(str(indi), list_save_path)
            utils.save_populations(gen_no, self.pops)
