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
#准备任务
seg_task, normal_task = tasks.get_tasks(cfg=cfg)
#准备模型
model = get_model(cfg, seg_task, normal_task)
#准备数据和加载参数
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
print('The whole dataset length: {}'.format(len(full_data)))
test_data = MultiTaskDataset(data_dir, image_mean, test_data_list_1, test_data_list_2, None, None,
                            False, False, False, ignore_label)
print('The test dataset length: {}'.format(len(test_data)))
num_data = len(full_data)
indices = list(range(num_data))
split = int(np.floor(0.5 * num_data))  # 训练数据的划分
#train_data = torch.utils.data.Subset(data, indices[:split])
#val_data = torch.utils.data.Subset(data, indices[split:num_data])
#print('train_dataset_length:{}'.format(len(train_data)))
#print('val_dataset_length:{}'.format(len(val_data)))
#模型在训练过程中需要优化和更新的参数
nddr_params = []
fc8_weights = []
fc8_bias = []
base_params = []

for k, v in model.named_net_parameters():
    if 'path' in k:
        nddr_params.append(v)
    elif model.net1.fc_id in k:
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



#训练
optimizer = optim.SGD(parameter_dict, lr=cfg.TRAIN.LR,momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
if cfg.TRAIN.SCHEDULE == 'Poly':
    if cfg.TRAIN.WARMUP > 0.:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                lambda step: min(1., float(step) / cfg.TRAIN.WARMUP) * (
                                                            1 - float(step) / cfg.TRAIN.STEPS) ** cfg.TRAIN.POWER,
                                                last_epoch=-1)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                lambda step: (1 - float(step) / cfg.TRAIN.STEPS) ** cfg.TRAIN.POWER,
                                                last_epoch=-1)
model.cuda()
model.train()
# criterion =
epoch_num = 91

batch_size = 3
# train_loader = torch.utils.data.DataLoader(
#         train_data, batch_size=batch_size,
#         pin_memory=True)
# val_loader = torch.utils.data.DataLoader(
#         train_data, batch_size=batch_size,
#         pin_memory=True)
# test_loader = torch.utils.data.DataLoader(
#     test_data, batch_size=batch_size, shuffle=False
# )
# train_iter = iter(train_loader)
data_loader = torch.utils.data.DataLoader(
        full_data, batch_size=batch_size,shuffle=True,
        pin_memory=True)
test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )
train_iter = iter(data_loader)
epoc = 1
while epoc < epoch_num:
    start = datetime.datetime.now()
    print("epoc:{},start_time{}".format(epoc, start))
    # train_data = torch.utils.data.Subset(data, indices[:split])
    # val_data = torch.utils.data.Subset(data, indices[split:num_data])

    # val_loader = torch.utils.data.DataLoader(
    #     train_data, batch_size=batch_size,
    #     pin_memory=True)


    for step, (image, label_1, label_2) in enumerate(data_loader):
        if torch.cuda.is_available():
            image, label_1, label_2 = image.cuda(), label_1.cuda(), label_2.cuda()
        optimizer.zero_grad()
        result = model.loss(image, (label_1, label_2))
        out1, out2 = result.out1, result.out2

        loss1 = result.loss1
        loss2 = result.loss2

        loss = result.loss
        # if epoc%10==0:
        #     print(loss.data.item(), loss1.data.item(), loss2.data.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        # Print out the loss periodically.
        if step % 265 == 0:
            print('Train Step: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss1: {:.6f}\tLoss2: {:.6f}'.format(
                epoc, step * len(image), len(data_loader.dataset),
                    100. * step / len(data_loader), loss.data.item(),
                loss1.data.item(), loss2.data.item()))

    if epoc % 2== 0:
        # checkpoint = {
        #             'cfg': cfg,
        #             'epoc': epoc,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'loss': loss,
        #             'loss1': loss1,
        #             'loss2': loss2,
        #             'task1_metric': None,
        #             'task2_metric': None,
        # }

        if cfg.TRAIN.EVAL_CKPT:

                model.eval()
                path = os.getcwd()+'/out10.dat'

                torch.cuda.empty_cache()
                task1_metric, task2_metric = evaluate(test_loader, model, seg_task, normal_task)
                # for k, v in task1_metric.items():
                #     writer.add_scalar('eval/{}'.format(k), v, steps)
                # for k, v in task2_metric.items():
                #     writer.add_scalar('eval/{}'.format(k), v, steps)
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


                # checkpoint['task1_metric'] = task1_metric
                # checkpoint['task2_metric'] = task2_metric
                model.train()
                torch.cuda.empty_cache()

        torch.save(model, os.path.join(cfg.SAVE_DIR, 'vgg_nyuv2_default',
                                            'ckpt-%s.pth' % str(epoc).zfill(5)))
    if epoc >= epoch_num:
        break
    epoc = epoc+1
    end = datetime.datetime.now()
    print("epoc:{},end_time{}".format(epoc, end))