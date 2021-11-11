import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Stage类：初始化参数：输出通道，层
作用：将传入的list类型的“层”，顺序组合起来，并形成前向传播
使用时，实例化一个个体，个体输入为某个x
'''
class Stage(nn.Module):
    def __init__(self, out_channels, layers):
        super(Stage, self).__init__()
        if isinstance(layers, list):
            self.feature = nn.Sequential(*layers)
        else:
            self.feature = layers
        self.out_channels = out_channels
        
    def forward(self, x):
        return self.feature(x)
    
'''
batch_norm函数：初始化参数：通道个数,eps,momentum
作用初始化BN层参数，并返回BN层输出的结果
'''
def batch_norm(num_features, eps=1e-3, momentum=0.05):
    bn = nn.BatchNorm2d(num_features, eps, momentum)
    nn.init.constant_(bn.weight, 1)
    nn.init.constant_(bn.bias, 0)
    return bn

'''
函数get_nddr_bn：初始化参数，cfg
作用：输出BN层内容
'''
def get_nddr_bn(cfg):
    if cfg.MODEL.NDDR_BN_TYPE == 'default':
        return lambda width: batch_norm(width, eps=1e-03, momentum=cfg.MODEL.BATCH_NORM_MOMENTUM)
    else:
        raise NotImplementedError
'''
函数get_nddr:输入参数：cfg, in_Channel, out_channel
'''
def get_nddr(cfg, in_channels, out_channels):
    return SingleSidedAsymmetricNDDR(cfg, in_channels, out_channels)




class NDDR(nn.Module):
    def __init__(self, cfg, out_channels):
        super(NDDR, self).__init__()
        init_weights = cfg.MODEL.INIT
        norm = get_nddr_bn()
        
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)
        
        # Initialize weight
        if len(init_weights):
            self.conv1.weight = nn.Parameter(torch.cat([
                torch.eye(out_channels) * init_weights[0],
                torch.eye(out_channels) * init_weights[1]
            ], dim=1).view(out_channels, -1, 1, 1))
            self.conv2.weight = nn.Parameter(torch.cat([
                torch.eye(out_channels) * init_weights[1],
                torch.eye(out_channels) * init_weights[0]
            ], dim=1).view(out_channels, -1, 1, 1))
        else:
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

        self.activation = nn.ReLU()

        self.bn1 = norm(out_channels)
        self.bn2 = norm(out_channels)

    def forward(self, feature1, feature2):
        x = torch.cat([feature1, feature2], 1)#拼接
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out1 = self.bn1(out1)
        out2 = self.bn2(out2)
        out1 = self.activation(out1)
        out2 = self.activation(out2)
        return out1, out2

'''
singlesidenddr类：cfg， outchannels, reverse 

'''
# class SingleSideNDDR(nn.Module):
#     def __init__(self, cfg, out_channels, reverse):
#         """
#         Net1 is main task, net2 is aux task
#         """
#         super(SingleSideNDDR, self).__init__()
#         init_weights = cfg.MODEL.INIT
#         norm = get_nddr_bn(cfg)
#
#         self.conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)
#
#         # Initialize weight
#         if len(init_weights):
#             self.conv.weight = nn.Parameter(torch.cat([
#                 torch.eye(out_channels) * init_weights[0],
#                 torch.eye(out_channels) * init_weights[1]
#             ], dim=1).view(out_channels, -1, 1, 1))
#         else:
#             nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
#
#         self.activation = nn.ReLU()
#
#         self.bn = norm(out_channels)
#
#         self.reverse = reverse
#
#     def forward(self, feature1, feature2):
#         if self.reverse:
#             out2 = feature2
#             out1 = torch.cat([feature1, feature2], 1)
#             out1 = self.conv(out1)
#             out1 = self.bn(out1)
#         else:
#             out1 = feature1
#             out2 = torch.cat([feature2, feature1], 1)
#             out2 = self.conv(out2)
#             out2 = self.bn(out2)
#         return out1, out2

    
    
class SingleSidedAsymmetricNDDR(nn.Module):
    def __init__(self, cfg, in_channels, out_channels):
        super(SingleSidedAsymmetricNDDR, self).__init__()
        init_weights = cfg.MODEL.INIT
        norm = get_nddr_bn(cfg)
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        assert in_channels >= out_channels
        # check if out_channel divides in_channels
        assert in_channels % out_channels == 0
        multipiler = in_channels / out_channels - 1
        
        # Initialize weight
        if len(init_weights):
            weight = [torch.eye(out_channels) * init_weights[0]] +\
                 [torch.eye(out_channels) * init_weights[1] / float(multipiler) for _ in range(int(multipiler))]
            self.conv.weight = nn.Parameter(torch.cat(weight, dim=1).view(out_channels, -1, 1, 1))
        else:
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        
        self.activation = nn.ReLU()
        self.bn = norm(out_channels)
        nn.init.constant_(self.bn.weight, 1.)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, features):
        """

        :param features: upstream feature maps
        :return:
        """
        x = torch.cat(features, 1)
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out
