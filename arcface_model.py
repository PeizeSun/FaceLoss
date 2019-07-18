# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

cfg = {
    'A': [0, 0, 0, 0],
    'C': [1, 2, 4, 1],
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class CosineLayer(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(CosineLayer, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.weight = Parameter(torch.Tensor(in_planes, out_planes))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, _input):
        '''
        :param _input: (B, F) , B is batch_size, F is feature_dim
        :return:
        '''
        x = _input                                          # (B, F)
        w = self.weight                                     # (F, C)
        xlen = x.pow(2).sum(1).pow(0.5)                     # (B)
        wlen = w.pow(2).sum(0).pow(0.5)                    # (C)

        inner_wx = x.mm(w)  # (B, C)
        cos_theta = inner_wx / xlen.view(-1, 1) / wlen.view(1, -1)  # (B, C)
        cos_theta = cos_theta.clamp(-1, 1)  # (B, C)

        output = (cos_theta, None)
        return output


class ArcMarginSoftmaxWithLoss(nn.Module):
    def __init__(self, s=30.0, m=0.50, easy_margin=False, gamma=0):
        super(ArcMarginSoftmaxWithLoss, self).__init__()
        self.gamma = gamma
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, _input, target):
        cos_theta, _ = _input               # (B, C)

        # cos(a+b)=cos(a)*cos(b)-size(a)*sin(b)
        # from IPython import embed; embed()

        sin_theta = torch.sqrt((1.0 - torch.pow(cos_theta, 2)).clamp(0, 1))
        phi_theta = cos_theta * self.cos_m - sin_theta * self.sin_m

        if self.easy_margin:
            phi_theta = torch.where(cos_theta > 0, phi_theta, cos_theta)
        else:
            phi_theta = torch.where(cos_theta > self.th, phi_theta, cos_theta - self.mm)

        target = target.view(-1, 1)                     # (B, 1)
        index = cos_theta.data * 0.0
        index.scatter_(1, target.data.view(-1, 1), 1)   # (B, C)

        output = index * phi_theta + (1 - index) * cos_theta
        output = self.s * output
        logit = F.log_softmax(output)
        logit = logit.gather(1, target).view(-1)

        pt = logit.data.exp()
        loss = -1 * (1 - pt) ** self.gamma * logit
        loss = loss.mean()

        return loss


class SphereFace(nn.Module):
    def __init__(self, block, layers, num_classes=10, feat_dim=4):
        '''

        :param block: residual units
        :param layers: number of residual units per stage
        :param num_classes:
        :param feat_dim:
        '''
        super(SphereFace, self).__init__()
        self.conv1 = conv3x3(1, 64, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(64, 128, 2)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = conv3x3(128, 256, 2)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = conv3x3(256, 512, 2)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, nr_blocks=layers[0])
        self.layer2 = self._make_layer(block, 128, nr_blocks=layers[1])
        self.layer3 = self._make_layer(block, 256, nr_blocks=layers[2])
        self.layer4 = self._make_layer(block, 512, nr_blocks=layers[3])
        self.fc5 = nn.Linear(512 * 2 * 2, 512)
        self.fc6 = nn.Linear(512, feat_dim)
        self.fc7 = CosineLayer(feat_dim, num_classes)

    def _make_layer(self, block, planes, nr_blocks, stride=1):
        if nr_blocks != 0:
            layers = list()
            for _ in range(0, nr_blocks):
                downsample = nn.Sequential(
                    conv1x1(planes, planes, stride),
                    nn.BatchNorm2d(planes),
                )
                layers.append(block(planes, planes, stride, downsample))
            return nn.Sequential(*layers)
        else:
            return None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        if self.layer1 is not None:
            x = self.layer1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        if self.layer2 is not None:
            x = self.layer2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        if self.layer3 is not None:
            x = self.layer3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        if self.layer4 is not None:
            x = self.layer4(x)

        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.fc6(x)
        y = self.fc7(x)

        return x, y


def ArcSphereFace4(**kwargs):
    '''
    Constructs a SphereFace4 model
    :param kwargs:
    :return:
    '''
    model = SphereFace(BasicBlock, cfg['A'], **kwargs)

    return model


def ArcSphereFace20(**kwargs):
    '''
    Constructs a SphereFace4 model
    :param kwargs:
    :return:
    '''
    model = SphereFace(BasicBlock, cfg['C'], **kwargs)

    return model

