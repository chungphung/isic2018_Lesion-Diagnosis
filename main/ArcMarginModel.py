import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import (AdaptiveAvgPool2d, AvgPool2d, BatchNorm1d, BatchNorm2d,
                      Conv2d, Dropout, Dropout2d, Linear, MaxPool2d, Module,
                      Parameter, PReLU, ReLU, Sequential, Sigmoid)


class ArcMarginModel(nn.Module):
    def __init__(self, device, m=0.8, s=30.0):
        super(ArcMarginModel, self).__init__()

        emb_size = 512
        num_classes = 7
        self.device = device
        self.weight = Parameter(torch.FloatTensor(num_classes, emb_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = False
        self.m = m
        self.prev_m = m
        self.s = s
        self.prev_s = s 
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label, phase='train'):

        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        if phase == 'train':
            if self.m!=self.prev_m or self.s!=self.prev_s:
                self.cos_m = math.cos(self.m)
                self.sin_m = math.sin(self.m)
                self.th = math.cos(math.pi - self.m)
                self.mm = math.sin(math.pi - self.m) * self.m
                self.prev_s = self.s 
                self.prev_m = self.m
            phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            one_hot = torch.zeros(cosine.size(), device=self.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
        else:
            phi = cosine * self.cos_m + sine * self.sin_m  # cos(theta - m)
            output = phi*self.s
        return output
