import math
from collections import OrderedDict

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


class BasicCNN(nn.Module):
    def __init__(self, num_init_features=64):

        super(BasicCNN, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

            ('conv1', nn.Conv2d(64, 128, kernel_size=5, stride=2,
                                padding=2, bias=False)),
            ('norm1', nn.BatchNorm2d(128)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

            ('conv2', nn.Conv2d(128, 128, kernel_size=5, stride=2,
                                padding=2, bias=False)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))


        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class ArcMarginModel_AutoMargin(nn.Module):
    def __init__(self, device, s=30.0):
        super(ArcMarginModel, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, 64, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

            ('conv1', nn.Conv2d(64, 128, kernel_size=5, stride=2,
                                padding=2, bias=False)),
            ('norm1', nn.BatchNorm2d(128)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

            ('conv2', nn.Conv2d(128, 128, kernel_size=5, stride=2,
                                padding=2, bias=False)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))


        self.classifier = nn.Linear(128, 1)

        emb_size = 512
        num_classes = 7
        self.device = device
        self.weight = Parameter(torch.FloatTensor(num_classes, emb_size))
        nn.init.xavier_uniform_(self.weight)

        self.m = m
        self.s = s
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, vector, x, label, phase='train'):
        x = F.normalize(vector)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        if phase == 'train':
            
            # calculate margin 
            features = self.features(x)
            feature_out = F.relu(features, inplace=True)
            feature_out = F.adaptive_avg_pool2d(feature_out, (1, 1))
            feature_out = torch.flatten(feature_out, 1)
            self.m = self.classifier(feature_out)
            ##################

            self.cos_m = math.cos(self.m)
            self.sin_m = math.sin(self.m)
            self.th = math.cos(math.pi - self.m)
            self.mm = math.sin(math.pi - self.m) * self.m

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



from preprocess import preproc
from torch.utils.data import DataLoader
from dataset import dataloader
def main():
    training_csv = './main/train.csv'
    data = '../../data/ISIC2018_Task3_Training_Input'
    training = dataloader(training_csv, data, preproc(), 'training')
    batch_iterator = iter(DataLoader(
                    training, 1, shuffle=True, num_workers=0))
    iteration = int(len(training)/1)
    model = BasicCNN()
    model.train() 
    model = model.to("cuda:0")
            
    for step in range(iteration):
        images, labels = next(batch_iterator)
        images = images.to("cuda:0")
        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            outputs = model(images)
        


if __name__ == "__main__":
    main()