from __future__ import division, print_function

import copy
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

from dataset import dataloader
from densenet import densenet121, densenet201
from preprocess import preproc
from ArcMarginModel import ArcMarginModel
from FocalLoss import FocalLoss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


training_csv = './densenet/train.csv'
validate_csv = './densenet/val.csv'
data = './data/ISIC2018_Task3_Training_Input'
labels_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

training = dataloader(training_csv, data, preproc(), 'training')
validation = dataloader(validate_csv, data, preproc(), 'validate')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataloaders = {'train': training, 'val': validation}


def visualizing(phase, epoch, step, epoch_loss, epoch_acc):
    # training visualizing
    if epoch == 0 and step == 0:
        writer.add_scalar(f'{phase} loss', epoch_loss, 0)
        writer.add_scalar(f'{phase} accuracy', 0, 0)
    ######################
    else:
        writer.add_scalar(f'{phase} loss',
                          epoch_loss,
                          epoch * len(dataloaders[phase]) + len(dataloaders[phase]))
        writer.add_scalar(f'{phase} accuracy',
                          epoch_acc,
                          epoch * len(dataloaders[phase]) + len(dataloaders[phase]))
    ######################


def train_model(model, criterion, optimizer, scheduler, writer, model_name, batch_size, arccos=None, num_epochs=25):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    lowest_val_loss = 100.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train_loss = 0.0
        val_loss = 0.0
        train_correct = 0
        val_correct = 0
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                # Iterate over data.
                batch_iterator = iter(DataLoader(
                    dataloaders[phase], batch_size, shuffle=True, num_workers=20))
            else:
                model.eval()   # Set model to evaluate mode
                batch_iterator = iter(DataLoader(
                    dataloaders[phase], batch_size, shuffle=False, num_workers=20))

            # for images, labels in dataloaders[phase]:
            iteration = int(len(dataloaders[phase])/batch_size)
            for step in range(iteration):
                images, labels = next(batch_iterator)
                images = images.to(device)
                labels = labels.squeeze(1)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    if arccos is not None:
                        outputs = arc_margin(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # training visualizing
                    if epoch == 0 and step == 0:
                        writer.add_graph(model, images[0].unsqueeze(0))
                        visualizing(phase, epoch, step, loss, 0)
                    ######################

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                if phase == 'train':
                    train_loss += loss.item() * images.size(0)
                    train_correct += torch.sum(preds == labels.data)
                    print(
                        f'epoch {epoch} - step {step}/{iteration} - TrainingLoss {loss}')
                else:
                    val_loss += loss.item() * images.size(0)
                    val_correct += torch.sum(preds == labels.data)
                    print(
                        f'epoch {epoch} - step {step}/{iteration} - ValidateLoss {loss}')

            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
            if phase == 'train':
                epoch_loss = train_loss / len(dataloaders[phase])
                epoch_acc = train_correct.double() / len(dataloaders[phase])
            else:
                epoch_loss = val_loss / len(dataloaders[phase])
                epoch_acc = val_correct.double() / len(dataloaders[phase])

            # training visualizing
            visualizing(phase, epoch, step, epoch_loss, epoch_acc)
            ######################

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss < lowest_val_loss:
                lowest_val_loss = epoch_loss
                best_model = copy.deepcopy(model)
                if arccos is not None:
                    best_arc_margin = copy.deepcopy(arc_margin)
                best_epoch = epoch

        # save full model last epoch
        if epoch == num_epochs-1:
            if arccos is None:
                torch.save(model, f'./weights/{model_name}_epoch{epoch}.pth')
            else:
                torch.save({'model': model, 'arccos': arc_margin, 'optimizer': optimizer},
                           f'./weights/{model_name}_epoch{epoch}.tar')

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(lowest_val_loss))
        print(f'Best epoch: {best_epoch}')

    # save best model
    if arccos is None:
        torch.save(best_model, f'./weights/{model_name}_epoch{best_epoch}.pth')
    else:
        torch.save({'model': best_model, 'arccos': best_arc_margin,
                    'optimizer': optimizer}, f'./weights/{model_name}_epoch{best_epoch}.tar')
    return model


if __name__ == "__main__":
    now = datetime.now()
    arccos = True
    if arccos:
        model_name = f'densenet121_ArcMargin_{now.date()}_{now.hour}-{now.minute}'
    else:
        model_name = f'densenet121_AutoWtdCE_{now.date()}_{now.hour}-{now.minute}'

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(f'runs/{model_name}')

    model_ft = densenet121(pretrained=True)
    num_ftrs = model_ft.classifier.in_features

    if arccos:
        model_ft.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512), nn.ReLU())
        model_ft = model_ft.to(device)
        arc_margin = ArcMarginModel(device, m=0.1, s=3.0).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.SGD([{'params': model_ft.parameters()}, {
                                 'params': arc_margin.parameters()}], lr=0.01, momentum=0.9)
    else:
        model_ft.classifier = nn.Linear(num_ftrs, 7)
        model_ft = model_ft.to(device)
        criterion = nn.CrossEntropyLoss(weight=training.weights.to(device))
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        arccos = None

    # Observe that all parameters are being optimized

    # Decay LR by a factor of 0.1 every 7 epochs
#     exp_lr_scheduler = lr_scheduler.StepLR(
#         optimizer_ft, step_size=10, gamma=0.1)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(
        optimizer_ft, milestones=[43], gamma=0.1)
#     exp_lr_scheduler = None
    model_ft = train_model(model_ft, criterion, optimizer_ft,
                           exp_lr_scheduler, writer, model_name, batch_size=32, arccos=arccos, num_epochs=100)
