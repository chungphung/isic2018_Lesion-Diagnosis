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

from dataset import dataloader
from balance_dataloader import BalancedBatchSampler
from densenet import densenet121, densenet201, densenet161
from preprocess import preproc
from ArcMarginModel import ArcMarginModel
from FocalLoss import FocalLoss
import timm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


training_csv = './main/train.csv'
validate_csv = './main/val.csv'
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


def train_model(model, criterion, optimizer, scheduler, writer, model_name, batch_size, num_epochs=25):

    since = time.time()
    lowest_val_loss = 100.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train_loss = 0.0
        val_loss = 0.0
        train_correct = 0
        val_correct = 0
        # Each epoch has a training and validation phase
        count = 0
        lowest_train_loss = 9999
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                # Iterate over data.
                batch_iterator = iter(DataLoader(
                    dataloaders[phase], batch_size, shuffle=True, num_workers=8))
                # balanced_batch_sampler = BalancedBatchSampler(training, training_csv, 4, batch_size)
                # batch_iterator = iter(DataLoader(
                #     dataloaders[phase], batch_sampler=balanced_batch_sampler, num_workers=20))

            else:
                model.eval()   # Set model to evaluate mode
                batch_iterator = iter(DataLoader(
                    dataloaders[phase], batch_size, shuffle=False, num_workers=8))

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
                    outputs = arc_margin(outputs, images, labels)
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
                
                if lowest_train_loss <= epoch_loss:
                    count+=1
                else:
                    lowest_train_loss = epoch_loss
                    
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
                best_arc_margin = copy.deepcopy(arc_margin)
                best_epoch = epoch

        # save full model last epoch
        if epoch == num_epochs-1:
            torch.save({'model': model, 'arccos': arc_margin, 'optimizer': optimizer}, f'./weights/{model_name}_epoch{epoch}.tar')

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(lowest_val_loss))
        print(f'Best epoch: {best_epoch}')

    # save best model
    torch.save({'model': best_model, 'arccos': best_arc_margin, 'batch_size': batch_size,
                'optimizer': optimizer}, f'./weights/{model_name}_epoch{best_epoch}.tar')
    return model


if __name__ == "__main__":
    
    cont = False
    
    now = datetime.now()
    model_name = f'ViT_ArcMargin_{now.date()}_{now.hour}-{now.minute}'

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(f'runs/{model_name}')
    
    if not cont:
        model_ft = timm.create_model(
        'vit_base_patch16_224', pretrained=True)
        num_ftrs = model_ft.head.in_features

        model_ft.head = nn.Sequential(nn.Dropout(0.2),
            nn.Linear(num_ftrs, 512), nn.ReLU())
        model_ft = model_ft.to(device)
        arc_margin = ArcMarginModel(device, m=0.1, s=5.0).to(device)
    else:
        model = torch.load('./weights/efficientnet_b2_ArcMargin_2021-09-25_20-27_epoch30.tar')
        model_ft = model['model'].to(device)
        arc_margin = model['arccos'].to(device)
    criterion = nn.CrossEntropyLoss()
#     criterion = FocalLoss()
    
    optimizer_ft = optim.SGD([{'params': model_ft.parameters()}, {
                                'params': arc_margin.parameters(), 'weight_decay': 1e-3}], lr=0.001, momentum=0.9)


    # Observe that all parameters are being optimized
    exp_lr_scheduler = lr_scheduler.MultiStepLR(
        optimizer_ft, milestones=[25,40], gamma=0.1)
    
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, writer, model_name, batch_size=32, num_epochs=100)