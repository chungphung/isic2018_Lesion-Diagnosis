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
from densenet import densenet121
from preprocess import preproc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


training_csv = '../../data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'
training_data = '../../data/ISIC2018_Task3_Training_Input'
labels_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

training = dataloader(training_csv, training_data, preproc(), 'training', 0.8)
validation = dataloader(training_csv, training_data,
                        preproc(), 'validate', 0.2)

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
        print(f'{phase} loss  ', str(
            epoch * len(dataloaders[phase]) + len(dataloaders[phase])))

        writer.add_scalar(f'{phase} accuracy',
                          epoch_acc,
                          epoch * len(dataloaders[phase]) + len(dataloaders[phase]))
    ######################


def train_model(model, criterion, optimizer, scheduler, writer, model_name, batch_size, num_epochs=25):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    lowest_val_loss = 10.0

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
                    dataloaders[phase], batch_size, shuffle=True, num_workers=4))
            else:
                model.eval()   # Set model to evaluate mode
                batch_iterator = iter(DataLoader(
                    dataloaders[phase], batch_size, shuffle=True, num_workers=4))

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
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # training visualizing
                    if epoch == 0 and step == 0:
                        writer.add_graph(model, images[0].unsqueeze(0))
                        visualizing(phase, epoch, step, loss, 0)
                    ######################

                    # backward + optimize only if in training phase
                    if phase == 'train':
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
                best_model_wts = copy.deepcopy(model.state_dict())

        # save weight every epoch
        torch.save(model.state_dict(),
                   f'./weights/state_{model_name}_epoch{epoch}.pth')
        # save full model every epoch
        torch.save(model, f'./weights/full_{model_name}_epoch{epoch}.pth')

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(lowest_val_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/densenet_crossentropy_weighted_auto')

    now = datetime.now()
    model_name = f'densenet121_AutoWtdCE_{now.date()}_{now.hour}-{now.minute}'

    model_ft = densenet121(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.classifier = nn.Linear(num_ftrs, 7)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss(weight=training.weights.to(device))

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft,
                           exp_lr_scheduler, writer, model_name, batch_size=12, num_epochs=50)
