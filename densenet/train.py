from __future__ import division, print_function

import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import dataloader
from preprocess import preproc
from densenet import densenet121


# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')


training_csv = '../../data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'
training_data = '../../data/ISIC2018_Task3_Training_Input'
labels_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

training = dataloader(training_csv, training_data, preproc(), True, 0.01)
validation = dataloader(training_csv, training_data, preproc(), False, 0.01)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataloaders = {'train': training, 'val': validation}


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    batch_size = 2
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                # Iterate over data.
                batch_iterator = iter(DataLoader(dataloaders[phase], batch_size, shuffle=True, num_workers=0))
            else:
                model.eval()   # Set model to evaluate mode
                batch_iterator = iter(DataLoader(dataloaders[phase], batch_size, shuffle=True, num_workers=4))

            running_loss = 0.0
            running_corrects = 0

            
            # for images, labels in dataloaders[phase]:
            iteration = int(len(dataloaders[phase])/batch_size)
            for step in range(iteration):
                images, labels = next(batch_iterator)
                images = images.to(device)
                labels = labels.squeeze(1)
                labels = labels.to(device)
                
                writer.add_graph(model, images)
                writer.close()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)
                print(f'epoch {epoch} - step {step}/{iteration} - TrainingLoss {loss}')
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects.double() / len(dataloaders[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    model_ft = densenet121(pretrained=False)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss(weight=training.weights.to(device))

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=50)
