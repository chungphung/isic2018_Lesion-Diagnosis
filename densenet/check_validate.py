import os
import time
from os.path import basename

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import dataloader
from preprocess import preproc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# data and model paths
validate_csv = './Densenet/val.csv'
data = '../../data/ISIC2018_Task3_Training_Input'

model_path = './weights/full_densenet121_AutoWtdCE_2020-12-05_19-46_epoch49.pth'

labels_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# dataloader
validation = dataloader(validate_csv, data, preproc(), 'validate')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _confusion_matrix(y_true, y_pred):
    cfs = confusion_matrix(y_true, y_pred)
    print('{:10}\t{:6}\t{:6}\t{:6}\t{:6}\t{:6}\t{:6}\t{:6}'.format(
        'true\pred', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'))
    for i in range(7):
        tmp = '{:10}\t'.format(labels_names[i])
        row = cfs[i]
        for r in row:
            tmp += '{:<6}\t'.format(r)
        print(tmp)


def check(model, batch_size):

    result = {}
    since = time.time()

    print('-' * 10)

    model.eval()   # Set model to evaluate mode
    batch_iterator = iter(DataLoader(
        validation, batch_size, shuffle=False, num_workers=4))
    y_true = []
    y_pred = []
    # for images, labels in validation:
    iteration = int(len(validation)/batch_size)
    for step in tqdm(range(iteration), desc="Running..."):
        images, labels = next(batch_iterator)
        y_true += labels.squeeze(1).tolist()

        images = images.to(device)

        # run predictions
        with torch.set_grad_enabled(False):
            outputs = model(images)
            onehot = torch.argmax(outputs.cpu(), dim=1).tolist()

        y_pred += onehot
    _confusion_matrix(y_true, y_pred)

if __name__ == "__main__":
    model = torch.load(model_path)
    model = model.to(device)

    check(model, batch_size=12)
