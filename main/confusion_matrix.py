import os
import time
from os.path import basename

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ArcMarginModel import ArcMarginModel
from dataset import dataloader
from preprocess import preproc

import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

labels_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# data and model paths
validate_csv = './main/val.csv'
data = '../../data/ISIC2018_Task3_Training_Input'
model_path = './weights/densenet121_ArcMargin_2021-01-22_1-12_epoch99.tar'
# dataloader
test_loader = dataloader(validate_csv, data, preproc(), 'validate')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def confusion(model, batch_size, arccos=None, voting=True):
    since = time.time()

    print('-' * 10)

    model.eval()   # Set model to evaluate mode
    batch_iterator = iter(DataLoader(
        test_loader, batch_size, shuffle=False, num_workers=4))

    # for images, labels in test_loader:
    iteration = int(len(test_loader)/batch_size)
    ret = {}
    y_true = []
    y_pred = []
    for step in tqdm(range(iteration), desc="Running..."):
        images, labels = next(batch_iterator)
        images = images.to(device)

        # run predictions
        with torch.set_grad_enabled(False):
            outputs = model(images)
            if arccos != None:
                outputs = arc_margin(outputs, labels, phase='test')
            preds = torch.argmax(torch.softmax(outputs, dim=1).cpu(), dim=1)

        for i in range(batch_size):
            y_pred.append(int(preds[i]))
            y_true.append(int(labels[i]))
    

    cfs = confusion_matrix(y_true, y_pred)
    tmp = '{:^6}{:^6}{:^6}{:^6}{:^6}{:^6}{:^6}{:^6}\n'.format(' ',0,1,2,3,4,5,6)
    for i in range(7):
        tmp+='{:^6}'.format('true_' + str(i))
        for j in range(7):
            tmp+='{:^6}'.format(cfs[i][j])
        tmp+='\n'
    print('\nconfusion: \n')
    print(tmp)
    print('\nclassification_report:\n')
    print(classification_report(y_true, y_pred, labels=[0,1,2,3,4,5,6]))

    time_elapsed = time.time() - since
    print('Runnning complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

if __name__ == "__main__":
    arccos = True
    print(model_path)
    if arccos:
        model_ft = torch.load(model_path)
        model = model_ft['model'].to(device)
        arc_margin = model_ft['arccos'].to(device)
    else:
        model = torch.load(model_path)
        model = model.to(device)
        arc_margin = None
    confusion(model, batch_size=24, arccos=arc_margin, voting=False)
