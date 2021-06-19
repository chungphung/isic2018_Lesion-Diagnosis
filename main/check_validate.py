import os
import time
from os.path import basename

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import dataloader
from preprocess import preproc
import seaborn as sn
import pandas as pd
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# data and model paths
validate_csv = './main/val.csv'
data = './data/ISIC2018_Task3_Training_Input'

models = os.listdir('./weights')

scores = models
labels_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# dataloader
validation = dataloader(validate_csv, data, preproc(), 'validate')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _confusion_matrix(model_name, score, y_true, y_pred):
    cfs = confusion_matrix(y_true, y_pred)
    cfs.dtype = np.float
    for i in range(len(cfs)):
        cfs[i] = cfs[i]/sum(cfs[i])
    df_cm = pd.DataFrame(cfs, index=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'], columns=[
                         'pMEL', 'pNV', 'pBCC', 'pAKIEC', 'pBKL', 'pDF', 'pVASC'])
    plt.figure(figsize=(10, 7))
    sns_plot = sn.heatmap(df_cm, annot=True, square=True,
                          cmap="YlGnBu").set_title(f'Score: {score}')
    fig = sns_plot.get_figure()
    fig.savefig(f"./confusion_matrix/{model_name}.png", dpi=400)


def check(model, model_name, score, batch_size):

    since = time.time()

    print('-' * 10)

    model.eval()   # Set model to evaluate mode
    batch_iterator = iter(DataLoader(
        validation, batch_size, shuffle=False, num_workers=20))
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
            outputs = arc_margin(outputs, labels, phase='test')
            onehot = torch.argmax(outputs.cpu(), dim=1).tolist()

        y_pred += onehot
    _confusion_matrix(model_name, score, y_true, y_pred)


if __name__ == "__main__":
    for s, m in zip(scores, models):
        if '.ipynb_checkpoints' in m:
            continue
        print(m)
        model_path = f'./weights/{m}'
        model_name = basename(model_path)[:-4]

        model_ft = torch.load(model_path)
        model = model_ft['model'].to(device)
        arc_margin = model_ft['arccos'].to(device)

        check(model, model_name, s, batch_size=12)
