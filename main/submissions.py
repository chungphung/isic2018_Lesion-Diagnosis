import os
import time
from os.path import basename

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# from ArcMarginModel import ArcMarginModel
from dataset import dataloader
from preprocess import preproc, randaugment_preproc

import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# data and model paths
test_data = '../../data/ISIC2018_Task3_Test_Input'
model_path = './weights/efficientnet_b2_ArcMargin_2022-02-15_3-35_epoch84.tar'

labels_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# dataloader
test_loader = dataloader(None, test_data, preproc(), 'test')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def summision_generate(model, batch_size, arccos):

    result = {}
    since = time.time()

    print('-' * 10)

    model.eval()   # Set model to evaluate mode
    batch_iterator = iter(DataLoader(
        test_loader, batch_size, shuffle=False, num_workers=4))

    # for images, labels in test_loader:
    iteration = int(len(test_loader)/batch_size)
    for step in tqdm(range(iteration), desc="Running..."):
        images, labels = next(batch_iterator)

        images = images.to(device)

        # run predictions
        with torch.set_grad_enabled(False):
            outputs = model(images)
            if arccos != None:
                # outputs = arc_margin(outputs, None, labels, phase='test')
                outputs = arc_margin(outputs, labels, phase='test')
            preds = torch.softmax(outputs, dim=1)
            onehot = np.eye(7)[torch.argmax(preds.cpu(), dim=1)]

        for i in range(batch_size):
            if 'image' not in result.keys():
                result['image'] = [
                    basename(test_loader.imgs_path[step*batch_size:(step+1)*batch_size][i])[:-4]]
            else:
                result['image'].append(
                    basename(test_loader.imgs_path[step*batch_size:(step+1)*batch_size][i])[:-4])
            for j in range(7):
                if labels_names[j] not in result.keys():
                    result[labels_names[j]] = [onehot[i][j]]
                else:
                    result[labels_names[j]].append(onehot[i][j])
    keys = list(result.keys())
    df = pd.DataFrame(result[keys[0]], columns=[keys[0]])
    for k in keys[1:]:
        df[k] = result[k]

    # saving the dataframe
    df.to_csv(
        f'./submissions/{basename(model_path)[:-4]}_nocentercrop.csv', index=False)

    time_elapsed = time.time() - since
    print('Runnning complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    model_ft = torch.load(model_path)
    model = model_ft['model'].to(device)
    import torch.nn as nn 
    model.global_pool.flatten = nn.Flatten(1)
    arc_margin = model_ft['arccos'].to(device)
    summision_generate(model, batch_size=12, arccos=arc_margin)
