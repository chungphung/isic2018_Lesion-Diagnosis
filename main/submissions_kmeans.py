import os
import time
import warnings
from os.path import basename

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch.serialization import SourceChangeWarning
from torch.utils.data import DataLoader
from tqdm import tqdm

# from ArcMarginModel import ArcMarginModel
from dataset import dataloader
from preprocess import preproc
import pickle

warnings.filterwarnings("ignore", category=SourceChangeWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# data and model paths
test_data = '../../data/ISIC2018_Task3_Training_Input'
model_path = './weights/densenet121_ArcMargin_2021-01-22_1-12_epoch99.tar'

labels_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# dataloader
test_loader = dataloader(None, test_data, preproc(), 'test')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def full_crop(image):
    img_t = []
    for r in range(0, 4):
        for c in range(0, 5):
            img_t.append(image[:, 50*r:50*r+300, 50*c:50*c+400])
    return img_t


def summision_generate(model, batch_size, arccos=None, voting=True):

    result = {}
    since = time.time()

    print('-' * 10)

    model.eval()   # Set model to evaluate mode
    batch_iterator = iter(DataLoader(
        test_loader, batch_size, shuffle=False, num_workers=4))

    # for images, labels in test_loader:
    iteration = int(len(test_loader)/batch_size)
    X = []
    ref_dict = {}
    names = []
    for step in tqdm(range(iteration), desc="Running..."):
        images, labels = next(batch_iterator)
        if voting:
            img_t = full_crop(images[0].numpy())
            images = torch.tensor(img_t)

        images = images.to(device)

        # run predictions
        with torch.set_grad_enabled(False):
            outputs = model(images)
            X+=outputs.cpu().tolist()
            names+=list(labels[0])
    for i in range(len(names)):
        ref_dict[names[i]] = X[i]
    with open('ref_dict.pickle', 'wb') as f:
        pickle.dump(ref_dict, f)
    # kmeans = KMeans(n_clusters=7, random_state=0).fit(X)
            # if arccos != None:
            #     outputs = arc_margin(outputs, labels, phase='test')
    #         preds = torch.softmax(outputs, dim=1)
            



    #         onehot = np.eye(7)[torch.argmax(preds.cpu(), dim=1)]

    #         if voting:
    #             vote_onehot = onehot.copy()
    #             onehot = []
    #             for i in range(batch_size):
    #                 onehot.append(
    #                     np.eye(7)[np.argmax(sum(vote_onehot[i*20:(i+1)*20]))])
    #             onehot = np.array(onehot)

    #     for i in range(batch_size):
    #         if 'image' not in result.keys():
    #             result['image'] = [
    #                 basename(test_loader.imgs_path[step*batch_size:(step+1)*batch_size][i])[:-4]]
    #         else:
    #             result['image'].append(
    #                 basename(test_loader.imgs_path[step*batch_size:(step+1)*batch_size][i])[:-4])
    #         for j in range(7):
    #             if labels_names[j] not in result.keys():
    #                 result[labels_names[j]] = [onehot[i][j]]
    #             else:
    #                 result[labels_names[j]].append(onehot[i][j])
    # keys = list(result.keys())
    # df = pd.DataFrame(result[keys[0]], columns=[keys[0]])
    # for k in keys[1:]:
    #     df[k] = result[k]

    # # saving the dataframe
    # if voting:
    #     df.to_csv(
    #         f'./submissions/voting_{basename(model_path)[:-4]}.csv', index=False)
    # else:
    #     df.to_csv(
    #         f'./submissions/{basename(model_path)[:-4]}.csv', index=False)

    # time_elapsed = time.time() - since
    # print('Runnning complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    arccos = True
    if arccos:
        model_ft = torch.load(model_path)
        model = model_ft['model'].to(device)
        arc_margin = model_ft['arccos'].to(device)
    else:
        model = torch.load(model_path)
        model = model.to(device)
        arc_margin = None
    summision_generate(model, batch_size=5, arccos=arc_margin, voting=False)
