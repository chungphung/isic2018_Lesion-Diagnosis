from os import listdir
from os.path import join

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


class dataloader(data.Dataset):
    def __init__(self, txt_path, data_path, preproc=None, mode='training', percents=1):
        self.preproc = preproc
        self.imgs_path = []
        self.labels = []
        self.weights = []
        self.mode = mode
        if mode == 'training':
            df = pd.read_csv(txt_path)
            tmp = []
            for col in df.columns[1:]:
                tmp.append((df[col] == 1).sum())
            self.weights = torch.Tensor(tmp/sum(tmp))
            img_list = list(df['image'])[:int(len(df)*percents)]
        elif mode == 'validate':
            df = pd.read_csv(txt_path)
            tmp = []
            for col in df.columns[1:]:
                tmp.append((df[col] == 1).sum())
            img_list = list(df['image'])[-int(len(df)*percents):]
        else:
            img_list = listdir(data_path)
            del img_list[img_list.index('ATTRIBUTION.txt')]
            del img_list[img_list.index('LICENSE.txt')]
        for i in img_list:
            image_name = i if '.jpg' in i else i+'.jpg'
            self.imgs_path.append(join(data_path, image_name))
            if mode == 'training' or mode == 'validate':
                tmp_label = df[df['image'] == i].apply(
                    lambda x: df.columns[x == 1], axis=1).to_string(header=False)
                self.labels.append(tmp_label.split()[1][8:-3])

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])

        try:
            label = self.labels[index]
        except:
            label = 'NV'
        if self.preproc is not None:
            img, target = self.preproc(img, label)
        else:
            target = label
            img = (img/255.0).astype(np.float32)
        return torch.from_numpy(img.transpose(2, 0, 1)), target
