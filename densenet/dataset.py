from os.path import join

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


class dataloader(data.Dataset):
    def __init__(self, txt_path, data_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.labels = []
        self.weights = []
        df = pd.read_csv(txt_path)
        img_list = list(df['image'])
        for i in img_list:
            tmp_label = df[df['image']==i].apply(lambda x: df.columns[x==1], axis = 1).to_string(header=False)
            self.labels.append(tmp_label.split()[1][8:-3])
            self.imgs_path.append(join(data_path, i+'.jpg'))

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])

        label = self.labels[index]
        if self.preproc is not None:
            img, target = self.preproc(img, label)
        return torch.unsqueeze(torch.from_numpy(img), 0), target 
