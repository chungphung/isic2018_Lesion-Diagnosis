from os import mkdir
from os.path import exists, join
from shutil import copy

import pandas as pd

csv = ['./main/train.csv', './main/val.csv']
dst = '../../data/tf_data'
data = '../../data/ISIC2018_Task3_Training_Input'
for txt_path in csv:
    df = pd.read_csv(txt_path)
    for col in df.columns[1:]:
        print(col)
        if not exists(join(dst, txt_path.split('/')[-1][:-4], col)):
            mkdir(join(dst, txt_path.split('/')[-1][:-4], col))
        for i in df[df[col]==1].image:
            copy(join(data, i+'.jpg'), join(dst, txt_path.split('/')[-1][:-4], col))
