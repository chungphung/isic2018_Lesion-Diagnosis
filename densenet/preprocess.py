import os
import os.path

import cv2
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd 


class dataloader(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        flag_62 = False
        self.flags = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                flag_62 = False

                if line.startswith('# 62'):
#                     import pdb; pdb.set_trace()
                    flag_62 = False
                    
                if isFirst is True:
                    isFirst = False
                else:
                    
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    if flag_62:
                        self.words.append(labels.copy())
                        self.words.append(labels.copy())
                        self.words.append(labels.copy())
                        
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt','images/') + path
                self.imgs_path.append(path)
                if flag_62:
                    self.flags.append(True)
                    self.flags.append(True)
                    self.flags.append(True)
                    self.flags.append(True)
                    self.imgs_path.append(path)
                    self.imgs_path.append(path)
                    self.imgs_path.append(path)
                else:
                    self.flags.append(False)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)
        if flag_62:
            self.words.append(labels.copy())
            self.words.append(labels.copy())
            self.words.append(labels.copy())

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
#         import pdb;pdb.set_trace()
        
        img = cv2.imread(self.imgs_path[index])
        try:
            height, width, _ = img.shape
        except:
            print(self.imgs_path[index])

        ############## downscale ############## 
        img = cv2.resize(img, (int(width/2), int(height/2)))
        height, width, _ = img.shape
        #######################################
        
        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))

            ############## downscale ############## 
            annotation[0, 0] = int(label[0]/2) # x1
            annotation[0, 1] = int(label[1]/2) # y1
            annotation[0, 2] = int((label[0] + label[2])/2)  # x2
            annotation[0, 3] = int((label[1] + label[3])/2)  # y2

            # landmarks
            annotation[0, 4] = label[4]/2    # l0_x
            annotation[0, 5] = label[5]/2    # l0_y
            annotation[0, 6] = label[7]/2    # l1_x
            annotation[0, 7] = label[8]/2    # l1_y
            annotation[0, 8] = label[10]/2   # l2_x
            annotation[0, 9] = label[11]/2   # l2_y
            annotation[0, 10] = label[13]/2  # l3_x
            annotation[0, 11] = label[14]/2  # l3_y
            annotation[0, 12] = label[16]/2  # l4_x
            annotation[0, 13] = label[17]/2  # l4_y
            #######################################

            if (annotation[0, 4]<0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
#             print(self.flags[index])
            img, target = self.preproc(img, target, self.flags[index])

        # return torch.from_numpy(img), target
        return torch.unsqueeze(torch.from_numpy(img), 0), target 

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)