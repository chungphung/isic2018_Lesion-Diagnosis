import random
from itertools import cycle

import cv2
import imutils
import numpy as np


def _random_crop(image):
    r = random.randint(0, 3)
    c = random.randint(0, 4)
    img_t = image[50*r:50*r+300, 50*c:50*c+400]
    return img_t


def _distort(image):
    h, w, c = image.shape
    no_px = int(h*w*0.05)

    x = np.random.choice(h, 77, replace=False)  
    y = np.random.choice(w, 78, replace=False)  
    idx_list = np.array(list(zip(cycle(y),x))[:-6]).T.tolist()

    val = random.randint(-5, 5)
    
    image_t = np.array(image).T 
    image_t[idx_list]+=val
    image = image_t.T
    return image


def _mirror(image):
    flipcode = random.randint(-1, 1)
    image = cv2.flip(image, flipcode)
    return image


def _rotate(image):
    angle = random.randint(-3, 3)
    rotated = imutils.rotate(image, angle)
    return rotated 


class preproc(object):

    def __init__(self):
        pass 
    
    def __call__(self, image, targets):

        labels = targets[:, -1].copy()
#         print(labels.shape)
        image_t = _random_crop(image)
        image_t = _distort(image_t)
        image_t = _mirror(image_t)

        labels_t = np.expand_dims(labels, 1)

        return image_t, labels_t
