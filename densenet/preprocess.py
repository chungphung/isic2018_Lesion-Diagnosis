import random
from itertools import cycle

import cv2
import imutils
import numpy as np
import torch


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
    idx_list = np.array(list(zip(cycle(y), x))[:-6]).T.tolist()

    val = random.randint(-5, 5)

    image_t = image.transpose(1, 0, 2)
    image_t[idx_list] += np.uint8(val)
    image = image_t.transpose(1, 0, 2)
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
        self.labels_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

    def __call__(self, image, targets):
        labels = torch.Tensor([self.labels_names.index(targets)])
        image_t = _random_crop(image)
        if bool(random.getrandbits(1)):
            image_t = _distort(image_t)
        if bool(random.getrandbits(1)):
            image_t = _mirror(image_t)
        image_t = (image_t/255.0).astype(np.float32)
        return image_t, labels
