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


def _brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    rand_value = random.randint(-50, 50)
    if rand_value >= 0:
        v[v+rand_value >= 255] = 255
        v[v+rand_value < 255] += rand_value
    else:
        v[v+rand_value <= 0] = 0
        v[v-abs(rand_value) > 0] -= abs(rand_value)
    hsv[:, :, 2] = v
    img_t = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img_t


def _distort(image):
    h, w, c = image.shape

    x = np.random.choice(h, 77, replace=False)
    y = np.random.choice(w, 78, replace=False)
    idx_list = []
    for i in range(len(x)):
        for j in range(len(y)):
            idx_list.append((x[i], y[j]))

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

    def __call__(self, image, targets, phase):
        if phase == 'training' or phase == 'validate':
            labels = torch.tensor(
                np.array([self.labels_names.index(targets)]), dtype=torch.long)
            if phase == 'training':
                image_t = _random_crop(image)
                if bool(random.getrandbits(1)):
                    image_t = _brightness(image_t)
                if bool(random.getrandbits(1)):
                    image_t = _distort(image_t)
                if bool(random.getrandbits(1)):
                    image_t = _mirror(image_t)
                if bool(random.getrandbits(1)):
                    image_t = _rotate(image_t)
            else:
                image_t = image
            image_t = (image_t/255.0).astype(np.float32)
        else:
            labels = [targets]
            image_t = (image/255.0).astype(np.float32)
        return image_t, labels
