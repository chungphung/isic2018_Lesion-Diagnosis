import random

import cv2
import imutils
import numpy as np
import torch
from torchvision.transforms import RandAugment, RandomCrop, CenterCrop
from PIL import Image
from AutoAugment.autoaugment import AutoAugment, AutoAugmentPolicy

lowcost = AutoAugment(AutoAugmentPolicy.LCA)
cropper = RandomCrop(size=(224, 224))
center = CenterCrop(size=(224, 224))


def _random_crop(image):
    r = random.randint(0, 3)
    c = random.randint(0, 4)
    # img_t = image[25*r:25*r+350, 25*c:25*c+450]
#     img_t = image[25*r:25*r+224, 25*c:25*c+224]
    img_t = image[:, 25*c:25*c+450]
    return img_t


def _center_crop(image):
    # img_t = image[50:400, 75:525]
    #     img_t = image[113:337, 188:412]
    img_t = image[:, 75:525]
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
                if bool(random.getrandbits(1)):
                    image_t = _center_crop(image)
                else:
                    image_t = _random_crop(image)
                if bool(random.getrandbits(1)):
                    image_t = _brightness(image_t)
                if bool(random.getrandbits(1)):
                    image_t = _mirror(image_t)
                if bool(random.getrandbits(1)):
                    image_t = _rotate(image_t)
            else:  # validate will go here!!!
                image_t = _center_crop(image)
            image_t = (image_t/255.0).astype(np.float32)
        else:
            labels = [targets]
            # image = _center_crop(image)
            image_t = (image/255.0).astype(np.float32)
        return image_t, labels


class randaugment_preproc(object):

    def __init__(self):
        self.labels_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

    def __call__(self, image, targets, phase):
        if phase == 'training' or phase == 'validate':
            labels = torch.tensor(
                np.array([self.labels_names.index(targets)]), dtype=torch.long)
            if phase == 'training':
                image_t = RandAugment(image)
                if bool(random.getrandbits(1)):
                    image_t = _center_crop(image)
                else:
                    image_t = _random_crop(image)

            else:  # validate will go here!!!
                image_t = _center_crop(image)
            image_t = (image_t/255.0).astype(np.float32)
        else:
            labels = [targets]
            image = _center_crop(image)
            image_t = (image/255.0).astype(np.float32)
        return image_t, labels


class lowcost_center_preproc(object):

    def __init__(self):
        self.labels_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

    def __call__(self, image, targets, phase, image2):
        if phase == 'training' or phase == 'validate':
            labels = torch.tensor(
                np.array([self.labels_names.index(targets)]), dtype=torch.long)
            if phase == 'training':
                image_t = lowcost(image, image2)
                image_t = np.array(cropper(image_t), dtype=np.uint8)
                # print(image_t.shape)

            else:  # validate will go here!!!
                image_t = np.array(center(image), dtype=np.uint8)
                # print(image_t.shape)
            image_t = (image_t/255.0).astype(np.float32)
        else:
            labels = [targets]
            image_t = np.array(center(image), dtype=np.uint8)
            image_t = (image_t/255.0).astype(np.float32)
        return image_t, labels


class lowcost_16crop_preproc(object):

    def __init__(self):
        self.labels_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

    def __call__(self, image, targets, phase, image2):
        if phase == 'training' or phase == 'validate':
            labels = torch.tensor(
                np.array([self.labels_names.index(targets)]), dtype=torch.long)
            if phase == 'training':
                image_t = lowcost(image, image2)
                image_t = np.array(cropper(image_t), dtype=np.uint8)

            else:  # validate will go here!!!
                # image_t = np.array(center(image), dtype=np.uint8)
                pass
            image_t = (image_t/255.0).astype(np.float32)
        else:
            labels = [targets]
            # image = image_t = np.array(center(image), dtype=np.uint8)
            image_t = (image/255.0).astype(np.float32)
        return image_t, labels
