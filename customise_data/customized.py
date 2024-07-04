from torch.utils.data import Dataset
import pandas as pd
import torch
import os
from skimage import io, transform
import numpy as np
from utils import load_images


def train_features_labels():
    cat_train = load_images("dataset/data/train/cats")
    dog_train = load_images("dataset/data/train/dogs")

    features = np.append(cat_train, dog_train, axis=0)
    labels = np.array([0] * len(cat_train) + [1] * len(dog_train))
    return features, labels


# def test_features_labels():
#     features = load_images("dataset/data/test")
#     return features


class Cat_Dog_Dataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return (image, label)


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, cat_dog = sample["image"], sample["cat_dog"]

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        cat_dog = cat_dog * [new_w / w, new_h / h]

        return {"image": img, "cat_dog": cat_dog}


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        image, cat_dog = sample["image"], sample["cat_dog"]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top : top + new_h, left : left + new_w]

        cat_dog = cat_dog - [left, top]

        return {"image": image, "cat_dog": cat_dog}


class ToTensor(object):
    def __call__(self, sample):
        image, cat_dog = sample["image"], sample["cat_dog"]

        image = image.transpose((2, 0, 1))

        return {"image": torch.from_numpy(image), "cat_dog": torch.from_numpy(cat_dog)}
