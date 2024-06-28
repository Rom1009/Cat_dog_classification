from torch.utils.data import Dataset
import pandas as pd
import torch
import os
from skimage import io, transform
import numpy as np


class Cat_Dog_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.cat_dog_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.cat_dog_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.cat_dog_csv.iloc[idx, 0])
        image = io.imread(img_name)
        cat_dog = self.cat_dog_csv.iloc[idx, 1]
        cat_dog = np.asarray([cat_dog])
        cat_dog = cat_dog.astype("float")
        sample = {"image": image, "cat_dog": cat_dog}

        if self.transform:
            sample = self.transform(sample)

        return sample


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
