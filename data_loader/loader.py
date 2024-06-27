from torch.utils.data import Dataset
import pandas as pd


class Dog_Cat_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        pass


class Rescale(object):
    def __init__(self, output_size):
        pass

    def __call__(self, sample):
        pass


class RandomCrop(object):
    def __init__(self, output_size):
        pass

    def __call__(self, sample):
        pass


class ToTensor(object):
    def __call__(self, sample):
        pass
