# Import libraries
import zipfile
from pathlib import Path
import os
import urllib.request
import torch.nn as nn
import cv2
from tqdm import tqdm


# Take data raw from github link (self generated)
def create_file(filename: str):
    # Create path zip folder
    zip_dir = Path(os.path.join("dataset/data", filename))

    # Check if zip folder exists
    if not zip_dir.is_file():

        # Create folder containing zip file
        Path("dataset/data").mkdir(parents=True, exist_ok=True)

        # Create link to zip file
        url = os.path.join(
            "https://github.com/Rom1009/Data/raw/main/cat_dog_data", filename
        )
        url = url.replace("\\", "/")
        print(url)
        # Request zip file on github server to download
        urllib.request.urlretrieve(url, zip_dir)

        # Unzip zip file
        with zipfile.ZipFile("dataset/data/" + filename, "r") as data_file:
            data_file.extractall("dataset/data")


def load_images(path):
    list_of_images = os.listdir(path)
    images = []
    for img in tqdm(list_of_images):
        if img == "_DS_Store":
            continue
        image = cv2.imread(os.path.join(path, img))
        image = cv2.resize(image, dsize=(100, 100))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return images


# Change inital weights
def weights_inital(m, init_type="xavier"):
    if type(m) == "Linear":
        if init_type == "xavier":
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill(0.01)
        elif init_type == "normal":
            nn.init.normal_(m.weight)
            m.bias.data.fill(0.01)
        elif init_type == "he":
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill(0.01)
