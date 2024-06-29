# Import libraries
import zipfile
from pathlib import Path
import os
import urllib.request
import csv


# Take data raw from github link (self generated)
def create_file(path_download: str, filename: str):
    # Create path zip folder
    zip_dir = Path(os.path.join("dataset/data", filename))

    # Check if zip folder exists
    if not zip_dir.is_file():

        # Create folder containing zip file
        Path("dataset/data").mkdir(parents=True, exist_ok=True)

        # Create link to zip file
        url = os.path.join(path_download, filename)
        url = url.replace("\\", "/")

        # Request zip file on github server to download
        urllib.request.urlretrieve(url, zip_dir)

        # Unzip zip file
        with zipfile.ZipFile("dataset/data/" + filename, "r") as data_file:
            data_file.extractall("dataset/data")


# Create csv file to store image and labels of dataset and Use for Dataset class
def csv_file():
    root_dir = "dataset/data/train"

    csv_name = "dataset/cat_dog.csv"

    classes = ["dogs", "cats"]

    # Read csv file if not created
    with open(csv_name, "w", newline="") as file:
        # write csv file and headers
        writer = csv.writer(file)
        writer.writerow(["filename", "label"])

        # Loop throught classes to write the contain
        for label, class_name in enumerate(classes):
            class_dir = os.path.join(root_dir, class_name)

            # Loop in list "dataset/data/train" + class_dir
            for img_name in os.listdir(class_dir):
                # Check if the imgs are jpg, png,...
                if img_name.endswith((".jpg")):
                    writer.writerow([img_name, label])
