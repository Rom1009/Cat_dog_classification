# Image Dataset Loader and Cat-Dog Classifier

This project contains a Convolutional Neural Network (CNN) model for classifying images of dogs and cats. Additionally, it includes a web application built with Streamlit for uploading and detecting image classes.

## Features

- Load images from a dataset.
- Apply transformations to the images if specified.
- Use PyTorch DataLoader for efficient data handling.
- Implement and train a CNN model.
- Test the model on a validation set.
- Build a Streamlit web app to upload images and detect classes (dog or cat).

## Installation

To use this project, clone the repository and install the required dependencies.

```sh
git clone https://github.com/Rom1009/Cat_dog_classification.git
cd DOG_CAT_PYTORCH
pip install -r requirements.txt
```

I don't include dataset because it quite large so you must run and unlock all comment in main to get data, model

```sh
python main.py
```

You have two options to run web app

```sh
python -m streamlit run app.py
```

```sh
streamlit run app.py
```
