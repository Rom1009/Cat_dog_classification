#
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torchvision import datasets, transforms
import torch.optim as optim

# ----------------------------------------------------------------
# Import folders
from sklearn.model_selection import train_test_split

from customise_data.customized import train_features_labels
import customise_data
from train.train import train
from valid.test import test


def Data_Loader():
    features_train, labels_train = train_features_labels()

    X_train, X_test, y_train, y_test = train_test_split(
        features_train, labels_train, random_state=0, test_size=0.2
    )

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),  # convert to tensor
            transforms.RandomRotation(degrees=20),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.005),
            transforms.RandomGrayscale(p=0.2),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # squeeze to -1 and 1
        ]
    )

    train_data = customise_data.Cat_Dog_Dataset(X_train, y_train, transform_train)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)

    transforms_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

    test_data = customise_data.Cat_Dog_Dataset(X_test, y_test, transforms_test)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, drop_last=True)

    return train_loader, test_loader, len(features_train)


def Load_Model(optimizer, model, train_loader, test_loader, loss_criteria):
    optimizer.zero_grad()
    epoch_nums = []
    training_loss = []
    validation_loss = []
    epochs = 10
    for epoch in range(1, epochs + 1):
        # print the epoch number
        print("Epoch: {}".format(epoch))

        # Feed training data into the model to optimize the weights
        train_loss = train(model, train_loader, optimizer, loss_criteria)

        # Feed the test data into the model to check its performance
        test_loss = test(model, test_loader, loss_criteria)

        # Log the metrics for this epoch
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)

