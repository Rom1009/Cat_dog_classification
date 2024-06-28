import torch.nn as nn
import torch

# ----------------------------------------------------------------
# Import folders
# import customise_data


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Linear()
        self.layer2 = nn.Linear()
        self.layer3 = nn.Linear()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.softmax(self.layer3(x), dim=1)

    # def print():
    #     transform = transforms.Compose(
    #         [
    #             customise_data.Rescale(255),
    #             customise_data.RandomCrop(224),
    #             customise_data.ToTensor()
    #         ]
    #     )
    #     transform_data = customise_data.Cat_Dog_Dataset(
    #         csv_file="dataset/cat_dog.csv",
    #         root_dir="dataset/data/train/cat_dog_resized",
    #         transform=transform,
    #     )

    #     for i in range(len(transform_data)):
    #         sample = transform_data[i]

    #         print(i, sample["image"].size(), sample["cat_dog"].size())

    #         if i == 3:
    #             break
