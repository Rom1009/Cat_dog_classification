# Import libraries
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# Import folders
from utils import *
from Model_.model import *
import NerualNetwork


def main():
    train_loader, test_loader, len_features = Data_Loader()

    loss_criteria = nn.BCELoss()
    learning_rate = 0.001
    model = NerualNetwork.CNN()
    model.apply(lambda m: weights_inital(m, init_type="he"))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    Load_Model(optimizer, model, train_loader, test_loader, loss_criteria)

    model_file = "model.pt"
    torch.save(model.state_dict(), model_file)


if __name__ == "__main__":
    # create folder "dataset/data"  and unzip folder in its
    train_str = "train.zip"
    test_str = "test.zip"
    # create_file(train_str)
    # create_file(test_str)

    # -------------------------------
    main()
