import torch
def train(model, data_loader, optimizer, loss_criteria):
    model.train()
    train_loss = 0

    for batch, (data, target) in enumerate(data_loader):

        optimizer.zero_grad()
        output = model(data)

        loss = loss_criteria(output, torch.tensor(target.reshape(-1, 1)).float())
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = train_loss / (batch + 1)
    print("Training set: Average loss: {:.6f}".format(avg_loss))
    return avg_loss
