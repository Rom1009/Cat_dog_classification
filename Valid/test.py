import torch

def test(model, data_loader, loss_criteria):
    model.eval()
    correct = 0
    val_loss = 0

    with torch.no_grad():
        batch_count = 0
        for batch, (data, target) in enumerate(data_loader):
            batch_count += 1
            output = model(data)
            val_loss += loss_criteria(
                output, torch.tensor(target.reshape(-1, 1)).float()
            ).item()

            # Calculate accuracy
            rounded_test_preds = torch.round(output)
            correct += torch.sum(
                rounded_test_preds == torch.tensor(target.reshape(-1, 1)).float()
            )

    avg_loss = correct / batch_count
    print(
        "Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            avg_loss,
            correct,
            len(data_loader.dataset),
            100.0 * correct / len(data_loader.dataset),
        )
    )

    # return average loss for the epoch
    return avg_loss
