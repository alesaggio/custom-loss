import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split

from plotting_functions import plot_features, plot_loss_vs_epochs, plot_loss_vs_param


# Define the loss function
def forecast_loss(y_pred, y_true, stock, lambda_term=1.0):
    loss = torch.mean((y_pred - y_true) ** 2)
    dim = len(y_pred)
    penalty = torch.mean(torch.max(torch.zeros(dim), lambda_term * (y_pred - stock)))
    loss += penalty
    return loss


def calculate_loss_mse(model, dim, y_true):
    output = model(dim)
    lossfunc = nn.MSELoss()
    loss = lossfunc(output.view(-1, 1), y_true)
    return loss.item()


def calculate_loss_custom(model, dim, y_true, stock):
    output = model(dim)
    loss = forecast_loss(output, y_true, stock)
    return loss.item()


# Define the model class
class MeanModel(nn.Module):
    def __init__(self):
        super(MeanModel, self).__init__()
        self.mean = nn.Parameter(torch.randn(1))

    def forward(self, dim):
        return self.mean * torch.ones(dim)


# Define the dataset class
class MyDataset(Dataset):
    def __init__(self, tensor1, tensor2, tensor3):
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.tensor3 = tensor3
        self.length = len(tensor1)  # Assume that length is the same for all tensors

    def __getitem__(self, idx):
        return self.tensor1[idx], self.tensor2[idx], self.tensor3[idx]

    def __len__(self):
        return self.length


if __name__ == "__main__":
    # Create directory for plots
    outdir = "plots"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Define the input tensors for stocks and sales
    n = 1000
    torch.manual_seed(0)
    stocks = torch.randint(0, 10, (n,))
    demand = torch.poisson(torch.ones(n) * 2.0)
    sales = torch.min(demand, stocks)

    # Plot the features
    plot_features(stocks, demand, sales, outdir)

    # Create dataset
    dataset = MyDataset(stocks, demand, sales)
    print("Head of input tensors: \n", dataset[0:5])

    # Split dataset for train and test
    n_train = int(0.8 * len(dataset))  # 80% for training
    n_test = len(dataset) - n_train  # Remaining 20% for testing
    train_dataset, test_dataset = random_split(
        dataset, [n_train, n_test], generator=torch.Generator().manual_seed(200)
    )

    stocks_train = torch.tensor(
        [sample[0].item() for sample in train_dataset], dtype=torch.float32
    ).view(-1, 1)
    sales_train = torch.tensor(
        [sample[2].item() for sample in train_dataset], dtype=torch.float32
    ).view(-1, 1)

    stocks_test = torch.tensor(
        [sample[0].item() for sample in test_dataset], dtype=torch.float32
    ).view(-1, 1)
    sales_test = torch.tensor(
        [sample[2].item() for sample in test_dataset], dtype=torch.float32
    ).view(-1, 1)

    # Create model with built-in loss function (MSE)
    model_mse = MeanModel()
    optimizer_mse = torch.optim.SGD(model_mse.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Create model with custom loss function
    model_custom = MeanModel()
    optimizer_custom = torch.optim.SGD(model_custom.parameters(), lr=0.01)

    # Ensure that the two models are initialized with the same weights for safe comparison
    model_custom.load_state_dict(model_mse.state_dict())

    # Train the model
    epochs = 200
    total_mseloss = []
    total_customloss = []

    optimal_param_mse = 0.0
    optimal_param_custom = 0.0

    for i in range(epochs):
        optimizer_mse.zero_grad()
        outputs_mse = model_mse(n_train).view(-1, 1)
        loss_mse = criterion(outputs_mse, sales_train)
        loss_mse.backward()
        optimizer_mse.step()
        total_mseloss.append(loss_mse.item())

        optimizer_custom.zero_grad()
        outputs_custom = model_custom(n_train).view(-1, 1)
        loss_custom = forecast_loss(outputs_custom, sales_train, stocks_train)
        loss_custom.backward()
        optimizer_custom.step()
        total_customloss.append(loss_custom.item())

        if i == epochs - 1:
            optimal_param_mse = model_mse.mean.item()
            optimal_param_custom = model_custom.mean.item()

    # Plot losses
    plot_loss_vs_epochs(epochs, total_mseloss, total_customloss, outdir)

    # Predict sales on test dataset
    predictions_mse = model_mse(n_test)
    predictions_custom = model_custom(n_test)

    # In principle, one would round predictions since sales must be a round number.
    # However, we do not it in here since the two trained models only return one parameter each which are similar,
    # so by rounding them up there are chances that we would end up with the prediction for both the models, and
    # would not be able to compare the accuracies.

    # Compute custom error to compare performances
    mse_original = forecast_loss(predictions_mse, sales_test, stocks_test)
    mse_custom = forecast_loss(predictions_custom, sales_test, stocks_test)

    print("Accuracy on the model trained with MSE: ", round(mse_original.item(), 3))
    print(
        "Accuracy on the model trained with custom loss: ", round(mse_custom.item(), 3)
    )

    print("Parameter of the model trained with MSE: ", round(optimal_param_mse, 3))
    print(
        "Parameter of the model trained with custom loss",
        round(optimal_param_custom, 3),
    )

    # Plot loss as a function of the parameter, to see that the minimum corresponds to the trained model's parameter
    loss_mse_values = []
    loss_custom_values = []
    parameter_values = torch.linspace(-2, 5, 200)
    for param_val in parameter_values:
        model_mse.mean.data = torch.tensor([param_val])  # Set the parameter value
        model_custom.mean.data = torch.tensor([param_val])  # Set the parameter value
        loss_mse = calculate_loss_mse(model_mse, n_test, sales_test)
        loss_mse_values.append(loss_mse)
        loss_custom = calculate_loss_custom(
            model_custom, n_test, sales_test, stocks_test
        )
        loss_custom_values.append(loss_custom)

    # Plot loss vs parameter
    plot_loss_vs_param(parameter_values, loss_mse_values, loss_custom_values, outdir)
