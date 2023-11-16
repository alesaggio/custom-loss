import torch
from custom_loss import forecast_loss


# Ensure that the penalty is zero when predictions <= stocks
def test_custom_loss_pred_below_stock():
    predictions = torch.tensor([2.0, 3.0, 1.0])
    sales = torch.tensor([2.0, 3.0, 1.0])
    stocks = torch.tensor([4.0, 3.0, 2.0])
    loss_custom = forecast_loss(predictions, sales, stocks)
    criterion = torch.nn.MSELoss()
    loss_mse = criterion(predictions, sales)
    assert loss_custom == loss_mse


# Ensure that the penalty is non-zero when predictions > stocks
def test_custom_loss_pred_above_stock():
    predictions = torch.tensor([5.0, 4.0, 3.0])
    sales = torch.tensor([5.0, 4.0, 3.0])
    stocks = torch.tensor([4.0, 3.0, 2.0])
    loss_custom = forecast_loss(predictions, sales, stocks)
    criterion = torch.nn.MSELoss()
    loss_mse = criterion(predictions, sales)
    assert loss_custom > loss_mse
