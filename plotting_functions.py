import os
import numpy as np
import matplotlib.pyplot as plt


def plot_features(stocks, demand, sales, outdir):
    plt.figure()
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))

    axs[0, 0].hist(stocks.numpy(), bins=10, histtype="step")
    axs[0, 0].set_xlabel("Stocks")
    axs[0, 1].hist(demand.numpy(), bins=10, histtype="step")
    axs[0, 1].set_xlabel("Demand")
    axs[1, 0].hist(sales.numpy(), bins=10, histtype="step")
    axs[1, 0].set_xlabel("Sales")
    fig.delaxes(axs[1, 1])
    plt.savefig(os.path.join(outdir, "feature_distributions.png"))
    plt.close()


def plot_loss_vs_epochs(epochs, loss_mse, loss_custom, outdir):
    plt.figure()
    plt.plot(range(epochs), loss_mse, label="MSE Loss", color="red")
    plt.plot(range(epochs), loss_custom, label="Custom Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(outdir, "loss_vs_epochs.png"))
    plt.close()


def plot_loss_vs_param(parameters, loss_mse, loss_custom, outdir):
    plt.figure()
    plt.plot(parameters.numpy(), loss_mse, label="MSE Loss", color="red")
    plt.plot(parameters.numpy(), loss_custom, label="Custom Loss", color="blue")
    plt.xticks(np.arange(-2, 5, step=0.5))
    plt.xlabel("Parameter value")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(outdir, "loss_vs_parameter.png"))
    plt.close()
