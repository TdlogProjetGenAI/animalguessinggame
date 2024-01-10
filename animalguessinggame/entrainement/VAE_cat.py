"""Script for the VAE which was run on Imagine Server."""

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from math import log

path = "/home/tdlog"
data_path = "/home/tdlog/CAT3"
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
full_dataset = datasets.ImageFolder(root=data_path, transform=transform)
test_size = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - test_size
train_set, test_set = random_split(full_dataset, [train_size, test_size])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) class.

    Attributes:
        encoder (nn.Sequential): Encoder neural network.
        mean (nn.Conv2d): Mean layer for reparameterization.
        logvar (nn.Conv2d): Logvariance layer for reparameterization.
        decoder (nn.Sequential): Decoder neural network.
    """

    def __init__(self):
        """Initialize the VAE model with encoder, mean, logvar, and decoder."""
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
        self.mean = nn.Conv2d(3, 3, kernel_size=5, padding=2)
        self.logvar = nn.Conv2d(3, 3, kernel_size=5, padding=2)
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=8, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=9, padding=3),
            nn.Sigmoid()
        )

    def encode(self, x):
        """
        Encode the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Tuple containing mean and logvar.
        """
        y = self.encoder(x)
        return self.mean(y), self.logvar(y)

    def reparameterize(self, mu, logvar):
        """
        Reparameterizing trick to be differentiable for the backward porcess.

        Args:
            mu (torch.Tensor): Mean tensor.
            logvar (torch.Tensor): Logvariance tensor.

        Returns:
            torch.Tensor: Reparameterized tensor.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)

    def decode(self, z):
        """
        Decode the input tensor.

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Decoded tensor.
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

def test_loss(test, mod):
    """
    Calculate the test loss for the VAE.

    Args:
        test : DataLoader for the test set.
        mod : VAE model.

    Returns:
        float: Test loss.
    """
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(test):
        x = x.to(device)
        x_hat = mod(x)
        mean, log_var = mod.encode(x)
        loss = loss_function(x, x_hat, mean, log_var)
        overall_loss += loss.item()
    return overall_loss / (test_size)

def loss_function(x, recon_x, mu, logvar):
    """
    Calculate the loss function for the VAE.

    Args:
        x (torch.Tensor): Input tensor.
        recon_x (torch.Tensor): Reconstructed tensor.
        mu (torch.Tensor): Mean tensor.
        logvar (torch.Tensor): Logvariance tensor.

    Returns:
        torch.Tensor: Loss value.
    """
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld

def train(model, optimizer, epochs, device):
    """
    Train the VAE model.

    Args:
        model : VAE model.
        optimizer : Optimizer.
        epochs (int): Number of training epochs.
        device : Device for training.

    Returns:
        float: Overall loss.
    """
    loss_train_per_epoch = []
    loss_test_per_epoch = []
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = model(x)
            mean, log_var = model.encode(x)
            loss = loss_function(x, x_hat, mean, log_var)
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
        average_loss = overall_loss / (train_size)
        loss_train_per_epoch.append(log(average_loss))
        loss_test_per_epoch.append(log(test_loss(test_loader, model)))
        if epoch > 0 and (epoch + 1) % 1000 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 5
        if (1 + epoch) % 200 == 0:
            abs = [k for k in range(len(loss_train_per_epoch))]
            plt.plot(abs, loss_train_per_epoch)
            plt.plot(abs, loss_test_per_epoch)
            torch.save(model.state_dict(), path + "/model" + f"{epoch}" + ".pt")
            plt.savefig(path + "/courbe" + f"{epoch}" + ".png")
            plt.clf()
            loss_train_per_epoch = []
            loss_test_per_epoch = []
    return overall_loss


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train(model, optimizer, epochs=10000, device=device)
