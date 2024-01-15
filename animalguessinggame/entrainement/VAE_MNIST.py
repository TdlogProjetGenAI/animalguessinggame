"""Script for the VAE which was run to generate MNIST type number."""

import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import random


batch_size = 64
transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST training dataset
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) class.

    Architecture:
    - Encoder with two linear layers for mean and variance
    - Reparameterization trick for sampling from the latent space
    - Decoder with two linear layers for reconstruction

    """
    def __init__(self):
        """Initialize the VAE model with encoder, mean, logvar, and decoder."""
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 100)  # mean
        self.fc22 = nn.Linear(400, 100)  # variance
        self.fc3 = nn.Linear(100, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        """
        Encode the input to obtain the mean and variance of the latent space.

        Input:
        - x: Input tensor

        Output:
        - Tuple containing mean and variance tensors
        """
        h1 = f.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from the latent space.

        Input:
        - mu: Mean tensor
        - logvar: Log variance tensor

        Output:
        - Sampled latent variable tensor
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)

    def decode(self, z):
        """
        Decode the latent variable to reconstruct the input.

        Input:
        - z: Latent variable tensor

        Output:
        - Reconstructed image tensor
        """
        h3 = f.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        """
        Forward pass through the VAE.

        Input:
        - x: Input tensor

        Output:
        - Tuple containing reconstructed image, mean, and log variance tensors
        """
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(x, recon_x, mu, logvar):
    """
    Calculate the loss function for the VAE.

    Input:
    - x: Input tensor
    - recon_x: Reconstructed image tensor
    - mu: Mean tensor
    - logvar: Log variance tensor

    Output:
    - Total loss tensor
    """
    bce = f.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld

def train(model, optimizer, epochs, device):
    """
    Train the VAE model.

    Input:
    - model: VAE model
    - optimizer: Optimizer for model parameters
    - epochs: Number of training epochs
    - device: Device to use for training (cuda or cpu)

    Output:
    - Overall loss during training
    """
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, 784).to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("Epoch", epoch + 1, "Average Loss:", overall_loss / (batch_idx * batch_size))
    return overall_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train(model, optimizer, epochs=10, device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
train(model, optimizer, epochs=10, device=device)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


# Generate and display a reconstructed image from the test set
image, _ = test_set.__getitem__(random.randint(0, 10000))
with torch.no_grad():
    image = image.cpu()
    model = model.cpu()
    recon_image, mu, logvar = model(image.unsqueeze(0))
    image = np.reshape(image.numpy(), (28, 28))
    recon_image = recon_image.cpu()
    recon_image = np.reshape(recon_image.numpy(), (28, 28))

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original")
axes[1].imshow(recon_image, cmap='gray')
axes[1].set_title("Reconstructed")
plt.show()

# Generate an image from the latent space
with torch.no_grad():
    z = torch.randn(1, 100)  # 100 is the dimension of the latent space
    generated_image = model.decode(z)

# Display the generated image
generated_image = generated_image.view(1, 28, 28) 
plt.imshow(generated_image.squeeze().numpy(), cmap='gray')
plt.show()
