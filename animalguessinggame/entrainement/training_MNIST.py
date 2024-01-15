"""This code was used on Google Colab then adpated to train our classfier for MNIST database."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root=".", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.MNIST(root=".", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
batch_size = 64

class ClassifierMNIST(nn.Module):
    """MNSIT classifier for MNIST dataset."""
    def __init__(self, num_classes=10):
        """
        Initialize the MNIST Classifier.

        Args:
        - num_classes (int): Number of output classes.

        """
        super(ClassifierMNIST, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5, 1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.

        """
        x = self.convolution(x)
        x = x.view(x.size(0), -1)
        return self.linear_layers(x)

def test_loss(test_loader, model, criterion, device):
    """
    Calculate the test loss of the model.

    Args:
    - test_loader : DataLoader for the test dataset.
    - model : Trained model.
    - criterion : Loss criterion.
    - device : Device (CPU or GPU) on which to perform computations.

    Returns:
    - float: Average test loss.

    """
    model.eval()
    overall_loss = 0

    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(test_loader):
            x, labels = x.to(device), labels.to(device)
            prediction = model(x)
            prediction = prediction.to(device)
            labels = labels.to(device)
            loss = criterion(prediction, labels)
            overall_loss += loss.item()

    model.train()
    return overall_loss / (batch_idx * batch_size)

def train(model, optimizer, criterion, train_loader, test_loader, epochs, device):
    """
    Train the model.

    Args:
    - model : Model to be trained.
    - optimizer : Optimization algorithm.
    - criterion : Loss criterion.
    - train_loader : DataLoader for the training dataset.
    - test_loader : DataLoader for the test dataset.
    - epochs : Number of training epochs.
    - device : Device (CPU or GPU) on which to perform computations.

    Returns:
    - float: Overall loss after training.

    """
    model.train()
    loss_train_per_epoch = []
    loss_test_per_epoch = []

    for epoch in range(epochs):
        overall_loss = 0

        for batch_idx, (x, labels) in enumerate(train_loader):
            x, labels = x.to(device), labels.to(device)

            optimizer.zero_grad()
            prediction = model(x)
            prediction = prediction.to(device)
            labels = labels.to(device)
            loss = criterion(prediction, labels)
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
            print(batch_idx / len(train_loader), loss.item())

        loss_train_per_epoch.append(overall_loss / len(train_loader.dataset))
        current_test_loss = test_loss(test_loader, model, criterion, device)
        loss_test_per_epoch.append(current_test_loss)

        if epoch >= 1 and current_test_loss < loss_test_per_epoch[-2]:
            file_path = os.path.join('animalguessinggame', 'models', 'classifierVF_MINST.pt')
            torch.save(model, file_path)

        print("Epoch", epoch + 1, "Average Loss:", overall_loss / len(train_loader.dataset))

    abs = [k for k in range(epochs)]
    plt.plot(abs, loss_train_per_epoch, color="red", label="train")
    plt.plot(abs, loss_test_per_epoch, color="blue", label="test")
    plt.legend()
    plt.show()

    return overall_loss


def model_evaluation(model, test_loader, device):
    """
    Evaluate the performance of the trained model on the test dataset.

    Args:
    - model : Trained model to be evaluated.
    - test_loader : DataLoader for the test dataset.
    - device : Device (CPU or GPU) on which to perform computations.

    Returns:
    - float: Accuracy percentage on the test dataset.

    """
    val = 0
    u = 0

    for batch_idx, (x, labels) in enumerate(test_loader):
        x = x.to(device)
        x = model(x)
        x = x.to(device)
        labels = labels.to(device)
        for k in range(x.size(0)):
            a = int(labels[k])
            b = int(torch.argmax(x[k]).item())
            u += 1
            if a == b:
                val += 1

    return (val * 100 / u)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ClassifierMNIST(num_classes=10)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)
train(model, optimizer, criterion, train_loader, test_loader, epochs=10, device=device)
