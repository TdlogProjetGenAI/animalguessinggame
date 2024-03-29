"""
This script was adapted from code initially used on Google Colab to train a classifier for the animals90 database.

The script defines a ResNet-based classifier, trains it on the animals90 dataset, and evaluates its performance.

"""

import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import os

T = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

folder_path = "path/to/the/database"  # Replace with the actual path
input_dim = 224
batch_size = 100
full_dataset = datasets.ImageFolder(root=folder_path, transform=T)
test_size = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - test_size
train_set, test_set = random_split(full_dataset, [train_size, test_size])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNetClassifier(nn.Module):
    """
    ResNet-based classifier for animals90 dataset.

    Args:
    - num_classes (int): Number of output classes.

    Attributes:
    - resnet : ResNet18 model with modified fully connected layer.

    """

    def __init__(self, num_classes=90):
        """
        Initialize the Resnet Classifier.

        Args:
        - num_classes (int): Number of output classes.

        """
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet18(weights='DEFAULT')
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the classifier.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.

        """
        x = self.resnet(x)
        return x


def test_loss(test_loader, model, criterion, device):
    """
    Calculate the test loss for the model.

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
        return overall_loss / len(test_loader)


def train(model, optimizer, criterion, train_loader, test_loader, epochs, device):
    """
    Train the ResNet classifier.

    Args:
    - model : ResNet classifier model.
    - optimizer : Optimizer for training.
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
            print(batch_idx/len(train_loader), loss.item())

        loss_train_per_epoch.append(overall_loss / len(train_loader.dataset))
        current_test_loss = test_loss(test_loader, model, criterion, device)
        loss_test_per_epoch.append(current_test_loss)

        if epoch >= 1 and current_test_loss < loss_test_per_epoch[-2]:
            file_path = os.path.join('animalguessinggame', 'models', 'classifierVF_animals90.pt')
            torch.save(model, file_path)

        print("Epoch", epoch + 1, "Average Loss:", overall_loss / len(train_loader.dataset))

    abs = [k for k in range(epochs)]
    plt.plot(abs, loss_train_per_epoch, color="red", label="train")
    plt.plot(abs, loss_test_per_epoch, color="blue", label="test")
    plt.legend()
    plt.show()

    return overall_loss


animaux_liste = [
    'antilope', 'blaireau', 'chauve-souris', 'ours', 'abeille', 'scarabée', 'bison', 'sanglier', 'papillon', 'chat',
    'chenille', 'chimpanzé', 'cafard', 'vache', 'coyote', 'crabe', 'corbeau', 'cerf', 'chien', 'dauphin', 'âne',
    'libellule', 'canard', 'aigle', 'éléphant', 'flamant rose', 'mouche', 'renard', 'chèvre', 'poisson rouge', 'oie',
    'gorille', 'sauterelle', 'hamster', 'lièvre', 'hérisson', 'hippopotame', 'calao', 'cheval', 'colibri', 'hyène',
    'méduse', 'kangourou', 'koala', 'coccinelles', 'léopard', 'lion', 'lézard', 'homard', 'moustique',
    'papillon de nuit', 'souris', 'pieuvre', 'okapi', 'orang-outan', 'loutre', 'hibou', 'bœuf', 'huître', 'panda',
    'perroquet', 'pélican', 'pingouin', 'cochon', 'pigeon', 'porc-épic', 'opossum', 'raton laveur', 'rat', 'renne',
    'rhinocéros', 'bécasse', 'hippocampe', 'phoque', 'requin', 'mouton', 'serpent', 'moineau', 'calmar', 'écureuil',
    'étoile de mer', 'cygne', 'tigre', 'dinde', 'tortue', 'baleine', 'loup', 'wombat', 'pic-vert', 'zèbre'
]

dict = {i: animal for i, animal in enumerate(animaux_liste)}

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

        for k in range(x.size(0)):

            a = dict[int(labels[k])]
            b = dict[torch.argmax(x[k]).item()]
            u += 1
            if a == b:
                val += 1
    print(val*100/u)
    print(u)


def display(model, test_loader, device):
    """
    Display each image along with its true label and model's prediction.

    Args:
    - model : Trained model to be evaluated.
    - test_loader : DataLoader for the test dataset.
    - device : Device (CPU or GPU) on which to perform computations.

    """
    for batch_idx, (images, labels) in enumerate(test_loader):
        for i in range(len(images)):
            image = images[i].numpy().transpose((1, 2, 0))
            true_label = dict[int(labels[i])]

            with torch.no_grad():
                model.eval()
                outputs = model(images.to(device))
                predictions = torch.argmax(outputs, dim=1)
                predicted_label = dict[int(predictions[i])]
            plt.imshow(image)
            plt.title(f"True Label: {true_label}, Predicted Label: {predicted_label}")
            plt.show()


resnet_model = ResNetClassifier(num_classes=90)
resnet_model = resnet_model.to(device)

optimizer_resnet = optim.Adam(resnet_model.parameters(), lr=1e-3)
criterion_resnet = nn.CrossEntropyLoss()

train(resnet_model, optimizer_resnet, criterion_resnet, train_loader, test_loader, epochs=10, device=device)
print(model_evaluation(resnet_model, test_loader, device))
display(resnet_model, test_loader, device)