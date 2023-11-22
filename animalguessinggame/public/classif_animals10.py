    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms, models
    import torch.optim as optim
    from PIL import Image
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt

    T = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_size=64
    class ResNetClassifier(nn.Module):
        def __init__(self, num_classes=10):
            super(ResNetClassifier, self).__init__()
            self.resnet = models.resnet18(weights='DEFAULT')
            in_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(in_features, num_classes)


        def forward(self, x):
            x=self.resnet(x)
            return x

    def classifie_animals10(image_path):
        model=ResNetClassifier()
        model=torch.load("C:/Users/maxim/Desktop/IMI/TDLOG/Projet_TdLog")

        image=open(image_path)
        image=T(image)
        
