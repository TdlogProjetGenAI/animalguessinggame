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
import os 
from flask import current_app


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet18(weights='DEFAULT')
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)


    def forward(self, x):
        x=self.resnet(x)
        return x


dict = {0: "chien", 1: "cheval", 
            2 : "elephant", 3: "papillon", 4: "poule", 
            5: "chat", 6: "vache", 7: "mouton", 8: "araignée", 
            9: "écureuil"}

def classifie_animals10(image_path):
    model_chemin = os.path.join('animalguessinggame', 'models', 'classifierVF_animals10.pt')
    model = torch.load(model_chemin, map_location='cpu')
    image=Image.open(image_path)
    T = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image=T(image)
    image=image.unsqueeze(0)
    with torch.no_grad():
        x = model(image)[0]
    predicted_class = int(torch.argmax(x).item())
    return dict[predicted_class]


######
def classifie_mnist(image__list_path):
    model=ResNetClassifier()
    model_chemin = os.path.join('animalguessinggame', 'models', 'classifierVF_minst.pt')
    model = torch.load(model_chemin, map_location='cpu')
    model.eval()
    ans=[]
    for x  in image__list_path:
        image=Image.open(x)
        T1 = transforms.Compose([
        transforms.ToTensor()
        ])
        image=T1(image)
        image=image.unsqueeze(0)
        with torch.no_grad():
            x = model(image)[0]

        ans.append(int(torch.argmax(x).item()))
    return concat(ans)

def concat(L):
    n = len(L)
    string =[]
    if n == 4:
        if L == [0,0,0,0]:
            return 'zéro'
        #gestion des milliers
        if L[0] != 0:
            string.append(chiffre[L[0]]+' mille')
        #gestion des centaines
        if L[1] != 0 and L[1] != 1:
            string.append(chiffre[L[1]]+" "+'cents')
        if L[1] == 1:
            string.append('cent')
        #gestion des dizaines et unités
        if L[2] != 7 and L[2] != 9 and L[2] != 0 and L[2] != 1:
            string.append(dizaine[L[2]])
            if L[3] != 0 :
                if L[3] == 1 and L[2] != 8:
                    string.append('et '+ chiffre[L[3]])
                else:
                    string.append(chiffre[L[3]])
        # cas soixante-dix et quatre vingt dix
        if L[2] == 7 or L[2] == 9:
            if L[3] == 0:
                string.append(dizaine[L[2]])
            elif L[3] == 1 and L[2] == 7:
                string.append("soixante et onze")
            else:
                string.append(dizaine[L[2]-1] + " " + dix_vingt[L[3]])
        # cas des dizaines egales à 1 
        if L[2] == 1:
            string.append(dix_vingt[L[3]])
        #cas des dizaines nulles
        if L[2] == 0:
            if L[3] != 0:
                string.append(chiffre[L[3]])
    if n == 3:
        if L == [0,0,0]:
            return 'zéro'
        #gestion des centaines
        if L[0] != 0 and L[0] != 1:
            string.append(chiffre[L[0]]+" "+'cents')
        if L[0] == 1:
            string.append('cent')
        #gestion des dizaines et unités
        if L[1] != 7 and L[1] != 9 and L[1] != 0 and L[1] != 1:
            string.append(dizaine[L[1]])
            if L[2] != 0 :
                if L[2] == 1 and L[1] != 8:
                    string.append('et '+ chiffre[L[2]])
                else:
                    string.append(chiffre[L[2]])
        # cas soixante-dix et quatre vingt dix
        if L[1] == 7 or L[1] == 9:
            if L[2] == 0:
                string.append(dizaine[L[1]])
            elif L[2] == 1 and L[1] == 7:
                string.append("soixante et onze")
            else:
                string.append(dizaine[L[1]-1] + " " + dix_vingt[L[2]])
        # cas des dizaines egales à 1 
        if L[1] == 1:
            string.append(dix_vingt[L[2]])
        #cas des dizaines nulles
        if L[1] == 0:
            if L[2] != 0:
                string.append(chiffre[L[2]])
    if n == 2:
        if L == [0,0]:
            return 'zéro'
    #gestion des dizaines et unités
        if L[0] != 7 and L[0] != 9 and L[0] != 0 and L[0] != 1:
            string.append(dizaine[L[0]])
            if L[1] != 0 :
                if L[1] == 1 and L[0] != 8:
                    string.append('et '+ chiffre[L[1]])
                else:
                    string.append(chiffre[L[1]])
        # cas soixante-dix et quatre vingt dix
        if L[0] == 7 or L[0] == 9:
            if L[1] == 0:
                string.append(dizaine[L[0]])
            elif L[1] == 1 and L[0] == 7:
                string.append("soixante et onze")
            else:
                string.append(dizaine[L[0]-1] + " " + dix_vingt[L[1]])
        # cas des dizaines egales à 1 
        if L[0] == 1:
            string.append(dix_vingt[L[1]])
        #cas des dizaines nulles
        if L[0] == 0:
            if L[1] != 0:
                string.append(chiffre[L[1]])
    if n == 1:
        string.append(chiffre[L[0]])
    if n not in [1,2,3,4]:
        return 'ERROR'
    rep = ''
    for i in string:
        rep += i+' '
    return rep[:-1]