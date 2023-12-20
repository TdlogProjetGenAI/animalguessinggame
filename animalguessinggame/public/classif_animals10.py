import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os 
from flask import current_app


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=10):
        """
        Initialize a ResNet classifier with a specified number of classes.

        Args:
            num_classes (int): Number of output classes.
        """
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet18(weights='DEFAULT')
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the ResNet classifier.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.resnet(x)
        return x
class VAE(nn.Module):
    def __init__(self):
        """
        Initialize a Variational Autoencoder (VAE) model.
        """
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)  
        self.fc22 = nn.Linear(400, 20)  
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        """
        Initialize a Variational Autoencoder (VAE) model.
        """
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)  

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for VAE.

        Args:
            mu (torch.Tensor): Mean tensor.
            logvar (torch.Tensor): Log variance tensor.

        Returns:
            torch.Tensor: Reparameterized tensor.
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu) 

    def decode(self, z):
        """
        Decode latent variable into the reconstructed input.

        Args:
            z (torch.Tensor): Latent variable tensor.

        Returns:
            torch.Tensor: Reconstructed output tensor.
        """
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))  

    def forward(self, x):
        """
        Forward pass of the VAE.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Tuple containing reconstructed output, mean, and log variance tensors.
        """
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    

class Classifier_mnist(nn.Module):
    def __init__(self, num_classes=10):
        """
        Initialize a simple classifier for MNIST images.

        Args:
            num_classes (int): Number of output classes.
        """
        super(Classifier_mnist, self).__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5, 1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.mod2 = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        """
        Forward pass of the MNIST classifier.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.mod(x)
        x = x.view(x.size(0), -1)
        return self.mod2(x)
    

dict = {0: "chien", 1: "cheval", 
        2: "éléphant", 3: "papillon", 4: "poule", 
        5: "chat", 6: "vache", 7: "mouton", 8: "araignée", 
        9: "écureuil"}
dict_eng = {0: "dog", 1: "horse", 
            2: "elephant", 3: "butterfly", 4: "chicken", 
            5: "cat", 6: "cow", 7: "sheep", 8: "spider", 
            9: "squirrel"}


def classifie_animals10(image_path):
    """
    Classify an animal image into one of 10 classes using a pre-trained model.

    Args:
        image_path (str): Path to the input image.

    Returns:
        list: A list containing the predicted class label in French and English.
    """
    model_chemin = os.path.join(current_app.root_path, 'models', 'classifierVF_animals10.pt')
    model = torch.load(model_chemin, map_location='cpu')
    model.eval()
    image_path = 'animalguessinggame/static'+image_path
    image = Image.open(image_path)
    T = transforms.Compose([transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])
    image = T(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        x = model(image)[0]
    predicted_class = int(torch.argmax(x).item())
    return [dict[predicted_class], dict_eng[predicted_class]]

def classifie_animals90(image_path):
    """
    Classify an animal image into one of 90 classes using a pre-trained model.

    Args:
        image_path (str): Path to the input image.

    Returns:
        list: A list containing the predicted class label in French and English.
    """
    model_chemin = os.path.join(current_app.root_path, 'models', 'classifier_animals90.pt')
    model = torch.load(model_chemin, map_location='cpu')
    model.eval()
    image_path = 'animalguessinggame/static'+image_path
    image = Image.open(image_path)
    T = transforms.Compose([transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])
    image = T(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        x = model(image)[0]
    predicted_class = int(torch.argmax(x).item())
    return [animaux[predicted_class], animaux_eng[predicted_class]]
######
def classifie_mnist(image__list_path):
    """
    Classify an number image into one of 10 classes using a pre-trained model.

    Args:
        image_path (str): Path to the input image.

    Returns:
        list: A list containing the predicted class label in French and English.
    """
    model = Classifier_mnist()
    model_chemin = os.path.join('animalguessinggame', 'models', 'classifierVF_minst.pt')
    model = torch.load(model_chemin, map_location='cpu')
    model.eval()
    ans = []
    for x in image__list_path:
        x = 'animalguessinggame/static' + x
        image = Image.open(x)
        T1 = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
        image = T1(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            x = model(image)[0]

        ans.append(int(torch.argmax(x).item()))
    return [concat(ans), concat_eng(ans)]
########


animaux_liste_eng = [
    'antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat',
    'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog',
    'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox',
    'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus',
    'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard',
    'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter',
    'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine',
    'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep',
    'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf',
    'wombat', 'woodpecker', 'zebra'
]

animaux_eng = {i: animal for i, animal in enumerate(animaux_liste_eng)}

animaux_liste = [
    'antilope', 'blaireau', 'chauve-souris', 'ours', 'abeille', 'scarabée', 'bison', 'sanglier', 'papillon', 'chat',
    'chenille', 'chimpanzé', 'cafard', 'vache', 'coyote', 'crabe', 'corbeau', 'cerf', 'chien', 'dauphin', 'âne',
    'libellule', 'canard', 'aigle', 'éléphant', 'flamant rose', 'mouche', 'renard', 'chèvre', 'poisson rouge', 'oie',
    'gorille', 'sauterelle', 'hamster', 'lièvre', 'hérisson', 'hippopotame', 'calao', 'cheval', 'colibri', 'hyène',
    'méduse', 'kangourou', 'koala', 'coccinelles', 'léopard', 'lion', 'lézard', 'homard', 'moustique', 
    'papillon de nuit', 'souris', 'pieuvre', 'okapi', 'orang-outan', 'loutre', 'hibou', 'bœuf', 'huître', 
    'panda', 'perroquet', 'pélican', 'pingouin', 'cochon', 'pigeon', 'porc-épic', 'opossum', 'raton laveur', 
    'rat', 'renne', 'rhinocéros', 'bécasse', 'hippocampe', 'phoque', 'requin', 'mouton', 'serpent', 'moineau', 
    'calmar', 'écureuil', 'étoile de mer', 'cygne', 'tigre', 'dinde', 'tortue', 'baleine', 'loup', 'wombat', 
    'pic-vert', 'zèbre'
]

animaux = {i: animal for i, animal in enumerate(animaux_liste)}


######
chiffre = {0: 'zéro',
           1: 'un',
           2: 'deux',
           3: 'trois',
           4: 'quatre',
           5: 'cinq',
           6: 'six',
           7: 'sept',
           8: 'huit',
           9: 'neuf',
           }

dizaine = {1: 'dix',
           2: 'vingt',
           3: 'trente',
           4: 'quarante',
           5: 'cinquante',
           6: 'soixante',
           7: 'soixante dix',
           8: 'quatre vingt',
           9: 'quatre vingt dix'
           }

dix_vingt = {0: 'dix',
             1: 'onze',
             2: 'douze',
             3: 'treize',
             4: 'quatorze',
             5: 'quinze',
             6: 'seize',
             7: 'dix sept',
             8: 'dix huit',
             9: 'dix neuf'}

# flake8: noqa:C901    
def concat(L):
    """
    Convert a list of integers to its French representation in words.

    Args:
        L (list): List of integers representing a number. The length of L should be 1, 2, 3, or 4.

    Returns:
        str: French representation of the input number in words.
    """
    n = len(L)
    string = []
    if n == 4:
        if L == [0, 0, 0, 0]:
            return 'zéro'
        # gestion des millierss
        if L[0] != 0:
            string.append(chiffre[L[0]]+' mille')
        # gestion des centaines
        if L[1] != 0 and L[1] != 1:
            if L[2] == 0 and L[3] == 0:
                string.append(chiffre[L[1]] + " " +'cents')
            else:
                string.append(chiffre[L[1]] + " " + 'cent')
        if L[1] == 1:
            string.append('cent')
        # gestion des dizaines et unités
        if L[2] != 7 and L[2] != 9 and L[2] != 0 and L[2] != 1:
            if L[2] != 8:
                string.append(dizaine[L[2]])
            if L[2] == 8:
                if L[3] == 0:
                    string.append("quatre vingts")
                if L[3] != 0:
                    string.append(dizaine[L[2]])
            if L[3] != 0:
                if L[3] == 1 and L[2] != 8:
                    string.append('et ' + chiffre[L[3]])
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
        # cas des dizaines nulles
        if L[2] == 0:
            if L[3] != 0:
                string.append(chiffre[L[3]])
    if n == 3:
        if L == [0, 0, 0]:
            return 'zéro'
        # gestion des centaines
        if L[0] != 0 and L[0] != 1:
            if L[1] == 0 and L[2] == 0:
                string.append(chiffre[L[0]]+" "+'cents')
            else:
                string.append(chiffre[L[0]]+" " + "cent")
        if L[0] == 1:
            string.append('cent')
        # gestion des dizaines et unités
        if L[1] != 7 and L[1] != 9 and L[1] != 0 and L[1] != 1:
            if L[1] != 8:
                string.append(dizaine[L[1]])
            if L[1] == 8:
                if L[2] == 0:
                    string.append("quatre vingts")
                if L[2] != 0:
                    string.append(dizaine[L[1]])
            if L[2] != 0:
                if L[2] == 1 and L[1] != 8:
                    string.append('et ' + chiffre[L[2]])
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
        # cas des dizaines nulles
        if L[1] == 0:
            if L[2] != 0:
                string.append(chiffre[L[2]])
    if n == 2:
        if L == [0, 0]:
            return 'zéro'
    # gestion des dizaines et unités
        if L[0] != 7 and L[0] != 9 and L[0] != 0 and L[0] != 1:
            if L[0] != 8:
                string.append(dizaine[L[0]])
            if L[0] == 8:
                if L[1] == 0:
                    string.append("quatre vingts")
                if L[1] != 0:
                    string.append(dizaine[L[0]])
            if L[1] != 0:
                if L[1] == 1 and L[0] != 8:
                    string.append('et ' + chiffre[L[1]])
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
        # cas des dizaines nulles
        if L[0] == 0:
            if L[1] != 0:
                string.append(chiffre[L[1]])
    if n == 1:
        string.append(chiffre[L[0]])
    if n not in [1, 2, 3, 4]:
        return 'ERROR'
    rep = ''
    for i in string:
        rep += i + ' '
    return rep[:-1]

def concat_eng(L):
    """
    Convert a list of integers to its English representation in words.

    Args:
        L (list): List of integers representing a number. The length of L should be 1, 2, 3, or 4.

    Returns:
        str: English representation of the input number in words.
    """
    n = len(L)
    string = []
    if n == 4:
        if L == [0, 0, 0, 0]:
            return 'zero'
        # Handling thousands
        if L[0] != 0:
            string.append(digits[L[0]] + ' thousand')
        # Handling hundreds
        if L[1] != 0 and L[1] != 1:
            string.append(digits[L[1]] + " " + 'hundred')
        if L[1] == 1:
            string.append('one hundred')
        # Handling tens and units
        if L[2] != 0:
            if L[2] == 1:
                if L[3] == 0:
                    string.append('ten')
                else:
                    string.append(teens[L[3]])
            else:
                string.append(tens[L[2]])
                if L[3] != 0:
                    string.append(digits[L[3]])
        elif L[3] != 0:
            string.append(digits[L[3]])
    elif n == 3:
        if L == [0, 0, 0]:
            return 'zero'
        # Handling hundreds
        if L[0] != 0 and L[0] != 1:
            string.append(digits[L[0]] + " " + 'hundred')
        if L[0] == 1:
            string.append('one hundred')
        # Handling tens and units
        if L[1] != 0:
            if L[1] == 1:
                if L[2] == 0:
                    string.append('ten')
                else:
                    string.append(teens[L[2]])
            else:
                string.append(tens[L[1]])
                if L[2] != 0:
                    string.append(digits[L[2]])
        elif L[2] != 0:
            string.append(digits[L[2]])
    elif n == 2:
        if L == [0, 0]:
            return 'zero'
        # Handling tens and units
        if L[0] != 0:
            if L[0] == 1:
                if L[1] == 0:
                    string.append('ten')
                else:
                    string.append(teens[L[1]])
            else:
                string.append(tens[L[0]])
                if L[1] != 0:
                    string.append(digits[L[1]])
        elif L[1] != 0:
            string.append(digits[L[1]])
    elif n == 1:
        string.append(digits[L[0]])
    else:
        return 'ERROR'

    rep = ''
    for i in string:
        rep += i + ' '
    return rep[:-1]


digits = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}
tens = {2: 'twenty', 3: 'thirty', 4: 'forty', 5: 'fifty', 6: 'sixty', 7: 'seventy', 8: 'eighty', 9: 'ninety'}
teens = {0: 'ten', 1: 'eleven', 2: 'twelve', 3: 'thirteen', 4: 'fourteen', 5: 'fifteen', 6: 'sixteen', 7: 'seventeen', 
         8: 'eighteen', 9: 'nineteen'}
