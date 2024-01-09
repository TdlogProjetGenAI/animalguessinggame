"""This code was used to generate artificial pictures on Google Colab."""

import keras_cv
import matplotlib.pyplot as plt


model = keras_cv.models.StableDiffusion(img_height=512, img_width=512, jit_compile=True)

def save_image(prompt, k, folder):
    """
    Save an image generated by the model based on the given prompt.

    Parameters:
    - prompt (str): The prompt used to generate the image.
    - k (int): The index or identifier for the image.
    - folder (str): The folder where the image will be saved.

    Returns:
    None

    This function generates an image using the provided prompt with the help of the model's
    text_to_image method. The generated image is then displayed using Matplotlib,
    and it is saved to the specified folder with a filename based on the index 'k'.
    """
    image = model.text_to_image(prompt=prompt, batch_size=1)
    plt.imshow(image[0])
    plt.axis("off")
    plt.savefig(f'/animalguessinggame/static/IA_notIA/{folder}/{k}.png')


elem_objects = [
    "Arbre", "Voiture", "Mer", "Plage", "Pomme", "Ordinateur", "Montagne", "Lampe", "Chien", "Café",
    "Livre", "Avion", "Soleil", "Étoiles", "Tasse", "Chaussures", "Parc", "Horloge", "Papillon", "Téléphone",
    "Lune", "Piano", "Océan", "Chaise", "Télévision", "Appareil photo", "Pizza", "Fleurs", "Pont", "Écouteurs",
    "Vélo", "Lunettes de soleil", "Lac", "Chat", "Guitare", "Sac à dos", "Nuages", "Bus", "Feu de camp", "Trousse",
    "Journal", "Ballon", "Porte", "Feuille", "Échelle", "Bureau", "Parapluie", "Caméra", "Portefeuille", "Champ",
    "Table", "Arc-en-ciel", "Microphone", "Légumes", "Plante", "Piscine", "Papier", "Globe", "Veste",
    "Casquette", "Cour", "Champignon", "Feu de circulation", "Bulles", "Miroir", "Bateau", "Chalet", "Évier",
    "Pont suspendu", "Carte", "Glace", "Château de sable", "Tennis", "Boussole", "Chemineé", "Gâteau", "Collier",
    "Radio", "Ampoule", "Clé", "Aspirateur", "Skateboard", "Bouteille", "Voie ferrée", "Épices", "Brosse à dents",
    "Réfrigérateur", "Nuage de pluie", "Statue", "Sac à main", "Crayon", "Douche", "Couronne", "Gants", "Croissant",
    "Fusée", "Table de picnic", "Volcan", "Tapis"
]

for i, obj in enumerate(elem_objects, start=1):
    prompt = f'{obj} super réaliste'
    save_image(prompt, i, "IA")

elem_emotions = [
    "Émotion", "Couleurs pastel", "Mélodie de la nature", "Port",
    "Horizon infini", "Soleil levant", "Douce brise"
]

for i, emo in enumerate(elem_emotions, start=1):
    prompt = f'peinture {emo} façon Monet'
    save_image(prompt, i, "IA_Monet")
