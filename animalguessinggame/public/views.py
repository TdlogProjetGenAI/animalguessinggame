# -*- coding: utf-8 -*-
"""Public section, including homepage and signup."""
from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
    jsonify
)
import random

#import pretty_midi
from scipy.io import wavfile
import IPython

from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import glob

import os
from flask_login import login_required, login_user, logout_user

from animalguessinggame.extensions import login_manager
from animalguessinggame.public.forms import LoginForm
from animalguessinggame.user.forms import RegisterForm
from animalguessinggame.user.models import User
from animalguessinggame.utils import flash_errors
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os 
from flask import current_app
#from pydub import AudioSegment
import numpy as np
from flask import redirect, url_for, render_template
from flask import jsonify
from flask import render_template, request
import time
from flask import redirect
from flask import session
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField

import numpy as np
from scipy.io.wavfile import write
from .classif_animals10 import ResNetClassifier, classifie_animals10, classifie_animals90, Classifier_mnist, VAE, classifie_mnist
from .levenstein import distance_levenstein
#from .bach import F_get_max_temperature, F_convert_midi_2_list, F_sample_new_sequence
blueprint = Blueprint("public", __name__, static_folder="../static")

import __main__

class compt():
    def __init__(self):
        self.k = 0

    def incr(self):
        self.k += 1

    def value(self):
        return self.k
    
    def value_to_sero(self):
        self.k = 0
        

setattr(__main__, "ResNetClassifier", ResNetClassifier)
setattr(__main__, "Classifier_mnist", Classifier_mnist)
setattr(__main__, "VAE", VAE)
setattr(__main__, "compteur", compt)

class GenerateImageForm(FlaskForm):
    username = StringField('Username') 
    password = PasswordField('Password')  
    prompt = StringField('Prompt')
    submit = SubmitField('Soumettre')


@login_manager.user_loader
def load_user(user_id):
    """Load user by ID."""
    return User.get_by_id(int(user_id))


@blueprint.route("/", methods=["GET", "POST"])
def home():
    """Home page."""
    form = LoginForm(request.form)
    current_app.logger.info("Hello from the home page!")
    # Handle logging in
    if request.method == "POST":
        if form.validate_on_submit():
            login_user(form.user)
            flash("You are logged in.", "success")
            redirect_url = request.args.get("next") or url_for("user.members")
            return redirect(redirect_url)
        else:
            flash_errors(form)
    return render_template("public/home.html", form=form)


@blueprint.route("/logout/")
@login_required
def logout():
    """Logout."""
    logout_user()
    flash("You are logged out.", "info")
    return redirect(url_for("public.home"))


@blueprint.route("/register/", methods=["GET", "POST"])
def register():
    """Register new user."""
    form = RegisterForm(request.form)
    if form.validate_on_submit():
        User.create(
            username=form.username.data,
            email=form.email.data,
            password=form.password.data,
            active=True,
        )
        flash("Thank you for registering. You can now log in.", "success")
        return redirect(url_for("public.home"))
    else:
        flash_errors(form)
    return render_template("public/register.html", form=form)


@blueprint.route("/about/")
def about():
    """About page."""
    form = LoginForm(request.form)
    return render_template("public/about.html", form=form)



##animals10
@blueprint.route('/generate_image/', methods=['GET', 'POST'])
def generate_image():
    form = GenerateImageForm()
    image_path = session.get('current_image', get_random_image_path())
    congratulations_message = None

    if form.validate_on_submit():
        prompt_value = form.prompt.data.lower()
        anws = classifie_animals10(image_path)
        if prompt_value == anws[0] or prompt_value == anws[1]:
            congratulations_message = "Félicitations, vous avez gagné !"
        elif (distance_levenstein(prompt_value, anws[0]) <= 2 or distance_levenstein(prompt_value, anws[1]) <= 2):
            congratulations_message = "Tu chauffes !"
        else:
            congratulations_message = "Essaie encore !"
    session['current_image'] = image_path

    return render_template('public/image_page.html', image_path=image_path, congratulations_message=congratulations_message, form=form)

@blueprint.route('/replay/', methods=['GET'])
def replay():
    session.pop('current_image', None)
    return redirect(url_for('public.generate_image'))


# Votre route Flask pour afficher la liste des animaux
@blueprint.route('/liste_animals10', methods=['GET'])
def liste_animals10():
    animals10_dict = {0: "chien", 1: "cheval", 2: "éléphant", 3: "papillon", 4: "poule", 5: "chat", 6: "vache", 7: "mouton", 8: "araignée", 9: "écureuil"}
    return render_template('public/liste_animals10.html', animals10_dict=animals10_dict)



####animals90


@blueprint.route('/generate_image_hard/', methods=['GET', 'POST'])
def generate_image_hard():
    form = GenerateImageForm()
    image_path = session.get('current_image_hard', get_random_image_hard_path())
    congratulations_message = None

    if form.validate_on_submit():
        prompt_value = form.prompt.data.lower()
        anws = classifie_animals90(image_path)
        if prompt_value == anws[0] or prompt_value == anws[1]:
            congratulations_message = "Félicitations, vous avez gagné!"
        elif (distance_levenstein(prompt_value, anws[0]) <= 2 or distance_levenstein(prompt_value, anws[1]) <= 2):
            congratulations_message = "Tu chauffes !"           
        else:
            congratulations_message = "Essaie encore"
    session['current_image_hard'] = image_path

    return render_template('public/image_page_hard.html', image_path=image_path, congratulations_message=congratulations_message, form=form)

@blueprint.route('/replay_hard/', methods=['GET'])
def replay_hard():
    session.pop('current_image_hard', None)
    return redirect(url_for('public.generate_image_hard'))


@blueprint.route('/liste_animals90', methods=['GET'])
def liste_animals90():
    animals90 = [
    'antilope', 'blaireau', 'chauve-souris', 'ours', 'abeille', 'scarabée', 'bison', 'sanglier', 'papillon', 'chat',
    'chenille', 'chimpanzé', 'cafard', 'vache', 'coyote', 'crabe', 'corbeau', 'cerf', 'chien', 'dauphin', 'âne',
    'libellule', 'canard', 'aigle', 'éléphant', 'flamant rose', 'mouche', 'renard', 'chèvre', 'poisson rouge', 'oie',
    'gorille', 'sauterelle', 'hamster', 'lièvre', 'hérisson', 'hippopotame', 'calao', 'cheval', 'colibri', 'hyène',
    'méduse', 'kangourou', 'koala', 'coccinelles', 'léopard', 'lion', 'lézard', 'homard', 'moustique', 'papillon de nuit',
    'souris', 'pieuvre', 'okapi', 'orang-outan', 'loutre', 'hibou', 'bœuf', 'huître', 'panda', 'perroquet', 'pélécaniformes',
    'pingouin', 'cochon', 'pigeon', 'porc-épic', 'opossum', 'raton laveur', 'rat', 'renne', 'rhinocéros', 'bécasse',
    'hippocampe', 'phoque', 'requin', 'mouton', 'serpent', 'moineau', 'calmar', 'écureuil', 'étoile de mer', 'cygne',
    'tigre', 'dinde', 'tortue', 'baleine', 'loup', 'wombat', 'pic-vert', 'zèbre'
    ]
    animals90_dict = {i: animal for i, animal in enumerate(animals90)}
    return render_template('public/liste_animals90.html', animals90_dict=animals90_dict)


###number
@blueprint.route('/generate_number/', methods=['GET', 'POST'])
def generate_number():
    form = GenerateImageForm()
    current_method_name = session.get('current_method', 'get_random_gen_number_path')
    current_method = globals()[current_method_name]
    images_list_path = session.get('current_images', current_method())
    congratulations_message = None
    if form.validate_on_submit():
        prompt_value = form.prompt.data.lower()
        anws = classifie_mnist(images_list_path)
        if prompt_value == anws[0] or prompt_value == anws[1]:
            congratulations_message = "Félicitations, vous avez gagné!"
        elif (distance_levenstein(prompt_value, anws[0]) <= 2 or distance_levenstein(prompt_value, anws[1]) <= 2):
            congratulations_message = "Tu chauffes !"
        else:
            congratulations_message = "Essaie encore"
    session['current_images'] = images_list_path
    return render_template('public/number_page.html', images_list_path=images_list_path, congratulations_message=congratulations_message, form=form)

@blueprint.route('/replay_number/', methods=['GET'])
def replay_number():
    session.pop('current_images', None)
    return redirect(url_for('public.generate_number'))

@blueprint.route('/toggle_method/', methods=['POST'])
def toggle_method():
    current_method_name = session.get('current_method', 'get_random_gen_number_path')
    
    # Change the method name based on the current state
    if current_method_name == 'get_random_gen_number_path':
        session['current_method'] = 'get_random_number_path'
    else:
        session['current_method'] = 'get_random_gen_number_path'

    # Redirect back to the previous page or the main page
    return redirect(request.referrer or url_for('public.generate_number'))


def get_random_image_path():
    images_folder = os.path.join(current_app.root_path, 'static', 'images_animals10')
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if image_files:
        random_image = random.choice(image_files)
        return f'/images_animals10/{random_image}'
    else:
        return None

def get_random_image_hard_path():
    images_folder = os.path.join(current_app.root_path, 'static', 'images_animals90')
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if image_files:
        random_image = random.choice(image_files)
        return f'/images_animals90/{random_image}'
    else:
        return None
    

def get_random_number_path():
    images_folder = os.path.join(current_app.root_path, 'static', 'images_number')
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images_list = []
    if image_files:
        number_images = random.randint(1,4)
        for k in range(number_images):
            random_image = random.choice(image_files)
            images_list.append(f'/images_number/{random_image}')
        return images_list
    else:
        return None
    

def get_random_gen_number_path():
    images_list = []
    number_images = random.randint(1,4)
    for k in range(number_images):
        random_image_path = gen_number_path()
        images_list.append(random_image_path)
    return images_list
k = compt()
def gen_number_path():
    
    model_chemin = os.path.join('animalguessinggame', 'models', 'VAE_MINST.pt')
    model = torch.load(model_chemin, map_location='cpu')
    model_chemin_classif = os.path.join('animalguessinggame', 'models', 'classifierVF_MINST.pt')
    classif = torch.load(model_chemin_classif, map_location='cpu')
    z3 = torch.randn(20, 20)
    with torch.no_grad():
        generated_images = model.decode(z3)
    gen = [x.view(1, 28, 28).unsqueeze(0) for x in generated_images]
    with torch.no_grad():
        x = [classif(x) for x in gen]
    h = [float(torch.max(prob)) for prob in x]
    best_gen_index = h.index(max(h))
    image_gen_vf = generated_images[best_gen_index]
    output_directory = os.path.join(current_app.root_path, 'static', 'images_number')
    #os.makedirs(output_directory, exist_ok=True)
    image_tensor = image_gen_vf.view(28, 28).cpu().numpy()
    image_pil = Image.fromarray((image_tensor * 255).astype('uint8'))
    k.incr()
    if k.value() >= 10:
        k.value_to_sero()
    output_filename = os.path.join(output_directory, 'output_image'+f'{k.value()}'+'.png')
    image_pil.save(output_filename)
    
    output_filename = f'/images_number/output_image{k.value()}.png'
    return output_filename


# Configurez le répertoire de téléchargement
UPLOAD_FOLDER = 'animalguessinggame/static/images_animals10'
UPLOAD_FOLDER_HARD = 'animalguessinggame/static/images_animals90'
UPLOAD_FOLDER_NUMBER = 'animalguessinggame/static/images_number'
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Liste des extensions de fichiers autorisées
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


from werkzeug.utils import secure_filename




@blueprint.route('/upload_images', methods=['POST'])
def upload_images():
    if 'images' not in request.files:
        flash('Aucun fichier téléchargé')
        return redirect(url_for('public.generate_image'))

    files = request.files.getlist('images')

    if not files or all(file.filename == '' for file in files):
        flash('Aucun fichier sélectionné')
        return redirect(url_for('public.generate_image'))

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

    flash('Images téléchargées avec succès')
    return redirect(url_for('public.generate_image'))

@blueprint.route('/upload_images_hard', methods=['POST'])
def upload_images_hard():
    if 'images' not in request.files:
        flash('Aucun fichier téléchargé')
        return redirect(url_for('public.generate_image_hard'))

    files = request.files.getlist('images')

    if not files or all(file.filename == '' for file in files):
        flash('Aucun fichier sélectionné')
        return redirect(url_for('public.generate_image_hard'))

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER_HARD, filename)
            file.save(filepath)

    flash('Images téléchargées avec succès')
    return redirect(url_for('public.generate_image_hard'))



@blueprint.route('/upload_images_number', methods=['POST'])
def upload_images_number():
    if 'images' not in request.files:
        flash('Aucun fichier téléchargé')
        return redirect(url_for('public.generate_number'))

    files = request.files.getlist('images')

    if not files or all(file.filename == '' for file in files):
        flash('Aucun fichier sélectionné')
        return redirect(url_for('public.generate_number'))

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER_NUMBER, filename)
            file.save(filepath)

    flash('Images téléchargées avec succès')
    return redirect(url_for('public.generate_number'))





########################music gen
# @blueprint.route('/generate_music/', methods=['GET', 'POST'])
# def generate_music():
#     max_midi_T_x = 1000
#     DIR = './'
#     import urllib.request
#     midi_file_l = ['cs1-2all.mid', 'cs5-1pre.mid', 'cs4-1pre.mid', 'cs3-5bou.mid', 'cs1-4sar.mid', 'cs2-5men.mid', 'cs3-3cou.mid', 'cs2-3cou.mid', 'cs1-6gig.mid', 'cs6-4sar.mid', 'cs4-5bou.mid', 'cs4-3cou.mid', 'cs5-3cou.mid', 'cs6-5gav.mid', 'cs6-6gig.mid', 'cs6-2all.mid', 'cs2-1pre.mid', 'cs3-1pre.mid', 'cs3-6gig.mid', 'cs2-6gig.mid', 'cs2-4sar.mid', 'cs3-4sar.mid', 'cs1-5men.mid', 'cs1-3cou.mid', 'cs6-1pre.mid', 'cs2-2all.mid', 'cs3-2all.mid', 'cs1-1pre.mid', 'cs5-2all.mid', 'cs4-2all.mid', 'cs5-5gav.mid', 'cs4-6gig.mid', 'cs5-6gig.mid', 'cs5-4sar.mid', 'cs4-4sar.mid', 'cs6-3cou.mid']
#     for midi_file in midi_file_l:
#     #if os.path.isfile(DIR + midi_file) is None:
#         urllib.request.urlretrieve ("http://www.jsbach.net/midi/" + midi_file, DIR + midi_file)

#     midi_file_l = glob.glob(DIR + 'cs*.mid')

#     X_list = [] 
#     X_list = F_convert_midi_2_list(midi_file_l, max_midi_T_x)
#     model_chemin=os.path.join('animalguessinggame', 'models', 'bach_modele.h5')
#     model = load_model(model_chemin)

#     sum_v = np.zeros(n_x)
#     for X_ohe in X_list: 
#         sum_v += np.sum(X_list[0], axis=0)
#         prior_v = sum_v/np.sum(sum_v)

#     note_l, prediction_l = F_sample_new_sequence(model, prior_v)

#     new_midi_data = pretty_midi.PrettyMIDI()
#     cello_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
#     cello = pretty_midi.Instrument(program=cello_program)
#     time = 0
#     step = 0.3
#     for note_number in note_l:
#         myNote = pretty_midi.Note(velocity=100, pitch=note_number, start=time, end=time+step)
#         cello.notes.append(myNote)
#         time += step
#     new_midi_data.instruments.append(cello)
    

#     # Assume that new_midi_data.synthesize(fs=44100) returns a NumPy array
#     audio_data_np = new_midi_data.synthesize(fs=44100)

#     # Normalize the audio data to the range [-32768, 32767] for int16 format
#     normalized_audio_data = (audio_data_np * 32767).astype(np.int16)
#     import base64
#     audio_base64 = base64.b64encode(normalized_audio_data).decode('utf-8')

#     # Return the base64-encoded audio data as JSON
#     return jsonify({'audio_data': audio_base64})