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


import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from flask_login import login_required, login_user, logout_user, current_user

from animalguessinggame.extensions import login_manager
from animalguessinggame.public.forms import LoginForm, GenerateImageForm2
from animalguessinggame.user.forms import RegisterForm
from animalguessinggame.user.models import User
from animalguessinggame.utils import flash_errors
from animalguessinggame.database import Score, ScoreHard,ScoreHardClock, ScoreNum
#from animalguessinggame.app import db
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from flask import current_app
#from pydub import AudioSegment
from flask import redirect, url_for, render_template
from flask import jsonify
from flask import render_template, request
import time
from flask import redirect
from flask import session
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from flask import Flask
import numpy as np
from scipy.io.wavfile import write
from .classif_animals10 import ResNetClassifier, classifie_animals10, classifie_animals90, Classifier_mnist, VAE, classifie_mnist
from .levenstein import distance_levenstein
#from flask_sqlalchemy import SQLAlchemy
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
    
    def value_to_zero(self):
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
# Ajoutez cette importation à votre fichier Python
from flask import session

# ...

@blueprint.route('/generate_image/', methods=['GET', 'POST'])
def generate_image():
    form = GenerateImageForm()
    image_path = session.get('current_image', get_random_image_path())
    congratulations_message = None
    attempts = session.get('attempts', 3)
    win = session.get('win', False)
    played = session.get('played', False)
    score = session.get('score', 0)
    top_scores = Score.get_top_scores()
    sound_file = 'sound_animals10/chat.mp3'
    play_win_sound = True
    if attempts>0 and form.validate_on_submit():
        prompt_value = form.prompt.data.lower()
        anws = classifie_animals10(image_path)
        if prompt_value == anws[0] or prompt_value == anws[1]:
            congratulations_message = "Félicitations, vous avez gagné !"
            win = True
            play_win_sound = True
            sound_file=f'sound_animals10/{anws[0]}.mp3'         
            if attempts == 3 and not played:
                score+=8
            elif attempts == 2 and not played:
                score+=5
            elif attempts == 1 and not played:
                score+=3
            played = True
        else:
            attempts -= 1
            if attempts == 0:
                congratulations_message = f"Dommage. La réponse était {anws[0]}."
                user_id = current_user.id if current_user.is_authenticated else "invite"
                new_score = Score(user_id=user_id, score_value=score)
                new_score.save()
                score=0
                played=True
            elif (distance_levenstein(prompt_value, anws[0]) <= 2 or distance_levenstein(prompt_value, anws[1]) <= 2):
                congratulations_message = f"Tu chauffes ! Il vous reste {attempts} essais."
                play_win_sound = True
                sound_file=f'sound_animals10/tu_chauffes.mp3'
            else:
                congratulations_message = f"Essaie encore ! Il vous reste {attempts} essais."
                play_win_sound = True
                sound_file=f'sound_animals10/essaie_encore.mp3'
        top_scores = Score.get_top_scores()
    session['win'] = win
    session['played'] = played
    session['attempts'] = attempts
    session['current_image'] = image_path
    session['score'] = score

    return render_template('public/image_page.html', image_path=image_path, congratulations_message=congratulations_message, form=form, score=score, top_scores=top_scores,play_win_sound = play_win_sound, sound_file = sound_file)


@blueprint.route('/replay/', methods=['GET'])
def replay():

    win = session.get('win', False)
    if not win:
        # L'utilisateur n'a pas encore gagné, réinitialisez le score
        user_id = current_user.id if current_user.is_authenticated else "invite"
        new_score = Score(user_id=user_id, score_value=session['score'])
        new_score.save()
        top_scores = Score.get_top_scores()
        session['score'] = 0

    session['attempts'] = 3
    session['current_image'] = get_random_image_path()
    session['win'] = False  # Réinitialisez la variable win à False
    session['played'] = False
    return redirect(url_for('public.generate_image'))

@blueprint.route('/liste_animals10', methods=['GET'])
def liste_animals10():
    animals10_dict = {0: "chien", 1: "cheval", 2: "éléphant", 3: "papillon", 4: "poule", 5: "chat", 6: "vache", 7: "mouton", 8: "araignée", 9: "écureuil"}
    score = session.get('score', 0)
    return render_template('public/liste_animals10.html', animals10_dict=animals10_dict, score=score)

@blueprint.route('/top_scores/', methods=['GET'])
def top_scores():
    top_scores = Score.query.order_by(Score.score_value.desc()).limit(10).all()
    return render_template('public/top_scores.html', top_scores=top_scores)

####animals90

@blueprint.route('/generate_image_hard', methods=['GET','POST'])
def generate_image_hard():
    form = GenerateImageForm()
    image_path = session.get('current_image_hard', get_random_image_hard_path())
    congratulations_message = None
    attempts = session.get('attempts_hard', 3)
    win = session.get('win', False)
    played = session.get('played', False)
    score = session.get('score', 0)
    top_scores = ScoreHard.get_top_scores()


    if attempts>0 and form.validate_on_submit():
        prompt_value = form.prompt.data.lower()
        anws = classifie_animals90(image_path)
        if prompt_value == anws[0] or prompt_value == anws[1]:
            congratulations_message = "Félicitations, vous avez gagné !"
            win = True
            
            if attempts == 3 and not played:
                score+=8
            elif attempts == 2 and not played:
                score+=5
            elif attempts == 1 and not played:
                score+=3
            played = True

        else:
            attempts -= 1
            if attempts == 0:
                congratulations_message = f"Dommage. La réponse était {anws[0]}."
                user_id = current_user.id if current_user.is_authenticated else "invite"
                new_score = ScoreHard(user_id=user_id, score_value=score)
                new_score.save()
                score=0
                played=True
            elif (distance_levenstein(prompt_value, anws[0]) <= 2 or distance_levenstein(prompt_value, anws[1]) <= 2):
                congratulations_message = f"Tu chauffes ! Il vous reste {attempts} essais."
            else:
                congratulations_message = f"Essaie encore ! Il vous reste {attempts} essais."
        top_scores = ScoreHard.get_top_scores()
    session['attempts_hard'] = attempts
    session['current_image_hard'] = image_path
    session['win'] = win
    session['played'] = played
    session['score'] = score
    return render_template('public/image_page_hard.html', image_path=image_path, congratulations_message=congratulations_message, form=form, score = score, top_scores = top_scores)




@blueprint.route('/replay_hard', methods=['GET'])
def replay_hard():
    session.pop('current_image_hard', None)
    win = session.get('win', False)
    if not win:
        # L'utilisateur n'a pas encore gagné, réinitialisez le score
        user_id = current_user.id if current_user.is_authenticated else "invite"
        new_score = ScoreHard(user_id=user_id, score_value=session['score'])
        new_score.save()
        top_scores = ScoreHard.get_top_scores()
        session['score'] = 0

    session['attempts_hard'] = 3
    session['current_image'] = get_random_image_path()
    session['win'] = False  # Réinitialisez la variable win à False
    session['played'] = False
    return redirect(url_for('public.generate_image_hard'))

@blueprint.route('/liste_animals90', methods=['GET'])
def liste_animals90():
    animals90 = [
    'abeille', 'aigle', 'âne', 'antilope', 'baleine', 'bécasse', 'blaireau', 'bison', 'bœuf', 'calamar', 'calao', 'canard',
    'cafard', 'cerf', 'chat', 'chauve-souris', 'cheval', 'chenille', 'chimpanzé', 'chien', 'cochon', 'colibri', 'coq',
    'corbeau', 'coyote', 'crapaud', 'crabe', 'cygne', 'dauphin', 'dinosaure', 'dindon', 'éléphant', 'écureuil',
    'étoile de mer', 'flamant rose', 'fourmi', 'gorille', 'hamster', 'hérisson', 'hippocampe', 'hibou', 'homard', 'hyène',
    'kangourou', 'koala', 'léopard', 'lézard', 'lion', 'loup', 'loutre', 'méduse', 'moineau', 'moustique', 'mouche',
    'mouton', 'octopus', 'okapi', 'oie', 'opossum', 'orque', 'ours', 'panda', 'papillon', 'papillon de nuit', 'paon',
    'perroquet', 'phoque', 'pigeon', 'pingouin', 'pieuvre', 'poisson rouge', 'porc-épic', 'rat', 'raton laveur', 'renard',
    'renne', 'requin', 'rhinocéros', 'sanglier', 'sauterelle', 'scarabée', 'serpent', 'souris', 'tigre', 'tortue', 'vache',
    'wombat', 'zèbre'
    ]
    animals90_dict = {i: animal for i, animal in enumerate(animals90)}
    return render_template('public/liste_animals90.html', animals90_dict=animals90_dict)

#############Clock#####################

@blueprint.route('/generate_image_hard_clock', methods=['GET','POST'])
def generate_image_hard_clock():
    form = GenerateImageForm()
    image_path = session.get('current_image_hard_clock', get_random_image_hard_path())
    congratulations_message = None
    attempts = session.get('attempts_hard_clock', 1)
    win = session.get('win_clock', False)
    played = session.get('played_clock', False)
    score = session.get('score_clock', 0)
    top_scores = ScoreHardClock.get_top_scores()

    if attempts>0 and form.validate_on_submit():
        prompt_value = form.prompt.data.lower()
        anws = classifie_animals90(image_path)
        if prompt_value == anws[0] or prompt_value == anws[1]:
            congratulations_message = "Félicitations, vous avez gagné !"
            win = True
            
            score += 1
            played = True

        else:
            attempts -= 1
            if attempts == 0:
                congratulations_message = f"Dommage. La réponse était {anws[0]}."
            elif (distance_levenstein(prompt_value, anws[0]) <= 2 or distance_levenstein(prompt_value, anws[1]) <= 2):
                congratulations_message = f"Tu chauffes ! Il vous reste {attempts} essais."
            else:
                congratulations_message = f"Essaie encore ! Il vous reste {attempts} essais."
    session['attempts_hard_clock'] = attempts
    session['current_image_hard_clock'] = image_path
    session['win_clock'] = win
    session['played_clock'] = played
    session['score_clock'] = score
    return render_template('public/image_page_hard_clock.html', image_path=image_path, congratulations_message=congratulations_message, form=form, score = score)




@blueprint.route('/replay_hard_clock', methods=['GET'])
def replay_hard_clock():
    session.pop('current_image_hard_clock', None)
    win = session.get('win_clock', False)
    if not win:
        # L'utilisateur n'a pas encore gagné, réinitialisez le score
        user_id = current_user.id if current_user.is_authenticated else "invite"
        new_score = ScoreHardClock(user_id=user_id, score_value=session['score_clock'])
        new_score.save()
        session['score_clock'] = 0

    session['attempts_hard_clock'] = 3
    session['current_image_clock'] = get_random_image_path()
    session['win_clock'] = False  # Réinitialisez la variable win à False
    session['played_clock'] = False
    return redirect(url_for('public.generate_image_hard_clock'))


    
@blueprint.route('/upload_images_hard_clock', methods=['POST'])
def upload_images_hard_clock():
    if 'images' not in request.files:
        flash('Aucun fichier téléchargé')
        return redirect(url_for('public.generate_image_hard_clock'))

    files = request.files.getlist('images')

    if not files or all(file.filename == '' for file in files):
        flash('Aucun fichier sélectionné')
        return redirect(url_for('public.generate_image_hard_clock'))

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER_HARD, filename)
            file.save(filepath)

    flash('Images téléchargées avec succès')
    return redirect(url_for('public.generate_image_hard_clock'))




#################number#######################
@blueprint.route('/generate_number/', methods=['GET', 'POST'])
def generate_number():
    form = GenerateImageForm()
    current_method_name = session.get('current_method', 'get_random_gen_number_path')
    current_method = globals()[current_method_name]
    images_list_path = session.get('current_images', current_method())
    anws = classifie_mnist(images_list_path)
    congratulations_message = None
    attempts = session.get('attempts_number', 3)
    win = session.get('win', False)
    played = session.get('played', False)
    score = session.get('score', 0)
    top_scores = ScoreNum.get_top_scores()

    if attempts>0 and form.validate_on_submit():
        prompt_value = form.prompt.data.lower()
        
        if prompt_value == anws[0] or prompt_value == anws[1]:
            congratulations_message = "Félicitations, vous avez gagné !"
            win = True
            
            if attempts == 3 and not played:
                score+=8
            elif attempts == 2 and not played:
                score+=5
            elif attempts == 1 and not played:
                score+=3
            played = True

        else:
            attempts -= 1
            if attempts == 0:
                congratulations_message = f"Dommage. La réponse était {anws[0]}."
                user_id = current_user.id if current_user.is_authenticated else "invite"
                new_score = ScoreNum(user_id=user_id, score_value=score)
                new_score.save()
                score=0
                played=True
            elif (distance_levenstein(prompt_value, anws[0]) <= 2 or distance_levenstein(prompt_value, anws[1]) <= 2):
                congratulations_message = f"Tu chauffes ! Il vous reste {attempts} essais."
            else:
                congratulations_message = f"Essaie encore ! Il vous reste {attempts} essais."
        top_scores = ScoreNum.get_top_scores()
    session['attempts_number'] = attempts
    session['current_images'] = images_list_path
    session['score'] = score
    session['win'] = win
    session['played'] = played
    return render_template('public/number_page.html', images_list_path=images_list_path, congratulations_message=congratulations_message, form=form, score = score, top_scores = top_scores)

@blueprint.route('/replay_number/', methods=['GET'])
def replay_number():
    session['attempts_number'] = 3
    session.pop('current_images', None)
    win = session['win']
    if not win:
        # L'utilisateur n'a pas encore gagné, réinitialisez le score
        user_id = current_user.id if current_user.is_authenticated else "invite"
        new_score = ScoreNum(user_id=user_id, score_value=session['score'])
        new_score.save()
        top_scores = ScoreNum.get_top_scores()
        session['score'] = 0
    session['win'] = False  # Réinitialisez la variable win à False
    session['played'] = False
    return redirect(url_for('public.generate_number'))


@blueprint.route('/toggle_method/', methods=['POST'])
def toggle_method():
    current_method_name = session.get('current_method', 'get_random_gen_number_path')
    if current_method_name == 'get_random_gen_number_path':
        session['current_method'] = 'get_random_number_path'
    else:
        session['current_method'] = 'get_random_gen_number_path'
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

def number_path():
    images_folder = os.path.join(current_app.root_path, 'static', 'images_number')
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if image_files:
        random_image = random.choice(image_files)
        return f'images_number/{random_image}'
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
    image_tensor = image_gen_vf.view(28, 28).cpu().numpy()
    image_pil = Image.fromarray((image_tensor * 255).astype('uint8'))
    k.incr()
    if k.value() >= 20:
        k.value_to_zero()
    output_filename = os.path.join(output_directory, 'output_image'+f'{k.value()}'+'.png')
    image_pil.save(output_filename)
    
    output_filename = f'/images_number/output_image{k.value()}.png'
    return output_filename


UPLOAD_FOLDER = 'animalguessinggame/static/images_animals10'
UPLOAD_FOLDER_HARD = 'animalguessinggame/static/images_animals90'
UPLOAD_FOLDER_NUMBER = 'animalguessinggame/static/images_number'
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







@blueprint.route('/guessai/', methods=['GET', 'POST'])
def guessai():
    form = GenerateImageForm2()
   # played = session.get('played', False)
    if random.choice([True, False]):
        AI = session.get('AI', True)
        image_path = session.get('current_image_guessai', gen_number_path())
    else:
        AI = session.get('AI', False)
        image_path = session.get('current_image_guessai', number_path())

    congratulations_message = None

    if form.validate_on_submit():

        is_ia = form.is_ia
        if AI:
            if is_ia:
                congratulations_message = "Félicitations ! L'image a été générée par notre IA" #elle
            else:
                congratulations_message = "Perdu ! L'image a été générée par notre IA" 
        else:
            if is_ia:
                congratulations_message = "Perdu ! L'image n'a pas été générée par notre IA" #elle
            else:
                congratulations_message = "Félicitations ! L'image n'a pas été générée par notre IA"
#        congratulations_message = "boucle2"

    session['current_image_guessai'] = image_path
    return render_template('public/guessai.html', image_path=image_path, congratulations_message=congratulations_message, form=form)


@blueprint.route('/replay_new_game/', methods=['GET'])
def replay_guessai():

    session.pop('image_guessai', None)
    if random.choice([True, False]):
        session['AI'] = True
        session['current_image_guessai'] = gen_number_path()
    else:
        session['AI'] = False
        session['current_image_guessai'] = number_path()
    return redirect(url_for('public.guessai'))


@blueprint.route('/guessai_cifar/', methods=['GET', 'POST'])
def guessai_cifar():
    form = GenerateImageForm2()
   # played = session.get('played', False)
    if random.choice([True, False]):
        AI = session.get('AI', True)
        image_path = session.get('current_image_cifar', get_random_image_cifar_ai())
    else:
        AI = session.get('AI', False)
        image_path = session.get('current_image_cifar', get_random_image_cifar_real())

    congratulations_message = None

    if form.validate_on_submit():

        is_ia = form.is_ia
        if AI:
            if is_ia:
                congratulations_message = "Félicitations ! L'image a été générée par notre IA" #elle
            else:
                congratulations_message = "Perdu ! L'image a été générée par notre IA" 
        else:
            if is_ia:
                congratulations_message = "Perdu ! L'image n'a pas été générée par notre IA" #elle
            else:
                congratulations_message = "Félicitations ! L'image n'a pas été générée par notre IA"
#        congratulations_message = "boucle2"

    session['current_image_cifar'] = image_path
    return render_template('public/guessai_cifar.html', image_path=image_path, congratulations_message=congratulations_message, form=form)

@blueprint.route('/replay_new_game_cifar/', methods=['GET'])
def replay_guessai_cifar():
    if random.choice([True, False]):
        session['AI_cifar'] = True
        session['current_image_cifar'] = get_random_image_cifar_ai()
    else:
        session['AI_cifar']=False
        session['current_image_cifar']=get_random_image_cifar_real()    
    session.pop('current_image_cifar',None)
    

    return redirect(url_for('public.guessai_cifar'))

def get_random_image_cifar_ai():
    images_folder = os.path.join(current_app.root_path, 'static', 'cifar','FAKE')
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if image_files:
        random_image = random.choice(image_files)
        return f'/cifar/FAKE/{random_image}'
    else:
        return None
    
def get_random_image_cifar_real():
    images_folder = os.path.join(current_app.root_path, 'static', 'cifar','REAL')
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if image_files:
        random_image = random.choice(image_files)
        return f'/cifar/REAL/{random_image}'
    else:
        return None
    

