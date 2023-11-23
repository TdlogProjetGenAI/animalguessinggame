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

import os
from flask_login import login_required, login_user, logout_user

from animalguessinggame.extensions import login_manager
from animalguessinggame.public.forms import LoginForm
from animalguessinggame.user.forms import RegisterForm
from animalguessinggame.user.models import User
from animalguessinggame.utils import flash_errors
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

from .classif_animals10 import ResNetClassifier, classifie_animals10

blueprint = Blueprint("public", __name__, static_folder="../static")

# Assurez-vous d'importer StringField et SubmitField
# forms.py
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField

class GenerateImageForm(FlaskForm):
    username = StringField('Username')  # Vous pouvez personnaliser le libellé si nécessaire
    password = PasswordField('Password')  # Vous pouvez personnaliser le libellé si nécessaire
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


from flask import redirect, url_for, render_template


from flask import jsonify


from flask import render_template, request


import time


from flask import redirect


from flask import session


@blueprint.route('/generate_image/', methods=['GET', 'POST'])
def generate_image():
    form = GenerateImageForm()
    image_path = session.get('current_image', get_random_image_path())
    congratulations_message = None

    if form.validate_on_submit():
        prompt_value = form.prompt.data.lower()
        if prompt_value == classifie_animals10(image_path):
            congratulations_message = "Félicitations, vous avez gagné!"
        else:
            congratulations_message = "Essaie encore"
    session['current_image'] = image_path

    return render_template('public/image_page.html', image_path=image_path, congratulations_message=congratulations_message, form=form)

@blueprint.route('/replay/', methods=['GET'])
def replay():
    session.pop('current_image', None)
    return redirect(url_for('public.generate_image'))

@blueprint.route('/generate_number/', methods=['GET', 'POST'])
def generate_number():
    form = GenerateImageForm()
    images_list_path = session.get('current_images', get_random_number_path())
    congratulations_message = None


    if form.validate_on_submit():
        prompt_value = form.prompt.data.lower()
        if prompt_value == "chat":
            congratulations_message = "Félicitations, vous avez gagné!"
        else:
            congratulations_message = "Essaie encore"
    session['current_images'] = images_list_path

    return render_template('public/number_page.html', images_list_path=images_list_path, congratulations_message=congratulations_message, form=form)


@blueprint.route('/replay_number/', methods=['GET'])
def replay_number():
  
    session.pop('current_images', None)
    return redirect(url_for('public.generate_number'))


def get_random_image_path():
    images_folder = os.path.join(current_app.root_path, 'static', 'images_animals10')
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if image_files:
        random_image = random.choice(image_files)
        return f'/images_animals10/{random_image}'
    else:
        return None


def get_random_number_path():
    images_folder = os.path.join(current_app.root_path, 'static', 'images_number')
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images_list=[]
    if image_files:
        number_images=random.randint(1,4)
        for k in range(number_images):
            random_image = random.choice(image_files)
            images_list.append(f'/images_number/{random_image}')
        return images_list
    else:
        return None
########################



dict = {0: "chien", 1: "cheval", 
            2 : "elephant", 3: "papillon", 4: "poule", 
            5: "chat", 6: "vache", 7: "mouton", 8: "araignée", 
            9: "écureuil"}

def classifie_animals10(image_path):  
    
    model_chemin = os.path.join('animalguessinggame', 'models', 'classifierVF_animals10.pt')
    model=torch.load(model_chemin, map_location='cpu')
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
    
