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

# Exemple dans votre view.py

from flask import redirect, url_for, render_template

# Ajoutez cela à votre fichier view.py
from flask import jsonify

# Importez les modules nécessaires
from flask import jsonify

# ...

# Importez le module render_template depuis Flask
from flask import render_template, request
# @blueprint.route('/generate_number', methods=['GET', 'POST'])
# def generate_number():
#     #images_list_path = get_random_number_path()
#     return render_template('public/number_page.html')

# from flask_wtf import FlaskForm
# from wtforms import StringField, SubmitField

# class GenerateImageForm(FlaskForm):
#     prompt = StringField('Prompt')
#     submit = SubmitField('Soumettre')

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
        if prompt_value == "chat":
            congratulations_message = "Félicitations, vous avez gagné!"
        else:
            congratulations_message = "Essaie encore"
    session['current_image'] = image_path

    return render_template('public/image_page.html', image_path=image_path, congratulations_message=congratulations_message, form=form)

@blueprint.route('/replay/', methods=['GET'])
def replay():
    # Réinitialisez la variable de session pour le prompt et générez une nouvelle image
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
    # Réinitialisez la variable de session pour la liste d'images et générez une nouvelle liste
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
    
def animal(): 
    dict = {0: "chien", 1: "cheval", 
            2 : "elephant", 3: "papillon", 4: "poule", 
            5: "chat", 6: "vache", 7: "mouton", 8: "araignée", 
            9: "écureuil"}
    k=random.randint(0,9)
    return(dict[k])

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
    
