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
# Importez le module render_template depuis Flask
from flask import render_template, request

# Importez le module render_template depuis Flask
from flask import render_template, request

@blueprint.route('/generate_image/', methods=['GET', 'POST'])
def generate_image():
    
    if request.method == 'POST':
        prompt_value = request.form.get('prompt', '')
        if prompt_value.lower() == "chat":
            return render_template('public/image_page.html', image_path=image_path, congratulations_message="Félicitations, vous avez gagné!")

    
    image_path = get_random_image_path()
    return render_template('public/image_page.html', image_path=image_path)


def get_random_image_path():
    images_folder = os.path.join(current_app.root_path, 'static', 'images')
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if image_files:
        random_image = random.choice(image_files)
        return f'/images/{random_image}'
    else:
        return None
    
def animal(): 
    dict = {0: "chien", 1: "cheval", 
            2 : "elephant", 3: "papillon", 4: "poule", 
            5: "chat", 6: "vache", 7: "mouton", 8: "araignée", 
            9: "écureuil"}
    k=random.randint(0,9)
    return(dict[k])