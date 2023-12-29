# -*- coding: utf-8 -*-
"""Public section, including homepage and signup."""
from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for
)
import random
import __main__
import os
from flask_login import login_required, login_user, logout_user, current_user
from animalguessinggame.extensions import login_manager
from animalguessinggame.public.forms import LoginForm, GenerateImageFormIA, GenerateImageForm
from animalguessinggame.user.forms import RegisterForm
from animalguessinggame.user.models import User
from animalguessinggame.utils import flash_errors
from animalguessinggame.database import Score, ScoreHard, ScoreHardClock, ScoreNum
from PIL import Image
import torch
from flask import session
from .classif_animals10 import ResNetClassifier, classifie_animals10, classifie_animals90 
from .classif_animals10 import Classifier_mnist, VAE, classifie_mnist
from .levenstein import distance_levenstein
from werkzeug.utils import secure_filename

blueprint = Blueprint("public", __name__, static_folder="../static")


class Compt():
    """
    A simple class representing a global counter.

    It will be used for the storage of generated number in order to
    avoid the storage of all the data generated but just the useful one.

    Attributes:
    - k (int): The current value of the counter.

    Methods:
    - incr(): Increment the counter by 1.
    - value(): Get the current value of the counter.
    - value_to_zero(): Reset the counter to zero.
    """

    def __init__(self):
        """ Initialize the counter with a value of 0. """
        self.k = 0

    def incr(self):
        """ Increment the counter by 1. """
        self.k += 1

    def value(self):
        """
        Get the current value of the counter.

        Returns:
        - int: The current value of the counter.
        """
        return self.k

    def value_to_zero(self):
        """ Reset the counter to zero. """
        self.k = 0


setattr(__main__, "ResNetClassifier", ResNetClassifier)
setattr(__main__, "Classifier_mnist", Classifier_mnist)
setattr(__main__, "VAE", VAE)
setattr(__main__, "compteur", Compt)

@login_manager.user_loader
def load_user(user_id):
    """Load user by ID."""
    return User.get_by_id(int(user_id))


@blueprint.route("/", methods=["GET", "POST"])
def home():
    """Home page."""
    form = LoginForm(request.form)
    current_app.logger.info("Hello from the home page!")
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

# animals10
@blueprint.route('/generate_image/', methods=['GET', 'POST'])
def generate_image():
    """
    Draw a new image in a dataset of animals and allows the user to guess the name of the animal in the image.

    Supported HTTP Methods:
        - GET: Displays a new image.
        - POST: Validates the user-provided answer.

    Returns:
        flask.render_template: HTML page displaying the image, form, congratulations messages,
                              score, and top scores.
    """
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
    if attempts > 0 and form.validate_on_submit():
        prompt_value = form.prompt.data.lower()
        anws = classifie_animals10(image_path)
        if prompt_value == anws[0] or prompt_value == anws[1]:
            congratulations_message = "Félicitations, vous avez gagné !"
            win = True
            play_win_sound = True
            sound_file = f'sound_animals10/{anws[0]}.mp3'         
            if attempts == 3 and not played:
                score += 8
            elif attempts == 2 and not played:
                score += 5
            elif attempts == 1 and not played:
                score += 3
            played = True
        else:
            attempts -= 1
            if attempts == 0:
                congratulations_message = f"Dommage. La réponse était {anws[0]}."
                user_id = current_user.id if current_user.is_authenticated else "invite"
                new_score = Score(user_id=user_id, score_value=score)
                new_score.save()
                score = 0
                played = True
            elif (distance_levenstein(prompt_value, anws[0]) <= 2 or distance_levenstein(prompt_value, anws[1]) <= 2):
                congratulations_message = f"Tu chauffes ! Il vous reste {attempts} essais."
                play_win_sound = True
                sound_file = 'sound_animals10/tu_chauffes.mp3'
            else:
                congratulations_message = f"Essaie encore ! Il vous reste {attempts} essais."
                play_win_sound = True
                sound_file = 'sound_animals10/essaie_encore.mp3'
        top_scores = Score.get_top_scores()
    session['win'] = win
    session['played'] = played
    session['attempts'] = attempts
    session['current_image'] = image_path
    session['score'] = score

    return render_template('public/image_page.html', image_path=image_path,
                           congratulations_message=congratulations_message, form=form,
                           score=score, top_scores=top_scores, play_win_sound=play_win_sound,
                           sound_file=sound_file)


@blueprint.route('/replay/', methods=['GET'])
def replay():
    """
    Resets the session to allow the user to replay after completing a game.

    Returns:
        flask.redirect: Redirects to the 'public.generate_image' route.
    """
    win = session.get('win', False)
    if not win:
        user_id = current_user.id if current_user.is_authenticated else "invite"
        new_score = Score(user_id=user_id, score_value=session['score'])
        new_score.save()
        top_scores = Score.get_top_scores() # noqa
        session['score'] = 0

    session['attempts'] = 3
    session['current_image'] = get_random_image_path()
    session['win'] = False 
    session['played'] = False
    return redirect(url_for('public.generate_image'))

@blueprint.route('/liste_animals10', methods=['GET'])
def liste_animals10():
    """
    Displays a predefined list of 10 animals.

    Returns:
        flask.render_template: HTML page showing the list of 10 animals and the current score.
    """
    animals10_dict = {0: "chien", 1: "cheval", 2: "éléphant", 3: "papillon", 4: "poule", 5: "chat",
                      6: "vache", 7: "mouton", 8: "araignée", 9: "écureuil"}
    score = session.get('score', 0)
    return render_template('public/liste_animals10.html', animals10_dict=animals10_dict, score=score)

@blueprint.route('/top_scores/', methods=['GET'])
def top_scores():
    """
    Displays the top scores recorded in the database.

    Returns:
        flask.render_template: HTML page showing the top scores.
    """
    top_scores = Score.query.order_by(Score.score_value.desc()).limit(10).all()
    return render_template('public/top_scores.html', top_scores=top_scores)

# animals90

@blueprint.route('/generate_image_hard', methods=['GET', 'POST'])
def generate_image_hard():
    """
    Draw a new image in a larger dataset of animals and allows the user to guess the name of the animal in the image.

    Supported HTTP Methods:
        - GET: Displays a new image.
        - POST: Validates the user-provided answer.

    Returns:
        flask.render_template: HTML page displaying the image, form, congratulations messages,
                              score, and top scores.
    """
    form = GenerateImageForm()
    image_path = session.get('current_image_hard', get_random_image_hard_path())
    congratulations_message = None
    attempts = session.get('attempts_hard', 3)
    win = session.get('win', False)
    played = session.get('played', False)
    score = session.get('score', 0)
    top_scores = ScoreHard.get_top_scores()
    if attempts > 0 and form.validate_on_submit():
        prompt_value = form.prompt.data.lower()
        anws = classifie_animals90(image_path)
        if prompt_value == anws[0] or prompt_value == anws[1]:
            congratulations_message = "Félicitations, vous avez gagné !"
            win = True
            if attempts == 3 and not played:
                score += 8
            elif attempts == 2 and not played:
                score += 5
            elif attempts == 1 and not played:
                score += 3
            played = True

        else:
            attempts -= 1
            if attempts == 0:
                congratulations_message = f"Dommage. La réponse était {anws[0]}."
                user_id = current_user.id if current_user.is_authenticated else "invite"
                new_score = ScoreHard(user_id=user_id, score_value=score)
                new_score.save()
                score = 0
                played = True
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
    return render_template('public/image_page_hard.html', image_path=image_path,
                           congratulations_message=congratulations_message, form=form,
                           score=score, top_scores=top_scores)

@blueprint.route('/replay_hard', methods=['GET'])
def replay_hard():
    """
    Resets the session to allow the user to replay after completing a game.

    Returns:
        flask.redirect: Redirects to the 'public.generate_image_hard' route.
    """
    session.pop('current_image_hard', None)
    win = session.get('win', False)
    if not win:
        # L'utilisateur n'a pas encore gagné, réinitialisez le score
        user_id = current_user.id if current_user.is_authenticated else "invite"
        new_score = ScoreHard(user_id=user_id, score_value=session['score'])
        new_score.save()
        top_scores = ScoreHard.get_top_scores() # noqa
        session['score'] = 0

    session['attempts_hard'] = 3
    session['current_image'] = get_random_image_path()
    session['win'] = False  # Réinitialisez la variable win à False
    session['played'] = False
    return redirect(url_for('public.generate_image_hard'))

@blueprint.route('/liste_animals90', methods=['GET'])
def liste_animals90():
    """
    Displays a predefined list of 90 animals.

    Returns:
        flask.render_template: HTML page showing the list of 90 animals and the current score.
    """

    animals90 = [
        'abeille', 'aigle', 'âne', 'antilope', 'baleine', 'bécasse', 'blaireau', 'bison', 'boeuf',
        'calao', 'canard', 'cerf', 'chauve-souris', 'chat', 'chèvre', 'chenille', 'cheval', 'chien', 'chimpanzé',
        'chouette', 'cochon', 'coyote', 'crapaud', 'crabe', 'coccinelle', 'cygne', 'dauphin', 'dinde', 'écureuil',
        'éléphant', 'étoile de mer', 'flamant rose', 'fourmi', 'gazelle', 'girafe', 'gorille', 'guépard', 'guêpe',
        'hamster', 'hérisson', 'hippopotame', 'hirondelle', 'huître', 'hyène', 'kangourou', 'koala',
        'léopard', 'lion', 'loup', 'loutre', 'méduse', 'mille-pattes', 'mouche', 'mouton', 'moineau', 'moustique',
        'mule', 'opossum', 'orang-outan', 'oie', 'okapi', 'ours', 'panda', 'papillon', 'papillon de nuit', 'perroquet',
        'phoque', 'pic-vert', 'pigeon', 'pingouin', 'poisson rouge', 'porc-épic', 'ragondin', 'rat', 'raton laveur',
        'renard', 'renne', 'requin', 'salamandre', 'sanglier', 'scarabée', 'serpent', 'souris', 'sauterelle', 'singe',
        'tigre', 'tortue', 'vache', 'wombat', 'zèbre'
    ]
    animals90_dict = {i: animal for i, animal in enumerate(animals90)}
    return render_template('public/liste_animals90.html', animals90_dict=animals90_dict)

# ############Clock#####################

@blueprint.route('/reset_score_hard_clock', methods=['GET'])
def reset_score_hard_clock():
    """
    Resets the score when button "Réinitialiser Timer" is pressed.

    Returns:
        A modification to the page generate_image_hard_clock
    """
    session['score_clock'] = 0
    return redirect(url_for('public.generate_image_hard_clock'))


@blueprint.route('/generate_image_hard_clock', methods=['GET', 'POST'])
def generate_image_hard_clock():
    """
    Generates an image for the game, keeps track of the score, generates a new image when ann answer is submitted.

    Returns:
        A modification to the game page
    """
    form = GenerateImageForm()
    image_path = session.get('current_image_hard_clock', get_random_image_hard_path())
    congratulations_message = None
    win = session.get('win_clock', False)
    played = session.get('played_clock', False)
    score_clock = session.get('score_clock', 0)

    session['current_image_hard_clock'] = image_path
    if form.validate_on_submit():
        prompt_value = form.prompt.data.lower()
        anws = classifie_animals90(image_path)
        if prompt_value == anws[0] or prompt_value == anws[1]:
            congratulations_message = "Bonne réponse !"
            win = True

            score_clock += 1
            played = True
            session['current_image_hard_clock'] = get_random_image_hard_path()

        else:
            new_image_path = get_random_image_hard_path()
            congratulations_message = f"Dommage. La réponse était {anws[0]}."
            session['current_image_hard_clock'] = new_image_path

    session['win_clock'] = win
    session['played_clock'] = played
    session['score_clock'] = score_clock

    return render_template('public/image_page_hard_clock.html', image_path=session['current_image_hard_clock'],
                           congratulations_message=congratulations_message, form=form, score_clock=score_clock)

@blueprint.route('/replay_hard_clock', methods=['GET'])
def replay_hard_clock():
    """
    Updates the image when the "Changer d'image" button is pressed.

    Returns:
        _type_: _description_
    """
    session.pop('current_image_hard_clock', None)
    win = session.get('win_clock', False)
    if not win:
        # L'utilisateur n'a pas encore gagné, réinitialisez le score
        user_id = current_user.id if current_user.is_authenticated else "invite"
        new_score = ScoreHardClock(user_id=user_id, score_value=session['score_clock'])
        new_score.save()
        session['score_clock'] = 0

    session['current_image_hard_clock'] = get_random_image_path()
    session['win_clock'] = False  # Réinitialisez la variable win à False
    session['played_clock'] = False
    return redirect(url_for('public.generate_image_hard_clock'))

@blueprint.route('/upload_images_hard_clock', methods=['POST'])
def upload_images_hard_clock():
    """
    Handles image upload via a POST request. The images uploaded go in the animals10 dataset.

    Expects images in the 'images' field of the request. Displays flash messages for
    no files uploaded or all files having empty filenames. Valid files are saved to
    UPLOAD_FOLDER after security checks.

    Returns:
        flask.redirect: Redirects to 'public.generate_image' after processing the upload.

    Flash Messages:
        - 'Aucun fichier téléchargé': No files included in the request.
        - 'Aucun fichier sélectionné': All included files have empty filenames.
        - 'Images téléchargées avec succès': Successful upload of images.
    """
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

# ################number#######################
@blueprint.route('/generate_number/', methods=['GET', 'POST'])
def generate_number():
    """
    Generates a new number or draw a number in a dataset of numbers (according to the method choosen by the player)
    and allows the user to guess the name of the animal in the image.

    Supported HTTP Methods:
        - GET: Displays a new image.
        - POST: Validates the user-provided answer.

    Returns:
        flask.render_template: HTML page displaying the image, form, congratulations messages,
                              score, and top scores.
    """
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

    if attempts > 0 and form.validate_on_submit():
        prompt_value = form.prompt.data.lower()

        if prompt_value == anws[0] or prompt_value == anws[1]:
            congratulations_message = "Félicitations, vous avez gagné !"
            win = True

            if attempts == 3 and not played:
                score += 8
            elif attempts == 2 and not played:
                score += 5
            elif attempts == 1 and not played:
                score += 3
            played = True

        else:
            attempts -= 1
            if attempts == 0:
                congratulations_message = f"Dommage. La réponse était {anws[0]}."
                user_id = current_user.id if current_user.is_authenticated else "invite"
                new_score = ScoreNum(user_id=user_id, score_value=score)
                new_score.save()
                score = 0
                played = True
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
    return render_template('public/number_page.html', images_list_path=images_list_path,
                           congratulations_message=congratulations_message, form=form,
                           score=score, top_scores=top_scores)

@blueprint.route('/replay_number/', methods=['GET'])
def replay_number():
    """
    Resets the session to allow the user to replay after completing a game.

    Returns:
        flask.redirect: Redirects to the 'public.generate_number' route.
    """
    session['attempts_number'] = 3
    session.pop('current_images', None)
    win = session['win']
    if not win:
        # L'utilisateur n'a pas encore gagné, réinitialisez le score
        user_id = current_user.id if current_user.is_authenticated else "invite"
        new_score = ScoreNum(user_id=user_id, score_value=session['score'])
        new_score.save()
        top_scores = ScoreNum.get_top_scores() # noqa
        session['score'] = 0
    session['win'] = False  # Réinitialisez la variable win à False
    session['played'] = False
    return redirect(url_for('public.generate_number'))


@blueprint.route('/toggle_method/', methods=['POST'])
def toggle_method():
    """
    Toggles between two methods for either generating random numbers or drawing randomly numbers in a dataset in the
    session.

    The function switches between 'get_random_gen_number_path' and 'get_random_number_path' methods
    to display random numbers and updates the session accordingly.

    Returns:
        flask.redirect: Redirects to the previous page or the 'public.generate_number' route.
    """
    current_method_name = session.get('current_method', 'get_random_gen_number_path')
    if current_method_name == 'get_random_gen_number_path':
        session['current_method'] = 'get_random_number_path'
    else:
        session['current_method'] = 'get_random_gen_number_path'
    return redirect(request.referrer or url_for('public.generate_number'))


def get_random_image_path():
    """
    Returns a randomly chosen image path from the 'images_animals10' folder.

    The function looks for image files  in the
    'images_animals10' folder and returns a randomly selected image path.

    Returns:
        str or None: A string representing the path to a random image, or None if no images are found.
    """
    images_folder = os.path.join(current_app.root_path, 'static', 'images_animals10')
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if image_files:
        random_image = random.choice(image_files)
        return f'/images_animals10/{random_image}'
    else:
        return None

def get_random_image_hard_path():
    """
    Returns a randomly chosen image path from the 'images_animals90' folder.

    The function looks for image files (with extensions '.png', '.jpg', or '.jpeg') in the
    'images_animals90' folder and returns a randomly selected image path.

    Returns:
        str or None: A string representing the path to a random image, or None if no images are found.
    """
    images_folder = os.path.join(current_app.root_path, 'static', 'images_animals90')
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if image_files:
        random_image = random.choice(image_files)
        return f'/images_animals90/{random_image}'
    else:
        return None

def get_random_number_path():
    """
    Returns a list of randomly chosen image paths from the 'images_number' folder.

    The function looks for image files (with extensions '.png', '.jpg', or '.jpeg') in the
    'images_number' folder and returns a list of randomly selected image paths, with the
    number of images varying between 1 and 4.

    Returns:
        list or None: A list of strings representing paths to random images, or None if no images are found.
    """
    images_folder = os.path.join(current_app.root_path, 'static', 'images_number')
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images_list = []
    if image_files:
        number_images = random.randint(1, 4)
        for k in range(number_images):
            random_image = random.choice(image_files)
            images_list.append(f'/images_number/{random_image}')
        return images_list
    else:
        return None

def number_path():
    """
    Returns a randomly chosen image path from the 'images_number' folder.

    The function looks for image files (with extensions '.png', '.jpg', or '.jpeg') in the
    'images_number' folder and returns a randomly selected image path.

    Returns:
        str or None: A string representing the path to a random image, or None if no images are found.
    """
    images_folder = os.path.join(current_app.root_path, 'static', 'images_number')
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if image_files:
        random_image = random.choice(image_files)
        return f'images_number/{random_image}'
    else:
        return None

def get_random_gen_number_path():
    """
    Generates a list of randomly generated image paths using the 'gen_number_path' function.

    Returns:
        list: A list of strings representing randomly generated image paths.
    """
    images_list = []
    number_images = random.randint(1, 4)
    for k in range(number_images):
        random_image_path = gen_number_path()
        images_list.append(random_image_path)
    return images_list


k = Compt()


def gen_number_path():
    """
    Generates a random image using a Variational Autoencoder (VAE) and a classifier model.

    This function employs a Variational Autoencoder (VAE) to generate a random image in a latent space.
    The generated image is then classified using a pre-trained classifier model. The classifier assigns
    confidence scores to each potential image, and the image with the highest confidence score is selected.

    Returns:
        str: The path to the generated image.
    """
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
    """
    Check if the given filename has an allowed extension.

    Parameters:
        filename (str): The name of the file to be checked.

    Returns:
        bool: True if the filename has an allowed extension, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@blueprint.route('/upload_images', methods=['POST'])
def upload_images():
    """
    Handles image upload via a POST request.

    The images uploaded go in the animals10 dataset.
    Expects images in the 'images' field of the request. Displays flash messages for
    no files uploaded or all files having empty filenames. Valid files are saved to
    UPLOAD_FOLDER after security checks.

    Returns:
        flask.redirect: Redirects to 'public.generate_image' after processing the upload.

    Flash Messages:
        - 'Aucun fichier téléchargé': No files included in the request.
        - 'Aucun fichier sélectionné': All included files have empty filenames.
        - 'Images téléchargées avec succès': Successful upload of images.
    """
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
    """
    Handles image upload via a POST request.

    The images uploaded go in the animals90 dataset.
    Expects images in the 'images' field of the request. Displays flash messages for
    no files uploaded or all files having empty filenames. Valid files are saved to
    UPLOAD_FOLDER after security checks.

    Returns:
        flask.redirect: Redirects to 'public.generate_image_hard' after processing the upload.
    """
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
    """
    Handles image upload via a POST request.

    The images uploaded go in the number dataset.

    Expects images in the 'images' field of the request. Displays flash messages for
    no files uploaded or all files having empty filenames. Valid files are saved to
    UPLOAD_FOLDER after security checks.

    Returns:
        flask.redirect: Redirects to 'public.generate_number' after processing the upload.
    """
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
    """
    Handles the guessing game for AI-generated images of numbers.

    Displays an image and allows users to guess whether it was generated by AI.
    Compares the user's guess with the ground truth and provides a congratulations message.

    Returns:
        flask.render_template: Renders the 'public/guessai.html' template.
    """
    form = GenerateImageFormIA()
    ai = session.get('AI')
    if ai is None:
        ai = random.choice([True, False])
        session['AI'] = ai

    image_path = session.get('current_image_guessai')
    if image_path is None:
        image_path = gen_number_path() if ai else number_path()
        session['current_image_guessai'] = image_path

    congratulations_message = None

    if form.validate_on_submit():
        is_ia = form.is_ia.data  
        if ai:
            if is_ia:
                congratulations_message = "Félicitations ! C'était de IA"
            else:      
                congratulations_message = "Perdu ! C'était de IA "
        else:
            if is_ia:
                congratulations_message = "Perdu ! Ce n'était pas de IA"
            else:
                congratulations_message = "Félicitations ! Ce n'était pas de IA"

    session['current_image_guessai'] = image_path
    return render_template('public/guessai.html', image_path=image_path,
                           congratulations_message=congratulations_message, form=form)

@blueprint.route('/replay_guessai/', methods=['GET'])
def replay_guessai():
    """
    Initiates a new round of the guessing game.

    Generates a new AI status and selects a random image for the next round.

    Returns:
        flask.redirect: Redirects to the 'public.guessai' route.
    """
    ai = random.choice([True, False])
    session['AI'] = ai

    if ai:
        session['current_image_guessai'] = gen_number_path()
    else:
        session['current_image_guessai'] = number_path()

    return redirect(url_for('public.guessai'))

@blueprint.route('/guessai_hard/', methods=['GET', 'POST'])
def guessai_hard():
    """
    Handles the guessing game for AI-generated every-day life images and paintings.

    Displays an image and allows users to guess whether it was generated by AI.
    Compares the user's guess with the ground truth and provides a congratulations message.

    Returns:
        flask.render_template: Renders the 'public/guessai_hard.html' template.
    """
    form = GenerateImageFormIA()

    ai = session.get('AI_hard')
    if ai is None:
        ai = random.choice([True, False])
        session['AI_hard'] = ai

    image_path = session.get('current_image_guess_hard')
    if image_path is None:
        image_path = get_random_image_hard_ai() if ai else get_random_image_hard_real()
        session['current_image_guess_hard'] = image_path

    congratulations_message = None

    if form.validate_on_submit():
        is_ia = form.is_ia.data  

        ground_truth = session.get('ground_truth_hard')
        if ground_truth is None:
            ground_truth = ai
            session['ground_truth_hard'] = ground_truth

        if is_ia == ground_truth:
            congratulations_message = "Gagné ! C'était de l'IA." if ground_truth else "Gagné ! Ce n'était pas de l'IA."
        else:
            congratulations_message = "Perdu ! C'était de l'IA." if ground_truth else "Perdu ! Ce n'était pas de l'IA."

        session.pop('current_image_guess_hard', None)

    return render_template('public/guessai_hard.html', image_path=image_path,
                           congratulations_message=congratulations_message, form=form)

@blueprint.route('/replay_new_game_hard/', methods=['GET'])
def replay_guessai_hard():
    """
    Initiates a new round of the guessing game.

    Generates a new AI status and selects a random image for the next round.

    Returns:
        flask.redirect: Redirects to the 'public.guessai_hard' route.
    """
    ai = random.choice([True, False])
    session['AI_hard'] = ai

    if ai:
        session['current_image_guess_hard'] = get_random_image_hard_ai()
    else:
        session['current_image_guess_hard'] = get_random_image_hard_real()

    session.pop('ground_truth_hard', None)

    return redirect(url_for('public.guessai_hard'))

def get_random_image_hard_ai():
    """
    Get a random image path from the 'FAKE' folder for the AI mode.

    Returns:
        str: The path to a random image generated by AI in the 'FAKE' folder.
            Returns None if no AI-generated images are available.
    """
    images_folder = os.path.join(current_app.root_path, 'static', 'IA_notIA', 'FAKE')
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.JPEG'))]

    if image_files:
        random_image = random.choice(image_files)
        return f'/IA_notIA/FAKE/{random_image}'
    else:
        return None

def get_random_image_hard_real():
    """
    Get a random image path from the 'REAL' folder for the real mode.

    Returns:
        str: The path to a random image not generated by AI in the 'REAL' folder.
            Returns None if no real player images are available.
    """
    images_folder = os.path.join(current_app.root_path, 'static', 'IA_notIA', 'REAL')
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.JPEG'))]

    if image_files:
        random_image = random.choice(image_files)
        return f'/IA_notIA/REAL/{random_image}'
    else:
        return None
