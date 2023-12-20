# -*- coding: utf-8 -*-
"""Public forms."""
from flask_wtf import FlaskForm
from wtforms import PasswordField, StringField, SubmitField
from wtforms.validators import DataRequired
from animalguessinggame.user.models import User


class LoginForm(FlaskForm):
    """Login form."""

    username = StringField("Username", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])

    def __init__(self, *args, **kwargs):
        """Create instance."""
        super(LoginForm, self).__init__(*args, **kwargs)
        self.user = None

    def validate(self, **kwargs):
        """Validate the form."""
        initial_validation = super(LoginForm, self).validate()
        if not initial_validation:
            return False

        self.user = User.query.filter_by(username=self.username.data).first()
        if not self.user:
            self.username.errors.append("Unknown username")
            return False

        if not self.user.check_password(self.password.data):
            self.password.errors.append("Invalid password")
            return False

        if not self.user.active:
            self.username.errors.append("User not activated")
            return False
        return True


class GenerateImageForm_IA(FlaskForm):
    is_ia = SubmitField('IA')
    username = StringField('Username') 
    password = PasswordField('Password')  
    prompt = StringField('Prompt')
    submit = SubmitField('Soumettre')

class GenerateImageForm(FlaskForm):
    username = StringField('Username') 
    password = PasswordField('Password')  
    prompt = StringField('Prompt')
    submit = SubmitField('Soumettre')