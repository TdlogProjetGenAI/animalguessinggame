# -*- coding: utf-8 -*-
"""Database module, including the SQLAlchemy database object and DB-related utilities."""
from typing import Optional, Type, TypeVar

from .compat import basestring
from .extensions import db

T = TypeVar("T", bound="PkModel")

# Alias common SQLAlchemy names
Column = db.Column
relationship = db.relationship


class CRUDMixin(object):
    """Mixin that adds convenience methods for CRUD (create, read, update, delete) operations."""

    @classmethod
    def create(cls, **kwargs):
        """Create a new record and save it the database."""
        instance = cls(**kwargs)
        return instance.save()

    def update(self, commit=True, **kwargs):
        """Update specific fields of a record."""
        for attr, value in kwargs.items():
            setattr(self, attr, value)
        if commit:
            return self.save()
        return self

    def save(self, commit=True):
        """Save the record."""
        db.session.add(self)
        if commit:
            db.session.commit()
        return self

    def delete(self, commit: bool = True) -> None:
        """Remove the record from the database."""
        db.session.delete(self)
        if commit:
            return db.session.commit()
        return


class Model(CRUDMixin, db.Model):
    """Base model class that includes CRUD convenience methods."""

    __abstract__ = True


class PkModel(Model):
    """Base model class that includes CRUD convenience methods, plus adds a 'primary key' column named ``id``."""

    __abstract__ = True
    id = Column(db.Integer, primary_key=True)

    @classmethod
    def get_by_id(cls: Type[T], record_id) -> Optional[T]:
        """Get record by ID."""
        if any(
            (
                isinstance(record_id, basestring) and record_id.isdigit(),
                isinstance(record_id, (int, float)),
            )
        ):
            return cls.query.get(int(record_id))
        return None

class Score(db.Model):
    """
    Model for storing user scores.

    Attributes:
    - id (int): Primary key for the score.
    - user_id (int): Foreign key referencing the user.
    - score_value (int): The value of the user's score.
    """
    __tablename__ = 'scores'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    score_value = db.Column(db.Integer)

    user = db.relationship('User', backref='scores', lazy=True)

    def __init__(self, user_id, score_value):
        """
        Initialize a new score.

        Parameters:
        - user_id (int): The user's ID.
        - score_value (int): The value of the user's score.
        """
        self.user_id = user_id
        self.score_value = score_value

    def save(self):
        """
        Save the score to the database.
        """
        db.session.add(self)
        db.session.commit()
        
    @staticmethod
    def get_top_scores(limit=10):
        """
        Retrieve the top scores.

        Parameters:
        - limit (int): The maximum number of scores to retrieve (default is 10).

        Returns:
        - list of Score: A list of Score objects representing the top scores, ordered by score value in descending 
        order.
        """
        return Score.query.order_by(Score.score_value.desc()).limit(limit).all()

class ScoreHard(db.Model):
    """
    Model for storing user scores.

    Attributes:
    - id (int): Primary key for the score.
    - user_id (int): Foreign key referencing the user.
    - score_value (int): The value of the user's score.
    """
    __tablename__ = 'scores_hard'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    score_value = db.Column(db.Integer)

    user = db.relationship('User', backref='scores_hard', lazy=True)

    def __init__(self, user_id, score_value):
        """
        Initialize a new score.

        Parameters:
        - user_id (int): The user's ID.
        - score_value (int): The value of the user's score.
        """
        self.user_id = user_id
        self.score_value = score_value

    def save(self):
        """
        Save the score to the database.
        """
        db.session.add(self)
        db.session.commit()

    @staticmethod
    def get_top_scores(limit=10):
        """
        Retrieve the top scores.

        Parameters:
        - limit (int): The maximum number of scores to retrieve (default is 10).

        Returns:
        - list of Score: A list of Score objects representing the top scores, ordered by score value in descending 
        order.
        """
        return ScoreHard.query.order_by(ScoreHard.score_value.desc()).limit(limit).all()
    
# ###### Score Clock #########
class ScoreHardClock(db.Model):
    """
    Model for storing user scores.

    Attributes:
    - id (int): Primary key for the score.
    - user_id (int): Foreign key referencing the user.
    - score_value (int): The value of the user's score.
    """
    __tablename__ = 'scores_hard_clock'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    score_value = db.Column(db.Integer)

    user = db.relationship('User', backref='scores_hard_clock', lazy=True)

    def __init__(self, user_id, score_value):
        """
        Initialize a new score.

        Parameters:
        - user_id (int): The user's ID.
        - score_value (int): The value of the user's score.
        """
        self.user_id = user_id
        self.score_value = score_value

    def save(self):
        """
        Save the score to the database.
        """
        db.session.add(self)
        db.session.commit()

    @staticmethod
    def get_top_scores(limit=10):
        """
        Retrieve the top scores.

        Parameters:
        - limit (int): The maximum number of scores to retrieve (default is 10).

        Returns:
        - list of Score: A list of Score objects representing the top scores, ordered by score value in descending 
        order.
        """
        return ScoreHardClock.query.order_by(ScoreHardClock.score_value.desc()).limit(limit).all()    

class ScoreNum(db.Model):
    """
    Model for storing user scores.

    Attributes:
    - id (int): Primary key for the score.
    - user_id (int): Foreign key referencing the user.
    - score_value (int): The value of the user's score.
    """
    __tablename__ = 'scores_num'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    score_value = db.Column(db.Integer)

    user = db.relationship('User', backref='scores_num', lazy=True)

    def __init__(self, user_id, score_value):
        """
        Initialize a new score.

        Parameters:
        - user_id (int): The user's ID.
        - score_value (int): The value of the user's score.
        """
        self.user_id = user_id
        self.score_value = score_value

    def save(self):
        """
        Save the score to the database.
        """
        db.session.add(self)
        db.session.commit()

    @staticmethod
    def get_top_scores(limit=10):
        """
        Retrieve the top scores.

        Parameters:
        - limit (int): The maximum number of scores to retrieve (default is 10).

        Returns:
        - list of Score: A list of Score objects representing the top scores, ordered by score value in descending 
        order.
        """
        return ScoreNum.query.order_by(ScoreNum.score_value.desc()).limit(limit).all()

def reference_col(
    tablename, nullable=False, pk_name="id", foreign_key_kwargs=None, column_kwargs=None
):
    """Column that adds primary key foreign key reference.

    Usage: ::

        category_id = reference_col('category')
        category = relationship('Category', backref='categories')
    """
    foreign_key_kwargs = foreign_key_kwargs or {}
    column_kwargs = column_kwargs or {}

    return Column(
        db.ForeignKey(f"{tablename}.{pk_name}", **foreign_key_kwargs),
        nullable=nullable,
        **column_kwargs,
    )
