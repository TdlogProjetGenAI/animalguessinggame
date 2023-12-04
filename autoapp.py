# -*- coding: utf-8 -*-
"""Create an application instance."""
from animalguessinggame.app import create_app
#from watchdog.events import EVENT_TYPE_OPENED

app = create_app()
# UPLOAD_FOLDER = 'animalguessinggame/static/images_animals10'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER