# -*- coding: utf-8 -*-
"""Create an application instance."""
from animalguessinggame.app import create_app
#from watchdog.events import EVENT_TYPE_OPENED
from flask import Flask

app = create_app()


