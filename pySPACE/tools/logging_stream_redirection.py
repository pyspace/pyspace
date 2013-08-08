""" This module contains a class to redirect the logging output from one logger to another one.

It has to be used with the logger class.


:Author: Hendrik Woehrle (hendrik.woehrle@dfki.de)
:Created: 2011/12/05
"""

import logging

class RedirectionHandler(logging.Handler):
    
    def __init__(self, destination):
        logging.Handler.__init__(self)
        self.destination = destination

    def emit(self, record):
        self.destination.handle(record)



