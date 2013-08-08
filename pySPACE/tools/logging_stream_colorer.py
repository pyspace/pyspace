""" This module contains a class to change the color of the logger output.

It has to be used with the logger class.

Example of usage:

.. code-block:: python

    # Custom logger class with multiple destinations
    class ColoredLogger(logging.Logger):
        FORMAT = "[$BOLD%(name)-20s$RESET][%(levelname)-18s]  %(message)s ($BOLD%(filename)s$RESET:%(lineno)d)"
        COLOR_FORMAT = formatter_message(FORMAT, True)
        
        def __init__(self, name):
            logging.Logger.__init__(self, name, logging.DEBUG)
    
            color_formatter = ColoredFormatter(self.COLOR_FORMAT)
    
            console = logging.StreamHandler()
            console.setFormatter(color_formatter)
    
            self.addHandler(console)
            return
    
    
    logging.setLoggerClass(ColoredLogger)


Adopted version of http://stackoverflow.com/questions/384076/how-can-i-make-the-python-logging-output-to-be-colored .
This is distributed under creative commons: http://creativecommons.org/licenses/by-sa/3.0/

:Author: Hendrik Woehrle (hendrik.woehrle@dfki.de)
:Created: 2011/03/22
"""

import logging

# These are the supported colors
class COLORS:
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)


# The background is set with 40 plus the number of the color, and the foreground with 30

# These are the sequences needed to get colored output
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

LEVEL_COLOR = {
    'WARNING': COLORS.YELLOW,
    'INFO': COLORS.GREEN,
    'DEBUG': COLORS.WHITE,
    'CRITICAL': COLORS.YELLOW,
    'ERROR': COLORS.RED
}

def formatter_message(message, use_color = True):
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message

class ColoredLevelFormatter(logging.Formatter):
    
    def __init__(self, msg, use_color = True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in LEVEL_COLOR:
            levelname_color = COLOR_SEQ % (30 + LEVEL_COLOR[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)
    
    
class ColorFormatter(logging.Formatter):
    
    def __init__(self, msg, color = COLORS.WHITE):
        logging.Formatter.__init__(self, msg)
        self.color = color

    def format(self, record):
        msg = record.msg
        msg_color = COLOR_SEQ % (30 + self.color) + msg + RESET_SEQ
        record.msg = msg_color
        return logging.Formatter.format(self, record)
