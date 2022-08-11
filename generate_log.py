import os
import logging
from logging import handlers

logger = logging.getLogger()


def log_init(file_name):
    if not os.path.exists(file_name):
        file = open(file_name, mode='w')
        file.close()

    #  Create a handler to save log to a file
    fh = handlers.RotatingFileHandler(filename=file_name)
    #  Create a handler to print log to console
    ch = logging.StreamHandler()

    #  Set format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.setLevel(level=logging.INFO)
    #  Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
