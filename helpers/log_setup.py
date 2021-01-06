import logging
from logging.handlers import TimedRotatingFileHandler
FORMAT = '%(asctime)s %(funcName)s %(lineno)d %(levelname)s - %(message)s'
log_format = logging.Formatter(FORMAT)

def get_logger(name, file_path):
    logger = logging.getLogger(name)
    file_handler = TimedRotatingFileHandler(file_path, when='D')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    # logger.addHandler(logging.StreamHandler())  - For adding STDOUT as well 
    return logger
