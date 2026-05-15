import logging
import os


def get_logger(name: str, log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt_file = logging.Formatter('%(asctime)s  %(levelname)-5s  %(message)s', datefmt='%H:%M:%S')
    fmt_con = logging.Formatter('%(message)s')

    fh = logging.FileHandler(os.path.join(log_dir, f'{name}.log'), mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_con)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
