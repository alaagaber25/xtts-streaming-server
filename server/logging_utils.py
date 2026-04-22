import logging


LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def get_logger(name: str = "xtts") -> logging.Logger:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
