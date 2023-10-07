import logging


def get_logger(name, level=logging.INFO):
    # Create a logger
    logger = logging.getLogger(name)

    # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(level)

    # Create a console handler and set the level
    console_handler = logging.StreamHandler()

    # Create a formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)
    return logger
