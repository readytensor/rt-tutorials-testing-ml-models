import logging
import traceback


def get_logger(log_file_path: str, task_name: str, reset_file: bool = False) -> logging.Logger:
    """
    Returns a logger object with handlers to log messages to both the console and a specified log file.

    Args:
        log_file_path (str): The file path to write the log messages to.
        task_name (str): The name of the task to include in the log messages.
        reset_file (bool): If True, clears the log file when initializing the logger.

    Returns:
        logging.Logger: A logger object with the specified handlers.
    """
    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # Clear the log file if reset_file is set to True
    if reset_file:
        open(log_file_path, "w", encoding="utf-8").close()
    
    mode = "a"
    log_file_handler = logging.FileHandler(log_file_path, mode=mode)
    log_file_handler.setLevel(logging.INFO)
    log_file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(log_file_handler)

    return logger


def log_error(message: str, error: Exception, error_fpath: str) -> None:
    """
    Log an error message and write the error to a specified file
    Args:
        message (str): The error message.
        error (Exception): The exception instance.
        error_fpath (str, optional): The file path to write the error message to. Defaults to None.
    """
    with open(error_fpath, 'w', encoding='utf-8') as file:
        err_msg = f"{message} Error: {str(error)}"
        file.write(err_msg)
        file.write('\n')
        traceback_msg = ''.join(traceback.format_exception(etype=type(error), value=error, tb=error.__traceback__))
        file.write(traceback_msg)


def close_handlers(logger: logging.Logger):
    """
    Closes the handlers of a logger object to allow a log file to be deleted.

    Args:
        logger (logging.Logger): The logger object to close handlers for.
    """
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
